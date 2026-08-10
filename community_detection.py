"""Community Detection on Sample Social Graph via NetworkX and community-louvain"""
import re
import json
import csv
import hashlib
from collections import defaultdict, Counter
from urllib.parse import quote, unquote
import requests
import os
from typing import Dict, Any, List
from pyvis.network import Network
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from node2vec import Node2Vec
from cleanup_root import clean_project_root
from community_naming import (
    DEFAULT_COMMUNITY_LABEL,
    build_community_classification_prompt,
    build_community_profile,
    coerce_allowed_label,
    get_community_label_color,
    parse_community_classification_response,
)
from elastic_query import DEFAULT_LOOKBACK_DAYS
from llm_client import call_llm_with_fallback


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


COMMUNITY_RANDOM_SEED = env_int("COMMUNITY_RANDOM_SEED", 42)


ENABLE_LLM_NAMING = env_flag("ENABLE_LLM_NAMING", default=True)
ENABLE_NAMING_TEXT_FETCH = env_flag(
    "ENABLE_NAMING_TEXT_FETCH",
    default=ENABLE_LLM_NAMING,
)

# ---- Echo Chamber Metrics ----

def compute_ei_index(g, community_nodes):
    """
    E-I Index = (E - I) / (E + I)
    E: edges from community to outside
    I: edges within community
    Range: [-1, +1]
    """
    internal_edges = 0
    external_edges = 0
    community_set = set(community_nodes)

    for u in community_nodes:
        for v in g.neighbors(u):
            if v in community_set:
                internal_edges += 1
            else:
                external_edges += 1

    # each internal edge counted twice
    internal_edges /= 2
    total = internal_edges + external_edges
    if total == 0:
        return 0.0
    return (external_edges - internal_edges) / total


def compute_conductance(g, community_nodes):
    """
    Conductance = cut(S, V-S) / min(vol(S), vol(V-S))
    Lower value => more isolated community
    """
    community_set = set(community_nodes)
    cut_edges = 0
    vol_s = 0
    vol_rest = 0

    for u in g.nodes():
        deg_u = g.degree(u)
        if u in community_set:
            vol_s += deg_u
            for v in g.neighbors(u):
                if v not in community_set:
                    cut_edges += 1
        else:
            vol_rest += deg_u

    denom = min(vol_s, vol_rest)
    if denom == 0:
        return 0.0
    return cut_edges / denom


DEFAULT_ECHO_THRESHOLDS = {
    # Structural isolation
    "ei_index_max": -0.2,   # lower (more negative) => more internal ties
    "conductance_max": 0.3, # lower => more isolated
    # Content homogeneity
    "homogeneity_min": 0.6, # dominant stance share
    # Size guardrail
    "min_size": 5,
}


def compute_content_homogeneity(stance_info):
    """Return dominant stance share based on stance_info counts."""
    if not stance_info:
        return None
    counts = stance_info.get("counts") or {}
    total = counts.get("total", 0) or 0
    if total <= 0:
        return None
    return max(counts.get("pos", 0), counts.get("neg", 0), counts.get("neu", 0)) / total


def classify_echo_chamber(ei_index, conductance, homogeneity, size, thresholds=None):
    """Classify echo chamber based on structural + content thresholds."""
    th = dict(DEFAULT_ECHO_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    if size is None or size < th["min_size"]:
        return False
    if ei_index is None or conductance is None:
        return False
    if ei_index > th["ei_index_max"]:
        return False
    if conductance > th["conductance_max"]:
        return False
    if homogeneity is None or homogeneity < th["homogeneity_min"]:
        return False
    return True


def compute_temporal_stability(prev_partition, curr_partition):
    """
    Compute AMI and NMI between two partitions
    """
    common_nodes = set(prev_partition.keys()) & set(curr_partition.keys())
    if len(common_nodes) < 2:
        return {"AMI": None, "NMI": None}

    prev_labels = [prev_partition[n] for n in common_nodes]
    curr_labels = [curr_partition[n] for n in common_nodes]

    return {
        "AMI": adjusted_rand_score(prev_labels, curr_labels),
        "NMI": normalized_mutual_info_score(prev_labels, curr_labels)
    }
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from elastic_client import es, INDEX


def fetch_community_texts(accounts: List[str], start_date: str = None,
                        end_date: str = None, max_texts: int = 10) -> Dict[str, List[str]]:
    """Fetch recent texts for a list of accounts from Elasticsearch.

    Robust behavior:
    - Reuse `es` and `INDEX` from elastic_client.py
    - Try username fields: user_name.keyword, user_name, sender.keyword, sender
    - Prefer `normalized_text` then fallback to `text`/`content`
    - Try sorting by `date` but fall back to unsorted if mapping missing
    - Print a 3-item debug preview for the first account (center) so we can verify
    """
    try:
        es_client = es
    except Exception:
        raise RuntimeError("Elasticsearch client `es` not available from elastic_client.py")

    index = INDEX
    username_fields = ["user_name.keyword", "user_name", "sender.keyword", "sender"]
    results: Dict[str, List[str]] = {acct: [] for acct in accounts}

    for i, acct in enumerate(accounts):
        hits = []
        for field in username_fields:
            if field.endswith('.keyword'):
                body = {"query": {"term": {field: acct}}}
            else:
                body = {"query": {"match": {field: acct}}}

            # apply date range if provided
            if start_date and end_date:
                body = {"bool": {"must": [body["query"]], "filter": [{"range": {"date": {"gte": start_date, "lte": end_date}}}]}}

            # try sorted query first, then fallback to unsorted if ES complains
            try:
                body_with_sort = dict(body)
                body_with_sort["sort"] = [{"date": {"order": "desc"}}]
                resp = es_client.search(index=index, body=body_with_sort, size=max_texts)
            except Exception:
                try:
                    resp = es_client.search(index=index, body=body, size=max_texts)
                except Exception:
                    resp = {"hits": {"hits": []}}

            hits = resp.get("hits", {}).get("hits", [])
            if hits:
                break

        tweets = []
        for h in hits:
            src = h.get("_source", {}) or {}
            text = src.get("normalized_text") or src.get("text") or src.get("content") or ""
            if text and isinstance(text, str):
                tweets.append(text.strip())

        results[acct] = tweets[:max_texts]

        # debug preview for the first account (likely the center)
        if i == 0:
            print(f"[debug] Sample normalized tweets for {acct} (count={len(tweets)}):")
            for t in tweets[:3]:
                print("   →", t[:200])

    return results


_ai_name_cache = {}
META_FIELDS = {
    "political_label": "گرایش سیاسی",
    "category": "موضوع",
    "emotion": "احساس",
    "sentiment": "حس",
    "language": "زبان"
}

STANCE_TARGETS = {
    "رهبری": ["رهبری", "رهبر", "خامنه", "khamenei", "khamenei_ir"]
}

SENTIMENT_MAP = {
    "positive": "pos",
    "pos": "pos",
    "مثبت": "pos",
    "negative": "neg",
    "neg": "neg",
    "منفی": "neg",
    "neutral": "neu",
    "neu": "neu",
    "خنثی": "neu",
    "مختلط": "neu"
}

POS_CUES = [
    "حامی", "حمایت", "زنده باد", "درود", "قهرمان", "افتخار",
    "عالی", "خوب", "مثبت", "درست", "حق", "همراهی"
]

NEG_CUES = [
    "مرگ بر", "لعنت", "نفرت", "ننگ", "خائن", "دیکتاتور",
    "بد", "افتضاح", "فاسد", "کثیف", "منفور", "مجرم",
    "محاکمه", "اعدام", "اعتراض"
]


def iter_res_documents(file_path="res.json"):
    """Yield docs from res.json (supports JSON array or JSON lines)."""
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = None
        while first_char is None:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
        f.seek(0)
        if first_char == "[":
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            if isinstance(data, list):
                for doc in data:
                    if isinstance(doc, dict):
                        yield doc
        else:
            for line in f:
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(doc, dict):
                    yield doc


def normalize_sentiment(value):
    """Normalize sentiment labels to pos/neg/neu."""
    if not value:
        return None
    return SENTIMENT_MAP.get(str(value).strip().lower())


def infer_sentiment_from_text(text):
    """Lightweight lexicon-based sentiment inference."""
    if not text:
        return None
    lower_text = str(text).lower()
    pos_hits = sum(1 for cue in POS_CUES if cue in lower_text)
    neg_hits = sum(1 for cue in NEG_CUES if cue in lower_text)
    if pos_hits == 0 and neg_hits == 0:
        return "neu"
    if pos_hits > neg_hits:
        return "pos"
    if neg_hits > pos_hits:
        return "neg"
    return "neu"


def load_user_context(file_path="res.json"):
    """Load user-level metadata and stance signals from res.json."""
    if not os.path.exists(file_path):
        print(f"[meta] {file_path} not found; skipping metadata load.")
        return {}, {}

    meta_counts = defaultdict(lambda: {field: Counter() for field in META_FIELDS})
    stance_map = defaultdict(lambda: defaultdict(Counter))
    for doc in iter_res_documents(file_path):
        username = doc.get("user_name") or doc.get("sender")
        if not username:
            continue
        for field in META_FIELDS:
            value = doc.get(field)
            if value:
                val_norm = str(value).strip()
                if val_norm.lower() in {"unknown", "نامشخص", "none", "null"}:
                    continue
                meta_counts[username][field][val_norm] += 1

        text = doc.get("normalized_text") or doc.get("content") or ""
        sentiment_raw = doc.get("sentiment") or doc.get("sentiment.label")
        sentiment = normalize_sentiment(sentiment_raw)
        if not sentiment:
            sentiment = infer_sentiment_from_text(text)
        if text and sentiment:
            stance_map[username]["__topic__"][sentiment] += 1
            lower_text = str(text).lower()
            for target, keywords in STANCE_TARGETS.items():
                if any(k.lower() in lower_text for k in keywords):
                    stance_map[username][target][sentiment] += 1

    meta_map = {}
    for username, counts in meta_counts.items():
        meta_map[username] = {
            field: (counter.most_common(1)[0][0] if counter else None)
            for field, counter in counts.items()
        }

    print(f"[meta] Loaded metadata for {len(meta_map)} users.")
    return meta_map, stance_map


def attach_metadata_to_graph(g, meta_map, stance_map=None, topic_label=None):
    """Attach metadata to graph for tooltip and community analysis."""
    if not meta_map and not stance_map:
        return
    if meta_map:
        g.graph["meta_map"] = meta_map
    for node in g.nodes():
        meta = meta_map.get(node)
        if meta:
            g.nodes[node]["meta"] = meta
    if stance_map:
        g.graph["stance_map"] = stance_map
    if topic_label:
        g.graph["topic_label"] = topic_label


def normalize_method_key(method):
    """Normalize visualization method names for graph-level label storage."""
    return str(method or "unknown").strip().lower()


def get_stored_community_profile(g, method, comm_id):
    """Return a saved method-specific community profile from the graph."""
    if g is None:
        return None
    method_key = normalize_method_key(method)
    store = g.graph.get("community_profiles", {})
    method_profiles = store.get(method_key, {})
    return (
        method_profiles.get(str(comm_id)) or
        method_profiles.get(comm_id)
    )


def get_stored_community_label(g, method, comm_id, node=None):
    """Return a saved method-specific community label."""
    profile = get_stored_community_profile(g, method, comm_id)
    if isinstance(profile, dict) and profile.get("label"):
        return profile["label"]

    if g is not None and node in g.nodes:
        method_key = normalize_method_key(method)
        method_label = g.nodes[node].get(f"community_label_{method_key}")
        if method_label:
            return method_label
        if method_key == "unknown":
            return g.nodes[node].get("community_label")
    return None


def save_community_profile_to_graph(g, method, comm_id, members, profile):
    """Save method-specific profile data for dashboard styling and reports."""
    if g is None or not profile:
        return
    method_key = normalize_method_key(method)
    graph_profiles = g.graph.setdefault("community_profiles", {})
    method_profiles = graph_profiles.setdefault(method_key, {})
    method_profiles[str(comm_id)] = {
        "label": profile.name,
        "confidence": profile.confidence,
        "reasoning": profile.reasoning,
        "description": profile.description,
    }
    for node in members:
        if node in g.nodes:
            g.nodes[node][f"community_label_{method_key}"] = profile.name
            g.nodes[node][f"community_confidence_{method_key}"] = profile.confidence


def fallback_label_from_metadata(members, meta_map):
    """Pick an allowed fallback label from members' dominant political tags."""
    if not meta_map:
        return DEFAULT_COMMUNITY_LABEL

    counts = Counter()
    for member in members:
        meta = meta_map.get(member, {})
        for field in ("political_label", "politic_group"):
            label = coerce_allowed_label(meta.get(field))
            if label:
                counts[label] += 1
                break

    if not counts:
        return DEFAULT_COMMUNITY_LABEL
    return counts.most_common(1)[0][0]


def validate_llm_community_label(candidate, dominant_political_label):
    """Reject a gray LLM result when member metadata has a specific label."""
    parsed = parse_community_classification_response(candidate, default=None)
    if not parsed:
        return None
    if (
        parsed["selected_label"] == DEFAULT_COMMUNITY_LABEL
        and dominant_political_label != DEFAULT_COMMUNITY_LABEL
    ):
        print(
            "[Community naming] Rejected gray label; dominant political metadata is "
            f"{dominant_political_label}"
        )
        return None
    return parsed


def prefer_specific_political_label(label, members, meta_map):
    """Prevent gray display labels when a specific political tag is available."""
    canonical_label = coerce_allowed_label(label, default=DEFAULT_COMMUNITY_LABEL)
    if canonical_label != DEFAULT_COMMUNITY_LABEL:
        return canonical_label
    metadata_label = fallback_label_from_metadata(members, meta_map)
    if metadata_label != DEFAULT_COMMUNITY_LABEL:
        return metadata_label
    return canonical_label


def aggregate_community_metadata(members, meta_map):
    """Aggregate metadata for a community and return summary + counts."""
    field_counts = {field: Counter() for field in META_FIELDS}
    for member in members:
        meta = meta_map.get(member, {})
        for field in META_FIELDS:
            value = meta.get(field)
            if value:
                field_counts[field][value] += 1

    summary = {
        field: (counter.most_common(1)[0][0] if counter else None)
        for field, counter in field_counts.items()
    }
    return summary, field_counts


def format_meta_summary(summary):
    """Format metadata summary for tooltips/legend."""
    parts = []
    stance = summary.get("stance")
    if isinstance(stance, dict):
        target = stance.get("target")
        stance_label = stance.get("stance")
        if stance_label:
            if target:
                parts.append(f"موضع نسبت به {target}: {stance_label}")
            else:
                parts.append(f"موضع: {stance_label}")
    for field, label in META_FIELDS.items():
        value = summary.get(field)
        if value:
            parts.append(f"{label}: {value}")
    echo = summary.get("echo")
    if isinstance(echo, dict):
        is_echo = echo.get("is_echo_chamber")
        if is_echo is not None:
            parts.append(f"اتاق پژواک: {'بله' if is_echo else 'خیر'}")
        ei_index = echo.get("ei_index")
        if isinstance(ei_index, (int, float)):
            parts.append(f"E-I: {ei_index:.2f}")
        conductance = echo.get("conductance")
        if isinstance(conductance, (int, float)):
            parts.append(f"هدایت: {conductance:.2f}")
        homogeneity = echo.get("content_homogeneity")
        if isinstance(homogeneity, (int, float)):
            parts.append(f"همگنی: {homogeneity:.2f}")
    return " | ".join(parts)


def detect_target_from_label(label):
    """Detect stance target based on label keywords."""
    if not label:
        return None
    for target, keywords in STANCE_TARGETS.items():
        if any(k in label for k in keywords):
            return target
    return None


def compute_community_stance(label, members, stance_map, topic_label=None):
    """Compute stance summary for a community."""
    if not stance_map:
        return None
    target = detect_target_from_label(label)
    target_key = target or "__topic__"
    counts = Counter()
    for member in members:
        counts.update(stance_map.get(member, {}).get(target_key, Counter()))
    total = sum(counts.values())
    if total == 0:
        return None
    pos = counts.get("pos", 0)
    neg = counts.get("neg", 0)
    neu = counts.get("neu", 0)
    if neu / total >= 0.7:
        stance = "بی‌جهت‌گیری"
    elif pos / total >= 0.6:
        stance = "حامی"
    elif neg / total >= 0.6:
        stance = "منتقد"
    else:
        stance = "ترکیبی"
    return {
        "target": target or topic_label,
        "stance": stance,
        "counts": {"pos": pos, "neg": neg, "neu": neu, "total": total}
    }


def compute_community_similarity(community_summaries):
    """Compute pairwise similarity (Jaccard) from summary labels."""
    comm_ids = list(community_summaries.keys())
    signatures = {}
    for comm_id in comm_ids:
        signature = set()
        for field in META_FIELDS:
            value = community_summaries[comm_id].get(field)
            if value:
                signature.add((field, value))
        signatures[comm_id] = signature

    edges = []
    for i in range(len(comm_ids)):
        for j in range(i + 1, len(comm_ids)):
            a = comm_ids[i]
            b = comm_ids[j]
            sig_a = signatures[a]
            sig_b = signatures[b]
            union = sig_a | sig_b
            if not union:
                continue
            similarity = len(sig_a & sig_b) / len(union)
            if similarity <= 0:
                continue
            edges.append({
                "source": a,
                "target": b,
                "similarity": similarity,
                "distance": 1 - similarity
            })
    return edges


def load_topic_label_from_elastic(file_path="elastic.py"):
    """Extract topic query from elastic.py multi_match query."""
    config = load_pipeline_config()
    if config.get("topic_label"):
        return config["topic_label"]
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        match = re.search(
            r'"multi_match"\s*:\s*\{[^}]*?"query"\s*:\s*"([^"]+)"',
            content,
            re.S
        )
        if not match:
            match = re.search(
                r"'multi_match'\s*:\s*\{[^}]*?'query'\s*:\s*'([^']+)'",
                content,
                re.S
            )
        if match:
            return match.group(1).strip()
    except Exception:
        return None
    return None


def load_pipeline_config(file_path="pipeline_config.json"):
    """Load topic/date/slot configuration written by elastic.py."""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[config] Failed to read {file_path}: {e}")
        return {}


def recent_date_range(days=DEFAULT_LOOKBACK_DAYS):
    """Return an inclusive recent date range ending today."""
    end = datetime.now().date()
    start = end - timedelta(days=max(1, days) - 1)
    return (
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.min.time()),
    )


def parse_config_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d")
    except Exception:
        return None


def get_pipeline_date_range(config):
    """Get configured dates, falling back to the latest 14 inclusive days."""
    start = parse_config_date(config.get("start_date"))
    end = parse_config_date(config.get("end_date"))
    if start and end:
        return start, end
    days = int(config.get("lookback_days") or DEFAULT_LOOKBACK_DAYS)
    return recent_date_range(days)


def inclusive_day_count(start_date, end_date):
    return (end_date.date() - start_date.date()).days + 1


def select_slot_modes_for_range(start_date, end_date):
    """Pick timeline granularities from the full requested date range."""
    days = inclusive_day_count(start_date, end_date)
    if days <= 8:
        return ["daily", "hourly"]
    if days <= 45:
        return ["weekly", "daily"]
    if days <= 90:
        return ["monthly", "weekly", "daily"]
    if days <= 180:
        return ["monthly", "weekly"]
    return ["quarterly", "monthly", "weekly"]


def apply_edge_widths(net, g, min_width=0.5, max_width=6.0):
    """Scale edge widths by interaction weights."""
    if not hasattr(net, "edges"):
        return
    weights = []
    for edge in net.edges:
        data = g.get_edge_data(edge.get("from"), edge.get("to"), default={})
        weight = data.get("weight", 1)
        edge["weight"] = weight
        weights.append(weight)

    if not weights:
        return
    min_weight = min(weights)
    max_weight = max(weights)

    for edge in net.edges:
        weight = edge.get("weight", 1)
        if max_weight == min_weight:
            width = min_width
        else:
            width = min_width + (weight - min_weight) * (max_width - min_width) / (max_weight - min_weight)
        edge["width"] = width
        edge["title"] = f"وزن تعامل: {weight}"


# --- Helper: Extract recent texts from Elasticsearch or file ---
def get_recent_texts(center_node: str, max_samples: int = 3, max_chars: int = 300):
    """Get sample texts from a center node's recent activity."""
    try:
        # Get texts from Elasticsearch
        account_texts = fetch_community_texts([center_node], max_texts=max_samples)

        # Get texts for this account
        texts = account_texts.get(center_node, [])

        if not texts:
            # Fallback to file if ES fails
            with open("interactions.json", "r", encoding="utf-8") as f:
                for line in f:
                    msg = json.loads(line)
                    sender = msg.get("sender")
                    text = msg.get("text") or msg.get("content") or ""
                    if sender == center_node and text:
                        texts.append(text.strip())
                        if len(texts) >= max_samples:
                            break

        if not texts:
            return ""

        # Join and truncate
        snippet = " ".join(texts[:max_samples])
        return snippet[:max_chars]

    except Exception as e:
        print(f"[error] getting texts for {center_node}: {e}")
        return ""


def is_valid_label(text):
    """Basic validation of generated community names."""
    if coerce_allowed_label(text):
        return True
    if not text or len(text.split()) > 4:  # Allow up to 4 words
        return False

    # Must have Persian characters
    if not any('\u0600' <= c <= '\u06FF' for c in text):
        return False

    # No punctuation or special characters except dash
    if any(c in text for c in '()[]{}«»,؛.!?:'):
        return False

    return True


def clean_label(text, default="ناشناس"):
    """Clean and normalize community labels."""
    allowed = coerce_allowed_label(text)
    if allowed:
        return allowed

    if not text or len(text) < 3:
        return default

    # Basic cleanup
    text = text.strip().split("\n")[0].strip()
    text = re.sub(r'[\[\](){}""\'\'«»\-–—]', ' ', text)
    text = re.sub(r'\*{1,2}|_{1,2}|`', '', text)
    text = " ".join(text.split())

    # Define Persian content check
    def has_persian(s):
        return bool(re.search('[\u0600-\u06FF]', s))

    # Media/News patterns
    news_patterns = {
        'news': 'خبرگزاری',
        'press': 'مطبوعات',
        'media': 'رسانه',
        'radio': 'رادیو',
        'tv': 'تلویزیون',
        'channel': 'شبکه',
        'daily': 'روزنامه',
        'agency': 'خبرگزاری'
    }

    # Political patterns
    political_patterns = {
        'front': 'جبهه',
        'movement': 'جنبش',
        'party': 'حزب',
        'group': 'گروه',
        'council': 'شورای',
        'alliance': 'ائتلاف'
    }

    # Social patterns
    social_patterns = {
        'activists': 'فعالان',
        'supporters': 'حامیان',
        'critics': 'منتقدان',
        'society': 'انجمن',
        'community': 'جامعه',
        'network': 'شبکه'
    }

    # Cultural patterns
    cultural_patterns = {
        'art': 'هنری',
        'cinema': 'سینمایی',
        'theater': 'تئاتری',
        'music': 'موسیقی',
        'literary': 'ادبی'
    }

    # If text has no Persian content, try pattern matching
    if not has_persian(text):
        text_lower = text.lower()

        # Try each pattern set
        for patterns in [news_patterns, political_patterns, 
                        social_patterns, cultural_patterns]:
            for eng, fa in patterns.items():
                if eng in text_lower:
                    return f"{fa} {text}"

        # If no patterns match, use default
        return default

    # Split on common separators
    parts = re.split(r'[|،\-:؛/]', text)
    parts = [p.strip() for p in parts]

    # Keep first meaningful Persian part
    persian_parts = [p for p in parts if has_persian(p) and len(p) > 2]
    if persian_parts:
        text = persian_parts[0]

    # Normalize whitespace
    text = " ".join(text.split())

    # Keep first 4 words
    if len(text.split()) > 4:
        text = " ".join(text.split()[:4])

    # Final cleanup and validation
    text = text.strip()
    return coerce_allowed_label(text) or (text if (text and has_persian(text)) else default)


def get_active_members(g, community_nodes, top_n=5):
    """Get the most active members of a community based on interaction counts."""
    # Count interactions for each node
    counts = {
        node: sum(1 for _ in g.edges(node))
        for node in community_nodes
    }

    # Get top N most active members
    # Sort by activity count then alphabetically
    active = sorted(
        counts.items(),
        key=lambda x: (x[1], x[0]),
        reverse=True
    )
    return [m for m, _ in active[:top_n]]


def load_few_shot_examples(num_examples=20):
    """Load random few-shot learning examples from the examples file."""
    try:
        with open("community_examples.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            examples = data.get("examples", [])
            if not examples:
                return ""
            
            # Randomly select examples
            import random
            selected = random.sample(examples, min(num_examples, len(examples)))
            
            # Format each example
            formatted = []
            template = (
                "مرکز: @{center}\n"
                "اعضای فعال: {members}\n"
                "نمونه توییت‌ها:\n{tweets}\n"
                "نام گروه: {label}\n"
            )
            
            for ex in selected:
                tweet_lines = "\n".join(f"- {t}" for t in ex['tweets'])
                members = ", ".join(f"@{m}" for m in ex['active_members'])
                
                formatted.append(template.format(
                    center=ex['center_node'],
                    members=members,
                    tweets=tweet_lines,
                    label=ex['label']
                ))
            
            return "\n---\n".join(formatted)
            
    except Exception as e:
        print(f"[warning] Could not load examples: {e}")
        return ""


def save_communities(communities, filename=None):
    """Save community details to a JSON file."""
    if not filename:
        filename = "community_details.json"
        
    try:
        # Convert to list if it's a generator
        communities_list = list(communities)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Add timestamp and metadata
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_communities": len(communities_list),
            "communities": communities_list
        }
        
        # Save with proper encoding and formatting
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(sanitize_json(data), f, ensure_ascii=False, indent=2)
            
        print(f"Saved {len(communities_list)} communities to {filename}")
        return True
        
    except Exception as e:
        print(f"[error] Failed to save communities: {e}")
        return False


def load_communities(filename=None):
    """
    Load previously saved community details.
    Returns tuple of (communities list, metadata dict)
    """
    if not filename:
        filename = "community_details.json"
        
    try:
        if not os.path.exists(filename):
            print(f"No saved communities found at {filename}")
            return [], {}
            
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        communities = data.get("communities", [])
        metadata = {
            "timestamp": data.get("timestamp"),
            "total_communities": data.get("total_communities")
        }
        
        print(f"Loaded {len(communities)} communities from {filename}")
        return communities, metadata
        
    except Exception as e:
        print(f"[error] Failed to load communities: {e}")
        return [], {}


def analyze_community_texts(texts, members):
    """Analyze community texts to find common themes."""
    if not texts:
        return None
        
    # Common themes to look for
    themes = {
        # Political themes
        r'سیاس|دولت|نظام|اصلاح|انقلاب': 'سیاسی',
        r'حقوق|آزاد|عدالت|دموکراس': 'حقوق بشر',
        r'اعتراض|تظاهرات|تجمع': 'اعتراضی',
        
        # Social themes
        r'اجتماع|مردم|جامعه|شهروند': 'اجتماعی',
        r'زنان|جنسیت|برابری': 'حقوق زنان',
        r'کارگر|معیشت|اقتصاد': 'کارگری',
        
        # Media themes
        r'خبر|گزارش|رسان|اطلاع': 'رسانه‌ای',
        r'تحلیل|بررسی|نقد|دیدگاه': 'تحلیلی',
        r'فرهنگ|هنر|ادب|سینما': 'فرهنگی'
    }
    
    # Join all texts
    full_text = ' '.join(texts)
    
    # Find matching themes
    matched_themes = []
    for pattern, theme in themes.items():
        if re.search(pattern, full_text, re.I):
            matched_themes.append(theme)
            
    return matched_themes[0] if matched_themes else None


def build_community_naming_prompt(context: str, few_shot: str) -> str:
    """Build the canonical prompt used by naming backends."""
    examples_block = few_shot.strip() if few_shot else "(no examples available)"
    context_block = context.strip() if context.strip() else "(no context available)"
    return f"""
You are a professional analyst in social network analysis. Provide a single, concise Persian community name (max 3-4 words) that reflects the community's dominant theme or role. Avoid usernames and any extraneous punctuation. Return only the label text.

Examples:
{examples_block}

Context (sample tweets):
{context_block}

Answer (label only):
"""


def ai_name_community(center_node, neighbors, node_label_map, comm_id, method, time_period=None, g=None):
    """
    Classify a community into one of COMMUNITY_LABELS.

    The graph stores labels per method because Louvain and Hybrid partitions can
    assign the same node to different communities.
    """
    try:
        members = [center_node] + list(neighbors)
        meta_map = g.graph.get("meta_map", {}) if g is not None else {}
        fallback_label = (
            fallback_label_from_metadata(members, meta_map) or
            DEFAULT_COMMUNITY_LABEL
        )
        cached = get_stored_community_profile(g, method, comm_id)
        if isinstance(cached, dict) and cached.get("label"):
            cached_label = prefer_specific_political_label(
                cached["label"],
                members,
                meta_map,
            )
            if cached_label == cached["label"]:
                return cached_label

        summary, _ = aggregate_community_metadata(members, meta_map)

        sample_neighbors = neighbors[:3] if neighbors else []
        nodes = [center_node] + sample_neighbors
        parts = []
        if ENABLE_NAMING_TEXT_FETCH:
            texts_map = {}
            try:
                texts_map = fetch_community_texts(nodes, max_texts=6)
            except Exception as e:
                print(f"[warning] text fetch failed for community {comm_id}: {e}")

            for n in nodes:
                tlist = texts_map.get(n, [])
                if tlist:
                    parts.append(f"{n}: {' | '.join(tlist[:2])}")

        meta_summary = format_meta_summary(summary)
        if meta_summary:
            parts.append(f"Metadata: {meta_summary}")

        context = "\n".join(parts).strip()
        if not context:
            context = f"Dominant metadata label: {fallback_label}"

        parsed = {
            "selected_label": fallback_label,
            "confidence": 0,
            "reasoning": "metadata fallback"
        }
        if ENABLE_LLM_NAMING:
            prompt = build_community_classification_prompt(
                members=members,
                text_content=context,
                dominant_political_label=fallback_label,
            )
            parsed, _ = call_llm_with_fallback(
                prompt,
                validator=lambda candidate: validate_llm_community_label(
                    candidate,
                    fallback_label,
                ),
                org_temperature=0.2,
                local_temperature=0.1,
                max_tokens=180,
            )

            if not parsed:
                parsed = {
                    "selected_label": fallback_label,
                    "confidence": 0,
                    "reasoning": "dominant political metadata fallback"
                }

        if (
            parsed["selected_label"] == DEFAULT_COMMUNITY_LABEL
            and fallback_label != DEFAULT_COMMUNITY_LABEL
        ):
            parsed = {
                "selected_label": fallback_label,
                "confidence": 0,
                "reasoning": "gray label overridden by dominant political metadata",
            }

        profile = build_community_profile(members, context, parsed)
        final_label = profile.name

        save_community_profile_to_graph(g, method, comm_id, members, profile)
        try:
            save_community_name(final_label, center_node, neighbors, comm_id, method, time_period)
        except Exception:
            pass

        if not hasattr(ai_name_community, "communities"):
            ai_name_community.communities = []
        ai_name_community.communities.append({
            "id": comm_id,
            "name": final_label,
            "confidence": profile.confidence,
            "reasoning": profile.reasoning,
            "description": profile.description,
            "center_node": center_node,
            "members": neighbors,
            "method": method,
            "member_count": len(neighbors)
        })

        return final_label

    except Exception as e:
        print(f"[error] naming community {comm_id}: {e}")
        return DEFAULT_COMMUNITY_LABEL


# Map of key accounts to their community identities
NODE_LABEL_MAP = {
    # Media/News
    "bbcpersian": "رسانه بی‌بی‌سی فارسی",
    "manototv": "شبکه تلویزیونی من و تو",
    "IranIntl": "شبکه خبری ایران اینترنشنال",
    "VOAIran": "رسانه صدای آمریکا",
    "RadioFarda": "رسانه رادیو فردا",
    "AFP": "خبرگزاری فرانسه",
    "Reuters": "خبرگزاری رویترز",
    
    # Political
    "khamenei_ir": "دفتر رهبری",
    "Rouhani": "حامیان دولت روحانی",
    "alilarijani": "جریان لاریجانی",
    "mostafataj": "تحلیلگران سیاسی",
    
    # Opposition
    "farashgard": "جنبش ققنوس",
    "ICHRI": "فعالان حقوق بشر",
    
    # International
    "netanyahu": "شبکه نتانیاهو",
    "palestineintl": "رسانه فلسطینی",
    
    # Cultural/Social
    "shahrvand": "روزنامه شهروند",
    "honaronline": "رسانه هنری",
    "ketabism": "انجمن قلم",
    
    # Additional Media
    "TheIndyPersian": "ایندیپندنت فارسی",
    "dw_persian": "دویچه‌وله فارسی",
    "ir_voanews": "صدای آمریکا فارسی",
    
    # Additional Political
    "ebtekarnews": "خبرگزاری ابتکار",
    "entekhab_news": "رسانه انتخاب",
    "etemadonline": "روزنامه اعتماد",
    
    # Social/Cultural
    "isna_farsi": "خبرگزاری ایسنا",
    "mehrnews_fa": "خبرگزاری مهر",
    "tasnimnews_fa": "خبرگزاری تسنیم"
}


# Find central node in each community
def get_community_centers(g, partition, centrality_measure='degree'):
    """
    Given a graph and partition, return the most central node in each community.
    centrality_measure: 'degree', 'pagerank', or 'betweenness'
    """
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    centers = {}
    for comm_id, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        if centrality_measure == 'pagerank':
            centrality = nx.pagerank(subgraph)
        elif centrality_measure == 'betweenness':
            centrality = nx.betweenness_centrality(subgraph)
        else:
            centrality = dict(subgraph.degree())
        central_node = max(centrality.items(), key=lambda x: x[1])[0]
        centers[comm_id] = central_node
    return centers


# Interactive Graph Visualization with PyVis
def visualize_graph_interactively(
        g, partition=None, title="Graph", filename="graph.html"):
    """Create an interactive visualization of the graph using PyVis."""
    # Initialize network
    net = Network(
        height="800px",
        width="100%",
        notebook=False,
        bgcolor="#ffffff",
        font_color="#333333"
    )
    
    # Configure network options before adding graph
    net.options = {
        "configure": {
            "enabled": True,
            "filter": "physics"
        },
        "physics": {
            "stabilization": {
                "enabled": True,
                "iterations": 100,
                "updateInterval": 50
            },
            "barnesHut": {
                "gravitationalConstant": -2000,
                "springConstant": 0.04,
                "springLength": 150
            }
        },
        "edges": {
            "color": "#999999",
            "width": 0.5,
            "smooth": {
                "enabled": False,
                "type": "continuous"
            }
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
            "zoomView": True,
            "dragNodes": True,
            "dragView": True
        }
    }
    
    # Add the graph after options are set
    net.from_nx(g)
    apply_edge_widths(net, g)

    # Optional: color nodes by community/cluster
    if partition:
        centers = get_community_centers(g, partition)
        for node, community_id in partition.items():
            center_node = centers.get(community_id)
            label = coerce_allowed_label(
                NODE_LABEL_MAP.get(center_node),
                default=DEFAULT_COMMUNITY_LABEL,
            )
            net.get_node(node)['color'] = get_community_label_color(label)
            net.get_node(node)['title'] = f"""
<div style="max-width: 300px; padding: 8px; text-align: right;">
    <strong style="font-size: 14px; display: block; margin-bottom: 4px;">{node}</strong>
    <span style="color: #666; font-size: 12px;">گروه: {label}</span>
</div>
"""

    # Always add metadata tags if available, for all nodes
    if hasattr(g, "graph") and "meta_map" in g.graph:
        for node in g.nodes():
            meta = g.graph["meta_map"].get(node)
            if meta:
                meta_info = "<br>".join(f"{k}: {v}" for k, v in meta.items())
                net_node = net.get_node(node)
                existing_title = net_node.get("title", node)
                net_node["title"] = f"{existing_title}<br>{meta_info}"

    # Show node labels if label_map is present, regardless of partition
    if hasattr(g, "graph") and "label_map" in g.graph:
        label_map = g.graph["label_map"]
        for node in g.nodes():
            label = label_map.get(node)
            if label:
                title = net.get_node(node).get("title", node)
                net.get_node(node)['title'] = f"{title} ({label})"

    save_network_html(net, filename)
    print(f"Interactive graph saved to {filename}")


# Visualization for Embeddings (PCA + Clusters)
def visualize_embeddings(embeddings, labels, nodes, ax):
    """
    Visualize node embeddings in 2D using PCA and color them by cluster labels.
    """
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    unique_labels = list(set(labels))
    num_clusters = len(unique_labels)
    cmap = plt.cm.get_cmap("tab10", num_clusters)

    label_to_color = {label: cmap(i / (num_clusters - 1) if num_clusters > 1 else 0)
                      for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]

    for i, (x, y) in enumerate(reduced_embeddings):
        ax.scatter(x, y, color=colors[i], label=f"Node {nodes[i]}" if nodes else None)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   label=f'Cluster {label}', markersize=8)
        for label, color in label_to_color.items()
    ]
    ax.legend(handles=handles, title="KMeans Clusters", loc="best")
    ax.set_title("Node Embeddings PCA with KMeans Clustering (Hybrid Method)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True)


# CLASSIC Approach: Louvain Detector
def detect_communities_louvain(g, random_seed=COMMUNITY_RANDOM_SEED):
    """
    Run Louvain algorithm on graph G and return partition.
    G: networkx.Graph
    Returns: dict mapping node → community_id
    """
    partition = community_louvain.best_partition(
        g,
        weight='weight',
        random_state=random_seed,
    )
    return partition


# Main Louvain
def main_louvain(g):
    """
    Using Louvain algorithm for community detection.
    g: networkx.Graph
    """
    if g.number_of_nodes() == 0:
        print("Graph is empty. Louvain cannot proceed.")
        return
    # Run Louvain algorithm
    partition = detect_communities_louvain(g)

    # Print community results
    for node, comm in partition.items():
        print(f"Node {node}: Community {comm}")

    return partition


# HYBRID Approach: Node2Vec + Clustering
def build_user_graph(messages):
    """
    Build a user interaction graph from message data.
    """
    g = nx.Graph()
    edge_weights = defaultdict(int)

    for msg in messages:
        sender = msg.get("sender")
        target = msg.get("target")
        if not sender or not target or sender == target:
            continue

        # Treat every interaction type the same: weight = 1
        edge_weights[(sender, target)] += 1

    # Filter and add strong edges
    for (sender, target), total_weight in sorted(edge_weights.items()):
        if total_weight >= 1:
            g.add_edge(sender, target, weight=total_weight)
    print(f"Graph has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
    return g


def get_node_embeddings(g, dimensions=64, random_seed=COMMUNITY_RANDOM_SEED):
    """
    Generate Node2Vec embeddings for the graph.
    """
    workers = int(os.getenv("NODE2VEC_WORKERS", "1"))
    node2vec = Node2Vec(g, dimensions=dimensions, walk_length=20, num_walks=100,
                        workers=workers, seed=random_seed)
    model = node2vec.fit(window=10, min_count=1, seed=random_seed)

    # Ensure embeddings are generated for all nodes
    embeddings = []
    nodes = sorted(g.nodes())
    for node in nodes:
        try:
            embeddings.append(model.wv[str(node)])
        except KeyError:
            # Use a zero vector for missing embeddings
            embeddings.append([0] * dimensions)

    # Validate embedding lengths
    for i, emb in enumerate(embeddings):
        if len(emb) != dimensions:
            raise ValueError(f"Embedding at index {i} has incorrect dimension "
                             f"{len(emb)}; expected {dimensions}.")

    embeddings = np.array([np.array(emb) for emb in embeddings])
    return embeddings, nodes


# Data Collection: Load Interactions Dataset
def load_interactions(file_path="interactions.json"):
    """
    Load sender-target interactions from a JSON lines file.
    Each line is a dict with keys: sender, target, type
    """
    messages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                msg = json.loads(line)
                sender = msg.get("sender")
                target = msg.get("target")
                if sender and target and sender != target:
                    messages.append(msg)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(messages)} interactions.")
    return messages


def get_interactions_date_range(file_path="interactions.json"):
    """Return min/max datetime from interactions.json date fields."""
    min_dt = None
    max_dt = None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                date_val = msg.get("date")
                if not date_val:
                    continue
                date_str = str(date_val)[:10]
                if len(date_str) != 10:
                    continue
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                except Exception:
                    continue
                if min_dt is None or dt < min_dt:
                    min_dt = dt
                if max_dt is None or dt > max_dt:
                    max_dt = dt
    except FileNotFoundError:
        return None, None
    return min_dt, max_dt


# Glue It All Together
def run_kmeans(embeddings, n_clusters=5, random_seed=COMMUNITY_RANDOM_SEED):
    from sklearn.cluster import KMeans
    n_samples = embeddings.shape[0]
    if n_samples < n_clusters:
        n_clusters = max(1, n_samples)  # adjust cluster count dynamically
        msg = (
            f"[auto-adjust] n_clusters reduced to {n_clusters}"
            f" due to small sample size ({n_samples})."
        )
        print(msg)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        n_init=10,
    )
    return kmeans.fit_predict(embeddings)


def summarize_partition_sizes(partition):
    """Summarize community sizes for a partition dict."""
    if not partition:
        return {"num_clusters": 0, "total_nodes": 0, "sizes": []}
    counts = Counter(partition.values())
    sizes = sorted(counts.values(), reverse=True)
    arr = np.array(sizes, dtype=float)
    return {
        "num_clusters": len(sizes),
        "total_nodes": int(arr.sum()),
        "min_size": int(arr.min()),
        "max_size": int(arr.max()),
        "mean_size": float(arr.mean()),
        "median_size": float(np.median(arr)),
        "sizes": sizes
    }


def compute_modularity_safe(g, partition):
    """Compute modularity with guardrails."""
    if not partition or g.number_of_edges() == 0:
        return None
    try:
        return float(community_louvain.modularity(partition, g, weight="weight"))
    except Exception:
        return None


def compute_echo_metrics(g, partition, stance_map=None, topic_label=None, thresholds=None):
    """Compute echo chamber metrics for a partition."""
    communities = defaultdict(list)
    for node, cid in partition.items():
        communities[cid].append(node)

    echo_metrics = {}
    for cid, members in communities.items():
        stance_info = None
        if stance_map:
            stance_info = compute_community_stance(None, members, stance_map, topic_label=topic_label)
        homogeneity = compute_content_homogeneity(stance_info)
        ei_index = compute_ei_index(g, members)
        conductance = compute_conductance(g, members)
        echo_metrics[cid] = {
            "ei_index": ei_index,
            "conductance": conductance,
            "content_homogeneity": homogeneity,
            "size": len(members),
            "is_echo_chamber": classify_echo_chamber(
                ei_index, conductance, homogeneity, len(members), thresholds=thresholds
            ),
        }
        if stance_info:
            echo_metrics[cid]["stance"] = stance_info

    return echo_metrics


def summarize_echo_metrics(echo_metrics, thresholds=None):
    """Print summary stats for echo metrics vs thresholds."""
    th = dict(DEFAULT_ECHO_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    total = len(echo_metrics)
    if total == 0:
        print("[echo] No communities to summarize.")
        return
    counts = {
        "size_ok": 0,
        "ei_ok": 0,
        "conductance_ok": 0,
        "homogeneity_ok": 0,
        "all_ok": 0,
        "homogeneity_missing": 0,
    }
    for info in echo_metrics.values():
        size = info.get("size", 0)
        ei = info.get("ei_index")
        conductance = info.get("conductance")
        homogeneity = info.get("content_homogeneity")
        size_ok = size >= th["min_size"]
        ei_ok = ei is not None and ei <= th["ei_index_max"]
        conductance_ok = conductance is not None and conductance <= th["conductance_max"]
        if homogeneity is None:
            counts["homogeneity_missing"] += 1
        homogeneity_ok = homogeneity is not None and homogeneity >= th["homogeneity_min"]
        if size_ok:
            counts["size_ok"] += 1
        if ei_ok:
            counts["ei_ok"] += 1
        if conductance_ok:
            counts["conductance_ok"] += 1
        if homogeneity_ok:
            counts["homogeneity_ok"] += 1
        if size_ok and ei_ok and conductance_ok and homogeneity_ok:
            counts["all_ok"] += 1
    print(
        "[echo] Summary:",
        f"total={total},",
        f"size_ok={counts['size_ok']},",
        f"ei_ok={counts['ei_ok']},",
        f"conductance_ok={counts['conductance_ok']},",
        f"homogeneity_ok={counts['homogeneity_ok']},",
        f"homogeneity_missing={counts['homogeneity_missing']},",
        f"echo_true={counts['all_ok']}"
    )


def build_hybrid_report(g, embeddings, nodes, hybrid_labels, louvain_partition=None,
                        start_date=None, end_date=None, stance_map=None,
                        topic_label=None, thresholds=None):
    """Build a comparable report for hybrid vs Louvain partitions."""
    hybrid_partition = {node: int(hybrid_labels[i]) for i, node in enumerate(nodes)}
    report = {
        "timeframe": {"start": start_date, "end": end_date},
        "hybrid": {
            "size_summary": summarize_partition_sizes(hybrid_partition),
            "modularity": compute_modularity_safe(g, hybrid_partition)
        }
    }

    report["hybrid"]["echo_metrics"] = compute_echo_metrics(
        g,
        hybrid_partition,
        stance_map=stance_map,
        topic_label=topic_label,
        thresholds=thresholds,
    )
    report["hybrid"]["echo_thresholds"] = dict(DEFAULT_ECHO_THRESHOLDS, **(thresholds or {}))

    # Silhouette score (embedding quality proxy)
    unique_labels = set(hybrid_labels)
    if len(unique_labels) > 1 and len(embeddings) > 1:
        try:
            report["hybrid"]["silhouette"] = float(silhouette_score(embeddings, hybrid_labels))
        except Exception:
            report["hybrid"]["silhouette"] = None
    else:
        report["hybrid"]["silhouette"] = None

    if louvain_partition:
        report["louvain"] = {
            "size_summary": summarize_partition_sizes(louvain_partition),
            "modularity": compute_modularity_safe(g, louvain_partition)
        }
        report["louvain"]["echo_metrics"] = compute_echo_metrics(
            g,
            louvain_partition,
            stance_map=stance_map,
            topic_label=topic_label,
            thresholds=thresholds,
        )
        report["louvain"]["echo_thresholds"] = dict(DEFAULT_ECHO_THRESHOLDS, **(thresholds or {}))
        # Compare partitions on shared nodes
        louvain_labels = [louvain_partition.get(node) for node in nodes]
        if all(label is not None for label in louvain_labels):
            try:
                report["comparison"] = {
                    "nmi": float(normalized_mutual_info_score(louvain_labels, hybrid_labels)),
                    "ari": float(adjusted_rand_score(louvain_labels, hybrid_labels))
                }
            except Exception:
                report["comparison"] = {"nmi": None, "ari": None}
            # Temporal stability proxy (same window, different methods)
            report["comparison"]["stability"] = {
                "AMI": report["comparison"]["ari"],
                "NMI": report["comparison"]["nmi"]
            }

    return report


def save_hybrid_report(report, output_dir="communities"):
    """Persist hybrid report to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start = report.get("timeframe", {}).get("start", "unknown")
    end = report.get("timeframe", {}).get("end", "unknown")
    filename = f"{output_dir}/hybrid_report_{start}_to_{end}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(sanitize_json(report), f, ensure_ascii=False, indent=2)
    print(f"[saved] {filename}")
    return filename


def main_hybrid(g, ax):
    """
    Using Node2Vec and KMeans for community detection.
    g: networkx.Graph
    ax: matplotlib.axes.Axes
    """

    if g.number_of_nodes() == 0:
        print("Graph is empty. Hybrid method cannot proceed.")
        return

    # Generate embeddings and run clustering
    embeddings, nodes = get_node_embeddings(g)
    print(f"Generated {len(embeddings)} embeddings for {len(nodes)} nodes.")

    # Check if it's safe to cluster
    if len(embeddings) < 2:
        print("Not enough embeddings for clustering.")
        return

    labels = run_kmeans(embeddings, n_clusters=5)

    print(f"Number of nodes in graph: {len(g.nodes())}")
    print(f"Number of embeddings: {len(embeddings)}")
    # Add interactive visualization
    partition = {node: labels[i] for i, node in enumerate(nodes)}
    visualize_graph_interactively(g, partition, title="Hybrid Graph",
                                  filename="hybrid_graph.html")


# ---- Helper: Clean Graph ----
def clean_graph(g):
    """Clean graph by removing isolated nodes and taking largest component.
    
    Removes nodes with no edges and returns largest connected subgraph."""
    low_degree_nodes = [n for n, d in g.degree() if d < 1]
    g.remove_nodes_from(low_degree_nodes)
    if g.number_of_nodes() == 0:
        return g
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
    return g


# ---- Helper: Style Partition ----
def style_partition(g, net, partition, method, _base_hue):
    """Style nodes using stored labels and return legend data with matching colors."""
    if g.number_of_nodes() == 0 or not partition:
        return {}

    method_key = normalize_method_key(method)
    meta_map = g.graph.get("meta_map", {})
    stance_map = g.graph.get("stance_map", {})
    topic_label = g.graph.get("topic_label")
    echo_by_comm = compute_echo_metrics(
        g,
        partition,
        stance_map=stance_map,
        topic_label=topic_label,
    )
    centers = get_community_centers(g, partition)
    communities = []
    for comm_id, center in centers.items():
        members = [n for n, c in partition.items() if c == comm_id]
        if not members:
            continue
        label = get_stored_community_label(g, method_key, comm_id, center)
        if not label:
            neighbors = [n for n in members if n != center]
            label = ai_name_community(center, neighbors, NODE_LABEL_MAP, comm_id, method_key, g=g)
        if not label:
            label = fallback_label_from_metadata(members, meta_map)
        label = prefer_specific_political_label(label, members, meta_map)
        stance_info = compute_community_stance(label, members, stance_map, topic_label=topic_label)
        if stance_info:
            if "حامیان" in label and stance_info["stance"] == "منتقد":
                label = label.replace("حامیان", "منتقدان")
            elif "حامیان" in label and stance_info["stance"] == "ترکیبی":
                label = label.replace("حامیان", "بحث درباره")
        for node in members:
            g.nodes[node][f"community_label_{method_key}"] = label
        summary, _ = aggregate_community_metadata(members, meta_map)
        if stance_info:
            summary["stance"] = stance_info
        echo_info = echo_by_comm.get(comm_id)
        if isinstance(echo_info, dict):
            summary["echo"] = {
                "is_echo_chamber": echo_info.get("is_echo_chamber"),
                "ei_index": echo_info.get("ei_index"),
                "conductance": echo_info.get("conductance"),
                "content_homogeneity": echo_info.get("content_homogeneity"),
            }
        communities.append({
            "id": comm_id,
            "label": label,
            "center": center,
            "members": members,
            "summary": summary
        })

    if not communities:
        return {}

    sorted_communities = sorted(communities, key=lambda c: len(c["members"]), reverse=True)

    label_sizes = Counter()
    label_meta = {}
    for community in sorted_communities:
        label_sizes[community["label"]] += len(community["members"])
        label_meta.setdefault(community["label"], community.get("summary", {}))

    label_colors: Dict[str, str] = {}
    for label in label_sizes:
        label_colors[label] = get_community_label_color(label)

    legend_map: Dict[str, Dict[str, Any]] = {
        label: {
            "color": color,
            "count": label_sizes[label],
            "meta": label_meta.get(label, {})
        }
        for label, color in label_colors.items()
    }

    for community in sorted_communities:
        display_label = community["label"]
        color = label_colors[display_label]
        for node in community["members"]:
            node_data = net.get_node(node)
            if not node_data:
                continue
            degree_boost = min(g.degree(node) * 2, 25)
            role_boost = 15 if node == community["center"] else 0
            size = 15 + degree_boost + role_boost
            meta_summary = format_meta_summary(community.get("summary", {}))
            title = f"{node}<br>جامعه: {display_label}"
            if meta_summary:
                title = f"{title}<br>{meta_summary}"
            node_data.update({
                "color": color,
                "label": node,
                "title": title,
                "size": size,
                "shape": "dot",
                "community_label": display_label,
                "borderWidth": 2 if node == community["center"] else 1,
                "borderWidthSelected": 3,
                "font": {"size": 14, "face": "Vazirmatn"},
                "physics": True,
            })

    apply_edge_widths(net, g)
    return legend_map


def save_community_similarity_graph(g, partition, method, base_filename):
    """Generate a community-level similarity graph from metadata summaries."""
    meta_map = g.graph.get("meta_map", {})
    if not meta_map or not partition:
        return None

    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    summaries = {}
    labels = {}
    for comm_id, members in communities.items():
        summary, _ = aggregate_community_metadata(members, meta_map)
        summaries[comm_id] = summary
        label = get_stored_community_label(g, method, comm_id)
        labels[comm_id] = prefer_specific_political_label(label, members, meta_map)

    edges = compute_community_similarity(summaries)
    if not edges:
        return None

    output_html = base_filename.replace("dashboard_", f"{method}_similarity_")
    output_json = output_html.replace(".html", ".json")

    net = Network(
        height="600px",
        width="100%",
        notebook=False,
        bgcolor="#ffffff",
        font_color="#333333"
    )
    net.options = {
        "physics": {
            "enabled": True,
            "stabilization": {"enabled": True, "iterations": 100}
        },
        "edges": {"color": "#7f8c8d"},
        "interaction": {"hover": True}
    }

    for comm_id, members in communities.items():
        label = labels.get(comm_id, f"گروه {comm_id}")
        summary_text = format_meta_summary(summaries.get(comm_id, {}))
        title = f"{label}<br>اندازه: {len(members)}"
        if summary_text:
            title = f"{title}<br>{summary_text}"
        net.add_node(
            str(comm_id),
            label=label,
            title=title,
            color=get_community_label_color(label),
            size=15 + min(len(members), 30)
        )

    for edge in edges:
        width = 1 + edge["similarity"] * 6
        title = f"شباهت: {edge['similarity']:.2f} | فاصله: {edge['distance']:.2f}"
        net.add_edge(
            str(edge["source"]),
            str(edge["target"]),
            width=width,
            title=title
        )

    save_network_html(net, output_html)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(sanitize_json(edges), f, ensure_ascii=False, indent=2)
    print(f"[saved] {output_html}, {output_json}")
    return output_html


# ---- Helper: Visualize or Dummy ----
def generate_legend_data(g, partition, method, colors):
    """Generate legend data for a community partition."""
    if g.number_of_nodes() == 0:
        return {}
    
    # Count nodes per community
    community_counts = defaultdict(int)
    for node, label in partition.items():
        community_counts[label] += 1
    
    # Create legend data structure
    legend_data = {}
    for label, color in colors.items():
        legend_data[label] = {
            "color": color,
            "count": community_counts.get(label, 0)
        }
    
    return legend_data


def numpy_to_python(obj):
    """Convert numpy types to standard Python types."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32,
        np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def sanitize_json(obj):
    """Recursively convert numpy/scalar types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json(v) for v in obj]
    return numpy_to_python(obj)


def write_summary_file(communities, start_date, end_date):
    """Write a human-readable summary of communities."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"community_summary_{start_date}_{end_date}_{timestamp}.txt"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"=== خلاصه جوامع ({start_date} تا {end_date}) ===\n\n")
            
            # Group by method
            by_method = {}
            for comm in communities:
                method = comm.get("method", "unknown")
                by_method.setdefault(method, []).append(comm)
            
            # Write each method's communities
            for method, comms in by_method.items():
                f.write(f"\n### جوامع {method} ###\n")
                for comm in sorted(comms, key=lambda x: x.get("member_count", 0), reverse=True):
                    f.write(f"\n- جامعه {comm['id']}:\n")
                    f.write(f"  نام: {comm['name']}\n")
                    f.write(f"  مرکز: {comm['center_node']}\n")
                    f.write(f"  تعداد اعضا: {comm['member_count']}\n")
                    if comm.get("sample_members"):
                        f.write(f"  نمونه اعضا: {', '.join(comm['sample_members'][:3])}\n")
                    f.write("\n")
            
        print(f"\n[saved] Community summary written to {filename}")
    except Exception as e:
        print(f"[error] writing summary: {e}")


def format_slot_value(dt, slot_mode=None):
    if slot_mode == "hourly":
        return dt.strftime("%Y-%m-%d %H:%M")
    return dt.strftime("%Y-%m-%d")


def format_slot_filename_part(dt, slot_mode=None):
    if slot_mode == "hourly":
        return dt.strftime("%y%m%d%H")
    return dt.strftime("%y%m%d")


def parse_message_datetime(value):
    if not value:
        return None
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).replace(tzinfo=None)
    except Exception:
        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d")
        except Exception:
            return None


def get_account_party(account, meta_map):
    if not account:
        return None
    meta = meta_map.get(account, {}) if meta_map else {}
    return coerce_allowed_label(meta.get("political_label"), default=None)


def infer_interaction_topic(msg, counterpart):
    for field in ("topic", "category", "hashtag", "event", "organization", "person", "location"):
        value = msg.get(field)
        if isinstance(value, list) and value:
            return str(value[0])
        if value:
            return str(value)
    interaction_labels = {
        "mention": "ذکر",
        "quote": "نقل‌قول",
        "repost": "بازنشر",
        "reply": "پاسخ",
    }
    interaction_type = interaction_labels.get(str(msg.get("type") or "").lower(), "تعامل")
    if counterpart:
        return f"{interaction_type} @{counterpart}"
    return interaction_type


def build_party_focus_for_messages(messages, meta_map, top_n=6):
    """Summarize selected political groups across interaction axes for one slot."""
    parties = defaultdict(lambda: {
        "total": 0,
        "incoming": 0,
        "outgoing": 0,
        "topics": Counter(),
    })

    for msg in messages:
        sender = msg.get("sender")
        target = msg.get("target")
        sender_party = get_account_party(sender, meta_map)
        target_party = get_account_party(target, meta_map)

        if sender_party:
            topic = infer_interaction_topic(msg, target)
            parties[sender_party]["total"] += 1
            parties[sender_party]["outgoing"] += 1
            parties[sender_party]["topics"][topic] += 1

        if target_party and target_party != sender_party:
            topic = infer_interaction_topic(msg, sender)
            parties[target_party]["total"] += 1
            parties[target_party]["incoming"] += 1
            parties[target_party]["topics"][topic] += 1

    return {
        party: {
            "total": data["total"],
            "incoming": data["incoming"],
            "outgoing": data["outgoing"],
            "topics": [
                {"label": label, "count": count}
                for label, count in data["topics"].most_common(top_n)
            ],
        }
        for party, data in sorted(
            parties.items(),
            key=lambda item: item[1]["total"],
            reverse=True,
        )
        if data["total"] > 0
    }


def collect_node_party_changes(
        g, previous_parties, current_slot, slot_mode, topic_label):
    """Build party-transition events and update in-memory node state.

    State intentionally lives only for the duration of the process. Persistence
    is delegated to the backend API; this function never writes an output file.
    """
    events = []
    for node in sorted(g.nodes()):
        party = get_stored_community_label(g, "hybrid", None, node=node)
        if not party:
            continue

        previous = previous_parties.get(node)
        if previous and previous["party"] != party:
            identity = {
                "node_id": str(node),
                "from_party": previous["party"],
                "to_party": party,
                "previous_slot": previous["slot"],
                "current_slot": current_slot,
                "slot_mode": slot_mode,
                "topic": topic_label,
            }
            event_id = hashlib.sha256(
                json.dumps(
                    identity,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            events.append({
                "event_id": event_id,
                **identity,
                "detected_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            })

        previous_parties[node] = {
            "party": party,
            "slot": current_slot,
        }
    return events


def post_party_change_events(events):
    """POST party changes to the configured backend without local fallback."""
    if not events:
        return True

    api_url = (os.getenv("PARTY_CHANGE_API_URL") or "").strip()
    if not api_url:
        print(
            f"[party-change] Detected {len(events)} transition(s); "
            "API is not configured, so they were not persisted."
        )
        return False

    headers = {"Content-Type": "application/json"}
    api_token = (os.getenv("PARTY_CHANGE_API_TOKEN") or "").strip()
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    timeout = float(os.getenv("PARTY_CHANGE_API_TIMEOUT", "10"))
    try:
        response = requests.post(
            api_url,
            json={"events": events},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        print(f"[party-change] Posted {len(events)} transition(s) to backend API.")
        return True
    except requests.RequestException as exc:
        print(
            f"[party-change] API request failed for {len(events)} "
            f"transition(s): {exc}"
        )
        return False


def visualize_or_dummy(slot_start, slot_end, g, louvain=None, hybrid=None, slot_mode=None):
    """Visualize or use an empty graph if g is empty."""
    # Format filename for the time slot
    start_str = format_slot_filename_part(slot_start, slot_mode)
    end_str = format_slot_filename_part(slot_end, slot_mode)
    mode_part = f"{slot_mode}_" if slot_mode else ""
    filename = f"dashboard_{mode_part}{start_str}_to_{end_str}.html"
    legend_path = filename.replace(".html", "_legend.json")
    
    # Initialize empty partitions if needed
    if g.number_of_nodes() == 0:
        print("[warning] Empty graph - skipping community naming")
        g = nx.Graph()
        partition_louvain, partition_hybrid = {}, {}
    else:
        partition_louvain = louvain if louvain else {}
        partition_hybrid = hybrid if hybrid else {}

    legend_data = {
        "hybrid": {"groups": {}},
    }

    try:
        # Reset community store before detection
        ai_name_community.communities = []

        # IMPORTANT: Call ai_name_community for each community
        time_period = f"{format_slot_value(slot_start, slot_mode)} تا {format_slot_value(slot_end, slot_mode)}"
        
        # Get community centers for Hybrid
        if partition_hybrid:
            print(f"\n[hybrid] Detecting {len(set(partition_hybrid.values()))} communities...")
            hybrid_centers = get_community_centers(g, partition_hybrid)
            
            for comm_id, center_node in hybrid_centers.items():
                # Get all members of this community
                neighbors = [
                    n for n, c in partition_hybrid.items() 
                    if c == comm_id and n != center_node
                ]
                
                # Call naming function
                print(f"[hybrid] Naming community {comm_id}: center={center_node}, members={len(neighbors)}")
                ai_name_community(
                    center_node, neighbors, NODE_LABEL_MAP,
                    comm_id, "hybrid", time_period, g=g
                )

        # Save community details
        if hasattr(ai_name_community, "communities"):
            write_summary_file(
                ai_name_community.communities,
                format_slot_value(slot_start, slot_mode).replace(":", ""),
                format_slot_value(slot_end, slot_mode).replace(":", "")
            )
            
            os.makedirs("communities", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"communities/{timestamp}_{start_str}_to_{end_str}.json"
            
            communities_data = {
                "timeframe": {
                    "start": format_slot_value(slot_start, slot_mode),
                    "end": format_slot_value(slot_end, slot_mode)
                },
                "communities": [
                    {k: numpy_to_python(v) for k, v in comm.items()}
                    for comm in ai_name_community.communities
                ]
            }
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sanitize_json(communities_data), f, ensure_ascii=False, indent=2)
            print(f"[saved] {output_path}")

        # Create base legend data
        legend_data = {
            "hybrid": {
                "groups": {str(k): numpy_to_python(v) 
                          for k, v in partition_hybrid.items()}
            }
        }

        with open(legend_path, 'w', encoding='utf-8') as f:
            json.dump(sanitize_json(legend_data), f, ensure_ascii=False, indent=2)
        
        # Visualize the graph
        visualize_combined_dashboard(
            g, partition_louvain, partition_hybrid, filename
        )
        return filename
        
    except Exception as e:
        print(f"[error] Visualization failed for {start_str} to {end_str}: {e}")
        import traceback
        traceback.print_exc()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("<html><body>Error generating visualization</body></html>")
        with open(legend_path, 'w', encoding='utf-8') as f:
            json.dump(sanitize_json(legend_data), f)
        return filename


# ---- Combined Dashboard Visualization ----
def create_community_network():
    """Create a network with optimal visualization settings."""
    net = Network(
        height="100vh",
        width="100%",
        notebook=False,
        heading="",
        bgcolor="#ffffff",
        font_color="#333333"
    )
    
    # Configure network options for optimal visualization
    net.options = {
        "physics": {
            "enabled": True,
            "stabilization": {
                "enabled": True,
                "iterations": 100,
                "updateInterval": 50
            },
            "barnesHut": {
                "gravitationalConstant": -2000,
                "springConstant": 0.04,
                "springLength": 150
            }
        },
        "edges": {
            "color": "#999999",
            "width": 0.5,
            "smooth": False
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
            "zoomView": True,
            "dragNodes": True,
            "dragView": True
        }
    }
    return net


def save_network_html(net, filename):
    """Write a PyVis network without trying to open a browser."""
    net.write_html(filename)
    try:
        with open(filename, "r", encoding="utf-8") as f:
            html = f.read()
        if "codex-pyvis-fullscreen-css" not in html:
            html = html.replace(
                "<style type=\"text/css\">",
                """<style type=\"text/css\">
             /* codex-pyvis-fullscreen-css */
             html, body {
                 width: 100%;
                 height: 100%;
                 margin: 0;
                 padding: 0;
                 overflow: hidden;
             }
             body {
                 position: fixed;
                 inset: 0;
             }
             body > .card,
             .card,
             .card-body {
                 width: 100% !important;
                 height: 100% !important;
                 margin: 0 !important;
                 padding: 0 !important;
                 border: 0 !important;
                 border-radius: 0 !important;
                 box-shadow: none !important;
             }
             #mynetwork {
                 width: 100vw !important;
                 height: 100vh !important;
                 min-width: 100vw !important;
                 min-height: 100vh !important;
                 border: 0 !important;
                 float: none !important;
             }
""",
                1,
            )
        if "codex-pyvis-fit-to-frame" not in html:
            html = html.replace(
                "                  return network;",
                """                  // codex-pyvis-fit-to-frame
                  function codexFitNetwork() {
                      if (!network || !container) return;
                      network.setSize('100%', '100%');
                      network.redraw();
                      network.fit({
                          animation: false,
                          minZoomLevel: 0.08,
                          maxZoomLevel: 3.5
                      });
                  }
                  network.once("stabilizationIterationsDone", function() {
                      setTimeout(codexFitNetwork, 60);
                  });
                  window.addEventListener("load", function() {
                      setTimeout(codexFitNetwork, 120);
                  });
                  window.addEventListener("resize", function() {
                      setTimeout(codexFitNetwork, 60);
                  });
                  if (window.ResizeObserver) {
                      new ResizeObserver(function() {
                          setTimeout(codexFitNetwork, 60);
                      }).observe(container);
                  }

                  return network;""",
                1,
            )
        if html != open(filename, "r", encoding="utf-8").read():
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)
    except Exception as exc:
        print(f"[warning] Could not apply fullscreen CSS to {filename}: {exc}")


def visualize_combined_dashboard(g, louvain_partition, hybrid_partition, filename):
    """Generate the user-facing Hybrid visualization.

    Louvain stays in the pipeline for internal comparisons and reports, but no
    Louvain graph, similarity graph, legend, or iframe is written to root.
    """
    hybrid_html = filename.replace("dashboard_", "hybrid_graph_")
    legend_path = filename.replace(".html", "_legend.json")

    net2 = create_community_network()

    if g.number_of_nodes() == 0:
        empty_data = {"hybrid": {"groups": {}}}
        with open(legend_path, 'w', encoding='utf-8') as f:
            json.dump(empty_data, f, ensure_ascii=False, indent=2)
        save_network_html(net2, hybrid_html)
        print(f"[saved] {hybrid_html}, {legend_path}")
        return

    # Build only the visual partition shown to users.
    hybrid_colors = build_community_visualization(g, hybrid_partition, net2, "Hybrid")
    hybrid_colors = style_partition(g, net2, hybrid_partition, "hybrid", 200)

    save_network_html(net2, hybrid_html)
    print(f"[saved] {hybrid_html}")

    # Save only the user-facing community-level similarity graph.
    save_community_similarity_graph(g, hybrid_partition, "hybrid", filename)

    legend_data = {
        "hybrid": {"groups": hybrid_colors}
    }

    # Overwrite legend file (always regenerate)
    try:
        if os.path.exists(legend_path):
            os.remove(legend_path)
    except Exception:
        pass

    with open(legend_path, 'w', encoding='utf-8') as f:
        json.dump(sanitize_json(legend_data), f, ensure_ascii=False, indent=2)
    print(f"[saved] {legend_path}")

    # read utils and generate dashboard
    with open("lib/bindings/utils.js", encoding='utf-8') as f:
        utils_js = f.read()

    dashboard_html = generate_dashboard_html(
        hybrid_html, legend_path, utils_js, legend_data
    )
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    print(f"[saved] {filename}")
        # Build legend HTML for each method
def build_legend_html(method, community_data):
    """Build enhanced HTML for community legend with detailed information."""
    legend_html = []
    legend_html.append('<div class="legend">')
    legend_html.append(f'<h3>جوامع {method}</h3>')
    legend_html.append('<div class="legend-grid">')
    
    # Sort communities by size
    sorted_communities = sorted(
        community_data.items(),
        key=lambda x: x[1].get("size", 0),
        reverse=True
    )
    
    # Add enhanced legend items
    for name, info in sorted_communities:
        size = info.get("size", 0)
        color = info.get("color", "#cccccc")
        center = info.get("center", "")
        active_members = info.get("active_members", [])
        main_types = info.get("main_types", [])
        
        # Build tooltip content
        tooltip_parts = []
        if center:
            tooltip_parts.append(f"مرکز: {center}")
        if active_members:
            tooltip_parts.append(f"اعضای فعال: {', '.join(active_members[:3])}")
        if main_types:
            tooltip_parts.append(f"ویژگی‌ها: {' و '.join(main_types)}")
            
        tooltip = " | ".join(tooltip_parts)
        
        # Generate legend item HTML
        item_html = f'''
        <div class="legend-item" title="{tooltip}">
            <span class="color-box" style="background:{color}"></span>
            <span class="label">{name}</span>
            <span class="count">({size} عضو)</span>
        </div>
        '''
        legend_html.append(item_html.strip())
    
    legend_html.append('</div>')
    legend_html.append('</div>')
    
    return '\n'.join(legend_html)        # Build legend data with structured counting
def count_community_members(partition, net, label):
    """Count members of a community with given label."""
    return sum(
        1 for node in partition.keys()
        if label in net.get_node(node)['title']
    )

def process_community_data(method_name, partition, centers, community_texts):
    """Process community data for legend generation with detailed information."""
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    community_info = {}
    community_labels = set()  # Track used labels to avoid duplicates
    
    for comm_id, members in communities.items():
        center = centers.get(comm_id)
        if not center:
            continue
            
        # Get key active members and their texts
        all_member_texts = []
        member_activity = {}
        
        for member in members:
            texts = [t for t in community_texts if member in t]
            member_activity[member] = len(texts)
            all_member_texts.extend(texts)
        
        active_members = sorted(
            member_activity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Analyze community content deeply
        all_texts = " ".join(all_member_texts).lower()
        
        # Define hierarchical content patterns
        content_patterns = {
            "خبری": {
                "خبرگزاری": ["خبرگزاری", "روزنامه", "نشریه"],
                "خبرنگاران": ["خبرنگار", "گزارشگر", "روزنامه‌نگار"],
                "اطلاع‌رسانی": ["پوشش", "گزارش", "اطلاع‌رسانی"]
            },
            "تحلیلی": {
                "تحلیلگران سیاسی": ["تحلیل سیاسی", "تحلیلگر سیاسی", "تحلیل"],
                "پژوهشگران": ["پژوهش", "مطالعات", "تحقیقات"],
                "منتقدان": ["نقد", "ارزیابی", "بررسی تخصصی"]
            },
            "سیاسی": {
                "سیاست داخلی": ["دولت", "مجلس", "انتخابات", "وزیر"],
                "دیپلماسی": ["دیپلماسی", "روابط بین‌الملل", "سیاست خارجی"],
                "اصلاح‌طلبان": ["اصلاح‌طلب", "اصلاحات"],
                "اصولگرایان": ["اصولگرا", "ارزشی"],
                "فعالان سیاسی": ["فعال", "کنشگر"]
            },
            "اجتماعی": {
                "مدیریت شهری": ["شهرداری", "شورای شهر", "مدیریت شهری"],
                "حقوق شهروندی": ["حقوق شهروندی", "مطالبات مردمی", "حقوق مردم"],
                "آسیب‌شناسان": ["آسیب اجتماعی", "مشکلات", "معضلات"]
            },
            "فرهنگی": {
                "سینماگران": ["سینما", "فیلم", "کارگردان"],
                "اهالی تئاتر": ["تئاتر", "نمایش", "صحنه"],
                "نویسندگان": ["کتاب", "داستان", "رمان"],
                "شاعران": ["شعر", "شاعر", "ادبیات"],
                "موسیقی": ["موسیقی", "آهنگ", "خواننده"]
            },
            "اقتصادی": {
                "تحلیلگران بورس": ["بورس", "سهام", "بازار سرمایه"],
                "اقتصاددانان": ["اقتصاد", "تورم", "ارز"],
                "کارآفرینان": ["استارتاپ", "کارآفرینی", "کسب و کار"]
            },
            "مدنی": {
                "فعالان حقوق بشر": ["حقوق بشر", "عدالت", "حقوق زنان"],
                "کنشگران مدنی": ["کنشگر مدنی", "فعال مدنی", "مطالبه‌گر"],
                "فعالان اجتماعی": ["فعال اجتماعی", "جامعه مدنی"]
            },
            "فناوری": {
                "فعالان فناوری": ["آی‌تی", "فناوری اطلاعات", "تکنولوژی"],
                "استارتاپی": ["استارتاپ", "نوآوری", "فین‌تک"],
                "تولیدکنندگان محتوا": ["محتوا", "پادکست", "یوتیوب"]
            }
        }
        
        # Analyze content patterns hierarchically
        category_scores = defaultdict(int)
        subcategory_details = defaultdict(dict)
        
        for category, subcats in content_patterns.items():
            for subcat, patterns in subcats.items():
                score = sum(1 for p in patterns if p in all_texts)
                if score > 0:  # Only count if pattern appears
                    subcategory_details[category][subcat] = score
                    category_scores[category] += score
        
        # Find primary and secondary categories
        sorted_categories = sorted(
            [(k, v) for k, v in category_scores.items() if v > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Generate label based on most specific match
        if sorted_categories:
            main_category = sorted_categories[0][0]
            subcats = subcategory_details[main_category]
            top_subcat = max(subcats.items(), key=lambda x: x[1])
            
            base_name = top_subcat[0]
            suffix = None
            
            # Add meaningful secondary characteristic if available
            if len(sorted_categories) > 1:
                second_category = sorted_categories[1][0]
                second_subcats = subcategory_details[second_category]
                if second_subcats:
                    second_top = max(second_subcats.items(), key=lambda x: x[1])
                    if second_top[1] > 0:
                        suffix = second_top[0]
            
            # Build unique name
            name = base_name
            if suffix:
                name = f"{base_name} و {suffix}"
            
            # Ensure uniqueness
            original_name = name
            counter = 1
            while name in community_labels:
                counter += 1
                name = f"{original_name} ({counter})"
            
            community_labels.add(name)
        else:
            # Fallback: use center node characteristics
            name = DEFAULT_COMMUNITY_LABEL
            community_labels.add(name)

        name = coerce_allowed_label(name, default=DEFAULT_COMMUNITY_LABEL)
        
        # Store detailed community info
        active_member_info = [
            (member, count) 
            for member, count in active_members 
            if count > 0
        ]
        
        community_info[name] = {
            "size": len(members),
            "center": center,
            "active_members": [m[0] for m in active_member_info],
            "main_types": [
                k for k, v in sorted_categories[:2] 
                if v > 0
            ]
        }
    
    return community_info


def generate_legend_data(g, louvain_partition, hybrid_partition, 
                        net1, net2, louvain_colors, hybrid_colors):
    """Generate legend data in the correct format for dashboard display."""
    if g.number_of_nodes() == 0:
        return {
            "hybrid": {"groups": {}}
        }

    # Process Louvain communities
    louvain_groups = {}
    for node, comm_id in louvain_partition.items():
        node_title = net1.get_node(node).get('title', '')
        for label, color in louvain_colors.items():
            if f"({label})" in node_title:
                if label not in louvain_groups:
                    louvain_groups[label] = {
                        "color": color,
                        "count": 1
                    }
                else:
                    louvain_groups[label]["count"] += 1

    # Process Hybrid communities
    hybrid_groups = {}
    for node, comm_id in hybrid_partition.items():
        node_title = net2.get_node(node).get('title', '')
        for label, color in hybrid_colors.items():
            if f"({label})" in node_title:
                if label not in hybrid_groups:
                    hybrid_groups[label] = {
                        "color": color,
                        "count": 1
                    }
                else:
                    hybrid_groups[label]["count"] += 1

    return {"hybrid": {"groups": hybrid_groups}}


def initialize_community_names():
    """Initialize the community names file."""
    try:
        with open("community_names.txt", "w", encoding="utf-8") as f:
            f.write("=== نام‌های جوامع ===\n\n")
        print("فایل نام‌های جوامع پاکسازی شد.")
    except Exception as e:
        print(f"خطا در پاکسازی فایل نام‌ها: {e}")


def build_community_visualization(g, partition, network, method="Unknown"):
    """Build network visualization - uses names from ai_name_community."""
    # Initialize network with graph data
    network.from_nx(g)
    apply_edge_widths(network, g)
    method_key = normalize_method_key(method)
    meta_map = g.graph.get("meta_map", {})
    
    # Find community centers
    centers = get_community_centers(g, partition)
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    comm_info = {}
    
    # Simple labeling - real names already saved by ai_name_community
    for comm_id, members in communities.items():
        center = centers.get(comm_id)
        if not center:
            continue
        
        label = get_stored_community_label(g, method_key, comm_id, center)
        if not label:
            neighbors = [n for n in members if n != center]
            label = ai_name_community(center, neighbors, NODE_LABEL_MAP, comm_id, method_key, g=g)
        label = prefer_specific_political_label(label, members, meta_map)
        
        # Get active members
        active = sorted(members, key=lambda x: g.degree(x), reverse=True)[:3]
        
        comm_info[comm_id] = {
            'label': label,
            'size': len(members),
            'center': center,
            'members': members,
            'active': active,
            'color': None  # Will be assigned below
        }
    
    colors = {}  # Will map labels to colors
    
    # Sort communities by size
    sorted_comms = sorted(
        comm_info.items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )
    
    # Assign colors and style nodes
    stance_map = g.graph.get("stance_map", {})
    topic_label = g.graph.get("topic_label")
    for comm_id, info in sorted_comms:
        color = get_community_label_color(info['label'])
        info['color'] = color
        colors[info['label']] = color
        
        # Style community nodes
        summary, _ = aggregate_community_metadata(info["members"], meta_map)
        stance_info = compute_community_stance(info["label"], info["members"], stance_map, topic_label=topic_label)
        if stance_info:
            summary["stance"] = stance_info
        meta_summary = format_meta_summary(summary)
        for node in info['members']:
            is_center = (node == info['center'])
            is_active = node in info['active']
            
            # Calculate node size
            base_size = 15
            degree_boost = min(g.degree(node) * 2, 25)
            role_boost = 15 if is_center else 10 if is_active else 0
            size = base_size + degree_boost + role_boost
            
            # Build tooltip
            roles = []
            if is_center:
                roles.append("مرکز جامعه")
            if is_active:
                roles.append("عضو فعال")
                
            tooltip = f"{node}"
            if roles:
                tooltip += f" ({' - '.join(roles)})"
            tooltip += f"\nجامعه: {info['label']}"
            if meta_summary:
                tooltip += f"\n{meta_summary}"
            
            # Update node styling
            node_data = network.get_node(node)
            node_data.update({
                'color': color,
                'label': node,
                'title': tooltip,
                'size': size,
                'community_label': info['label'],
                'borderWidth': 2 if is_center else 1,
                'borderWidthSelected': 3,
                'font': {'size': 14, 'face': 'Vazirmatn'}
            })
    
    # Optimize network display
    network_options = {
        "physics": {
            "stabilization": {
                "enabled": True,
                "iterations": 100
            },
            "barnesHut": {
                "gravitationalConstant": -2000,
                "springConstant": 0.04,
                "springLength": 150
            }
        },
        "edges": {
            "color": {"inherit": False, "color": "#cccccc"},
            "width": 0.5,
            "smooth": {"enabled": False}
        }
    }
    if hasattr(network.options, "update"):
        network.options.update(network_options)
    else:
        network.options = network_options
    
    return colors


def generate_dashboard_html(hybrid_html, legend_path, utils_js, legend_data=None):
    """Generate the HTML for the user-facing Hybrid dashboard.
    
    Args:
        hybrid_html: Path to the hybrid algorithm HTML visualization file
        legend_path: Path to the JSON file containing legend data
        utils_js: JavaScript utility functions to include in the page
        
    Returns:
        str: The complete HTML document as a string
    
    Notes:
        Louvain stays available for internal comparison metrics, but it is not
        rendered or linked in the visual output.
    """
    style = '''
        html, body {
            height: 100%;
        }
        body {
            font-family: Vazirmatn, Tahoma, Arial, sans-serif;
            margin: 0;
            padding: 0;
            direction: rtl;
            background-color: #ffffff;
            color: #2c3e50;
            overflow: hidden;
        }
        h1, h2, h3 {
            color: #2c3e50;
            text-align: center;
            margin: 0;
        }
        h1 {
            display: none;
        }
        .container {
            display: flex;
            flex-direction: column;
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 10px;
            box-sizing: border-box;
            max-width: none;
            min-height: 0;
        }
        .graph-section {
            background: #ffffff;
            border: 1px solid #dfe3e8;
            border-radius: 6px;
            padding: 8px;
            display: flex;
            flex-direction: column;
            flex: 1 1 auto;
            min-width: 0;
            min-height: 0;
            height: 100%;
        }
        .graph-section h2 {
            font-size: 18px;
            line-height: 1.4;
            padding: 0 0 8px;
            flex: 0 0 auto;
        }
        iframe {
            border: none;
            width: 100%;
            height: 100%;
            flex: 1 1 auto;
            min-height: 0;
            border-radius: 3px;
            margin: 0;
        }
        .legend {
            display: none;
        }
        .legend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .legend-item {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 4px;
            padding: 8px;
            background: white;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .color-box {
            width: 16px;
            height: 16px;
            border-radius: 4px;
            margin-left: 8px;
            border: 1px solid rgba(0,0,0,0.1);
        }
        .label {
            flex: 1;
            min-width: 120px;
            font-size: 14px;
        }
        .count {
            color: #666;
            font-size: 12px;
            margin-right: 8px;
        }
        .meta {
            display: block;
            flex-basis: 100%;
            font-size: 12px;
            color: #6c757d;
            margin-right: 24px;
            line-height: 1.6;
        }
        @media (max-width: 1200px) {
            .legend-grid {
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            }
        }
    '''
    
    embedded_legend_json = json.dumps(
        sanitize_json(legend_data or {}),
        ensure_ascii=False
    )

    js_code = f'''
        const embeddedLegendData = {embedded_legend_json};

        async function loadLegends() {{
            try {{
                let data = embeddedLegendData;
                if (!data || !data.hybrid) {{
                    const response = await fetch("{legend_path}");
                    data = await response.json();
                }}

                const metaLabels = {{
                    political_label: "گرایش سیاسی",
                    category: "موضوع",
                    emotion: "احساس",
                    sentiment: "حس",
                    language: "زبان"
                }};

                function formatMeta(meta) {{
                    if (!meta) return '';
                    const parts = [];
                    if (meta.stance) {{
                        if (typeof meta.stance === 'object') {{
                            const target = meta.stance.target;
                            const stance = meta.stance.stance;
                            if (stance) {{
                                parts.push(
                                    target
                                        ? `موضع نسبت به ${{target}}: ${{stance}}`
                                        : `موضع: ${{stance}}`
                                );
                            }}
                        }} else {{
                            parts.push(`موضع: ${{meta.stance}}`);
                        }}
                    }}
                    for (const [key, label] of Object.entries(metaLabels)) {{
                        if (meta[key]) {{
                            parts.push(`${{label}}: ${{meta[key]}}`);
                        }}
                    }}
                    if (meta.echo) {{
                        const echo = meta.echo;
                        if (echo.is_echo_chamber !== undefined && echo.is_echo_chamber !== null) {{
                            parts.push(`اتاق پژواک: ${{echo.is_echo_chamber ? "بله" : "خیر"}}`);
                        }}
                        if (typeof echo.ei_index === "number") {{
                            parts.push(`E-I: ${{echo.ei_index.toFixed(2)}}`);
                        }}
                        if (typeof echo.conductance === "number") {{
                            parts.push(`هدایت: ${{echo.conductance.toFixed(2)}}`);
                        }}
                        if (typeof echo.content_homogeneity === "number") {{
                            parts.push(`همگنی: ${{echo.content_homogeneity.toFixed(2)}}`);
                        }}
                    }}
                    return parts.join(' | ');
                }}
                
                function buildLegendHTML(groups) {{
                    const items = Object.entries(groups).map(([label, info]) =>
                        (function() {{
                            const meta = info.meta ? formatMeta(info.meta) : '';
                            return `<div class="legend-item">
                        <span class="color-box" style="background:${{info.color}}"></span>
                        <span class="label">${{label}}</span>
                        <span class="count">(${{info.count}} عضو)</span>
                        ${{meta ? `<span class="meta">${{meta}}</span>` : ''}}
                        </div>`;
                        }})()
                    );
                    return `<div class="legend-grid">${{items.join('')}}</div>`;
                }}
                
                const hybridLegend = document.getElementById("hybrid-legend");
                
                data.hybrid?.groups && 
                    hybridLegend &&
                    (hybridLegend.innerHTML = buildLegendHTML(data.hybrid.groups));
            }} catch (error) {{
                console.error("Error loading legends:", error);
            }}
        }}
        window.addEventListener("load", loadLegends);
    '''
    
    body = f'''
    <h1>تشخیص جوامع در شبکه اجتماعی</h1>
    <div class="container">
        <div class="graph-section">
            <h2>روش ترکیبی</h2>
            <iframe src="{hybrid_html}"></iframe>
        </div>
    </div>
    '''
    
    return f'''<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تشخیص جوامع در شبکه اجتماعی</title>
    <script type="text/javascript">{utils_js}</script>
    <style>{style}</style>
</head>
<body>{body}
<script>{js_code}</script>
</body>
</html>'''
def save_community_name(community_name: str, center_node: str, neighbors: list, 
                       comm_id: int, method: str, time_period: str = None):
    """Save a community name and its details to the community names file."""
    print(f"[DEBUG SAVE] Writing: {community_name} (center: {center_node})")
    try:
        with open("community_names.txt", "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 50 + "\n")
            if time_period:
                f.write(f"دوره زمانی: {time_period}\n")
            f.write(f"الگوریتم: {method}\n")
            f.write(f"نام جامعه: {community_name}\n")
            f.write(f"مرکز: {center_node}\n")
            f.write(f"تعداد اعضا: {len(neighbors)}\n")
            if len(neighbors) <= 5:
                f.write(f"اعضا: {', '.join(neighbors)}\n")
            else:
                active = neighbors[:5]
                f.write(f"نمونه اعضا: {', '.join(active)}\n")
            f.write("-" * 50 + "\n")
    except Exception as e:
        print(f"[error] Failed to write community name: {e}")


def fetch_community_texts_from_file(center_node, neighbors, filepath="res.json", size=15):
    """Fetch texts - fallback to Elasticsearch if file data doesn't match."""
    try:
        members = set([center_node] + list(neighbors))
        texts_by_member = defaultdict(list)
        
        # Try Elasticsearch first (has correct usernames)
        try:
            print(f"[DEBUG FETCH] Trying Elasticsearch for {len(members)} members...")
            es_texts = fetch_community_texts(
                list(members),
                max_texts=size
            )
            
            if es_texts:
                print(f"[DEBUG FETCH] ✓ Elasticsearch: Found texts for {len(es_texts)} members")
                for member, texts in es_texts.items():
                    if texts:
                        texts_by_member[member] = texts[:size]
                
                result = dict(texts_by_member)
                if result:
                    sample = list(result.keys())[0]
                    print(f"[DEBUG]   Sample from {sample}: {result[sample][0][:60]}...")
                return result
        except Exception as e:
            print(f"[DEBUG] Elasticsearch not available: {e}")
        
        # Fallback: try files (probably won't work but let's try)
        print(f"[DEBUG] Falling back to file-based lookup...")
        
        if os.path.exists("res.json"):
            with open("res.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for item in data:
                sender = item.get("user_name")
                text = (item.get("normalized_text") or item.get("content") or "").strip()
                
                if sender in members and text and len(text) > 10:
                    texts_by_member[sender].append(text)
        
        result = {m: texts[:size] for m, texts in texts_by_member.items()}
        
        if result:
            print(f"[DEBUG FETCH] ✓ Files: Found {len(result)} members")
        else:
            print(f"[DEBUG FETCH] ✗ No texts found in files or ES")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] fetch failed: {e}")
        return {}


def process_visualization(g, louvain_partition, hybrid_partition, filename):
    """Process the user-facing Hybrid visualization."""
    hybrid_html = filename.replace("dashboard_", "hybrid_graph_")
    legend_path = filename.replace(".html", "_legend.json")
    visualize_combined_dashboard(g, louvain_partition, hybrid_partition, filename)
    return hybrid_html, legend_path


def generate_time_slots(start_date, end_date, slot_type):
    slots = []
    range_start = datetime.combine(start_date.date(), datetime.min.time())
    range_end = datetime.combine(end_date.date(), datetime.max.time())
    total_days = inclusive_day_count(range_start, range_end)
    current = range_start

    if slot_type == "weekly" and 29 <= total_days <= 31:
        for idx in range(4):
            if current > range_end:
                break
            if idx == 3:
                slot_end = range_end
            else:
                slot_end = min(current + timedelta(days=7) - timedelta(microseconds=1), range_end)
            slots.append((current, slot_end))
            current = slot_end + timedelta(microseconds=1)
        return slots

    while current <= range_end:
        if slot_type == "hourly":
            next_slot = current + timedelta(hours=1)
            slot_end = min(next_slot - timedelta(microseconds=1), range_end)
        elif slot_type == "daily":
            next_slot = current + timedelta(days=1)
            slot_end = min(next_slot - timedelta(microseconds=1), range_end)
        elif slot_type == "weekly":
            next_slot = current + timedelta(days=7)
            slot_end = min(next_slot - timedelta(microseconds=1), range_end)
        elif slot_type == "monthly":
            next_slot = current + relativedelta(months=1)
            slot_end = min(next_slot - timedelta(microseconds=1), range_end)
        elif slot_type == "quarterly":
            next_slot = current + relativedelta(months=3)
            slot_end = min(next_slot - timedelta(microseconds=1), range_end)
        else:
            raise ValueError(f"Unsupported slot type: {slot_type}")
        slots.append((current, slot_end))
        current = next_slot
    return slots


# --- Timeline Dashboard Generator ---
import os

def normalize_topic_key(label):
    value = (label or "").strip()
    if not value:
        return "__topic__"
    key = re.sub(r"\s+", "_", value)
    key = re.sub(r"[^\w#آ-ی\u200c_-]+", "_", key, flags=re.UNICODE)
    return key.strip("_") or "__topic__"


def load_topic_options(file_path="topics_10.csv", active_label=None, active_query=None):
    """Load lightweight topic presets for the static dashboard selector."""
    options = []
    seen = set()

    def add_option(label, query="", has_data=False):
        clean_label = (label or "").strip()
        if not clean_label:
            return
        key = normalize_topic_key(clean_label)
        if key in seen:
            return
        seen.add(key)
        options.append({
            "key": key,
            "label": clean_label,
            "query": (query or clean_label).strip(),
            "hasData": bool(has_data),
        })

    add_option(active_label, active_query or active_label, has_data=True)

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    add_option(
                        row.get("topic_label"),
                        row.get("topic_query"),
                        has_data=False,
                    )
        except Exception as exc:
            print(f"[warning] Could not load topic options from {file_path}: {exc}")

    return options


def load_topic_output_manifests(base_dir="topic_outputs"):
    """Load previously generated per-topic timeline manifests."""
    manifests = []
    if not os.path.isdir(base_dir):
        return manifests

    for name in sorted(os.listdir(base_dir)):
        manifest_path = os.path.join(base_dir, name, "topic_manifest.json")
        if not os.path.exists(manifest_path):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("key") and manifest.get("modes"):
                for mode in (manifest.get("modes") or {}).values():
                    for item in mode.get("items", []):
                        dashboard = item.get("dashboard")
                        if dashboard:
                            item["dashboard"] = quote(
                                unquote(str(dashboard)),
                                safe="/._-",
                            )
                manifests.append(manifest)
        except Exception as exc:
            print(f"[warning] Could not load topic manifest {manifest_path}: {exc}")
    return manifests


def build_timeline_topics(topic_label, timeline_modes, topic_options=None):
    active_key = normalize_topic_key(topic_label)
    options = topic_options or load_topic_options(active_label=topic_label)
    topics = {}

    for option in options:
        key = option.get("key") or normalize_topic_key(option.get("label"))
        topics[key] = {
            "label": option.get("label") or key,
            "query": option.get("query") or option.get("label") or key,
            "hasData": bool(option.get("hasData")),
            "modes": {},
        }

    known_option_keys = {option.get("key") for option in options}
    for manifest in load_topic_output_manifests():
        key = manifest.get("key")
        label = manifest.get("label") or key
        topics[key] = {
            "label": label,
            "query": manifest.get("query") or label,
            "hasData": True,
            "modes": manifest.get("modes") or {},
        }
        if key not in known_option_keys:
            options.append({
                "key": key,
                "label": label,
                "query": manifest.get("query") or label,
                "hasData": True,
            })
            known_option_keys.add(key)

    topics.setdefault(active_key, {
        "label": topic_label,
        "query": topic_label,
        "hasData": True,
        "modes": {},
    })
    topics[active_key]["hasData"] = True
    topics[active_key]["modes"] = timeline_modes
    return active_key, topics, options


def generate_timeline_dashboard(output_file="timeline_dashboard.html", topic_label=None,
                                mode_dashboards=None, topic_options=None):
    """Generate a dashboard that can switch between real slot modes."""
    mode_labels = {
        "hourly": "ساعتی",
        "daily": "روزانه",
        "weekly": "هفتگی",
        "monthly": "ماهانه",
        "quarterly": "فصلی",
    }

    def parse_slot_range(filename: str):
        try:
            match = re.search(
                r"dashboard_(?:[a-z_]+_)?(\d{6,8})_to_(\d{6,8})\.html$",
                os.path.basename(filename),
            )
            if not match:
                return None, None
            start_raw, end_raw = match.group(1), match.group(2)
            start_fmt = "%y%m%d%H" if len(start_raw) == 8 else "%y%m%d"
            end_fmt = "%y%m%d%H" if len(end_raw) == 8 else "%y%m%d"
            start_dt = datetime.strptime(start_raw, start_fmt)
            end_dt = datetime.strptime(end_raw, end_fmt)
            return start_dt, end_dt
        except Exception:
            return None, None

    def load_legend(dashboard_file):
        legend_file = dashboard_file.replace(".html", "_legend.json")
        if not os.path.exists(legend_file):
            return {"hybrid": {"groups": {}}}
        try:
            with open(legend_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"hybrid": data.get("hybrid", {"groups": {}})}
        except Exception:
            return {"hybrid": {"groups": {}}}

    if mode_dashboards is None:
        mode_dashboards = {"weekly": []}
        for filename in sorted(
            f for f in os.listdir(".")
            if f.startswith("dashboard_") and f.endswith(".html") and "_to_" in f
        ):
            start_dt, end_dt = parse_slot_range(filename)
            if not start_dt or not end_dt:
                continue
            mode_dashboards["weekly"].append({
                "file": filename,
                "start": start_dt.strftime("%Y-%m-%d"),
                "end": end_dt.strftime("%Y-%m-%d"),
            })

    timeline_modes = {}
    for mode, entries in mode_dashboards.items():
        items = []
        for entry in entries:
            dashboard_file = entry.get("file")
            if not dashboard_file or not os.path.exists(dashboard_file):
                continue
            start = entry.get("start")
            end = entry.get("end")
            if not start or not end:
                start_dt, end_dt = parse_slot_range(dashboard_file)
                start = start_dt.strftime("%Y-%m-%d") if start_dt else ""
                end = end_dt.strftime("%Y-%m-%d") if end_dt else ""
            items.append({
                "dashboard": dashboard_file,
                "range": f"{start} تا {end}" if start and end else "نامشخص",
                "legends": load_legend(dashboard_file),
                "partyFocus": entry.get("party_focus", {}),
            })
        if items:
            base_label = mode_labels.get(mode, mode)
            timeline_modes[mode] = {
                "label": base_label,
                "items": items,
            }

    if not timeline_modes:
        print("No dashboard files found for timeline modes.")
        return

    if topic_label is None:
        topic_label = load_topic_label_from_elastic() or "#همکاری_ملی"

    preferred_defaults = ["weekly", "daily", "monthly", "quarterly", "hourly"]
    default_mode = next(
        (mode for mode in preferred_defaults if mode in timeline_modes),
        next(iter(timeline_modes))
    )
    total_slots = len(timeline_modes[default_mode]["items"])
    active_topic, timeline_topics, topic_options = build_timeline_topics(
        topic_label,
        timeline_modes,
        topic_options=topic_options,
    )

    with open("timeline_template.html", "r", encoding="utf-8") as f:
        template = f.read()

    html = (
        template
        .replace("{timeline_modes}", json.dumps(sanitize_json(timeline_modes), ensure_ascii=False))
        .replace("{timeline_topics}", json.dumps(sanitize_json(timeline_topics), ensure_ascii=False))
        .replace("{topic_options}", json.dumps(sanitize_json(topic_options), ensure_ascii=False))
        .replace("{default_topic}", json.dumps(active_topic, ensure_ascii=False))
        .replace("{default_mode}", json.dumps(default_mode, ensure_ascii=False))
        .replace("{topic_label}", topic_label)
        .replace("{total_slots}", str(total_slots))
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Timeline dashboard saved to {output_file}")


# Print community results
if __name__ == "__main__":
    removed = clean_project_root()
    if removed:
        print(f"[cleanup] Removed {len(removed)} generated artifacts from project root.")
    # پاکسازی فایل اسامی در شروع برنامه
    try:
        with open("community_names.txt", "w", encoding="utf-8") as f:
            f.write("=== لیست کامل نام‌های جوامع تشخیص داده شده ===\n\n")
        print("[init] Community names file initialized")
    except Exception as e:
        print(f"[error] Failed to initialize community names file: {e}")
    print("[start] Running Louvain and Hybrid community detection...")
    meta_map, stance_map = load_user_context()
    config = load_pipeline_config()
    topic_label = (
        os.getenv("TOPIC_LABEL_OVERRIDE")
        or config.get("topic_label")
        or load_topic_label_from_elastic()
        or "جنگ جمهوری اسلامی و آمریکا"
    )
    start_date, end_date = get_pipeline_date_range(config)
    if start_date > end_date:
        raise ValueError(
            f"Invalid date range: {start_date.date()} is after {end_date.date()}"
        )

    slot_modes = config.get("slot_modes") or select_slot_modes_for_range(start_date, end_date)
    if isinstance(slot_modes, str):
        slot_modes = [
            mode.strip()
            for mode in re.split(r"[,،;\s]+", slot_modes)
            if mode.strip()
        ]
    valid_modes = {"hourly", "daily", "weekly", "monthly", "quarterly"}
    slot_modes = [mode for mode in slot_modes if mode in valid_modes]
    if not slot_modes:
        slot_modes = select_slot_modes_for_range(start_date, end_date)

    print(
        "[config] topic={topic} range={start}..{end} modes={modes}".format(
            topic=topic_label,
            start=start_date.date(),
            end=end_date.date(),
            modes=", ".join(slot_modes),
        )
    )
    all_messages = load_interactions()
    mode_dashboards = {}

    for slot_mode in slot_modes:
        slots = generate_time_slots(start_date, end_date, slot_mode)
        mode_dashboards[slot_mode] = []
        previous_node_parties = {}
        print(f"\n[mode] {slot_mode}: {len(slots)} slots")

        for idx, (slot_start, slot_end) in enumerate(slots):
            _ai_name_cache.clear()
            print(
                f"\n[slot {idx+1}/{len(slots)} | {slot_mode}] "
                f"Processing: {slot_start.date()} to {slot_end.date()}..."
            )
            messages = []
            for msg in all_messages:
                date_str = msg.get("date")
                msg_dt = parse_message_datetime(date_str)
                if not msg_dt:
                    print(f"[skip] Invalid date format: {date_str}")
                    continue
                if slot_start <= msg_dt <= slot_end:
                    messages.append(msg)

            dashboard_file = None
            if not messages:
                print("[skip] No valid messages found in this time slot")
                dashboard_file = visualize_or_dummy(
                    slot_start, slot_end, nx.Graph(), slot_mode=slot_mode
                )
            else:
                g_slot = build_user_graph(messages)
                g_slot = clean_graph(g_slot)
                attach_metadata_to_graph(
                    g_slot, meta_map, stance_map, topic_label=topic_label
                )
                if g_slot.number_of_nodes() == 0:
                    print("[skip] Empty graph after filtering")
                    dashboard_file = visualize_or_dummy(
                        slot_start, slot_end, g_slot, slot_mode=slot_mode
                    )
                else:
                    partition_louvain = detect_communities_louvain(g_slot)
                    embeddings, nodes = get_node_embeddings(g_slot)
                    labels = run_kmeans(embeddings, n_clusters=5)
                    partition_hybrid = {
                        node: labels[i] for i, node in enumerate(nodes)
                    }
                    report = build_hybrid_report(
                        g_slot,
                        embeddings,
                        nodes,
                        labels,
                        louvain_partition=partition_louvain,
                        start_date=format_slot_value(slot_start, slot_mode),
                        end_date=format_slot_value(slot_end, slot_mode)
                    )
                    save_hybrid_report(report)
                    dashboard_file = visualize_or_dummy(
                        slot_start, slot_end, g_slot,
                        partition_louvain, partition_hybrid,
                        slot_mode=slot_mode,
                    )
                    party_changes = collect_node_party_changes(
                        g_slot,
                        previous_node_parties,
                        current_slot=format_slot_value(slot_start, slot_mode),
                        slot_mode=slot_mode,
                        topic_label=topic_label,
                    )
                    post_party_change_events(party_changes)

            if dashboard_file:
                mode_dashboards[slot_mode].append({
                    "file": dashboard_file,
                    "start": format_slot_value(slot_start, slot_mode),
                    "end": format_slot_value(slot_end, slot_mode),
                    "party_focus": build_party_focus_for_messages(messages, meta_map),
                })

    print("[done] Community detection completed.")
    generate_timeline_dashboard(
        mode_dashboards=mode_dashboards,
        topic_label=topic_label,
    )
    removed = clean_project_root()
    if removed:
        print(f"[cleanup] Rotated {len(removed)} generated artifacts from project root.")
