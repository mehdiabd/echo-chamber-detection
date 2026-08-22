"""Module providing a flexible Elasticsearch query script."""
import os
import sys
import subprocess
import json
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from collections import Counter
import logging
import re
from datetime import datetime

from elastic_query import (
    DEFAULT_LOOKBACK_DAYS,
    build_topic_clause,
    parse_topic_terms,
    resolve_date_range,
)

# Set up logging to both console and file
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# File handler for internal logs
file_handler = logging.FileHandler("debug.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Console handler for system logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Show warnings/errors from dependencies
console_formatter = logging.Formatter("%(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)

log.addHandler(file_handler)
log.addHandler(console_handler)

def extract_mentions(text):
    return re.findall(r"@(\w+)", text or "")


def normalize_tweet_type(value):
    normalized = str(value or "").strip().lower()
    if normalized in {"quote", "qoute", "quoted"}:
        return "quote"
    if normalized in {"repost", "retweet", "re-tweet", "rt"}:
        return "repost"
    if normalized in {"post", "tweet", "original"}:
        return "post"
    return normalized


def same_user(left, right):
    return bool(left and right and left.casefold() == right.casefold())


DEFAULT_TOPIC_LABEL = "جنگ جمهوری اسلامی و آمریکا"
PIPELINE_CONFIG_PATH = "pipeline_config.json"

DEFAULT_WAR_TOPIC_TERMS = [
    "#جنگ_ایران_آمریکا",
    "#جنگ_ایران_و_آمریکا",
    "#ایران_آمریکا",
    "#حمله_آمریکا",
    "#حمله_به_ایران",
    "#حمله_به_فردو",
    "#جنگ_۱۲_روزه",
    "#جنگ_دوازده_روزه",
    "#تنگه_هرمز",
    "#پایگاه_العدید",
    "جنگ ایران آمریکا",
    "جنگ ایران و آمریکا",
    "جنگ جمهوری اسلامی آمریکا",
    "جنگ جمهوری اسلامی و آمریکا",
    "حمله آمریکا به ایران",
    "حمله آمریکا",
    "حمله به ایران",
    "حمله به فردو",
    "حمله به نطنز",
    "حمله به اصفهان",
    "تاسیسات هسته‌ای ایران",
    "تاسیسات هسته ای ایران",
    "فردو",
    "نطنز",
    "اصفهان",
    "پایگاه العدید",
    "العدید",
    "تنگه هرمز",
    "عملیات چکش نیمه شب",
    "چکش نیمه شب",
    "Midnight Hammer",
    "Operation Midnight Hammer",
    "Fordow",
    "Natanz",
    "Isfahan",
    "Al Udeid",
    "Strait of Hormuz",
]


def select_slot_modes(start, end):
    days = (end - start).days + 1
    if days <= 8:
        return ["daily", "hourly"]
    if days <= 45:
        return ["weekly", "daily"]
    if days <= 90:
        return ["monthly", "weekly", "daily"]
    if days <= 180:
        return ["monthly", "weekly"]
    return ["quarterly", "monthly", "weekly"]


def get_nested_value(data, *path):
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_text_value(*values):
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_user_handle(value):
    if isinstance(value, dict):
        value = first_text_value(
            value.get("user_name"),
            value.get("username"),
            value.get("screen_name"),
            value.get("name"),
        )
    if not isinstance(value, str):
        return None
    handle = value.strip()
    while handle.startswith("@"):
        handle = handle[1:].strip()
    return handle or None


def get_reply_target(source):
    reply = source.get("reply")
    return normalize_user_handle(
        first_text_value(
            source.get("reply_to_user"),
            source.get("in_reply_to_user"),
            source.get("in_reply_to_screen_name"),
            get_nested_value(source, "reply_to", "user_name"),
            get_nested_value(source, "reply_to", "user", "user_name"),
            get_nested_value(reply, "user_name"),
            get_nested_value(reply, "user", "user_name"),
        )
    )


def get_quote_target(source):
    quote = source.get("quote")
    quoted_status = source.get("quoted_status")
    return normalize_user_handle(
        first_text_value(
            get_nested_value(quote, "user", "user_name"),
            get_nested_value(quote, "user_name"),
            get_nested_value(quoted_status, "user", "user_name"),
            get_nested_value(quoted_status, "user_name"),
        )
    )


def get_repost_target(source):
    repost = source.get("repost")
    retweet = source.get("retweet")
    retweeted_status = source.get("retweeted_status")
    return normalize_user_handle(
        first_text_value(
            get_nested_value(repost, "user", "user_name"),
            get_nested_value(repost, "user_name"),
            get_nested_value(retweet, "user", "user_name"),
            get_nested_value(retweet, "user_name"),
            get_nested_value(retweeted_status, "user", "user_name"),
            get_nested_value(retweeted_status, "user_name"),
        )
    )

# Prompt the user to select the authentication type
# print("Select the Elasticsearch server (authentication type):")
# print("1. Production Elasticsearch server (API Key Authentication)")
# print("2. Temp Elasticsearch server (Basic Authentication)")
# auth_type = input("Enter 1 or 2: ").strip()

# Set variables based on the selected authentication type
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--auth", type=str, choices=["1", "2"], default="1", help="Auth type: 1=prod, 2=temp")
parser.add_argument("--topic-label", default=os.getenv("TOPIC_LABEL", DEFAULT_TOPIC_LABEL))
parser.add_argument(
    "--topic-query",
    default=os.getenv("TOPIC_QUERY", ""),
    help="OR/comma/pipe/newline separated topic phrases. Defaults to Iran-US war terms.",
)
parser.add_argument("--start-date", default=os.getenv("START_DATE", ""))
parser.add_argument("--end-date", default=os.getenv("END_DATE", ""))
parser.add_argument(
    "--days",
    type=int,
    default=int(os.getenv("LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS)),
    help="Inclusive lookback window when --start-date is omitted (default: 14 days).",
)
parser.add_argument("--include-secondary", action="store_true", help="Compatibility flag; secondary metadata is already fetched.")
parser.add_argument("--max-scan-docs", type=int, default=int(os.getenv("MAX_SCAN_DOCS", "0")))
args = parser.parse_args()
auth_type = args.auth

if auth_type == "1":
    # API Key Authentication
    # print("You selected Production Elasticsearch server (API Key Authentication).")
    cur_path = os.path.dirname(__file__)
    CERTIFICATE = os.path.join(cur_path, "ca.crt")
    ELASTICSEARCH_URL = "https://192.168.59.79:9200"
    AUTH = "YXYyeVRKWUJKSFpwMVdrTnZWRDc6UHhqRHBQa2ZUYW1yMnBwWTV3Ri0xUQ=="
    INDEX = "twitter_temp_data"

    # Create Elasticsearch client
    es = Elasticsearch(
        ELASTICSEARCH_URL,
        api_key=AUTH,
        ca_certs=CERTIFICATE,
        verify_certs=True,
        ssl_show_warn=False
    )
elif auth_type == "2":
    # Basic Authentication
    # print("You selected Temp Elasticsearch server (Basic Authentication).")
    cur_path = os.path.dirname(__file__)
    CERTIFICATE = os.path.join(cur_path, "http_ca.crt")
    ELASTICSEARCH_URL = "https://192.168.59.26:9200/"
    USERNAME = "m.abdolahi"
    PASSWORD = "@bd0l@h12345"
    INDEX = "twitter_maroufi"

    # Create Elasticsearch client
    es = Elasticsearch(
        ELASTICSEARCH_URL,
        basic_auth=(USERNAME, PASSWORD),
        verify_certs=True,
        ca_certs=CERTIFICATE,
        ssl_show_warn=False,
        ssl_assert_hostname=False
    )
else:
    # print("Invalid selection. Please run the script again and select 1 or 2.")
    sys.exit(1)

lookback_days = max(1, args.days)
start_date, end_date = resolve_date_range(
    args.start_date,
    args.end_date,
    lookback_days=lookback_days,
)
range_days = (end_date - start_date).days + 1
topic_terms = parse_topic_terms(args.topic_query) or DEFAULT_WAR_TOPIC_TERMS

query_body = {
    "track_total_hits": True,
    "query": {
        "bool": {
            "must": [
                build_topic_clause(topic_terms)
            ],
            "filter": [
                {
                    "range": {
                        "date": {
                            "gte": start_date.isoformat(),
                            "lte": end_date.isoformat(),
                            "format": "yyyy-MM-dd"
                        }
                    }
                }
            ]
        }
    },
    "_source": [
        "user_name",
        "normalized_text",
        "text",
        "content",
        "text",
        "comment",
        "keywords",
        "user_title",
        "political_category.label",
        "date",
        "timestamp",
        "crawl_date",
        "type",
        "reply_to_user",
        "reply_to",
        "in_reply_to_user",
        "in_reply_to_screen_name",
        "reply",
        "quote",
        "repost",
        "retweet",
        "retweeted_status",
        "quoted_status",
        "entity.mention",
        "entity.hashtag",
        "reply",
        "reply_to",
        "reply_to_user",
        "in_reply_to_user",
        "in_reply_to_screen_name",
        "quote",
        "quoted_status",
        "repost",
        "retweet",
        "retweeted_status",
        "category.label",
        "emotion.label",
    ]
}

pipeline_config = {
    "topic_label": args.topic_label,
    "topic_terms": topic_terms,
    "start_date": start_date.isoformat(),
    "end_date": end_date.isoformat(),
    "lookback_days": range_days,
    "slot_modes": select_slot_modes(start_date, end_date),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}
with open(PIPELINE_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(pipeline_config, f, ensure_ascii=False, indent=2)
log.info(
    "Pipeline config: topic=%s range=%s..%s terms=%s",
    args.topic_label,
    start_date.isoformat(),
    end_date.isoformat(),
    len(topic_terms),
)

type_counter = Counter()

def _extract_user(obj):
    if not obj:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("user_name", "username", "screen_name", "user_id"):
            if obj.get(key):
                return obj.get(key)
        user = obj.get("user")
        if isinstance(user, dict):
            for key in ("user_name", "username", "screen_name", "user_id"):
                if user.get(key):
                    return user.get(key)
    return None

def _log_count(label, query):
    try:
        res = es.count(index=INDEX, body={"query": query})
        log.warning(f"[count] {label}: {res.get('count')}")
    except Exception as e:
        log.warning(f"[count] {label} failed: {e}")

# Diagnostics: check whether data exists in date range and whether query filters too hard
try:
    date_only_query = {"range": {"date": {"gte": start_date, "lte": end_date, "format": "yyyy-MM-dd"}}}
    _log_count("date_only", date_only_query)
    _log_count("query_only", query_body["query"]["bool"]["must"][0])
    _log_count("query_and_date", query_body["query"])
    # Index min/max date for alignment
    try:
        min_date = es.search(
            index=INDEX,
            body={"sort": [{"date": "asc"}], "_source": ["date"], "size": 1}
        )
        max_date = es.search(
            index=INDEX,
            body={"sort": [{"date": "desc"}], "_source": ["date"], "size": 1}
        )
        min_hit = (min_date.get("hits", {}).get("hits") or [{}])[0].get("_source", {})
        max_hit = (max_date.get("hits", {}).get("hits") or [{}])[0].get("_source", {})
        min_val = str(min_hit.get("date", ""))[:10]
        max_val = str(max_hit.get("date", ""))[:10]
        if min_val and max_val:
            log.warning(f"[date_range] index min={min_val} max={max_val}")
    except Exception as e:
        log.warning(f"[date_range] failed: {e}")
    # Simple term test to confirm field matches
    test_term = "ایران"
    test_query = {
        "simple_query_string": {
            "query": test_term,
            "fields": ["normalized_text", "text"],
            "default_operator": "OR",
            "flags": "OR|AND|NOT|PHRASE|PRECEDENCE",
            "lenient": True,
        }
    }
    _log_count("test_term_only", test_query)
    # Sample one doc to inspect available fields
    try:
        sample = es.search(index=INDEX, body={"query": {"match_all": {}}, "size": 1})
        hits = sample.get("hits", {}).get("hits", [])
        if hits:
            src = hits[0].get("_source", {})
            log.warning(f"[sample] keys: {sorted(src.keys())[:30]}")
            for k in ["normalized_text", "content", "text", "date", "type"]:
                if k in src:
                    val = src.get(k)
                    if isinstance(val, str):
                        log.warning(f"[sample] {k}: {val[:200]}")
                    else:
                        log.warning(f"[sample] {k}: {type(val).__name__}")
        else:
            log.warning("[sample] no hits for match_all")
    except Exception as e:
        log.warning(f"[sample] failed: {e}")
    # Sample docs with non-empty content-like fields
    for field in ["normalized_text", "text", "comment", "keywords", "content"]:
        try:
            sample_q = {"query": {"exists": {"field": field}}, "size": 1}
            sample = es.search(index=INDEX, body=sample_q)
            hits = sample.get("hits", {}).get("hits", [])
            if not hits:
                continue
            src = hits[0].get("_source", {})
            val = src.get(field)
            if isinstance(val, str) and val.strip():
                log.warning(f"[sample_nonempty] {field}: {val[:200]}")
                break
            if isinstance(val, list) and val:
                log.warning(f"[sample_nonempty] {field}: {str(val[:5])}")
                break
        except Exception:
            continue
except Exception as e:
    log.warning(f"[count] diagnostics skipped: {e}")

log.info("Starting initial scan query...")
primary_scan = scan(
    es,
    index=INDEX,
    query=query_body,
    preserve_order=True,
    scroll="10m"
)

# Extract all distinct author usernames.
usernames = set()
COUNT = 0
with open("res.json", "w", encoding="utf-8") as f:
    for doc in primary_scan:
        source = doc["_source"]
        raw_type = source.get("type")
        tweet_type = normalize_tweet_type(raw_type)
        if tweet_type not in {"post", "reply", "quote", "repost"}:
            log.warning(f"❓ Unrecognized tweet_type: {tweet_type} (raw={raw_type})")
        type_counter[tweet_type] += 1
        json.dump(source, f, ensure_ascii=False)
        f.write("\n")
        COUNT += 1
        uname = normalize_user_handle(source.get("user_name"))
        if uname:
            usernames.add(uname)
        if args.max_scan_docs and COUNT >= args.max_scan_docs:
            log.info(f"Reached --max-scan-docs={args.max_scan_docs} in initial scan.")
            break
log.info(f"Total hits: {COUNT}")
log.info(f"TWEET TYPE COUNTS IN SCAN 1: {dict(type_counter)}")

# Rewind and re-scan to collect interactions
log.info("Starting scan to collect interactions...")
interaction_scan = scan(
    es,
    index=INDEX,
    query=query_body,
    preserve_order=True
)

interaction_usernames = set(usernames)
with open("interactions.json", "w", encoding="utf-8") as f_interactions:
    written = 0
    scanned_docs = 0
    EXAMPLE_LIMIT = 10
    EXAMPLE_PRINTED = 0
    for doc in interaction_scan:
        scanned_docs += 1
        if args.max_scan_docs and scanned_docs > args.max_scan_docs:
            log.info(f"Reached --max-scan-docs={args.max_scan_docs} in interaction scan.")
            break
        source = doc["_source"]
        raw_type = source.get("type")
        tweet_type = normalize_tweet_type(raw_type)
        # Only warn for truly unrecognized tweet types
        if tweet_type not in {"post", "reply", "quote", "repost"}:
            log.warning(f"❓ Unrecognized tweet_type: {tweet_type} (raw={raw_type})")
        # Remove repetitive example doc logging
        # Remove repetitive info log for every doc
        # Remove debug log for every tweet type
        sender = normalize_user_handle(source.get("user_name"))
        date_str = source.get("date", "")

        if not sender or not tweet_type or not date_str:
            continue

        # Handle structured interaction types
        if tweet_type == "reply":
            target = get_reply_target(source)
            if not target:
                # Only warn once for missing target user
                # log.warning("⛔ Skipped reply: No target user.")
                continue
            elif same_user(target, sender):
                # log.warning("⛔ Skipped reply: Target same as sender.")
                continue
            else:
                interaction = {
                    "sender": sender,
                    "target": target,
                    "type": "reply",
                    "date": date_str
                }
                log.info(f"Writing interaction: {interaction}")
                json.dump(interaction, f_interactions, ensure_ascii=False)
                f_interactions.write("\n")
                interaction_usernames.add(target)
                written += 1
        elif tweet_type == "quote":
            quoted_user = get_quote_target(source)
            if not quoted_user:
                # log.warning("⛔ Skipped quote: No quoted user.")
                continue
            elif same_user(quoted_user, sender):
                # log.warning("⛔ Skipped quote: Self-quote.")
                continue
            else:
                interaction = {
                    "sender": sender,
                    "target": quoted_user,
                    "type": "quote",
                    "date": date_str
                }
                log.info(f"Writing interaction: {interaction}")
                json.dump(interaction, f_interactions, ensure_ascii=False)
                f_interactions.write("\n")
                interaction_usernames.add(quoted_user)
                written += 1

        elif tweet_type == "repost":
            retweeted_user = get_repost_target(source)
            if not retweeted_user:
                # log.warning("⛔ Skipped repost: No retweeted user.")
                continue
            elif same_user(retweeted_user, sender):
                # log.warning("⛔ Skipped repost: Self-repost.")
                continue
            else:
                interaction = {
                    "sender": sender,
                    "target": retweeted_user,
                    "type": "repost",
                    "date": date_str
                }
                log.info(f"Writing interaction: {interaction}")
                json.dump(interaction, f_interactions, ensure_ascii=False)
                f_interactions.write("\n")
                interaction_usernames.add(retweeted_user)
                written += 1

        elif tweet_type == "post":
            # log.warning("🟡 Processing a POST tweet...")
            entity = source.get("entity")
            mentions = entity.get("mention", []) if isinstance(entity, dict) else []
            if not mentions:
                text = (
                    source.get("normalized_text")
                    or source.get("text")
                    or source.get("content")
                    or ""
                )
                mentions = extract_mentions(text)
                # log.warning(f"🔍 Mentions found after fallback: {mentions}")
            
            if mentions:
                for mention in mentions:
                    target = normalize_user_handle(mention)
                    if target and not same_user(target, sender):
                        interaction = {
                            "sender": sender,
                            "target": target,
                            "type": "mention",
                            "date": date_str
                        }
                        log.info(f"Writing interaction: {interaction}")
                        json.dump(interaction, f_interactions, ensure_ascii=False)
                        f_interactions.write("\n")
                        interaction_usernames.add(target)
                        written += 1

            elif tweet_type == "post":
                # log.warning("🟡 Processing a POST tweet...")
                mentions = source.get("entity", {}).get("mention", [])
                normalized_mentions = []
                if isinstance(mentions, list):
                    for mention in mentions:
                        if isinstance(mention, str):
                            normalized_mentions.append(mention)
                        elif isinstance(mention, dict):
                            name = (
                                mention.get("user_name")
                                or mention.get("username")
                                or mention.get("screen_name")
                                or mention.get("name")
                            )
                            if name:
                                normalized_mentions.append(name)
                text_blob = source.get("normalized_text") or source.get("content") or ""
                if normalized_mentions:
                    mentions = normalized_mentions
                else:
                    mentions = extract_mentions(text_blob)
                    # log.warning(f"🔍 Mentions found after fallback: {mentions}")

                if mentions:
                    for mention in mentions:
                        target = mention
                        if target and target != sender:
                            interaction = {
                                "sender": sender,
                                "target": target,
                                "type": "mention",
                                "date": date_str
                            }
                            log.info(f"Writing interaction: {interaction}")
                            json.dump(interaction, f_interactions, ensure_ascii=False)
                            f_interactions.write("\n")
                            written += 1
                else:
                    # Only warn once for missing mentions
                    # log.warning("⛔ Skipped post: No mentions found (entity or text).")
                    continue

log.warning(f"✅ TOTAL INTERACTIONS WRITTEN: {written}")
log.info("Extracted interactions written to interactions.json")
log.info("Finished writing interactions.json.")
log.info("Finished processing all interaction types.")

# Query twitter_source for political tags
log.info("Querying twitter_source for political tags...")

def chunked(iterable, size):
    iterable = list(iterable)
    return [iterable[i:i+size] for i in range(0, len(iterable), size)]


def search_secondary_metadata(query, batch_label):
    try:
        return list(scan(
            es,
            index="twitter_source",
            query=query,
            size=1000,
            scroll="5m",
            request_timeout=60,
        ))
    except Exception as exc:
        log.warning(f"Skipping secondary metadata batch {batch_label}: {exc}")
        return []

log.info("Chunking sender and target usernames for exact political tagging...")
metadata_usernames = sorted(interaction_usernames, key=str.casefold)
username_chunks = chunked(metadata_usernames, 500)

total_batches = len(username_chunks)
BATCH_INDEX = 1
SECONDARY_COUNT = 0
WRITTEN_METADATA_USERS = set()

with open("res.json", "w", encoding="utf-8") as f:
    for chunk in username_chunks:
        log.info(f"Querying batch {BATCH_INDEX}/{total_batches} [usernames]")
        query = {
            "query": {
                # `user_name` is the verified exact field in twitter_source.
                "terms": {"user_name": chunk}
            },
            "_source": [
                "user_name",
                "normalized_text",
                "text",
                "content",
                "user_title",
                "political_category.label",
                "type",
                "entity.mention",
                "category.label",
                "emotion.label",
                "entity.hashtag",
                "language.label",
                "ner.event",
                "ner.location",
                "ner.nationality",
                "ner.organ",
                "ner.person",
                "ner.politic_group",
                "offensive.label",
                "sentiment.label"
            ]
        }
        results = search_secondary_metadata(query, f"{BATCH_INDEX}/{total_batches} usernames")
        for doc in results:
            source = doc["_source"]
            uname = source.get("user_name")
            dname = source.get("user_title")
            political = source.get("political_category", {})
            label = political.get("label") if isinstance(political, dict) else None
            normalized_uname = normalize_user_handle(uname)
            metadata_key = normalized_uname.casefold() if normalized_uname else None
            if normalized_uname and label and metadata_key not in WRITTEN_METADATA_USERS:
                output_doc = {
                    "user_name": normalized_uname,
                    "user_title": dname,
                    "political_label": label,
                    "normalized_text": source.get("normalized_text"),
                    "content": source.get("text") or source.get("content"),
                    "category": get_nested_value(source, "category", "label"),
                    "emotion": get_nested_value(source, "emotion", "label"),
                    "hashtag": get_nested_value(source, "entity", "hashtag"),
                    "language": get_nested_value(source, "language", "label"),
                    "event": get_nested_value(source, "ner", "event"),
                    "location": get_nested_value(source, "ner", "location"),
                    "nationality": get_nested_value(source, "ner", "nationality"),
                    "organization": get_nested_value(source, "ner", "organ"),
                    "person": get_nested_value(source, "ner", "person"),
                    "politic_group": get_nested_value(source, "ner", "politic_group"),
                    "offensive": get_nested_value(source, "offensive", "label"),
                    "sentiment": get_nested_value(source, "sentiment", "label")
                }
                json.dump(output_doc, f, ensure_ascii=False)
                f.write("\n")
                WRITTEN_METADATA_USERS.add(metadata_key)
                SECONDARY_COUNT += 1
        BATCH_INDEX += 1

log.info(f"Political tag hits: {SECONDARY_COUNT}")
log.info(
    "Political metadata coverage: %s/%s interaction accounts",
    SECONDARY_COUNT,
    len(metadata_usernames),
)
log.info("Finished writing political tags to res.json.")

# Show preview of res.json (last 3 lines)
with open("res.json", "r", encoding="utf-8") as f:
    lines = f.readlines()
    preview = lines[-3:] if len(lines) >= 3 else lines
    print("\n--- Preview of last 3 lines of res.json ---")
    for line in preview:
        print(line.strip())

log.info("All steps completed successfully.")
log.info("Finished full pipeline.")
