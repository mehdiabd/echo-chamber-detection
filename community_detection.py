"""Community Detection on Sample Social Graph via NetworkX and community-louvain"""
import re
import json
from collections import defaultdict
import ollama
import requests
import os
from typing import Dict, Any, List
from pyvis.network import Network
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from node2vec import Node2Vec
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from elastic import es, INDEX, log as es_log
from elasticsearch.helpers import scan


def fetch_community_texts(accounts: List[str], start_date: str = None, 
                        end_date: str = None, max_texts: int = 10) -> Dict[str, List[str]]:
    """Fetch recent texts for a list of accounts from Elasticsearch."""
    try:
        # Build query
        query = {
            "bool": {
                "must": [{"terms": {"sender": accounts}}]
            }
        }

        # Add date range if provided
        if start_date and end_date:
            query["bool"]["filter"] = [{
                "range": {
                    "date": {
                        "gte": start_date,
                        "lte": end_date
                    }
                }
            }]

        # Execute search
        results = scan(
            es,
            query={"query": query},
            index=INDEX,
            _source=["sender", "text", "date"],
            size=100
        )

        # Collect texts by account
        texts_by_account = defaultdict(list)
        for hit in results:
            source = hit["_source"]
            sender = source.get("sender")
            text = source.get("text", "").strip()
            if sender and text:
                texts_by_account[sender].append(text)

        # Trim to max_texts per account
        return {
            account: texts[:max_texts] 
            for account, texts in texts_by_account.items()
        }
 
    except Exception as e:
        es_log.error(f"Error fetching texts: {e}")
        return {}


_ai_name_cache = {}


# --- Detect company LLM
try:
    resp = requests.get("http://192.168.59.239:8002/v1/models", timeout=5)
    ORG_LLM_MODEL = resp.json()["data"][0]["id"]
    ORG_LLM_URL = "http://192.168.59.239:8002/v1/chat/completions"
    print(f"[Org LLM] Running model: {ORG_LLM_MODEL}")
except Exception as e:
    ORG_LLM_MODEL = None
    ORG_LLM_URL = None
    print(f"[Org LLM] Not available → {e}")


def call_llm(prompt: str, backend="org"):
    """
    Generic LLM caller.
    backend: "org" or "local_llama"
    """
    if backend == "org":
        if not ORG_LLM_MODEL or not ORG_LLM_URL:
            return None
        try:
            payload = {
                "model": ORG_LLM_MODEL,
                "messages": [
                    {"role": "system", "content":
                        "شما یک متخصص در نام‌گذاری جوامع و گروه‌های اجتماعی هستید."},
                    {"role": "user", "content": prompt}
                ]
            }
            resp = requests.post(ORG_LLM_URL, json=payload, timeout=30)
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Org LLM] Failed → {e}")
            return None
    elif backend == "local_llama":
        try:
            response = ollama.chat(
                model="llama3.1",
                    messages=[
                    {
                        "role": "system",
                        "content": (
                            "تو یک متخصص در نام‌گذاری جوامع و "
                            "گروه‌های اجتماعی رسانه‌های اجتماعی هستی."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0.1,
                    "num_predict": 10   # 10-token limit
                })
            return response['message']['content'].strip()
        except Exception as e:
            print(f"[Local Llama] Failed → {e}")
            return None
    else:
        raise ValueError(f"Unknown backend: {backend}")


def call_local_llama(prompt: str):
    return call_llm(prompt, backend="local_llama")


def call_org_llm(prompt: str):
    return call_llm(prompt, backend="org")


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
    return text if (text and has_persian(text)) else default


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
            json.dump(data, f, ensure_ascii=False, indent=2)
            
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


def ai_name_community(center_node, neighbors, node_label_map, comm_id, method, time_period=None):
    """Generate meaningful community name using advanced content analysis."""
    print(f"[DEBUG] ai_name_community called: center={center_node}, neighbors={len(neighbors)}, method={method}")
    try:
        # 1. Check predefined names
        if result := node_label_map.get(center_node):
            save_community_name(result, center_node, neighbors, comm_id, method, time_period)
            return result

        # 2. Fetch and analyze community texts
        texts = fetch_community_texts_from_file(
            center_node, neighbors, filepath="res.json", size=15
        ) or {}
        
        if not texts:
            return "جامعه کوچک ناشناس"
        
        # Combine all texts for analysis
        all_texts = []
        for node_texts in texts.values():
            if isinstance(node_texts, list):
                all_texts.extend(node_texts)
            else:
                all_texts.append(node_texts)
        
        full_text = " ".join(all_texts).lower()
        
        # 3. Advanced content categorization
        categories = {
            "خبری": {
                "patterns": ["خبر", "گزارش", "اطلاع", "رسان", "پوشش", "منتشر", "اعلام"],
                "roles": ["خبرگزاری", "خبرنگاران", "رسانه خبری"]
            },
            "تحلیلی": {
                "patterns": ["تحلیل", "بررسی", "پژوهش", "مطالعه", "ارزیابی"],
                "roles": ["تحلیلگران", "پژوهشگران", "منتقدان"]
            },
            "سیاسی": {
                "patterns": ["سیاست", "دولت", "مجلس", "انتخاب", "نظام", "حزب"],
                "roles": ["فعالان سیاسی", "اصلاح‌طلبان", "اصولگرایان"]
            },
            "اجتماعی": {
                "patterns": ["جامعه", "مردم", "شهروند", "اجتماع", "حقوق"],
                "roles": ["فعالان مدنی", "کنشگران اجتماعی", "حقوق‌بشری"]
            },
            "فرهنگی": {
                "patterns": ["فرهنگ", "هنر", "ادب", "سینما", "موسیق", "تئاتر"],
                "roles": ["هنرمندان", "سینماگران", "نویسندگان"]
            },
            "اقتصادی": {
                "patterns": ["اقتصاد", "بازار", "تورم", "ارز", "قیمت", "بورس"],
                "roles": ["تحلیلگران اقتصادی", "فعالان بازار", "اقتصاددانان"]
            },
            "بین‌المللی": {
                "patterns": ["جهان", "بین‌الملل", "دیپلماسی", "خارج", "کشور"],
                "roles": ["تحلیلگران بین‌المللی", "دیپلمات‌ها", "خبرنگاران خارجی"]
            }
        }
        
        # Calculate scores for each category
        category_scores = {}
        for cat, info in categories.items():
            score = sum(full_text.count(p) for p in info["patterns"])
            if score > 0:
                category_scores[cat] = score
        
        # 4. Generate meaningful name
        if category_scores:
            # Get top 2 categories
            top_categories = sorted(
                category_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:2]
            
            main_cat = top_categories[0][0]
            main_roles = categories[main_cat]["roles"]
            
            # Select appropriate role based on text analysis
            role_scores = {}
            for role in main_roles:
                role_words = role.split()
                role_score = sum(
                    full_text.count(word) 
                    for word in role_words
                )
                role_scores[role] = role_score
            
            # Get best matching role
            if role_scores:
                best_role = max(role_scores.items(), key=lambda x: x[1])[0]
            else:
                best_role = main_roles[0]
            
            # Add secondary category if significant
            if len(top_categories) > 1 and top_categories[1][1] >= top_categories[0][1] * 0.3:
                second_cat = top_categories[1][0]
                result = f"{best_role} با گرایش {second_cat}"
            else:
                result = best_role
        else:
            # Fallback: analyze center node name
            center_lower = center_node.lower()
            if any(k in center_lower for k in ["news", "خبر", "bbc", "voa"]):
                result = "شبکه خبری"
            elif any(k in center_lower for k in ["art", "هنر", "cinema"]):
                result = "شبکه فرهنگی"
            elif any(k in center_lower for k in ["polit", "سیاس"]):
                result = "گروه سیاسی"
            else:
                result = f"جامعه {center_node}"
        
        # 5. Save to file
        save_community_name(result, center_node, neighbors, comm_id, method, time_period)
        
        return result
        
    except Exception as e:
        print(f"[error] Community naming failed: {e}")
        return "جامعه کوچک ناشناس"


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

    # Optional: color nodes by community/cluster
    if partition:
        centers = get_community_centers(g, partition)
        for node, community_id in partition.items():
            # visually distinct hues
            color = f"hsl({(community_id * 47) % 360}, 70%, 60%)"
            net.get_node(node)['color'] = color
            center_node = centers.get(community_id)
            label = NODE_LABEL_MAP.get(center_node, f"گروه {community_id}")
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

    net.show(filename)
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
def detect_communities_louvain(g):
    """
    Run Louvain algorithm on graph G and return partition.
    G: networkx.Graph
    Returns: dict mapping node → community_id
    """
    partition = community_louvain.best_partition(g, weight='weight')
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

    # Add interactive visualization
    visualize_graph_interactively(g, partition, title="Louvain Graph",
                                  filename="louvain_graph.html")


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
    for (sender, target), total_weight in edge_weights.items():
        if total_weight >= 1:
            g.add_edge(sender, target, weight=total_weight)
    print(f"Graph has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
    return g


def get_node_embeddings(g, dimensions=64):
    """
    Generate Node2Vec embeddings for the graph.
    """
    node2vec = Node2Vec(g, dimensions=dimensions, walk_length=20, num_walks=100,
                        workers=2)
    model = node2vec.fit(window=10, min_count=1)

    # Ensure embeddings are generated for all nodes
    embeddings = []
    nodes = list(g.nodes())
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


# Glue It All Together
def run_kmeans(embeddings, n_clusters=5):
    from sklearn.cluster import KMeans
    n_samples = embeddings.shape[0]
    if n_samples < n_clusters:
        n_clusters = max(1, n_samples)  # adjust cluster count dynamically
        msg = (
            f"[auto-adjust] n_clusters reduced to {n_clusters}"
            f" due to small sample size ({n_samples})."
        )
        print(msg)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)


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
def style_partition(g, net, partition, method, base_hue):
    """Style nodes and add legend for community visualization.
    
    Applies colors, sizes and tooltips based on community structure."""
    if g.number_of_nodes() == 0:
        return {}

    # Get community centers and initialize structures
    centers = get_community_centers(g, partition)
    community_info = defaultdict(lambda: {
        "nodes": [],
        "count": 0,
        "color": None,
        "center": None
    })

    # First pass: analyze communities and assign initial labels
    for comm_id, center in centers.items():
        # Get community members
        community_nodes = [n for n, c in partition.items() if c == comm_id]
        neighbors = [n for n in community_nodes if n != center]
        
        # Generate unique label
        label = ai_name_community(center, neighbors, NODE_LABEL_MAP, comm_id, method, g)
        
        # Store complete community info
        info = community_info[label]
        info["nodes"].extend(community_nodes)
        info["count"] = len(community_nodes)
        info["center"] = center
        
    # Generate distinct colors
    sorted_communities = sorted(
        community_info.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    # Use golden ratio for optimal color distribution
    golden_ratio = 0.618033988749895
    num_communities = len(sorted_communities)
    
    for i, (label, info) in enumerate(sorted_communities):
        # Base hue distribution
        hue = (i * 360 * golden_ratio) % 360
        
        # Adjust saturation based on community size
        base_saturation = 75
        size_boost = min(info["count"] * 2, 15)  # Max 15% boost
        saturation = min(base_saturation + size_boost, 90)
        
        # Keep good lightness for visibility
        lightness = 60
        
        info["color"] = f"hsl({hue},{saturation}%,{lightness}%)"

    # Apply node styling with improved visualization
    pos = nx.spring_layout(g, seed=42)
    
    for node in g.nodes():
        # Find node's community
        comm_id = partition[node]
        
        # Get the label for this node's community
        node_label = None
        for label, info in community_info.items():
            if node in info["nodes"]:
                node_label = label
                break
                
        if not node_label:
            continue
            
        # Get community info
        info = community_info[node_label]
        
        # Calculate node size based on role and degree
        base_size = 15
        degree_boost = min(g.degree(node) * 2, 25)
        center_boost = 15 if node == info["center"] else 0
        size = base_size + degree_boost + center_boost
        
        # Node styling with improved tooltips
        node_data = {
            "id": node,
            "label": node,
            "color": info["color"],
            "title": f"{node} ({node_label})",
            "size": size,
            "x": float(pos[node][0] * 1000),
            "y": float(pos[node][1] * 1000),
            "physics": True,
            "borderWidth": 2 if node == info["center"] else 1,
            "borderWidthSelected": 3,
            "font": {
                "size": 14,
                "face": "Vazirmatn"
            }
        }
        
        # Add any available metadata
        if hasattr(g, "graph") and "meta_map" in g.graph:
            meta = g.graph["meta_map"].get(node)
            if meta:
                node_data["title"] += "<br>" + "<br>".join(
                    f"{k}: {v}" for k, v in meta.items()
                )
        
        # Update node in network
        net.get_node(node).update(node_data)
    
    # Return color mapping for legend
    return {label: info["color"] for label, info in community_info.items()}


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


def visualize_or_dummy(slot_start, slot_end, g, louvain=None, hybrid=None):
    """Visualize or use an empty graph if g is empty."""
    # Format filename for the time slot
    date_format = "%y%m%d"
    start_str = slot_start.strftime(date_format)
    end_str = slot_end.strftime(date_format)
    filename = f"dashboard_{start_str}_to_{end_str}.html"
    legend_path = filename.replace(".html", "_legend.json")
    
    # Initialize empty partitions if needed
    if g.number_of_nodes() == 0:
        print("[warning] Empty graph - skipping community naming")
        g = nx.Graph()
        partition_louvain, partition_hybrid = {}, {}
    else:
        partition_louvain = louvain if louvain else {}
        partition_hybrid = hybrid if hybrid else {}

    try:
        # Reset community store before detection
        if hasattr(ai_name_community, "communities"):
            ai_name_community.communities = []

        # IMPORTANT: Call ai_name_community for each community
        time_period = f"{slot_start.strftime('%Y-%m-%d')} تا {slot_end.strftime('%Y-%m-%d')}"
        
        # Get community centers for Louvain
        if partition_louvain:
            print(f"\n[louvain] Detecting {len(set(partition_louvain.values()))} communities...")
            louvain_centers = get_community_centers(g, partition_louvain)
            
            for comm_id, center_node in louvain_centers.items():
                # Get all members of this community
                neighbors = [
                    n for n, c in partition_louvain.items() 
                    if c == comm_id and n != center_node
                ]
                
                # Call naming function
                print(f"[louvain] Naming community {comm_id}: center={center_node}, members={len(neighbors)}")
                ai_name_community(center_node, neighbors, NODE_LABEL_MAP, comm_id, "louvain", time_period)

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
                ai_name_community(center_node, neighbors, NODE_LABEL_MAP, comm_id, "hybrid", time_period)

        # Save community details
        if hasattr(ai_name_community, "communities"):
            write_summary_file(
                ai_name_community.communities,
                slot_start.strftime("%Y-%m-%d"),
                slot_end.strftime("%Y-%m-%d")
            )
            
            os.makedirs("communities", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"communities/{timestamp}_{start_str}_to_{end_str}.json"
            
            communities_data = {
                "timeframe": {
                    "start": slot_start.strftime("%Y-%m-%d"),
                    "end": slot_end.strftime("%Y-%m-%d")
                },
                "communities": [
                    {k: numpy_to_python(v) for k, v in comm.items()}
                    for comm in ai_name_community.communities
                ]
            }
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(communities_data, f, ensure_ascii=False, indent=2)
            print(f"[saved] {output_path}")
            
            ai_name_community.communities = []

        # Create base legend data
        legend_data = {
            "louvain": {
                "groups": {str(k): numpy_to_python(v) 
                          for k, v in partition_louvain.items()}
            },
            "hybrid": {
                "groups": {str(k): numpy_to_python(v) 
                          for k, v in partition_hybrid.items()}
            }
        }

        with open(legend_path, 'w', encoding='utf-8') as f:
            json.dump(legend_data, f, ensure_ascii=False, indent=2)
        
        # Visualize the graph
        visualize_combined_dashboard(
            g, partition_louvain, partition_hybrid, filename
        )
        
    except Exception as e:
        print(f"[error] Visualization failed for {start_str} to {end_str}: {e}")
        import traceback
        traceback.print_exc()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("<html><body>Error generating visualization</body></html>")
        with open(legend_path, 'w', encoding='utf-8') as f:
            json.dump(legend_data, f)


# ---- Combined Dashboard Visualization ----
def create_community_network():
    """Create a network with optimal visualization settings."""
    net = Network(
        height="600px",
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


def visualize_combined_dashboard(g, louvain_partition, hybrid_partition, filename):
    """Generate visualizations for both Louvain and Hybrid methods."""
    # Generate paths
    louvain_html = filename.replace("dashboard_", "louvain_graph_")
    hybrid_html = filename.replace("dashboard_", "hybrid_graph_")
    legend_path = filename.replace(".html", "_legend.json")

    # Create networks
    net1 = create_community_network()
    net2 = create_community_network()

    # Handle empty graph case
    if g.number_of_nodes() == 0:
        empty_data = {"louvain": {"groups": {}}, "hybrid": {"groups": {}}}
        with open(legend_path, 'w', encoding='utf-8') as f:
            json.dump(empty_data, f, ensure_ascii=False, indent=2)
        
        # Save empty networks
        net1.save_graph(louvain_html)
        net2.save_graph(hybrid_html)
        return

    # Build visualizations
    louvain_colors = build_community_visualization(
        g, louvain_partition, net1, "Louvain"
    )
    hybrid_colors = build_community_visualization(
        g, hybrid_partition, net2, "Hybrid"
    )
    
    # Save networks
    net1.save_graph(louvain_html)
    net2.save_graph(hybrid_html)
    print(f"[saved] {louvain_html}, {hybrid_html}")

    # Generate legend data with improved counting
    legend_data = {
        "louvain": {"groups": {}},
        "hybrid": {"groups": {}}
    }
    
    # Process Louvain communities
    for label, color in louvain_colors.items():
        count = sum(
            1 for node in louvain_partition 
            if label in net1.get_node(node)['title']
        )
        if count > 0:
            legend_data["louvain"]["groups"][label] = {
                "color": color,
                "count": count
            }
    
    # Process Hybrid communities
    for label, color in hybrid_colors.items():
        count = sum(
            1 for node in hybrid_partition
            if label in net2.get_node(node)['title']
        )
        if count > 0:
            legend_data["hybrid"]["groups"][label] = {
                "color": color,
                "count": count
            }
    
    # Save legend data
    with open(legend_path, 'w', encoding='utf-8') as f:
        json.dump(legend_data, f, ensure_ascii=False, indent=2)
    print(f"[saved] {legend_path}")

    # Read utils.js content
    with open("lib/bindings/utils.js", encoding='utf-8') as f:
        utils_js = f.read()
        
    # Generate and write dashboard HTML
    dashboard_html = generate_dashboard_html(
        louvain_html, hybrid_html, legend_path, utils_js
    )
    
    # Write the final dashboard
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    print(f"[saved] {filename}")
    
    print("Visualization completed successfully.")    # Initialize networks
    net1 = create_community_network()
    net2 = create_community_network()

    # Process empty graph case
    if g.number_of_nodes() == 0:
        empty_data = {
            "louvain": {"groups": {}},
            "hybrid": {"groups": {}}
        }
        with open(legend_path, 'w', encoding='utf-8') as f:
            json.dump(empty_data, f, ensure_ascii=False, indent=2)
        
        # Save empty networks and return
        net1.save_graph(louvain_html)
        net2.save_graph(hybrid_html)
        return
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
            name = f"جامعه {center} و همفکران"
            
            # Ensure uniqueness for fallback names too
            while name in community_labels:
                name = f"جامعه {center} و همراهان"
            community_labels.add(name)
        
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
            "louvain": {"groups": {}},
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

    return {
        "louvain": {"groups": louvain_groups},
        "hybrid": {"groups": hybrid_groups}
    }
    
    # Initialize and process graphs
    for net, label, partition, filename in [
        (net1, "Louvain", louvain_partition, louvain_html),
        (net2, "Hybrid", hybrid_partition, hybrid_html)
    ]:
        net.from_nx(g)
        # Network options already applied during initialization
        colors = style_partition(g, net, partition, label, 47)
        net.save_graph(filename)
        print(f"[regenerated] {filename}")
        if label == "Louvain":
            louvain_label_colors = colors
        else:
            hybrid_label_colors = colors
    print(f"[regenerated] {hybrid_html}")

    # Generate optimized HTML with improved styling
    # Build legend HTML for each method
def initialize_community_names():
    """Initialize the community names file."""
    try:
        with open("community_names.txt", "w", encoding="utf-8") as f:
            f.write("=== نام‌های جوامع ===\n\n")
        print("فایل نام‌های جوامع پاکسازی شد.")
    except Exception as e:
        print(f"خطا در پاکسازی فایل نام‌ها: {e}")


def build_community_visualization(g, partition, network, method="Unknown"):
    """Build network visualization for a community detection method."""
    # Initialize network with graph data
    network.from_nx(g)
    
    # Find community centers and analyze community structure
    centers = get_community_centers(g, partition)
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    # Initialize community labels and info
    comm_labels = {}  # Maps comm_id to label
    comm_sizes = {}   # Maps comm_id to size
    
    # First pass: Generate meaningful labels for each community
    for comm_id, members in communities.items():
        center = centers.get(comm_id)
        if not center:
            continue
            
        # Get community texts for analysis
        neighbors = [n for n in members if n != center]
        texts = fetch_community_texts_from_file(
            center, neighbors, filepath="res.json", size=15
        ) or {}
        
        # Analyze text patterns for better naming
        text_content = " ".join([
            text for node_texts in texts.values() 
            for text in (node_texts if isinstance(node_texts, list) else [node_texts])
        ]).lower()
        
        # Define content categories
        categories = {
            "خبری": ["خبر", "گزارش", "اطلاع", "رسان", "منتشر"],
            "تحلیلی": ["تحلیل", "بررسی", "پژوهش", "مطالعه"],
            "سیاسی": ["سیاس", "دولت", "مجلس", "حکومت"],
            "اقتصادی": ["اقتصاد", "بازار", "تورم", "ارز"],
            "اجتماعی": ["اجتماع", "جامعه", "مردم", "شهروند"],
            "فرهنگی": ["فرهنگ", "هنر", "ادب", "موسیق"],
            "ورزشی": ["ورزش", "فوتبال", "بازی", "مسابق"],
            "علمی": ["علم", "دانش", "فناور", "تکنولوژ"]
        }
        
        # Determine primary category
        main_category = None
        max_matches = 0
        for category, keywords in categories.items():
            matches = sum(text_content.count(kw) for kw in keywords)
            if matches > max_matches:
                max_matches = matches
                main_category = category
                
        # Determine role/stance
        roles = {
            "رسانه": ["خبر", "گزارش", "اطلاع", "پوشش"],
            "شبکه": ["ارتباط", "انتشار", "اشتراک"],
            "گروه": ["عضو", "همراه", "فعال"],
            "جریان": ["حرکت", "پویش", "کمپین"],
            "فعالان": ["فعال", "مطالبه", "پیگیر", "کنش"],
            "حامیان": ["حمایت", "پشتیبان", "موافق"],
            "منتقدان": ["انتقاد", "مخالف", "معترض"]
        }
        
        role = None
        max_role_matches = 0
        for r, keywords in roles.items():
            matches = sum(text_content.count(kw) for kw in keywords)
            if matches > max_role_matches:
                max_role_matches = matches
                role = r
        
        # Generate label based on analysis
        if not main_category:
            main_category = "مستقل"
        if not role:
            role = "رسانه" if "خبر" in text_content else "گروه"
            
        # Combine role and category with proper spacing
        label = f"{role} {main_category}"
        
        # Print community naming details
        print(f"\n=== Community {comm_id} ===")
        print(f"Center: {center}")
        print(f"Members: {len(members)}")
        print(f"Category: {main_category}")
        print(f"Role: {role}")
        print(f"Final Label: {label}")
        print(f"Sample Members: {', '.join(members[:5])}")
        print("---")
        
        # Store community info
        comm_labels[comm_id] = label
        comm_sizes[comm_id] = len(members)
    
    # Generate color scheme
    sorted_comms = sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)
    colors = {}  # Will map labels to colors
    
    for i, (comm_id, size) in enumerate(sorted_comms):
        # Use golden ratio for optimal color distribution
        golden_angle = 0.618033988749895
        hue = (i * 360 * golden_angle) % 360
        
        # Larger communities get more saturated colors
        saturation = min(75 + (size * 1.5), 90)  # Base 75%, max 90%
        
        # Keep lightness balanced for visibility
        lightness = 60
        
        color = f"hsl({hue},{saturation}%,{lightness}%)"
        label = comm_labels[comm_id]
        colors[label] = color
        
        # Apply styling to community nodes
        for node in communities[comm_id]:
            is_center = (node == centers[comm_id])
            
            # Calculate node size based on degree and role
            degree = g.degree(node)
            base_size = 15
            degree_boost = min(degree * 2, 25)
            center_boost = 15 if is_center else 0
            size = base_size + degree_boost + center_boost
            
            # Update node properties
            node_data = network.get_node(node)
            node_data.update({
                'color': color,
                'label': node,
                'title': f"""
<div style="max-width: 300px; padding: 8px; text-align: right;">
    <strong style="font-size: 14px; display: block; margin-bottom: 4px;">{node}</strong>
    <span style="color: #666; font-size: 12px;">گروه: {label}</span>
</div>
""",
                'size': size,
                'borderWidth': 2 if is_center else 1,
                'borderWidthSelected': 3,
                'font': {'size': 14, 'face': 'Vazirmatn'}
            })
            
            # Add metadata if available
            if hasattr(g, "graph") and "meta_map" in g.graph:
                meta = g.graph["meta_map"].get(node)
                if meta:
                    node_data["title"] += "<br>" + "<br>".join(
                        f"{k}: {v}" for k, v in meta.items()
                    )
    
    # Optimize layout settings
    network.options.update({
        "physics": {
            "stabilization": {"enabled": True, "iterations": 100},
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
    })
    
    return colors


def generate_dashboard_html(louvain_html, hybrid_html, legend_path, utils_js):
    """Generate the HTML for the combined dashboard.
    
    Args:
        louvain_html: Path to the Louvain algorithm HTML visualization file
        hybrid_html: Path to the hybrid algorithm HTML visualization file
        legend_path: Path to the JSON file containing legend data
        utils_js: JavaScript utility functions to include in the page
        
    Returns:
        str: The complete HTML document as a string
    
    Notes:
        The dashboard displays two interactive network visualizations side by side:
        one for the Louvain community detection algorithm and one for the hybrid approach.
        Each visualization has its own legend showing community labels and member counts.
    """
    style = '''
        body {
            font-family: Vazirmatn, Tahoma, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            direction: rtl;
            background-color: #f5f6fa;
            color: #2c3e50;
        }
        h1, h2, h3 {
            color: #2c3e50;
            text-align: center;
            margin: 15px 0;
        }
        .container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px auto;
            max-width: 1800px;
        }
        .graph-section {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px;
            display: flex;
            flex-direction: column;
        }
        iframe {
            border: none;
            width: 100%;
            height: 500px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .legend {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-top: auto;
        }
        .legend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
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
            font-size: 14px;
        }
        .count {
            color: #666;
            font-size: 12px;
            margin-right: 8px;
        }
        @media (max-width: 1200px) {
            .container {
                grid-template-columns: 1fr;
            }
            .legend-grid {
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            }
        }
    '''
    
    js_code = f'''
        async function loadLegends() {{
            try {{
                const response = await fetch("{legend_path}");
                const data = await response.json();
                
                function buildLegendHTML(groups) {{
                    const items = Object.entries(groups).map(([label, info]) => 
                        `<div class="legend-item">
                        <span class="color-box" style="background:${{info.color}}"></span>
                        <span class="label">${{label}}</span>
                        <span class="count">(${{info.count}} عضو)</span>
                        </div>`
                    );
                    return `<div class="legend-grid">${{items.join('')}}</div>`;
                }}
                
                const [louvainLegend, hybridLegend] = [
                    document.getElementById("louvain-legend"),
                    document.getElementById("hybrid-legend")
                ];
                
                data.louvain?.groups && 
                    (louvainLegend.innerHTML = buildLegendHTML(data.louvain.groups));
                data.hybrid?.groups && 
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
            <h2>الگوریتم لووین</h2>
            <iframe src="{louvain_html}"></iframe>
            <div id="louvain-legend" class="legend"></div>
        </div>
        <div class="graph-section">
            <h2>روش ترکیبی</h2>
            <iframe src="{hybrid_html}"></iframe>
            <div id="hybrid-legend" class="legend"></div>
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
    html = ["""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تشخیص جوامع در شبکه اجتماعی</title>"""]
    html.append('    <script type="text/javascript">')
    html.append(utils_js)
    html.append('    </script>')
    html.append("""    <style>
        body {
            font-family: Vazirmatn, Tahoma, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            direction: rtl;
            background-color: #f5f6fa;
            color: #2c3e50;
        }
        h1, h2, h3 {
            color: #2c3e50;
            text-align: center;
            margin: 15px 0;
        }
        .container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px auto;
            max-width: 1800px;
        }
        .graph-section {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px;
            display: flex;
            flex-direction: column;
        }
        iframe {
            border: none;
            width: 100%;
            height: 500px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .legend {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-top: auto;
        }
        .legend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
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
            font-size: 14px;
        }
        .count {
            color: #666;
            font-size: 12px;
            margin-right: 8px;
        }
        @media (max-width: 1200px) {
            .container {
                grid-template-columns: 1fr;
            }
            .legend-grid {
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            }
        }
    </style>
</head>
<body>
    <h1>تشخیص جوامع در شبکه اجتماعی</h1>
    <div class="container">
        <div class="graph-section">
            <h2>الگوریتم لووین</h2>
            <iframe src="{louvain}"></iframe>
            <div id="louvain-legend" class="legend"><!-- Legend will be loaded here --></div>
        </div>
        <div class="graph-section">
            <h2>روش ترکیبی</h2>
            <iframe src="{hybrid}"></iframe>
            <div id="hybrid-legend" class="legend"><!-- Legend will be loaded here --></div>
        </div>
    </div>
    <script>
        async function loadLegends() {
            try {
                const response = await fetch("{legend}");
                const data = await response.json();
                
                function buildLegendHTML(groups) {
                    let html = ['<div class="legend-grid">'];
                    for (let [label, info] of Object.entries(groups)) {
                        html.push(`
                            <div class="legend-item">
                                <span class="color-box" style="background:${info.color}"></span>
                                <span class="label">${label}</span>
                                <span class="count">(${info.count} عضو)</span>
                            </div>
                        `);
                    }
                    html.push('</div>');
                    return html.join('\\n');
                }
                
                if (data.louvain && data.louvain.groups) {
                    document.getElementById("louvain-legend").innerHTML = buildLegendHTML(data.louvain.groups);
                }
                
                if (data.hybrid && data.hybrid.groups) {
                    document.getElementById("hybrid-legend").innerHTML = buildLegendHTML(data.hybrid.groups);
                }
            } catch (error) {
                console.error("Error loading legends:", error);
            }
        }
        
        window.addEventListener("load", loadLegends);
    </script>
</body>
</html>""".format(
        louvain=louvain_html,
        hybrid=hybrid_html,
        legend=legend_path
    )).join('\\n')
    
    return template.format(
        louvain_html=louvain_html,
        hybrid_html=hybrid_html,
        legend_path=legend_path,
        utils_js=utils_js
    )
    html.append(f'    <script>{utils_js}</script>')
    html.append('    <style>')
    html.append('        :root {')
    html.append('            --primary-color: #2c3e50;')
    html.append('            --secondary-color: #34495e;')
    html.append('            --border-color: #bdc3c7;')
    html.append('            --hover-color: #3498db;')
    html.append('        }')
    html.append('        body {')
    html.append('            font-family: Vazirmatn, Tahoma, Arial, sans-serif;')
    html.append('            margin: 0;')
    html.append('            padding: 20px;')
    html.append('            direction: rtl;')
    html.append('            background-color: #f5f6fa;')
    html.append('            color: var(--primary-color);')
    html.append('        }')
    html.append('        h1, h2, h3 {')
    html.append('            color: var(--primary-color);')
    html.append('            text-align: center;')
    html.append('            margin: 15px 0;')
    html.append('        }')
    html.append('        .container {')
    html.append('            display: grid;')
    html.append('            grid-template-columns: repeat(2, 1fr);')
    html.append('            gap: 20px;')
    html.append('            margin: 20px auto;')
    html.append('            max-width: 1800px;')
    html.append('        }')
    html.append('        .graph-section {')
    html.append('            background: white;')
    html.append('            border-radius: 8px;')
    html.append('            box-shadow: 0 2px 4px rgba(0,0,0,0.1);')
    html.append('            padding: 15px;')
    html.append('            display: flex;')
    html.append('            flex-direction: column;')
    html.append('        }')
    html.append('        iframe {')
    html.append('            border: none;')
    html.append('            width: 100%;')
    html.append('            height: 500px;')
    html.append('            border-radius: 4px;')
    html.append('            margin-bottom: 20px;')
    html.append('        }')
    html.append('        .legend {')
    html.append('            background: #f8f9fa;')
    html.append('            border-radius: 8px;')
    html.append('            padding: 15px;')
    html.append('            margin-top: auto;')
    html.append('        }')
    html.append('        .legend-grid {')
    html.append('            display: grid;')
    html.append('            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));')
    html.append('            gap: 10px;')
    html.append('            margin-top: 10px;')
    html.append('        }')
    html.append('        .legend-item {')
    html.append('            display: flex;')
    html.append('            align-items: center;')
    html.append('            padding: 8px;')
    html.append('            background: white;')
    html.append('            border-radius: 4px;')
    html.append('            box-shadow: 0 1px 3px rgba(0,0,0,0.1);')
    html.append('        }')
    html.append('        .color-box {')
    html.append('            width: 16px;')
    html.append('            height: 16px;')
    html.append('            border-radius: 4px;')
    html.append('            margin-left: 8px;')
    html.append('            border: 1px solid rgba(0,0,0,0.1);')
    html.append('        }')
    html.append('        .label {')
    html.append('            flex: 1;')
    html.append('            font-size: 14px;')
    html.append('        }')
    html.append('        .count {')
    html.append('            color: #666;')
    html.append('            font-size: 12px;')
    html.append('            margin-right: 8px;')
    html.append('        }')
    html.append('        @media (max-width: 1200px) {')
    html.append('            .container {')
    html.append('                grid-template-columns: 1fr;')
    html.append('            }')
    html.append('            .legend-grid {')
    html.append('                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));')
    html.append('            }')
    html.append('        }')
    html.append('    </style>')
    html.append('</head>')
    html.append('<body>')
    html.append('    <h1>تشخیص جوامع در شبکه اجتماعی</h1>')
    html.append('    <div class="container">')
    html.append('        <div class="graph-section">')
    html.append('            <h2>الگوریتم لووین</h2>')
    html.append(f'            <iframe src="{louvain_html}"></iframe>')
    html.append('            <div id="louvain-legend"><!-- Legend will be loaded here --></div>')
    html.append('        </div>')
    html.append('        <div class="graph-section">')
    html.append('            <h2>روش ترکیبی</h2>')
    html.append(f'            <iframe src="{hybrid_html}"></iframe>')
    html.append('            <div id="hybrid-legend"><!-- Legend will be loaded here --></div>')
    html.append('        </div>')
    html.append('    </div>')
    html.append('    ')
    html.append('    <script>')
    html.append('        async function loadLegends() {')
    html.append('            try {')
    html.append(f'                const response = await fetch("{legend_path}");')
    html.append('                const data = await response.json();')
    html.append('                ')
    html.append('                function buildLegendHTML(groups) {')
    html.append("                    let html = ['<div class=\"legend\">', '<div class=\"legend-grid\">'];")
    html.append('                    ')
    html.append('                    Object.entries(groups).forEach(([label, info]) => {')
    html.append('                        html.push(`')
    html.append('                            <div class="legend-item">')
    html.append('                                <span class="color-box" style="background:${info.color}"></span>')
    html.append('                                <span class="label">${label}</span>')
    html.append('                                <span class="count">(${info.count} عضو)</span>')
    html.append('                            </div>')
    html.append('                        `);')
    html.append('                    });')
    html.append('                    ')
    html.append("                    html.push('</div>', '</div>');")
    html.append("                    return html.join('\\n');")
    html.append('                }')
    html.append('                ')
    html.append('                // Update both legends')
    html.append('                if (data.louvain && data.louvain.groups) {')
    html.append('                    document.getElementById("louvain-legend").innerHTML = ')
    html.append('                        buildLegendHTML(data.louvain.groups);')
    html.append('                }')
    html.append('                ')
    html.append('                if (data.hybrid && data.hybrid.groups) {')
    html.append('                    document.getElementById("hybrid-legend").innerHTML = ')
    html.append('                        buildLegendHTML(data.hybrid.groups);')
    html.append('                }')
    html.append('                ')
    html.append('            } catch (error) {')
    html.append('                console.error("Error loading legends:", error);')
    html.append('            }')
    html.append('        }')
    html.append('        ')
    html.append('        // Load legends when the page loads')
    html.append('        window.addEventListener("load", loadLegends);')
    html.append('    </script>')
    html.append('</body>')
    html.append('</html>')
    
    return '\n'.join(html)


def save_community_name(community_name: str, center_node: str, neighbors: list, 
                       comm_id: int, method: str, time_period: str = None):
    """Save a community name and its details to the community names file."""
    print(f"[DEBUG] Saving community: {community_name} for {center_node}")
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


def fetch_community_texts_from_file(
    center_node: str,
    neighbors: list,
    filepath: str = "res.json",
    size: int = 15
) -> Dict[str, List[str]]:
    """Fetch recent texts for a community from a JSON file.
    
    Args:
        center_node: The central node of the community
        neighbors: List of neighbor nodes
        filepath: Path to the JSON file containing texts
        size: Maximum number of texts to return per user
        
    Returns:
        Dict mapping usernames to their recent texts
    """
    try:
        # Get all community members
        members = [center_node] + neighbors
        texts_by_member = defaultdict(list)
        
        # Read texts from file
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    msg = json.loads(line)
                    sender = msg.get("sender")
                    text = msg.get("text", "").strip()
                    
                    # Only collect texts from community members
                    if sender in members and text:
                        texts_by_member[sender].append(text)
                        
                        # Stop if we have enough texts for this member
                        if len(texts_by_member[sender]) >= size:
                            continue
                            
                except json.JSONDecodeError:
                    continue
                    
        return {
            member: texts[:size]
            for member, texts in texts_by_member.items()
        }
        
    except Exception as e:
        print(f"[error] reading texts from file: {e}")
        return {}


def process_visualization(g, louvain_partition, hybrid_partition, filename):
    """Process community detection visualization for both methods."""
    # Generate paths
    louvain_html = filename.replace("dashboard_", "louvain_graph_")
    hybrid_html = filename.replace("dashboard_", "hybrid_graph_")
    legend_path = filename.replace(".html", "_legend.json")

    # Create networks
    net1 = create_community_network()
    net2 = create_community_network()

    # Handle empty graph case
    if g.number_of_nodes() == 0:
        empty_data = {"louvain": {"groups": []}, "hybrid": {"groups": []}}
        with open(legend_path, 'w', encoding='utf-8') as f:
            json.dump(empty_data, f, ensure_ascii=False, indent=2)
        
        # Save empty networks
        net1.save_graph(louvain_html)
        net2.save_graph(hybrid_html)
        return

    # Build visualizations with updated legend data
    build_community_visualization(g, louvain_partition, net1, "Louvain")
    build_community_visualization(g, hybrid_partition, net2, "Hybrid")
    
    # Save networks
    net1.save_graph(louvain_html)
    net2.save_graph(hybrid_html)
    print(f"[saved] {louvain_html}, {hybrid_html}")
    
    # Extract legend data from network options
    legend_data = {
        "louvain": {"groups": net1.options.get('groups', [])},
        "hybrid": {"groups": net2.options.get('groups', [])}
    }

    # Save legend data
    with open(legend_path, 'w', encoding='utf-8') as f:
        json.dump(legend_data, f, ensure_ascii=False, indent=2)
    print(f"[saved] {legend_path}")

    # Generate dashboard HTML
    with open("lib/bindings/utils.js", encoding='utf-8') as f:
        utils_js = f.read()
        
    dashboard_html = generate_dashboard_html(
        louvain_html, hybrid_html, legend_path, utils_js
    )
    
    # Write the final dashboard
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    print(f"[saved] {filename}")
    
    return louvain_html, hybrid_html, legend_path

    with open(filename, "w", encoding="utf-8") as f:
        f.write(combined_html)
    
    # Write enhanced legend
    legend_data = {
        "louvain": {
            "title": "جوامع شناسایی شده توسط الگوریتم لووین",
            "groups": {
                label: {
                    "color": color,
                    "count": sum(1 for node, comm in louvain_partition.items()
                               if net1.get_node(node)['title'].endswith(f"({label})"))
                }
                for label, color in louvain_label_colors.items()
            }
        },
        "hybrid": {
            "title": "جوامع شناسایی شده توسط روش ترکیبی",
            "groups": {
                label: {
                    "color": color,
                    "count": sum(1 for node, comm in hybrid_partition.items()
                               if net2.get_node(node)['title'].endswith(f"({label})"))
                }
                for label, color in hybrid_label_colors.items()
            }
        }
    }
    
    with open(legend_path, "w", encoding="utf-8") as f:
        json.dump(legend_data, f, indent=2, ensure_ascii=False)
    print(f"[legend regenerated] {legend_path}")
    print(f"Combined dashboard saved to {filename}")


def generate_time_slots(start_date, end_date, slot_type):
    slots = []
    current = start_date
    while current <= end_date:
        if slot_type == "daily":
            next_slot = current + timedelta(days=1)
        elif slot_type == "weekly":
            next_slot = current + timedelta(weeks=1)
        else:  # monthly
            next_slot = current + relativedelta(months=1)

        slot_end = min(next_slot - timedelta(days=1), end_date)
        slots.append((current, slot_end))
        current = next_slot
    return slots


# --- Timeline Dashboard Generator ---
import os

def generate_timeline_dashboard(output_file="timeline_dashboard.html"):
    """
    Generates an interactive dashboard that allows switching between time slot graphs.
    """
    dashboards = []
    for f in sorted([f for f in os.listdir(".") if f.startswith("dashboard_") and f.endswith(".html") and "_to_" in f]):
        legend_file = f.replace(".html", "_legend.json")
        if not os.path.exists(legend_file):
            continue
        try:
            with open(legend_file, "r", encoding="utf-8") as lf:
                legends = json.load(lf)
            # skip if no graph + no legends
            if not legends.get("louvain") and not legends.get("hybrid"):
                print(f"[skip] {f} has no graph/legends → deleting")
                try:
                    os.remove(f)
                    if os.path.exists(legend_file):
                        os.remove(legend_file)
                except Exception as e:
                    print(f"[cleanup failed] {e}")
                continue
        except Exception:
            continue
        dashboards.append(f)

    if not dashboards:
        print("No dashboard_*.html files found.")
        return

    # Load legends for each dashboard
    legends = {}
    for dash in dashboards:
        legend_file = dash.replace(".html", "_legend.json")
        if os.path.exists(legend_file):
            with open(legend_file, "r", encoding="utf-8") as f:
                legends[dash] = json.load(f)
        else:
            legends[dash] = {"louvain": {}, "hybrid": {}}

    # Generate graph containers HTML
    graph_containers = []
    for i, dash in enumerate(dashboards):
        louvain_legend_html = ''.join(
            f'<span style="color: {str(color)}">{str(label)}</span>'
            for label, color in legends.get(dash, {}).get("louvain", {}).items()
        )
        hybrid_legend_html = ''.join(
            f'<span style="color: {str(color)}">{str(label)}</span>'
            for label, color in legends.get(dash, {}).get("hybrid", {}).items()
        )
        
        container = f'''
        <div id="container{i}" style="display: none;">
            <iframe id="frame{i}" src="{dash}" title="{dash}" style="width: 100%; height: 650px; border: none;"></iframe>
            <div class="legend">
                <strong>Louvain Community Labels ({dash}):</strong><br>
                {louvain_legend_html}
            </div>
            <div class="legend">
                <strong>Hybrid Community Labels ({dash}):</strong><br>
                {hybrid_legend_html}
            </div>
        </div>
        '''
        graph_containers.append(container)

    # Read template and fill in values
    with open("timeline_template.html", "r", encoding="utf-8") as f:
        template = f.read()
    
    # Replace placeholders
    html = template.replace("{dashboard_list}", json.dumps(dashboards))\
                  .replace("{graph_containers}", '\n'.join(graph_containers))
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Timeline dashboard saved to {output_file}")


# Print community results
if __name__ == "__main__":
    # پاکسازی فایل اسامی در شروع برنامه
    try:
        with open("community_names.txt", "w", encoding="utf-8") as f:
            f.write("=== لیست کامل نام‌های جوامع تشخیص داده شده ===\n\n")
        print("[init] Community names file initialized")
    except Exception as e:
        print(f"[error] Failed to initialize community names file: {e}")

    print("[start] Running Louvain and Hybrid community detection...")
    start_date = datetime.strptime("2025-03-01", "%Y-%m-%d")
    end_date = datetime.strptime("2025-07-20", "%Y-%m-%d")
    delta_days = (end_date - start_date).days
    if delta_days <= 7:
        slot_type = "daily"
    elif delta_days <= 60:
        slot_type = "weekly"
    else:
        slot_type = "monthly"
    print(f"[config] Slot type selected: {slot_type}")
    slots = generate_time_slots(start_date, end_date, slot_type)
    for idx, (slot_start, slot_end) in enumerate(slots):
        _ai_name_cache.clear()
        print(f"\n[slot {idx+1}/{len(slots)}] Processing: {slot_start.date()} to {slot_end.date()}...")
        messages = []
        all_messages = load_interactions()
        for msg in all_messages:
            date_str = msg.get("date")
            if not date_str:
                continue
            try:
                msg_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                if slot_start.date() <= msg_date <= slot_end.date():
                    messages.append(msg)
            except Exception as e:
                print(f"[skip] Invalid date format: {date_str} → {e}")
                continue
        if not messages:
            print("[skip] No valid messages found in this time slot")
            visualize_or_dummy(slot_start, slot_end, nx.Graph())
            continue
        g_slot = build_user_graph(messages)
        g_slot = clean_graph(g_slot)
        if g_slot.number_of_nodes() == 0:
            print("[skip] Empty graph after filtering")
            visualize_or_dummy(slot_start, slot_end, g_slot)
            continue
        partition_louvain = detect_communities_louvain(g_slot)
        embeddings, nodes = get_node_embeddings(g_slot)
        labels = run_kmeans(embeddings, n_clusters=5)
        partition_hybrid = {node: labels[i] for i, node in enumerate(nodes)}
        visualize_or_dummy(
            slot_start, slot_end, g_slot,
            partition_louvain, partition_hybrid
        )
    print("[done] Community detection completed.")
    generate_timeline_dashboard()
