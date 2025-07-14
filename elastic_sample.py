import os
import sys
import json
import subprocess
import re
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from collections import defaultdict

# --- Connect to Production Elasticsearch ---
print("Using Production Elasticsearch.")
cur_path = os.path.dirname(__file__)
CERTIFICATE = os.path.join(cur_path, "ca.crt")
ELASTICSEARCH_URL = "https://192.168.59.79:9200"
AUTH = "YXYyeVRKWUJKSFpwMVdrTnZWRDc6UHhqRHBQa2ZUYW1yMnBwWTV3Ri0xUQ=="
INDEX = "twitter_temp_data"

es = Elasticsearch(
    ELASTICSEARCH_URL,
    api_key=AUTH,
    ca_certs=CERTIFICATE,
    verify_certs=True,
    ssl_show_warn=False
)

# --- Fetch data from ES ---
query_body = {
    "query": {
        "range": {
            "date": {
                "gte": "2025-04-01",
                "lte": "2025-05-30"
            }
        }
    },
    "_source": [
        "user_name", "user_title", "normalized_text", "date",
        "reply_to_user", "quote", "retweet", "political_label"
    ]
}

print("Fetching real interactions from ES...")
result = scan(es, index=INDEX, query=query_body, preserve_order=True, scroll="10m")

interactions = []
user_connections = defaultdict(set)
user_info = {}
from collections import defaultdict
political_tag_users = defaultdict(set)

# --- Load user filter from p_res.txt ---
p_res_users = set()
with open("p_res.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                user_data = json.loads(line)
                user_name = user_data.get("user_name")
                if user_name:
                    p_res_users.add(user_name)
            except json.JSONDecodeError:
                continue
print(f"Loaded {len(p_res_users)} users from p_res.txt.")

for doc in result:
    src = doc["_source"]
    sender = src.get("user_name")
    date_str = src.get("date", "")[:10]
    text = src.get("normalized_text", "")
    title = src.get("user_title") or f"DisplayName_{sender}"
    political_label = src.get("political_label", "unknown")

    # Track user info with political tag
    if sender:
        user_info[sender] = {
            "user_name": sender,
            "user_title": title,
            "political_label": political_label
        }
        political_tag_users[political_label].add(sender)

    # Mentions
    mentions = re.findall(r"@(\w{1,50})", text)
    for target in mentions:
        if sender and target and sender != target:
            interactions.append({
                "sender": sender,
                "target": target,
                "type": "mention",
                "date": date_str
            })
            user_connections[sender].add(target)
            user_connections[target].add(sender)

    # Replies (simulated field)
    reply_to = src.get("reply_to_user")
    if reply_to and reply_to != sender:
        interactions.append({
            "sender": sender,
            "target": reply_to,
            "type": "reply",
            "date": date_str
        })
        user_connections[sender].add(reply_to)
        user_connections[reply_to].add(sender)

    # Quotes
    quote = src.get("quote", {})
    quoted_user = quote.get("user", {}).get("user_name") if isinstance(quote, dict) else None
    if quoted_user and quoted_user != sender:
        interactions.append({
            "sender": sender,
            "target": quoted_user,
            "type": "quote",
            "date": date_str
        })
        user_connections[sender].add(quoted_user)
        user_connections[quoted_user].add(sender)

    # Retweets
    retweet = src.get("retweet", {})
    retweeted_user = retweet.get("user", {}).get("user_name") if isinstance(retweet, dict) else None
    if retweeted_user and retweeted_user != sender:
        interactions.append({
            "sender": sender,
            "target": retweeted_user,
            "type": "retweet",
            "date": date_str
        })
        user_connections[sender].add(retweeted_user)
        user_connections[retweeted_user].add(sender)

print(f"Collected {len(interactions)} raw interactions.")

# Filter users by presence in p_res.txt
connected_users_set = set()
for i in interactions:
    if i["sender"] in p_res_users and i["target"] in p_res_users:
        connected_users_set.add(i["sender"])
        connected_users_set.add(i["target"])

# Select enough users to ensure all 4 interaction types are represented
final_users = set()
type_presence = {"mention": False, "reply": False, "quote": False, "retweet": False}
for i in interactions:
    type_presence[i["type"]] = True

if all(type_presence.values()) and len(connected_users_set) >= 500:
    final_users = list(connected_users_set)[:500]
else:
    # fallback: include more users to try and cover all interaction types
    final_users = list(connected_users_set)

connected_users_set = set(final_users)
print(f"Selected {len(final_users)} users from valid interaction pool.")

filtered_interactions = [
    i for i in interactions
    if i["sender"] in p_res_users and i["target"] in p_res_users
]

# --- Write interactions.json ---
with open("interactions.json", "w", encoding="utf-8") as f:
    for i in filtered_interactions:
        json.dump(i, f, ensure_ascii=False)
        f.write("\n")

type_counts = defaultdict(int)
for i in filtered_interactions:
    type_counts[i["type"]] += 1

print(f"✅ interactions.json written with {len(filtered_interactions)} interactions.")
print(f"Type counts: {dict(type_counts)}")

# --- Write res.json ---
with open("res.json", "w", encoding="utf-8") as f:
    for u in connected_users_set:
        json.dump(user_info[u], f, ensure_ascii=False)
        f.write("\n")
print(f"✅ res.json written with {len(connected_users_set)} users.")

# --- Run convertor.py ---
convertor_script = os.path.join(cur_path, "../Parham/convertor.py")
if os.path.exists(convertor_script):
    print("Running convertor.py...")
    subprocess.run(["python3", convertor_script], check=True)
    print("✅ convertor.py executed.")
else:
    print("⚠️ convertor.py not found.")
