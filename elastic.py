"""Module providing a flexible Elasticsearch query script."""
import os
import sys
import subprocess
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from collections import Counter
import logging
import re

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

# Prompt the user to select the authentication type
# print("Select the Elasticsearch server (authentication type):")
# print("1. Production Elasticsearch server (API Key Authentication)")
# print("2. Temp Elasticsearch server (Basic Authentication)")
# auth_type = input("Enter 1 or 2: ").strip()

# Set variables based on the selected authentication type
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--auth", type=str, choices=["1", "2"], default="1", help="Auth type: 1=prod, 2=temp")
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

query_body = {
    "track_total_hits": True,
    "query": {
        "bool": {
            "must": [
                {
                    "terms": {
                        "entity.hashtag": ["#همکاری_ملی"]
                    }
                }
            ],
            "filter": [
                {
                    "range": {
                        "date": {
                            "gte": "now-270d/d",
                            "lte": "now/d"
                        }
                    }
                }
            ]
        }
    },
    "_source": [
        "user_name",           # ← This is what community_texts.py queries
        "normalized_text",     # ← This is the tweet content
        "content",             # ← Fallback content
        "user_title",
        "political_category.label",
        "date",
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
        "sentiment.label",
        "reply_to_user",
        "quote",
        "repost"
    ],

}

type_counter = Counter()

log.info("Starting initial scan query...")
result = scan(
    es,
    index=INDEX,
    query=query_body,
    preserve_order=True,
    scroll="10m"
)

# Extract all distinct usernames and displaynames
usernames = set()
displaynames = set()
COUNT = 0
with open("res.json", "w", encoding="utf-8") as f:
    for doc in result:
        source = doc["_source"]
        raw_type = source.get("type")
        tweet_type = "quote" if raw_type in {"quote", "qoute"} else (raw_type or "").lower()
        if tweet_type not in {"post", "reply", "quote", "repost"}:
            log.warning(f"❓ Unrecognized tweet_type: {tweet_type} (raw={raw_type})")
        type_counter[tweet_type] += 1
        json.dump(source, f, ensure_ascii=False)
        f.write("\n")
        COUNT += 1
        uname = source.get("user_name")
        dname = source.get("user_title")
        if uname:
            usernames.add(uname)
        if dname:
            displaynames.add(dname)
log.info(f"Total hits: {COUNT}")
log.info(f"TWEET TYPE COUNTS IN SCAN 1: {dict(type_counter)}")

# Rewind and re-scan to collect interactions
log.info("Starting scan to collect interactions...")
result = scan(
    es,
    index=INDEX,
    query=query_body,
    preserve_order=True
)

with open("interactions.json", "w", encoding="utf-8") as f_interactions:
    written = 0
    EXAMPLE_LIMIT = 10
    EXAMPLE_PRINTED = 0
    for doc in result:
        source = doc["_source"]
        raw_type = source.get("type")
        tweet_type = "quote" if raw_type in {"quote", "qoute"} else (raw_type or "").lower()
        # Only warn for truly unrecognized tweet types
        if tweet_type not in {"post", "reply", "quote", "repost"}:
            log.warning(f"❓ Unrecognized tweet_type: {tweet_type} (raw={raw_type})")
        # Remove repetitive example doc logging
        # Remove repetitive info log for every doc
        # Remove debug log for every tweet type
        sender = source.get("user_name")
        date_str = source.get("date", "")[:10]

        if not sender or not tweet_type or not date_str:
            continue

        # Handle structured interaction types
        if tweet_type == "reply":
            target = source.get("reply_to_user")
            if not target:
                # Only warn once for missing target user
                # log.warning("⛔ Skipped reply: No target user.")
                continue
            elif target == sender:
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
                written += 1
        elif tweet_type == "quote":
            quote = source.get("quote", {})
            if not isinstance(quote, dict):
                # log.warning("⛔ Skipped quote: Not a dict.")
                continue
            else:
                quoted_user = quote.get("user", {}).get("user_name")
                if not quoted_user:
                    # log.warning("⛔ Skipped quote: No quoted user.")
                    continue
                elif quoted_user == sender:
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
                    written += 1

        elif tweet_type == "repost":
            repost = source.get("repost", {})
            if not isinstance(repost, dict):
                # log.warning("⛔ Skipped repost: Not a dict.")
                continue
            else:
                retweeted_user = repost.get("user", {}).get("user_name")
                if not retweeted_user:
                    # log.warning("⛔ Skipped repost: No retweeted user.")
                    continue
                elif retweeted_user == sender:
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
                    written += 1

        elif tweet_type == "post":
            # log.warning("🟡 Processing a POST tweet...")
            mentions = source.get("entity", {}).get("mention", [])
            if not mentions:
                mentions = extract_mentions(source.get("normalized_text", ""))
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

log.info("Chunking user_name and user_title lists for political tagging...")
username_chunks = chunked(usernames, 100)
displayname_chunks = chunked(displaynames, 100)

total_batches = len(username_chunks) + len(displayname_chunks)
BATCH_INDEX = 1
SECONDARY_COUNT = 0

with open("res.json", "w", encoding="utf-8") as f:
    for chunk in username_chunks:
        log.info(f"Querying batch {BATCH_INDEX}/{total_batches} [usernames]")
        query = {
            "query": {
                "bool": {
                    "should": [{"match": {"user_name": uname}} for uname in chunk],
                    "minimum_should_match": 1
                }
            },
            "_source": [
                "user_name",
                "normalized_text",     # ← This is the tweet content
                "content",             # ← Fallback content
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
        results = es.search(index="twitter_source", body=query, size=1000)["hits"]["hits"]
        for doc in results:
            source = doc["_source"]
            uname = source.get("user_name")
            dname = source.get("user_title")
            political = source.get("political_category", {})
            label = political.get("label") if isinstance(political, dict) else None
            if uname and dname and label:
                output_doc = {
                    "user_name": uname,
                    "user_title": dname,
                    "political_label": label,
                    "normalized_text": source.get("normalized_text"),
                    "content": source.get("content"),
                    "category": source.get("category", {}).get("label"),
                    "emotion": source.get("emotion", {}).get("label"),
                    "hashtag": source.get("entity", {}).get("hashtag"),
                    "language": source.get("language", {}).get("label"),
                    "event": source.get("ner", {}).get("event"),
                    "location": source.get("ner", {}).get("location"),
                    "nationality": source.get("ner", {}).get("nationality"),
                    "organization": source.get("ner", {}).get("organ"),
                    "person": source.get("ner", {}).get("person"),
                    "politic_group": source.get("ner", {}).get("politic_group"),
                    "offensive": source.get("offensive", {}).get("label"),
                    "sentiment": source.get("sentiment", {}).get("label")
                }
                json.dump(output_doc, f, ensure_ascii=False)
                f.write("\n")
                SECONDARY_COUNT += 1
        BATCH_INDEX += 1

    for chunk in displayname_chunks:
        log.info(f"Querying batch {BATCH_INDEX}/{total_batches} [displaynames]")
        query = {
            "query": {
                "bool": {
                    "should": [{"match": {"user_title": dname}} for dname in chunk],
                    "minimum_should_match": 1
                }
            },
            "_source": [
                "user_name",
                "normalized_text",     # ← This is the tweet content
                "content",             # ← Fallback content                "user_title",
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
        results = es.search(index="twitter_source", body=query, size=1000)["hits"]["hits"]
        for doc in results:
            source = doc["_source"]
            uname = source.get("user_name")
            dname = source.get("user_title")
            political = source.get("political_category", {})
            label = political.get("label") if isinstance(political, dict) else None
            if uname and dname and label:
                output_doc = {
                    "user_name": uname,
                    "user_title": dname,
                    "political_label": label,
                    "normalized_text": source.get("normalized_text"),
                    "content": source.get("content"),
                    "category": source.get("category", {}).get("label"),
                    "emotion": source.get("emotion", {}).get("label"),
                    "hashtag": source.get("entity", {}).get("hashtag"),
                    "language": source.get("language", {}).get("label"),
                    "event": source.get("ner", {}).get("event"),
                    "location": source.get("ner", {}).get("location"),
                    "nationality": source.get("ner", {}).get("nationality"),
                    "organization": source.get("ner", {}).get("organ"),
                    "person": source.get("ner", {}).get("person"),
                    "politic_group": source.get("ner", {}).get("politic_group"),
                    "offensive": source.get("offensive", {}).get("label"),
                    "sentiment": source.get("sentiment", {}).get("label")
                }
                json.dump(output_doc, f, ensure_ascii=False)
                f.write("\n")
                SECONDARY_COUNT += 1
        BATCH_INDEX += 1

log.info(f"Political tag hits: {SECONDARY_COUNT}")
log.info("Finished writing political tags to res.json.")

# Show preview of res.json (last 3 lines)
with open("res.json", "r", encoding="utf-8") as f:
    lines = f.readlines()
    preview = lines[-3:] if len(lines) >= 3 else lines
    print("\n--- Preview of last 3 lines of res.json ---")
    for line in preview:
        print(line.strip())

# Automatically call normalize_json.py
cur_path = os.path.dirname(__file__)
normalize_script = os.path.join(cur_path, "normalize_json.py")
subprocess.run(["python3", normalize_script], check=True)
log.info("Normalization complete.")
log.info("All steps completed successfully.")
log.info("Finished full pipeline.")
