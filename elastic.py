"""Module providing a flexible Elasticsearch query script."""
import os
import sys
import subprocess
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from collections import Counter
import logging

logging.basicConfig(
    filename="debug.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# Prompt the user to select the authentication type
print("Select the Elasticsearch server (authentication type):")
print("1. Production Elasticsearch server (API Key Authentication)")
print("2. Temp Elasticsearch server (Basic Authentication)")
auth_type = input("Enter 1 or 2: ").strip()

# Set variables based on the selected authentication type
if auth_type == "1":
    # API Key Authentication
    print("You selected Production Elasticsearch server (API Key Authentication).")
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
    print("You selected Temp Elasticsearch server (Basic Authentication).")
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
    print("Invalid selection. Please run the script again and select 1 or 2.")
    sys.exit(1)

query_body = {
    "query": {
        "bool": {
            "must": [
                {
                    "range": {
                        "date": {
                            "gte": "2025-04-01",
                            "lte": "2025-04-15"
                        }
                    }
                }
            ]
        }
    },
    "_source": [
        "user_name",
        "user_title",
        "normalized_text",
        "date",
        "type",
        "entity.mention"
    ]
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
        tweet_type = source.get("type")
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
    for doc in result:
        source = doc["_source"]
        log.info(f"{source.get('type')} | {source.get('user_name')} | mentions={len(source.get('entity', {}).get('mention', []))} | reply_to={source.get('reply_to_user')} | quote={bool(source.get('quote'))} | repost={bool(source.get('repost'))}")

        tweet_type = source.get("type")

        # DEBUG
        if tweet_type != "post":
            log.info(f"===>>>{tweet_type}")

        sender = source.get("user_name")
        date_str = source.get("date", "")[:10]

        if not sender or not tweet_type or not date_str:
            continue

        # Handle structured interaction types
        if tweet_type == "reply":
            target = source.get("reply_to_user")
            if target and target != sender:
                interaction = {
                    "sender": sender,
                    "target": target,
                    "type": "reply",
                    "date": date_str
                }
                json.dump(interaction, f_interactions, ensure_ascii=False)
                f_interactions.write("\n")
        elif tweet_type == "quote":
            quote = source.get("quote", {})
            quoted_user = quote.get("user", {}).get("user_name") if isinstance(quote, dict) else None
            if quoted_user and quoted_user != sender:
                interaction = {
                    "sender": sender,
                    "target": quoted_user,
                    "type": "quote",
                    "date": date_str
                }
                json.dump(interaction, f_interactions, ensure_ascii=False)
                f_interactions.write("\n")

        elif tweet_type == "repost":
            repost = source.get("repost", {})
            retweeted_user = repost.get("user", {}).get("user_name") if isinstance(
                repost, dict) else None
            if retweeted_user and retweeted_user != sender:
                interaction = {
                    "sender": sender,
                    "target": retweeted_user,
                    "type": "repost",
                    "date": date_str
                }
                json.dump(interaction, f_interactions, ensure_ascii=False)
                f_interactions.write("\n")

        elif tweet_type == "post":
            mentions = source.get("entity", {}).get("mention", [])
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
                        json.dump(interaction, f_interactions, ensure_ascii=False)
                        f_interactions.write("\n")
            else:
                continue

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
            "_source": ["user_name", "user_title", "political_category.label", "type", "entity.mention"]
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
                    "political_label": label
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
            "_source": ["user_name", "user_title", "political_category.label", "type", "entity.mention"]
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
                    "political_label": label
                }
                json.dump(output_doc, f, ensure_ascii=False)
                f.write("\n")
                SECONDARY_COUNT += 1
        BATCH_INDEX += 1

log.info(f"Political tag hits: {SECONDARY_COUNT}")
log.info("Finished writing political tags to res.json.")

# Automatically call normalize_json.py
cur_path = os.path.dirname(__file__)
normalize_script = os.path.join(cur_path, "normalize_json.py")
subprocess.run(["python3", normalize_script], check=True)
log.info("Normalization complete.")
log.info("All steps completed successfully.")
log.info("Finished full pipeline.")
