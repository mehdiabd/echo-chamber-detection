"""Module providing a flexible Elasticsearch query script."""
import os
import sys
import subprocess
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

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
                    "match": {
                        "date": "2025-04-20"
                    }
                }
            ]
        }
    },
    "_source": [
        "user.username",
        "user.displayname",
        "mentionedUsers",
        "retweetedTweet",
        "quotedTweet"
    ]
}

result = scan(
    es,
    index=INDEX,
    query=query_body,
    preserve_order=True
)

count = 0
with open("res.json", "w", encoding="utf-8") as f:
    for doc in result:
        f.write(f"{doc['_source']}\n")
        count += 1
print("Total hits:", count)

# Automatically call normalize_json.py
cur_path = os.path.dirname(__file__)
normalize_script = os.path.join(cur_path, "normalize_json.py")
subprocess.run(["python3", normalize_script], check=True)
print("Normalization complete.")
