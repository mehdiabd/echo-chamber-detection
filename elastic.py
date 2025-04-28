"""Module providing a sample Elasticsearch query."""
import os
import subprocess
from elasticsearch import Elasticsearch

cur_path = os.path.dirname(__file__)

CERTIFICATE = cur_path + "/ca.crt"
AUTH = "YXYyeVRKWUJKSFpwMVdrTnZWRDc6UHhqRHBQa2ZUYW1yMnBwWTV3Ri0xUQ=="
ELASTICSEARCH_URL = "https://192.168.59.79:9200"
INDEX = "twitter_temp_data"
es = Elasticsearch(
    ELASTICSEARCH_URL,
    api_key=AUTH,
    ca_certs=CERTIFICATE,
    verify_certs=True,
    ssl_show_warn=False
)

result = es.search(index=INDEX, body={"query": {"match_all": {}},
                                      "size": 10000,
                                      "track_total_hits": True})

# Write the result to res.json
with open("res.json", "w", encoding="utf-8") as f:
    f.write(f"Total hits: {result['hits']['total']['value']}\n")
    for doc in result['hits']['hits']:
        f.write(f"{doc['_source']}\n")

print("Total hits:", result['hits']['total']['value'])

# Automatically call normalize_json.py
normalize_script = os.path.join(cur_path, "normalize_json.py")
subprocess.run(["python3", normalize_script], check=True)
print("Normalization complete.")
