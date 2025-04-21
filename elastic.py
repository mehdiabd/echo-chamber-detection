"""Module providing a sample Elasticsearch query."""
import os
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
                                      "size": 10,
                                      "track_total_hits": True})

# Print the result
for doc in result['hits']['hits']:
    print(doc['_source'])
print("Total hits:", result['hits']['total']['value'])
