"""Module providing a sample Elasticsearch query."""
import os
from elasticsearch import Elasticsearch

cur_path = os.path.dirname(__file__)

CERTIFICATE = cur_path + "/ca.crt"
AUTH = "YXYyeVRKWUJKSFpwMVdrTnZWRDc6UHhqRHBQa2ZUYW1yMnBwWTV3Ri0xUQ=="
ELASTICSEARCH_URL = "https://192.168.59.79:9200"
INDEX = "twitter_temp_data"
es = Elasticsearch(ELASTICSEARCH_URL, ca_certs=CERTIFICATE, api_key=AUTH,
                   ssl_assert_hostname=False)

es.search(index=INDEX, query={"query": {"match_all": {}}})
