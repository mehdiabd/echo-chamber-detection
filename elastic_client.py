"""Lightweight Elasticsearch client setup for runtime imports.

This module is safe to import from other modules (no data pipeline side effects).
"""

from __future__ import annotations

import os
from typing import Tuple

from elasticsearch import Elasticsearch


def create_es_client(auth_type: str | None = None) -> Tuple[Elasticsearch, str]:
    """
    Create Elasticsearch client and return (client, index).

    `auth_type`:
    - "1": production server (API key)
    - "2": temp server (basic auth)
    Falls back to ELASTIC_AUTH env var, then "1".
    """
    selected = (auth_type or os.getenv("ELASTIC_AUTH") or "1").strip()
    cur_path = os.path.dirname(__file__)

    if selected == "1":
        # Public hostname behind Kong, which terminates TLS with a publicly trusted
        # certificate - so no ca_certs/local CA file is needed and none should be set.
        # The previous value pointed straight at https://192.168.59.79:9200 with
        # ca_certs=<repo>/ca.crt; because .gitignore excludes *.crt that file was never
        # in the image, and every pipeline run died with
        # "TlsError: SSLError([Errno 2] No such file or directory)".
        es = Elasticsearch(
            "https://elastic.synappse.ir",
            api_key="YXYyeVRKWUJKSFpwMVdrTnZWRDc6UHhqRHBQa2ZUYW1yMnBwWTV3Ri0xUQ==",
            verify_certs=True,
            ssl_show_warn=False,
        )
        index = "twitter_temp_data"
        return es, index

    if selected == "2":
        certificate = os.path.join(cur_path, "http_ca.crt")
        es = Elasticsearch(
            "https://192.168.59.26:9200/",
            basic_auth=("m.abdolahi", "@bd0l@h12345"),
            verify_certs=True,
            ca_certs=certificate,
            ssl_show_warn=False,
            ssl_assert_hostname=False,
        )
        index = "twitter_maroufi"
        return es, index

    raise ValueError(f"Unsupported auth type: {selected!r}")


es, INDEX = create_es_client()
