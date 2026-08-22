"""Lightweight Elasticsearch client setup for runtime imports.

This module is safe to import from other modules (no data pipeline side effects).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Tuple

from elasticsearch import Elasticsearch


def _unique_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def ca_cert_candidates(filename: str) -> list[str]:
    env = os.getenv("ELASTIC_CA_CERT", "").strip()
    here = os.path.dirname(os.path.abspath(__file__))
    return _unique_paths(
        [
            env,
            os.path.join(here, filename),
            os.path.join(os.getcwd(), filename),
            os.path.join("/app", filename),
            os.path.join("/certs", filename),
            os.path.join(here, "certs", filename),
            os.path.join(os.getcwd(), "runtime_data", filename),
            os.path.join("/app/runtime_data", filename),
        ]
    )


def resolve_ca_cert(filename: str) -> str | None:
    for path in ca_cert_candidates(filename):
        if os.path.isfile(path):
            return path
    return None


def ssl_client_kwargs(filename: str) -> dict[str, Any]:
    forced = os.getenv("ELASTIC_VERIFY_CERTS", "").strip().lower()
    if forced in {"0", "false", "no", "off"}:
        return {"verify_certs": False, "ssl_show_warn": False}
    cert = resolve_ca_cert(filename)
    if cert:
        return {"verify_certs": True, "ssl_show_warn": False, "ca_certs": cert}
    # ca.crt is gitignored and missing from the Docker image. Skipping verify
    # is the only way fetch can run until the cert is mounted.
    print(
        f"[elastic] {filename} not found; connecting without TLS verification. "
        "Mount the CA at /app/ca.crt or set ELASTIC_CA_CERT.",
        file=sys.stderr,
        flush=True,
    )
    return {"verify_certs": False, "ssl_show_warn": False}


def create_es_client(auth_type: str | None = None) -> Tuple[Elasticsearch, str]:
    """
    Create Elasticsearch client and return (client, index).

    `auth_type`:
    - "1": production server (API key)
    - "2": temp server (basic auth)
    Falls back to ELASTIC_AUTH env var, then "1".
    """
    selected = (auth_type or os.getenv("ELASTIC_AUTH") or "1").strip()

    if selected == "1":
        es = Elasticsearch(
            "https://192.168.59.79:9200",
            api_key="YXYyeVRKWUJKSFpwMVdrTnZWRDc6UHhqRHBQa2ZUYW1yMnBwWTV3Ri0xUQ==",
            **ssl_client_kwargs("ca.crt"),
        )
        return es, "twitter_temp_data"

    if selected == "2":
        es = Elasticsearch(
            "https://192.168.59.26:9200/",
            basic_auth=("m.abdolahi", "@bd0l@h12345"),
            ssl_assert_hostname=False,
            **ssl_client_kwargs("http_ca.crt"),
        )
        return es, "twitter_maroufi"

    raise ValueError(f"Unsupported auth type: {selected!r}")


es, INDEX = create_es_client()
