# Echo Chamber Detection

A project to detect and analyze echo chambers in social networks. The visual
dashboard shows the Hybrid Node2Vec + KMeans output; Louvain is kept as a
background comparison method for reports and metrics.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r dev-requirements.txt
```

## Docker (recommended for any local machine)

Requires Docker Desktop / Docker Engine with Compose v2.

```bash
cp .env.example .env   # optional; add SYNAPPSE_API_KEY if you use LLM naming
make docker-up         # party-change API on http://localhost:8765
make docker-elastic    # fetch Elasticsearch data into the project root
make docker-pipeline   # run community detection (writes dashboards into ./)
make docker-down
```

The compose stack bind-mounts the project directory, so `interactions.json`,
`res.json`, `pipeline_config.json`, `ca.crt`, and generated HTML stay on the
host. Runtime SQLite lives in the `runtime_data` Docker volume. Local Ollama is
reached via `host.docker.internal:11434`. Set `ENABLE_LLM_NAMING=false` in
`.env` to skip LLMs entirely.

## Usage

- Run notebooks for data collection, graph building, and community detection.
- Use Makefile commands for formatting, linting, and testing.

## Community naming LLMs

Community naming is enabled by default and tries these backends in order:

1. The Synappse organization endpoint.
2. The self-hosted organization endpoint at `192.168.59.239:8002`.
3. Local Ollama using `llama3.1`.
4. The dominant political label in community metadata.

Configure Synappse with `ORG_LLM_API_KEY` (or `SYNAPPSE_API_KEY`). Its URL can
be overridden with `ORG_LLM_URL`. The self-hosted URLs can be overridden with
`ORG_LLM_SELF_HOSTED_MODELS_URL` and
`ORG_LLM_SELF_HOSTED_CHAT_URL`. Set `ENABLE_LLM_NAMING=false` to skip all LLMs
and classify communities directly from their metadata.

## Project-root artifact rotation

Generated dashboards, graph HTML files, similarity JSON/HTML files, logs, and
temporary files are rotated by `cleanup_root.py`. The cleanup is precise: it
only touches known generated filenames in the project root, protects artifacts
that overlap the date range in `pipeline_config.json`, and keeps source files
plus core data files such as `res.json`, `interactions.json`, and
`pipeline_config.json`.

Preview cleanup:

```bash
make rotate-temp
```

Apply cleanup:

```bash
make rotate-temp-apply
```

Override visualization retention when needed:

```bash
python3 -B cleanup_root.py --apply --retention daily=14,weekly=8,ten_day=3,legacy=2
```

The community detection pipeline also rotates generated root artifacts before
and after a run. Use `--no-config-protect` only when you intentionally want old
artifacts from the configured date range removed too.

## Party-change API

Start the local backend:

```bash
make party-change-api
```

Point the detection pipeline at it:

```bash
export PARTY_CHANGE_API_URL=http://127.0.0.1:8765/api/v1/party-changes
python3 community_detection.py
```

Query detected transitions:

```bash
curl 'http://127.0.0.1:8765/api/v1/party-changes?limit=100'
curl 'http://127.0.0.1:8765/api/v1/party-changes?node_id=some_username'
```

The service stores data in `runtime_data/party_changes.sqlite3`, outside the
generated-output paths and ignored by Git. It rotates records during ingestion;
the defaults retain 365 days and at most 100,000 events. Configure these with
`PARTY_CHANGE_RETENTION_DAYS`, `PARTY_CHANGE_MAX_EVENTS`, and
`PARTY_CHANGE_DB_PATH`.

Set `PARTY_CHANGE_API_TOKEN` in both processes to require bearer-token
authentication. The health endpoint is `GET /health`.

## License

MIT
