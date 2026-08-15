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
make docker-up         # Echo Chamber API on http://localhost:8765
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

## HTTP API

The unified service in `echo_chamber_api.py` covers party-change events,
pipeline runs, echo-chamber reports, dashboard/timeline metadata, and human
review. Party-change ingestion from `community_detection.py` is unchanged.

Start it locally:

```bash
make api
```

`make docker-up` starts the same service on http://localhost:8765. Set
`PARTY_CHANGE_API_TOKEN` or `ECHO_API_TOKEN` to require bearer-token
authentication. The health endpoint is `GET /health`. Swagger UI is
`http://localhost:8765/docs`; the OpenAPI document is
`http://localhost:8765/openapi.json` and can be imported into Postman
(Import → Link). Use the Authorize button in `/docs` when a bearer token
is set. `/health`, `/docs`, `/redoc`, and `/openapi.json` stay public.

A frontend can rebuild the timeline from `GET /api/v1/dashboards`,
`GET /api/v1/topics`, and `GET /api/v1/files/{filename}` instead of
scraping `timeline_dashboard.html`. Screen-by-screen map:
[FRONTEND.md](FRONTEND.md).

Run the portable test suite locally with `make test`, or inside the API
image (the same command a Kubernetes Job can copy) with `make docker-test`.

Point the detection pipeline at party-change storage:

```bash
export PARTY_CHANGE_API_URL=http://127.0.0.1:8765/api/v1/party-changes
python3 community_detection.py
```

### Pipeline Run + Report

```bash
curl -X POST http://127.0.0.1:8765/api/v1/pipeline/runs \
  -H 'Content-Type: application/json' \
  -d '{"topic_label":"جنگ","start_date":"2026-01-01","end_date":"2026-01-14","fetch":true,"detect":true}'
curl 'http://127.0.0.1:8765/api/v1/pipeline/runs'
curl 'http://127.0.0.1:8765/api/v1/pipeline/config'
curl 'http://127.0.0.1:8765/api/v1/reports'
curl 'http://127.0.0.1:8765/api/v1/reports/<report_id>'
curl 'http://127.0.0.1:8765/api/v1/communities/<report_id>/<community_id>'
```

`POST /api/v1/pipeline/runs` starts Elasticsearch fetch (`elastic.py`) and
community detection (`community_detection.py`) in the background. Only one run
is active at a time; a second start returns `409`. Use `"fetch": false` to
reuse `interactions.json`, or `"detect": false` to fetch data only.

### Dashboard / Timeline

```bash
curl 'http://127.0.0.1:8765/api/v1/dashboards'
curl 'http://127.0.0.1:8765/api/v1/dashboards/daily_260101_to_260102'
curl 'http://127.0.0.1:8765/api/v1/topics'
curl 'http://127.0.0.1:8765/api/v1/files/dashboard_daily_260101_to_260102.html'
```

Dashboard slots return legend JSON, graph links, and date metadata so a client
can render the timeline without scraping static HTML.

### Human Review

```bash
curl -X POST http://127.0.0.1:8765/api/v1/review/samples \
  -H 'Content-Type: application/json' \
  -d '{"method":"both","sample_size":20,"min_size":1,"seed":42}'
curl -X POST http://127.0.0.1:8765/api/v1/review/labels \
  -H 'Content-Type: application/json' \
  -d '{"labels":[{"sample_id":"...","human_label":1,"notes":""}]}'
curl 'http://127.0.0.1:8765/api/v1/review/metrics'
```

Sampling reuses the balanced picker from `sample_echo_review.py`. Metrics are
precision, recall, F1, and accuracy against `human_label`.

### Party Change

```bash
curl 'http://127.0.0.1:8765/api/v1/party-changes?limit=100'
curl 'http://127.0.0.1:8765/api/v1/party-changes?node_id=some_username'
```

Party-change events stay in `runtime_data/party_changes.sqlite3`. Pipeline runs
and review labels use `runtime_data/echo_chamber.sqlite3`. Both are outside
generated-output paths and ignored by Git. Party-change rotation defaults to
365 days and at most 100,000 events (`PARTY_CHANGE_RETENTION_DAYS`,
`PARTY_CHANGE_MAX_EVENTS`, `PARTY_CHANGE_DB_PATH`).

`make party-change-api` still starts the narrower party-change-only server.

## License

MIT
