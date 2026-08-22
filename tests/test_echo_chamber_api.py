import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from echo_chamber_api import (
    ArtifactIndex,
    PipelineStore,
    ReviewStore,
    create_app,
    execute_pipeline,
    validate_pipeline_payload,
)
from party_change_api import PartyChangeStore


REPORT = {
    "timeframe": {"start": "2026-01-01", "end": "2026-01-02"},
    "hybrid": {
        "modularity": 0.41,
        "silhouette": 0.22,
        "echo_metrics": {
            "0": {
                "size": 10,
                "ei_index": -0.5,
                "conductance": 0.1,
                "content_homogeneity": 0.8,
                "is_echo_chamber": True,
                "stance": {
                    "target": "رهبری",
                    "stance": "neg",
                    "counts": {"pos": 1, "neg": 8, "neu": 1, "total": 10},
                },
            },
            "1": {
                "size": 6,
                "ei_index": 0.2,
                "conductance": 0.5,
                "content_homogeneity": 0.3,
                "is_echo_chamber": False,
            },
        },
    },
    "louvain": {
        "modularity": 0.33,
        "echo_metrics": {
            "0": {
                "size": 8,
                "ei_index": -0.1,
                "conductance": 0.2,
                "content_homogeneity": 0.7,
                "is_echo_chamber": True,
            }
        },
    },
}


def write_report(root: Path, name="hybrid_report_2026-01-01_to_2026-01-02_20260102_120000.json"):
    directory = root / "communities"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(REPORT), encoding="utf-8")
    return path


def write_dashboard(root: Path):
    (root / "dashboard_daily_260101_to_260102.html").write_text(
        "<html>dashboard</html>", encoding="utf-8"
    )
    (root / "dashboard_daily_260101_to_260102_legend.json").write_text(
        json.dumps({"hybrid": {"groups": {"alice": 0, "bob": 0, "carol": 1}}}),
        encoding="utf-8",
    )
    (root / "hybrid_graph_daily_260101_to_260102.html").write_text(
        "<html>graph</html>", encoding="utf-8"
    )
    (root / "hybrid_graph_daily_260101_to_260102.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "alice", "label": "alice", "community": 0, "color": "#2563eb"},
                    {"id": "bob", "label": "bob", "community": 0, "color": "#2563eb"},
                    {"id": "carol", "label": "carol", "community": 1, "color": "#16a34a"},
                ],
                "edges": [
                    {"from": "alice", "to": "bob", "weight": 2},
                    {"from": "bob", "to": "carol", "weight": 1},
                ],
                "partyFocus": {
                    "parties": [
                        {
                            "name": "جریان زن‌زندگی‌آزادی",
                            "stats": {"total": 438, "outgoing": 209, "incoming": 229},
                            "top_interactions": [
                                {
                                    "type": "نقل‌قول",
                                    "handle": "@CENTCOM",
                                    "count": 15,
                                },
                                {"type": "ذکر", "handle": "@PMN_Amy", "count": 10},
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def party_event(event_id="event-1", node_id="alice"):
    return {
        "event_id": event_id,
        "node_id": node_id,
        "from_party": "party-a",
        "to_party": "party-b",
        "previous_slot": "2026-01-01",
        "current_slot": "2026-01-02",
        "slot_mode": "daily",
        "topic": "topic",
        "detected_at": "2026-01-02T12:00:00Z",
    }


class ArtifactIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        write_report(self.root)
        write_dashboard(self.root)
        (self.root / "pipeline_config.json").write_text(
            json.dumps(
                {
                    "topic_label": "جنگ",
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-02",
                }
            ),
            encoding="utf-8",
        )
        self.index = ArtifactIndex(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_lists_and_loads_reports(self):
        total, reports = self.index.list_reports({}, 10, 0)
        self.assertEqual(total, 1)
        self.assertEqual(reports[0]["hybrid"]["echo_count"], 1)
        self.assertEqual(reports[0]["topic"], "جنگ")
        detail = self.index.get_report(reports[0]["id"])
        self.assertEqual(detail["report"]["hybrid"]["modularity"], 0.41)

    def test_community_includes_legend_members(self):
        community = self.index.get_community(
            "hybrid_report_2026-01-01_to_2026-01-02_20260102_120000",
            "0",
            "hybrid",
        )
        self.assertTrue(community["metrics"]["is_echo_chamber"])
        self.assertEqual(set(community["members"]), {"alice", "bob"})

    def test_dashboard_slot_metadata(self):
        total, dashboards = self.index.list_dashboards({}, 10, 0)
        self.assertEqual(total, 1)
        detail = self.index.get_dashboard("daily_260101_to_260102")
        self.assertEqual(detail["slot_mode"], "daily")
        self.assertEqual(detail["legends"]["hybrid"]["groups"]["alice"], 0)
        self.assertEqual(detail["graph"]["node_count"], 3)
        self.assertEqual(detail["graph"]["edge_count"], 2)
        self.assertEqual(detail["graph"]["nodes"][0]["id"], "alice")
        self.assertEqual(detail["graph"]["edges"][0]["from"], "alice")
        self.assertNotIn("partyFocus", detail["graph"])
        self.assertEqual(
            detail["partyFocus"]["parties"][0]["name"],
            "جریان زن‌زندگی‌آزادی",
        )
        self.assertEqual(detail["partyFocus"]["parties"][0]["stats"]["total"], 438)
        self.assertEqual(
            detail["partyFocus"]["parties"][0]["top_interactions"][0]["handle"],
            "@CENTCOM",
        )
        self.assertEqual(
            detail["files"]["graph"],
            "/api/v1/files/hybrid_graph_daily_260101_to_260102.json",
        )

    def test_dashboard_graph_empty_without_json(self):
        (self.root / "hybrid_graph_daily_260101_to_260102.json").unlink()
        detail = self.index.get_dashboard("daily_260101_to_260102")
        self.assertEqual(detail["graph"]["nodes"], [])
        self.assertEqual(detail["graph"]["edge_count"], 0)
        self.assertEqual(detail["partyFocus"], {"parties": []})
        self.assertIsNone(detail["files"]["graph"])

    def test_rejects_path_traversal_for_files(self):
        with self.assertRaises(ValueError):
            self.index.resolve_static_file("../.env")
        (self.root / "secret.env").write_text("nope", encoding="utf-8")
        with self.assertRaises(PermissionError):
            self.index.resolve_static_file("secret.env")


class PipelineStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = PipelineStore(Path(self.temp_dir.name) / "api.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_and_complete_run(self):
        run = self.store.create({"fetch": True, "detect": True, "topic_label": "x"})
        self.assertEqual(run["status"], "queued")
        self.assertTrue(self.store.has_active())
        self.store.mark_running(run["run_id"])
        self.store.mark_done(
            run["run_id"], {"reports": ["a.json"], "dashboards": ["d.html"]}
        )
        done = self.store.get(run["run_id"])
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["reports"], ["a.json"])
        self.assertFalse(self.store.has_active())

    def test_validation_requires_work(self):
        with self.assertRaises(ValueError):
            validate_pipeline_payload({"fetch": False, "detect": False})


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ReviewStore(Path(self.temp_dir.name) / "api.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_labels_and_metrics(self):
        batch = self.store.create_batch(
            [
                {
                    "report_file": "r.json",
                    "method": "hybrid",
                    "community_id": "0",
                    "predicted_is_echo": "1",
                    "size": "10",
                },
                {
                    "report_file": "r.json",
                    "method": "hybrid",
                    "community_id": "1",
                    "predicted_is_echo": "0",
                    "size": "6",
                },
            ],
            report_query="r.json",
            method="hybrid",
            sample_size=2,
            min_size=1,
            seed=1,
        )
        samples = batch["samples"]
        self.store.apply_labels(
            [
                {
                    "sample_id": samples[0]["sample_id"],
                    "human_label": 1,
                    "notes": "echo",
                },
                {
                    "sample_id": samples[1]["sample_id"],
                    "human_label": 1,
                    "notes": "miss",
                },
            ]
        )
        metrics = self.store.metrics(batch["batch_id"])
        self.assertEqual(metrics["rows"]["used"], 2)
        self.assertEqual(metrics["overall"]["tp"], 1)
        self.assertEqual(metrics["overall"]["fn"], 1)
        self.assertEqual(metrics["overall"]["precision"], 1.0)
        self.assertEqual(metrics["overall"]["recall"], 0.5)


class ExecutePipelineTests(unittest.TestCase):
    def test_runs_requested_scripts_and_snapshots_outputs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "elastic.py").write_text(
                "from pathlib import Path\n"
                "Path('interactions.json').write_text('[]')\n"
                "Path('pipeline_config.json').write_text('{\"topic_label\":\"x\"}')\n",
                encoding="utf-8",
            )
            (root / "community_detection.py").write_text(
                "from pathlib import Path\n"
                "Path('communities').mkdir(exist_ok=True)\n"
                "Path('communities/hybrid_report_2026-01-01_to_2026-01-02_20260102_120000.json')"
                ".write_text('{\"timeframe\":{\"start\":\"2026-01-01\",\"end\":\"2026-01-02\"}}')\n"
                "Path('dashboard_daily_260101_to_260102.html').write_text('<html></html>')\n",
                encoding="utf-8",
            )
            log_file = root / "run.log"
            artifacts = execute_pipeline(
                root,
                {
                    "fetch": True,
                    "detect": True,
                    "topic_label": "جنگ",
                    "slot_modes": ["daily"],
                },
                log_file,
                "python3",
            )
            self.assertEqual(
                artifacts["reports"],
                ["hybrid_report_2026-01-01_to_2026-01-02_20260102_120000.json"],
            )
            self.assertEqual(
                artifacts["dashboards"],
                ["dashboard_daily_260101_to_260102.html"],
            )
            config = json.loads(
                (root / "pipeline_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["slot_modes"], ["daily"])
            self.assertEqual(config["topic_label"], "جنگ")


class ApiServerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        write_report(self.root)
        write_dashboard(self.root)
        (self.root / "pipeline_config.json").write_text(
            json.dumps({"topic_label": "جنگ", "start_date": "2026-01-01"}),
            encoding="utf-8",
        )
        (self.root / "topics_10.csv").write_text(
            "topic_label,topic_query\nجنگ,query-war\nانتخابات,query-election\n",
            encoding="utf-8",
        )
        db_path = self.root / "runtime_data" / "api.sqlite3"
        self.party_store = PartyChangeStore(
            self.root / "runtime_data" / "party.sqlite3", 0, 100
        )
        self.pipeline_store = PipelineStore(db_path)
        self.review_store = ReviewStore(db_path)
        self.executor_calls = []

        def fake_executor(root, params, log_file, python_bin):
            self.executor_calls.append(params)
            Path(log_file).write_text("ok\n", encoding="utf-8")
            return {
                "reports": [
                    "hybrid_report_2026-01-01_to_2026-01-02_20260102_120000.json"
                ],
                "dashboards": ["dashboard_daily_260101_to_260102.html"],
            }

        self.fake_executor = fake_executor
        self.client = self.make_client()

    def make_client(self, api_token=""):
        app = create_app(
            party_store=self.party_store,
            pipeline_store=self.pipeline_store,
            review_store=self.review_store,
            root=self.root,
            api_token=api_token,
            pipeline_executor=self.fake_executor,
            pipeline_sync=True,
        )
        return TestClient(app)

    def tearDown(self):
        self.temp_dir.cleanup()

    def request(self, method, path, payload=None, status=200, client=None, token=None):
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        kwargs = {"headers": headers}
        if payload is not None:
            kwargs["json"] = payload
        response = (client or self.client).request(method, path, **kwargs)
        if response.status_code != status:
            self.fail(
                f"{method} {path} returned {response.status_code} {response.text}, "
                f"expected {status}"
            )
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            return response.json()
        return response

    def test_health_and_pipeline_config(self):
        self.assertEqual(self.request("GET", "/health")["status"], "ok")
        config = self.request("GET", "/api/v1/pipeline/config")
        self.assertTrue(config["exists"])
        self.assertEqual(config["config"]["topic_label"], "جنگ")

    def test_pipeline_run_and_conflict(self):
        created = self.request(
            "POST",
            "/api/v1/pipeline/runs",
            {
                "fetch": False,
                "detect": True,
                "topic_label": "جنگ",
                "slot_modes": ["daily"],
            },
            status=202,
        )
        self.assertEqual(created["status"], "done")
        self.assertEqual(self.executor_calls[0]["detect"], True)
        listed = self.request("GET", "/api/v1/pipeline/runs")
        self.assertEqual(listed["total"], 1)
        detail = self.request("GET", f"/api/v1/pipeline/runs/{created['run_id']}")
        self.assertTrue(detail["reports"][0].startswith("hybrid_report_"))

        blocked = self.pipeline_store.create(
            {"fetch": False, "detect": True, "topic_label": "جنگ"}
        )
        self.pipeline_store.mark_running(blocked["run_id"])
        conflict = self.request(
            "POST",
            "/api/v1/pipeline/runs",
            {"fetch": False, "detect": True},
            status=409,
        )
        self.assertIn("already queued or running", conflict["error"])

    def test_reports_dashboards_topics_and_files(self):
        reports = self.request("GET", "/api/v1/reports")
        self.assertEqual(reports["total"], 1)
        report_id = reports["reports"][0]["id"]
        detail = self.request("GET", f"/api/v1/reports/{report_id}")
        self.assertEqual(detail["hybrid"]["echo_count"], 1)
        community = self.request("GET", f"/api/v1/communities/{report_id}/0")
        self.assertEqual(community["member_count"], 2)
        louvain = self.request(
            "GET", f"/api/v1/communities/{report_id}/0?method=louvain"
        )
        self.assertEqual(louvain["method"], "louvain")
        self.assertTrue(louvain["metrics"]["is_echo_chamber"])
        dashboards = self.request("GET", "/api/v1/dashboards")
        self.assertEqual(dashboards["dashboards"][0]["id"], "daily_260101_to_260102")
        slot = self.request("GET", "/api/v1/dashboards/daily_260101_to_260102")
        self.assertIn("alice", slot["legends"]["hybrid"]["groups"])
        self.assertEqual(slot["graph"]["node_count"], 3)
        self.assertEqual(slot["graph"]["edges"][0]["from"], "alice")
        topics = self.request("GET", "/api/v1/topics")
        labels = {item["label"] for item in topics["topics"]}
        self.assertIn("جنگ", labels)
        self.assertIn("انتخابات", labels)
        downloaded = self.request(
            "GET", "/api/v1/files/dashboard_daily_260101_to_260102.html"
        )
        self.assertIn(b"dashboard", downloaded.content)

    def test_human_review_sample_label_and_metrics(self):
        batch = self.request(
            "POST",
            "/api/v1/review/samples",
            {"method": "hybrid", "sample_size": 0, "min_size": 1, "seed": 1},
            status=201,
        )
        self.assertEqual(batch["count"], 2)
        samples = {row["community_id"]: row for row in batch["samples"]}
        listed = self.request(
            "GET", f"/api/v1/review/samples?batch_id={batch['batch_id']}"
        )
        self.assertEqual(listed["total"], 2)
        detail = self.request(
            "GET", f"/api/v1/review/samples/{samples['0']['sample_id']}"
        )
        self.assertEqual(detail["community_id"], "0")
        unlabeled = self.request(
            "GET",
            f"/api/v1/review/samples?batch_id={batch['batch_id']}&labeled=false",
        )
        self.assertEqual(unlabeled["total"], 2)
        labeled = self.request(
            "POST",
            "/api/v1/review/labels",
            {
                "labels": [
                    {"sample_id": samples["0"]["sample_id"], "human_label": 1},
                    {"sample_id": samples["1"]["sample_id"], "human_label": 0},
                ]
            },
        )
        self.assertEqual(labeled["updated"], 2)
        metrics = self.request(
            "GET", f"/api/v1/review/metrics?batch_id={batch['batch_id']}"
        )
        self.assertEqual(metrics["overall"]["n"], 2)
        self.assertEqual(metrics["overall"]["tp"], 1)
        self.assertEqual(metrics["overall"]["tn"], 1)
        self.assertEqual(metrics["overall"]["precision"], 1.0)
        self.assertEqual(metrics["overall"]["recall"], 1.0)

    def test_party_change_still_works(self):
        accepted = self.request(
            "POST",
            "/api/v1/party-changes",
            {"events": [party_event()]},
        )
        self.assertEqual(accepted["accepted"], 1)
        listed = self.request("GET", "/api/v1/party-changes?node_id=alice")
        self.assertEqual(listed["total"], 1)
        detail = self.request("GET", "/api/v1/party-changes/event-1")
        self.assertEqual(detail["to_party"], "party-b")

    def test_bearer_auth_rejects_and_accepts(self):
        client = self.make_client(api_token="secret")
        health = self.request("GET", "/health", client=client)
        self.assertEqual(health["status"], "ok")
        docs = client.get("/openapi.json")
        self.assertEqual(docs.status_code, 200)
        denied = self.request(
            "GET", "/api/v1/pipeline/config", status=401, client=client
        )
        self.assertEqual(denied["error"], "unauthorized")
        allowed = self.request(
            "GET",
            "/api/v1/pipeline/config",
            client=client,
            token="secret",
        )
        self.assertTrue(allowed["exists"])

    def test_cors_preflight(self):
        response = self.client.options(
            "/api/v1/pipeline/config",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization, Content-Type",
            },
        )
        self.assertIn(response.status_code, {200, 204})
        self.assertEqual(response.headers.get("access-control-allow-origin"), "*")

    def test_invalid_payloads_return_400(self):
        pipeline = self.request(
            "POST",
            "/api/v1/pipeline/runs",
            {"fetch": False, "detect": False},
            status=400,
        )
        self.assertIn("fetch or detect", pipeline["error"])
        party = self.request(
            "POST",
            "/api/v1/party-changes",
            {"events": []},
            status=400,
        )
        self.assertIn("events", party["error"])
        community = self.request(
            "GET",
            "/api/v1/communities/missing/0?method=other",
            status=400,
        )
        self.assertIn("method", community["error"])

    def test_file_errors(self):
        forbidden = self.request("GET", "/api/v1/files/secret.txt", status=403)
        self.assertEqual(forbidden["error"], "file is not served")
        missing = self.request(
            "GET",
            "/api/v1/files/dashboard_daily_999999_to_999999.html",
            status=404,
        )
        self.assertEqual(missing["error"], "not found")
        unknown_report = self.request(
            "GET", "/api/v1/reports/does-not-exist", status=404
        )
        self.assertEqual(unknown_report["error"], "not found")


if __name__ == "__main__":
    unittest.main()
