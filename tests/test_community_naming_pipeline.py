import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import networkx as nx

import community_detection
from community_naming import build_community_classification_prompt


class CommunityNamingPipelineTests(unittest.TestCase):
    def test_rejects_gray_llm_result_when_specific_metadata_exists(self):
        candidate = """{
            "selected_label": "قشر خاکستری",
            "confidence": 80,
            "reasoning": "unclear"
        }"""

        parsed = community_detection.validate_llm_community_label(
            candidate,
            "احمدی‌نژادی‌ها",
        )

        self.assertIsNone(parsed)

    def test_gray_is_allowed_when_metadata_is_also_gray(self):
        candidate = '{"selected_label": "قشر خاکستری"}'

        parsed = community_detection.validate_llm_community_label(
            candidate,
            "قشر خاکستری",
        )

        self.assertEqual(parsed["selected_label"], "قشر خاکستری")

    def test_prompt_forbids_gray_when_dominant_label_is_known(self):
        prompt = build_community_classification_prompt(
            members=["member"],
            text_content="Metadata: political_label=ahmadinezhad",
            dominant_political_label="احمدی‌نژادی‌ها",
        )

        self.assertIn("احمدی‌نژادی‌ها", prompt)
        self.assertIn("MUST NOT select", prompt)

    def test_display_replaces_gray_with_dominant_ahmadinezhad_tag(self):
        label = community_detection.prefer_specific_political_label(
            "قشر خاکستری",
            members=["member"],
            meta_map={"member": {"political_label": "ahmadinezhad"}},
        )

        self.assertEqual(label, "احمدی‌نژادی‌ها")

    @patch("community_detection.save_community_name")
    @patch("community_detection.call_llm_with_fallback", return_value=(None, None))
    def test_llm_chain_failure_uses_dominant_political_tag(
        self,
        mocked_llm_chain,
        mocked_save,
    ):
        graph = nx.Graph()
        members = ["center", "member-1", "member-2"]
        graph.add_nodes_from(members)
        graph.graph["meta_map"] = {
            "center": {"political_label": "اصولگرا"},
            "member-1": {"political_label": "اصولگرا"},
            "member-2": {"political_label": "اصلاح‌طلب"},
        }

        with patch.object(community_detection, "ENABLE_LLM_NAMING", True), patch.object(
            community_detection,
            "ENABLE_NAMING_TEXT_FETCH",
            False,
        ):
            label = community_detection.ai_name_community(
                center_node="center",
                neighbors=["member-1", "member-2"],
                # A dictionary label must not override members' political tags.
                node_label_map={"center": "اصلاح‌طلب"},
                comm_id=1,
                method="test",
                g=graph,
            )

        self.assertEqual(label, "اصولگرا")
        mocked_llm_chain.assert_called_once()
        mocked_save.assert_called_once()

    def test_party_focus_splits_type_and_handle(self):
        messages = [
            {"sender": "alice", "target": "CENTCOM", "type": "quote"},
            {"sender": "alice", "target": "PMN_Amy", "type": "mention"},
            {"sender": "bob", "target": "alice", "type": "mention"},
        ]
        meta_map = {
            "alice": {"political_label": "زن زندگی آزادی"},
            "bob": {"political_label": "اصولگرا"},
        }
        payload = community_detection.build_party_focus_for_messages(
            messages, meta_map
        )
        by_name = {party["name"]: party for party in payload["parties"]}
        freedom = by_name["جریان زن‌زندگی‌آزادی"]
        self.assertEqual(freedom["stats"], {"total": 3, "outgoing": 2, "incoming": 1})
        self.assertEqual(
            freedom["top_interactions"],
            [
                {"type": "نقل‌قول", "handle": "@CENTCOM", "count": 1},
                {"type": "ذکر", "handle": "@PMN_Amy", "count": 1},
                {"type": "ذکر", "handle": "@bob", "count": 1},
            ],
        )
        self.assertEqual(by_name["اصولگرا"]["stats"]["outgoing"], 1)
        self.assertEqual(by_name["اصولگرا"]["stats"]["incoming"], 0)

    def test_party_focus_is_written_into_hybrid_graph_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_path = root / "hybrid_graph_daily_260101_to_260102.json"
            graph_path.write_text(
                json.dumps({"nodes": [], "edges": []}), encoding="utf-8"
            )
            community_detection.attach_party_focus_to_graph_json(
                str(root / "dashboard_daily_260101_to_260102.html"),
                {"parties": [{"name": "اصولگرا", "stats": {"total": 1}}]},
            )
            saved = json.loads(graph_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["nodes"], [])
            self.assertEqual(saved["partyFocus"]["parties"][0]["name"], "اصولگرا")

    def test_timeline_dashboard_skips_missing_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dashboard = root / "dashboard_daily_260101_to_260102.html"
            dashboard.write_text("<html></html>", encoding="utf-8")
            with patch.object(
                community_detection, "resolve_timeline_template", return_value=None
            ):
                community_detection.generate_timeline_dashboard(
                    output_file=str(root / "timeline_dashboard.html"),
                    topic_label="جنگ",
                    mode_dashboards={
                        "daily": [
                            {
                                "file": str(dashboard),
                                "start": "2026-01-01",
                                "end": "2026-01-02",
                            }
                        ]
                    },
                )
            self.assertFalse((root / "timeline_dashboard.html").exists())

    def test_timeline_dashboard_uses_bundled_template(self):
        template = community_detection.resolve_timeline_template()
        self.assertTrue(template)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dashboard = root / "dashboard_daily_260101_to_260102.html"
            dashboard.write_text("<html></html>", encoding="utf-8")
            output = root / "timeline_dashboard.html"
            community_detection.generate_timeline_dashboard(
                output_file=str(output),
                topic_label="جنگ",
                mode_dashboards={
                    "daily": [
                        {
                            "file": str(dashboard),
                            "start": "2026-01-01",
                            "end": "2026-01-02",
                        }
                    ]
                },
            )
            html = output.read_text(encoding="utf-8")
            self.assertIn("جنگ", html)
            self.assertIn("daily", html)

    def test_empty_visualize_does_not_overwrite_populated_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filename = "dashboard_daily_260101_to_260101.html"
            graph = root / "hybrid_graph_daily_260101_to_260101.json"
            graph.write_text(
                json.dumps(
                    {
                        "nodes": [{"id": "alice", "label": "alice"}],
                        "edges": [],
                        "partyFocus": {
                            "parties": [
                                {
                                    "name": "اصولگرا",
                                    "stats": {"total": 1, "outgoing": 1, "incoming": 0},
                                    "top_interactions": [],
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / filename).write_text("<html>keep-me</html>", encoding="utf-8")
            cwd = Path.cwd()
            try:
                os.chdir(root)
                kept = community_detection.visualize_or_dummy(
                    datetime(2026, 1, 1),
                    datetime(2026, 1, 1),
                    nx.Graph(),
                    slot_mode="daily",
                )
            finally:
                os.chdir(cwd)
            self.assertEqual(kept, filename)
            payload = json.loads(graph.read_text(encoding="utf-8"))
            self.assertEqual(payload["nodes"][0]["id"], "alice")
            self.assertEqual((root / filename).read_text(encoding="utf-8"), "<html>keep-me</html>")


if __name__ == "__main__":
    unittest.main()
