import json
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
