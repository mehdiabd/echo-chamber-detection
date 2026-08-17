import json
import os
import tempfile
import unittest
from pathlib import Path

from cleanup_root import clean_project_root, plan_project_root_cleanup


def touch(path: Path, mtime: int = 1) -> Path:
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def names(entries):
    return {entry.path.name for entry in entries}


class CleanupRootTests(unittest.TestCase):
    def test_rotates_visualization_sets_together(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            old_files = [
                "dashboard_daily_260101_to_260101.html",
                "dashboard_daily_260101_to_260101_legend.json",
                "louvain_graph_daily_260101_to_260101.html",
                "hybrid_graph_daily_260101_to_260101.html",
                "louvain_similarity_daily_260101_to_260101.json",
                "hybrid_similarity_daily_260101_to_260101.html",
            ]
            new_files = [
                "dashboard_daily_260102_to_260102.html",
                "dashboard_daily_260102_to_260102_legend.json",
                "louvain_graph_daily_260102_to_260102.html",
                "hybrid_graph_daily_260102_to_260102.html",
            ]
            for filename in old_files:
                touch(root_path / filename, mtime=1)
            for filename in new_files:
                touch(root_path / filename, mtime=2)

            plan = plan_project_root_cleanup(
                root=str(root_path),
                visualization_retention={"daily": 1, "unknown": 0},
                drop_louvain_outputs=False,
            )

            self.assertEqual(names(plan), set(old_files))

    def test_keeps_core_data_and_removes_known_temporary_files(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            touch(root_path / "res.json")
            touch(root_path / "interactions.json")
            touch(root_path / "debug.log")
            touch(root_path / "scratch.tmp")
            touch(root_path / "source.py")

            removed = clean_project_root(root=str(root_path), dry_run=True)

            self.assertEqual(
                {Path(path).name for path in removed},
                {"debug.log", "scratch.tmp"},
            )

    def test_pipeline_config_protects_current_date_range(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "pipeline_config.json").write_text(
                json.dumps(
                    {
                        "start_date": "2026-06-19",
                        "end_date": "2026-07-18",
                        "slot_modes": ["daily"],
                    }
                ),
                encoding="utf-8",
            )
            protected = "dashboard_daily_260619_to_260619.html"
            old = "dashboard_daily_260101_to_260101.html"
            touch(root_path / protected, mtime=1)
            touch(root_path / old, mtime=2)

            plan = plan_project_root_cleanup(
                root=str(root_path),
                visualization_retention={"daily": 0, "unknown": 0},
            )

            self.assertEqual(names(plan), {old})

    def test_louvain_visual_outputs_are_dropped_even_in_config_range(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "pipeline_config.json").write_text(
                json.dumps(
                    {
                        "start_date": "2026-06-19",
                        "end_date": "2026-07-18",
                        "slot_modes": ["daily"],
                    }
                ),
                encoding="utf-8",
            )
            louvain_files = [
                "louvain_graph_daily_260619_to_260619.html",
                "louvain_similarity_daily_260619_to_260619.html",
                "louvain_similarity_daily_260619_to_260619.json",
            ]
            touch(root_path / "hybrid_graph_daily_260619_to_260619.html")
            for filename in louvain_files:
                touch(root_path / filename)

            plan = plan_project_root_cleanup(root=str(root_path))

            self.assertEqual(names(plan), set(louvain_files))


if __name__ == "__main__":
    unittest.main()
