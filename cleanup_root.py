"""Project-root cleanup helpers for generated artifacts."""
from __future__ import annotations

import glob
import os
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_REMOVE_PATTERNS: List[str] = [
    "community_summary_*.txt",
    "community_details_*.json",
    "hybrid_report_*_to_*.json",
    "debug.log",
    "run.log",
]

DEFAULT_REMOVE_DIRS: List[str] = [
    "__pycache__",
]

DEFAULT_KEEP_FILES: List[str] = [
    "community_examples.json",
    "interactions.json",
    "res.json",
]

LATEST_ONLY_GROUPS: List[Tuple[str, int]] = [
    ("dashboard_*.html", 1),
    ("dashboard_*_legend.json", 1),
    ("louvain_graph_*.html", 1),
    ("hybrid_graph_*.html", 1),
    ("louvain_similarity_*.html", 1),
    ("louvain_similarity_*.json", 1),
    ("hybrid_similarity_*.html", 1),
    ("hybrid_similarity_*.json", 1),
]


def _normalize_keep(keep_paths: Optional[Iterable[str]], root: str) -> set:
    keep = set()
    if not keep_paths:
        return keep
    for path in keep_paths:
        abs_path = os.path.abspath(os.path.join(root, path))
        keep.add(abs_path)
    return keep


def _select_latest(paths: Sequence[str], keep_count: int) -> set:
    ranked = sorted(
        paths,
        key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
        reverse=True,
    )
    return set(ranked[:keep_count])


def clean_project_root(
    root: str = ".",
    remove_patterns: Optional[Iterable[str]] = None,
    remove_dirs: Optional[Iterable[str]] = None,
    keep_paths: Optional[Iterable[str]] = None,
    keep_files: Optional[Iterable[str]] = None,
    latest_only_groups: Optional[Iterable[Tuple[str, int]]] = None,
    dry_run: bool = False,
) -> List[str]:
    """
    Remove generated artifacts from the project root to keep it tidy.

    Returns a list of removed paths (or would-remove paths in dry_run).
    """
    root_abs = os.path.abspath(root)
    patterns = list(remove_patterns) if remove_patterns else DEFAULT_REMOVE_PATTERNS
    dirs = list(remove_dirs) if remove_dirs else DEFAULT_REMOVE_DIRS
    keep = _normalize_keep(keep_paths, root_abs)
    keep_files_list = list(keep_files) if keep_files else DEFAULT_KEEP_FILES
    keep.update(_normalize_keep(keep_files_list, root_abs))
    latest_groups = list(latest_only_groups) if latest_only_groups else LATEST_ONLY_GROUPS

    removed: List[str] = []

    # Keep only the newest artifacts for selected patterns.
    for pattern, keep_count in latest_groups:
        matches = [
            os.path.abspath(p)
            for p in glob.glob(os.path.join(root_abs, pattern))
            if os.path.isfile(p)
        ]
        to_keep = _select_latest(matches, keep_count)
        for path in matches:
            if path in keep or path in to_keep:
                continue
            if dry_run:
                removed.append(path)
                continue
            try:
                os.remove(path)
                removed.append(path)
            except OSError:
                continue

    # Remove matching files in root.
    for pattern in patterns:
        for path in glob.glob(os.path.join(root_abs, pattern)):
            abs_path = os.path.abspath(path)
            if abs_path in keep:
                continue
            if not os.path.isfile(abs_path):
                continue
            if dry_run:
                removed.append(abs_path)
                continue
            try:
                os.remove(abs_path)
                removed.append(abs_path)
            except OSError:
                continue

    # Remove matching directories in root (e.g., __pycache__).
    for dir_name in dirs:
        dir_path = os.path.join(root_abs, dir_name)
        abs_dir = os.path.abspath(dir_path)
        if abs_dir in keep:
            continue
        if not os.path.isdir(abs_dir):
            continue
        if dry_run:
            removed.append(abs_dir)
            continue
        for base, subdirs, files in os.walk(abs_dir, topdown=False):
            for f in files:
                try:
                    os.remove(os.path.join(base, f))
                except OSError:
                    pass
            for d in subdirs:
                try:
                    os.rmdir(os.path.join(base, d))
                except OSError:
                    pass
        try:
            os.rmdir(abs_dir)
            removed.append(abs_dir)
        except OSError:
            continue

    return removed
