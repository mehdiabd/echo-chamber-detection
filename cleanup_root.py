"""Precise project-root rotation for generated artifacts.

The cleanup is intentionally root-scoped. It only touches known generated
filenames in the project root plus explicitly named cache directories.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_KEEP_FILES: List[str] = [
    "community_examples.json",
    "community_details.json",
    "communities_250601_to_250630.json",
    "interactions.json",
    "pipeline_config.json",
    "res.json",
]

DEFAULT_REMOVE_DIRS: List[str] = [
    "__pycache__",
]

DEFAULT_REMOVE_PATTERNS: List[str] = [
    ".DS_Store",
    "debug.log",
    "run.log",
    "louvain_graph.html",
    "hybrid_graph.html",
    "graph.html",
    "*.tmp",
    "*.temp",
]

LOUVAIN_OUTPUT_PATTERNS: List[str] = [
    "louvain_graph*.html",
    "louvain_similarity*.html",
    "louvain_similarity*.json",
]

DEFAULT_SUMMARY_KEEP = 10
DEFAULT_BACKUP_KEEP = 1

DEFAULT_VISUALIZATION_RETENTION: Dict[str, int] = {
    "hourly": 24,
    "daily": 7,
    "weekly": 4,
    "ten_day": 2,
    "monthly": 2,
    "quarterly": 2,
    "legacy": 1,
    "unknown": 1,
}

VISUALIZATION_RE = re.compile(
    r"^(?P<prefix>dashboard|louvain_graph|hybrid_graph|"
    r"louvain_similarity|hybrid_similarity)"
    r"(?:_(?P<mode>hourly|daily|weekly|ten_day|monthly|quarterly))?"
    r"_(?P<start>\d{6,8})_to_(?P<end>\d{6,8})"
    r"(?P<legend>_legend)?\.(?P<ext>html|json)$"
)


@dataclass(frozen=True)
class RotationEntry:
    """One generated file selected for rotation."""

    path: Path
    reason: str
    size: int


def _normalize_keep(keep_paths: Optional[Iterable[str]], root: Path) -> Set[Path]:
    keep: Set[Path] = set()
    if not keep_paths:
        return keep
    for path in keep_paths:
        keep.add((root / path).resolve())
    return keep


def _safe_root_file(path: Path, root: Path) -> bool:
    return path.parent == root and path.is_file()


def _safe_root_dir(path: Path, root: Path) -> bool:
    return path.parent == root and path.is_dir() and not path.is_symlink()


def _mode_for_match(match: re.Match[str]) -> str:
    mode = match.group("mode")
    if mode:
        return mode
    return "legacy"


def _visualization_key(match: re.Match[str]) -> Tuple[str, str, str]:
    return (_mode_for_match(match), match.group("start"), match.group("end"))


def _parse_config_date(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _parse_token_date(value: str) -> Optional[datetime]:
    try:
        fmt = "%y%m%d%H" if len(value) == 8 else "%y%m%d"
        return datetime.strptime(value, fmt)
    except ValueError:
        return None


def _load_pipeline_config(root: Path, config_path: str) -> Dict[str, object]:
    path = (root / config_path).resolve()
    if path.parent != root or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _config_slot_modes(config: Dict[str, object]) -> Set[str]:
    raw_modes = config.get("slot_modes")
    if isinstance(raw_modes, str):
        return {
            mode.strip()
            for mode in re.split(r"[,،;\s]+", raw_modes)
            if mode.strip()
        }
    if isinstance(raw_modes, list):
        return {str(mode).strip() for mode in raw_modes if str(mode).strip()}
    return set()


def _overlaps_config_range(match: re.Match[str], config: Dict[str, object]) -> bool:
    start_date = _parse_config_date(config.get("start_date"))
    end_date = _parse_config_date(config.get("end_date"))
    if not start_date or not end_date:
        return False

    mode = _mode_for_match(match)
    slot_modes = _config_slot_modes(config)
    if slot_modes and mode not in slot_modes:
        return False
    if mode == "legacy":
        return False

    artifact_start = _parse_token_date(match.group("start"))
    artifact_end = _parse_token_date(match.group("end"))
    if not artifact_start or not artifact_end:
        return False

    config_end = end_date + timedelta(days=1) - timedelta(microseconds=1)
    return artifact_start <= config_end and artifact_end >= start_date


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _add_entry(entries: Dict[Path, RotationEntry], path: Path, reason: str) -> None:
    entries[path] = RotationEntry(path=path, reason=reason, size=_size(path))


def _parse_retention(value: Optional[str]) -> Dict[str, int]:
    retention = dict(DEFAULT_VISUALIZATION_RETENTION)
    if not value:
        return retention

    for part in value.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(
                "Retention overrides must use mode=count, e.g. daily=14,weekly=8"
            )
        mode, count = [piece.strip() for piece in part.split("=", 1)]
        if mode not in retention:
            raise ValueError(f"Unknown retention mode: {mode}")
        parsed = int(count)
        if parsed < 0:
            raise ValueError(f"Retention count for {mode} must be >= 0")
        retention[mode] = parsed
    return retention


def plan_project_root_cleanup(
    root: str = ".",
    remove_patterns: Optional[Iterable[str]] = None,
    remove_dirs: Optional[Iterable[str]] = None,
    keep_paths: Optional[Iterable[str]] = None,
    keep_files: Optional[Iterable[str]] = None,
    visualization_retention: Optional[Dict[str, int]] = None,
    summary_keep: int = DEFAULT_SUMMARY_KEEP,
    backup_keep: int = DEFAULT_BACKUP_KEEP,
    protect_pipeline_config: bool = True,
    pipeline_config_path: str = "pipeline_config.json",
    drop_louvain_outputs: bool = True,
) -> List[RotationEntry]:
    """Return generated project-root artifacts that should be removed."""

    root_path = Path(root).resolve()
    patterns = (
        list(remove_patterns)
        if remove_patterns is not None
        else list(DEFAULT_REMOVE_PATTERNS)
    )
    dirs = list(remove_dirs) if remove_dirs is not None else list(DEFAULT_REMOVE_DIRS)
    keep = _normalize_keep(keep_paths, root_path)
    keep_files_list = (
        list(keep_files) if keep_files is not None else list(DEFAULT_KEEP_FILES)
    )
    keep.update(_normalize_keep(keep_files_list, root_path))
    retention = (
        dict(visualization_retention)
        if visualization_retention is not None
        else dict(DEFAULT_VISUALIZATION_RETENTION)
    )

    entries: Dict[Path, RotationEntry] = {}
    visualization_sets: Dict[Tuple[str, str, str], List[Path]] = {}
    summaries: List[Path] = []
    backups: List[Path] = []
    pipeline_config = (
        _load_pipeline_config(root_path, pipeline_config_path)
        if protect_pipeline_config
        else {}
    )

    for path in root_path.iterdir():
        resolved = path.resolve()
        if resolved in keep:
            continue

        if _safe_root_file(path, root_path):
            if drop_louvain_outputs and any(
                fnmatch.fnmatch(path.name, pattern)
                for pattern in LOUVAIN_OUTPUT_PATTERNS
            ):
                _add_entry(entries, path, "Louvain visual output disabled")
                continue

            match = VISUALIZATION_RE.match(path.name)
            if match:
                if pipeline_config and _overlaps_config_range(match, pipeline_config):
                    continue
                visualization_sets.setdefault(_visualization_key(match), []).append(path)
                continue
            if fnmatch.fnmatch(path.name, "community_summary_*.txt"):
                summaries.append(path)
                continue
            if ".backup_" in path.name:
                backups.append(path)
                continue
            for pattern in patterns:
                if fnmatch.fnmatch(path.name, pattern):
                    _add_entry(entries, path, f"matched pattern {pattern}")
                    break

        elif _safe_root_dir(path, root_path) and path.name in dirs:
            entries[path] = RotationEntry(
                path=path,
                reason=f"matched directory {path.name}",
                size=_directory_size(path),
            )

    sets_by_mode: Dict[str, List[Tuple[Tuple[str, str, str], float, List[Path]]]] = {}
    for key, paths in visualization_sets.items():
        mode = key[0]
        newest_mtime = max(_mtime(path) for path in paths)
        sets_by_mode.setdefault(mode, []).append((key, newest_mtime, paths))

    for mode, grouped_sets in sets_by_mode.items():
        keep_count = retention.get(mode, retention.get("unknown", 0))
        grouped_sets.sort(key=lambda item: item[1], reverse=True)
        for key, _, paths in grouped_sets[keep_count:]:
            reason = (
                f"older {mode} visualization set "
                f"{key[1]}_to_{key[2]} beyond latest {keep_count}"
            )
            for path in paths:
                _add_entry(entries, path, reason)

    summaries.sort(key=_mtime, reverse=True)
    for path in summaries[max(summary_keep, 0):]:
        _add_entry(entries, path, f"older community summary beyond latest {summary_keep}")

    backups.sort(key=_mtime, reverse=True)
    for path in backups[max(backup_keep, 0):]:
        _add_entry(entries, path, f"older backup beyond latest {backup_keep}")

    return sorted(entries.values(), key=lambda entry: (entry.reason, entry.path.name))


def _directory_size(path: Path) -> int:
    total = 0
    for base, _, files in os.walk(path):
        for filename in files:
            total += _size(Path(base) / filename)
    return total


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def clean_project_root(
    root: str = ".",
    remove_patterns: Optional[Iterable[str]] = None,
    remove_dirs: Optional[Iterable[str]] = None,
    keep_paths: Optional[Iterable[str]] = None,
    keep_files: Optional[Iterable[str]] = None,
    latest_only_groups: Optional[Iterable[Tuple[str, int]]] = None,
    dry_run: bool = False,
    visualization_retention: Optional[Dict[str, int]] = None,
    summary_keep: int = DEFAULT_SUMMARY_KEEP,
    backup_keep: int = DEFAULT_BACKUP_KEEP,
    protect_pipeline_config: bool = True,
    pipeline_config_path: str = "pipeline_config.json",
    drop_louvain_outputs: bool = True,
) -> List[str]:
    """
    Rotate generated project-root artifacts.

    ``latest_only_groups`` is accepted for backward compatibility. When passed,
    it is converted to broad prefix retention; new callers should use
    ``visualization_retention``.
    """

    if latest_only_groups is not None and visualization_retention is None:
        visualization_retention = dict(DEFAULT_VISUALIZATION_RETENTION)
        for pattern, keep_count in latest_only_groups:
            if pattern.startswith("dashboard_"):
                visualization_retention["legacy"] = keep_count
            elif pattern.startswith(("louvain_", "hybrid_")):
                visualization_retention["unknown"] = keep_count

    plan = plan_project_root_cleanup(
        root=root,
        remove_patterns=remove_patterns,
        remove_dirs=remove_dirs,
        keep_paths=keep_paths,
        keep_files=keep_files,
        visualization_retention=visualization_retention,
        summary_keep=summary_keep,
        backup_keep=backup_keep,
        protect_pipeline_config=protect_pipeline_config,
        pipeline_config_path=pipeline_config_path,
        drop_louvain_outputs=drop_louvain_outputs,
    )

    removed: List[str] = []
    for entry in plan:
        if not dry_run:
            try:
                _remove_path(entry.path)
            except OSError:
                continue
        removed.append(str(entry.path))
    return removed


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rotate generated artifacts from the project root."
    )
    parser.add_argument("--root", default=".", help="Project root to clean.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete files. Without this flag the command is a dry run.",
    )
    parser.add_argument(
        "--retention",
        default=os.getenv("ROOT_ROTATION_RETENTION"),
        help="Comma-separated overrides, e.g. daily=14,weekly=8,legacy=2.",
    )
    parser.add_argument(
        "--summary-keep",
        type=int,
        default=int(os.getenv("ROOT_ROTATION_SUMMARY_KEEP", DEFAULT_SUMMARY_KEEP)),
        help="Number of newest community_summary_*.txt files to keep.",
    )
    parser.add_argument(
        "--backup-keep",
        type=int,
        default=int(os.getenv("ROOT_ROTATION_BACKUP_KEEP", DEFAULT_BACKUP_KEEP)),
        help="Number of newest *.backup_* files to keep.",
    )
    parser.add_argument(
        "--no-config-protect",
        action="store_true",
        help="Do not protect visualization files overlapping pipeline_config.json.",
    )
    parser.add_argument(
        "--keep-louvain-output",
        action="store_true",
        help="Keep old Louvain graph/similarity files instead of deleting them.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final count and reclaimable size.",
    )
    args = parser.parse_args(argv)

    retention = _parse_retention(args.retention)
    plan = plan_project_root_cleanup(
        root=args.root,
        visualization_retention=retention,
        summary_keep=args.summary_keep,
        backup_keep=args.backup_keep,
        protect_pipeline_config=not args.no_config_protect,
        drop_louvain_outputs=not args.keep_louvain_output,
    )
    total_size = sum(entry.size for entry in plan)

    if not args.quiet:
        action = "remove" if args.apply else "would remove"
        for entry in plan:
            print(
                f"[{action}] {entry.path} "
                f"({_format_bytes(entry.size)}) - {entry.reason}"
            )

    if args.apply:
        removed = clean_project_root(
            root=args.root,
            dry_run=False,
            visualization_retention=retention,
            summary_keep=args.summary_keep,
            backup_keep=args.backup_keep,
            protect_pipeline_config=not args.no_config_protect,
            drop_louvain_outputs=not args.keep_louvain_output,
        )
        print(
            f"[done] removed {len(removed)} paths, "
            f"reclaimed about {_format_bytes(total_size)}"
        )
    else:
        print(
            f"[dry-run] {len(plan)} paths, "
            f"about {_format_bytes(total_size)} reclaimable"
        )
        print("[dry-run] re-run with --apply to delete these generated artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
