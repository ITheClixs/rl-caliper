"""Run bookkeeping: every result carries the code state and configuration that produced it."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_dirty() -> bool | None:
    """Whether tracked files differed from HEAD, so a SHA alone does not name the code that ran."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def portable(value: Any) -> Any:
    """A configuration value with absolute paths made repository-relative.

    A path outside the repository names a directory on one machine and nothing anywhere else,
    so it is recorded by its last component only.
    """
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [portable(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        path = Path(value)
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return f"<outside the repository>/{path.name}"
    return value


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# Captured when the process starts, not when it saves. A long run can outlive many commits, and
# a SHA read at save time describes the working tree then rather than the code that produced the
# result, which is the opposite of what a manifest is for.
_STARTED_AT = time.time()
_STARTED_SHA = git_sha()
_STARTED_DIRTY = git_dirty()


def manifest(config: dict[str, Any]) -> dict[str, Any]:
    config = portable(config)
    return {
        "git_sha": _STARTED_SHA,
        "git_dirty": _STARTED_DIRTY,
        "config_hash": config_hash(config),
        "config": config,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_at": _STARTED_AT,
        "wall_clock": time.time(),
    }


def save(name: str, config: dict[str, Any], payload: dict[str, Any]) -> Path:
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{config_hash(portable(config))}.json"
    path.write_text(json.dumps({"manifest": manifest(config), "result": payload}, indent=2))
    return path


def load_all(name: str) -> list[dict[str, Any]]:
    """Every stored run of `name`, oldest first.

    Ordered by the wall clock recorded in the manifest rather than by filename, which is a
    configuration hash and carries no ordering.
    """
    out = RUNS / name
    if not out.exists():
        return []
    records = []
    for path in out.glob("*.json"):
        blob = json.loads(path.read_text())
        # a run store can also hold artefacts an experiment wrote for itself, such as a forecast
        # frozen before its validation seeds. Those are not records and have no manifest.
        if isinstance(blob, dict) and "manifest" in blob and "result" in blob:
            records.append(blob)
    return sorted(records, key=lambda r: r["manifest"].get("wall_clock", 0.0))
