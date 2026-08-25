"""Run bookkeeping: every result carries the code state and configuration that produced it."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

RUNS = Path(__file__).resolve().parents[3] / "runs"


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# Captured when the process starts, not when it saves. A long run can outlive many commits, and
# a SHA read at save time describes the working tree then rather than the code that produced the
# result, which is the opposite of what a manifest is for.
_STARTED_AT = time.time()
_STARTED_SHA = git_sha()


def manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "git_sha": _STARTED_SHA,
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
    path = out / f"{config_hash(config)}.json"
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
    records = [json.loads(p.read_text()) for p in out.glob("*.json")]
    return sorted(records, key=lambda r: r["manifest"].get("wall_clock", 0.0))
