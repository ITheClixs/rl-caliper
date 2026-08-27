"""Recover the half-sample forecasts a run printed but did not store.

`split_variance` was added to the record schema after the seven-billion runs were made, so for
those two the only place the numbers exist is the run's own stdout. This reads them out of the
kept log and writes them beside the run, once, so that every figure in the paper is generated
from a file rather than typed in. Runs made after the schema change carry the field themselves
and are skipped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(
    r"two half-sample estimates of the same forecast:\s*([0-9.]+)\s*and\s*([0-9.]+)"
)


def main() -> None:
    for log in sorted((ROOT / "runs" / "logs").glob("*.log")):
        found = PATTERN.search(log.read_text())
        if not found:
            continue
        out = ROOT / "runs" / "logs" / f"{log.stem}.split.json"
        out.write_text(
            json.dumps(
                {
                    "source": f"runs/logs/{log.name}",
                    "half_sample_std": [float(found.group(1)), float(found.group(2))],
                },
                indent=2,
            )
        )
        print(f"wrote {out.relative_to(ROOT)}: {found.group(1)}, {found.group(2)}")


if __name__ == "__main__":
    main()
