"""Which stored forecast grid a number is quoted from.

Several sweeps write to the `p2_metric_forecast` store: the headline grid, a replicate of it on
a smaller prompt pool, per-estimator sub-grids and smoke tests. Choosing among them by size or by
load order lets one silently replace another. That happened once: two 192-setting grids, at
pools of 24 and 16, were re-run in the same commit, a `max` broke the tie by load order, and the
paper's headline moved from the first to the second without anyone deciding it should. The
selection is therefore by identity, and an identity that matches anything but one record is an
error.
"""

from __future__ import annotations

from typing import Any

from caliper.runtime import io

STORE = "p2_metric_forecast"

# The pool size `experiments/p2_metric_forecast/run.py` runs by default, the grid the sizing
# benchmark is graded on, and the one the headline has quoted since it was first reported.
HEADLINE_POOL = 24
# The same grid on a pool of sixteen prompts, reported as a replicate.
REPLICATE_POOL = 16
MINIMUM_CELLS = 100


def forecast_grid(
    pool: int = HEADLINE_POOL, estimator: str = "rloo", records: list[dict] | None = None
) -> dict[str, Any]:
    """The one full prospective grid run at `pool` with `estimator`."""
    candidates = records if records is not None else io.load_all(STORE)
    hits = [
        record
        for record in candidates
        if len(record["result"].get("cells", [])) >= MINIMUM_CELLS
        and record["manifest"]["config"].get("estimator", "rloo") == estimator
        and record["manifest"]["config"].get("pool") == pool
    ]
    if len(hits) != 1:
        raise LookupError(
            f"expected exactly one {estimator} forecast grid at pool {pool} with "
            f"{MINIMUM_CELLS}+ settings, found {len(hits)}"
        )
    return hits[0]
