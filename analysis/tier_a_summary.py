"""Summary statistics for the Tier A results note and the paper's Table 1."""

from __future__ import annotations

import numpy as np

from caliper.runtime import io

BINS = [(1.00, 1.15), (1.15, 1.35), (1.35, 2.00), (2.00, 8.00)]


def latest(name: str, required_key: str) -> list[dict]:
    runs = [r for r in io.load_all(name) if required_key in r["result"]["cells"][0]]
    if not runs:
        raise SystemExit(f"no run of {name} carries {required_key!r}")
    return runs[-1]["result"]["cells"]


def main() -> None:
    cells = latest("a4_law", "r2")
    r2 = np.array([c["r2"] for c in cells])
    spread = np.array([c["predicted_range"] for c in cells])
    rho = np.array([c["spearman"] for c in cells])

    print("| predicted spread | cells | median R^2 | median Spearman |")
    print("|---|---|---|---|")
    for lo, hi in BINS:
        mask = (spread >= lo) & (spread < hi)
        if mask.sum():
            print(
                f"| {lo:.2f} - {hi:.2f} | {mask.sum()} | "
                f"{np.median(r2[mask]):.3f} | {np.median(rho[mask]):+.3f} |"
            )

    seen = {}
    for c in cells:
        seen[(c["band"], c["diversity"], c["replicate"])] = (
            c["exact_g_star"],
            c["measured_g_star"],
        )
    exact = np.array([v[0] for v in seen.values()])
    measured = np.array([v[1] for v in seen.values()])
    rel = np.abs(measured - exact) / exact
    print(
        f"\npools: {len(exact)}   G* covered {exact.min():.2f}..{exact.max():.2f}   "
        f"median relative error {np.median(rel) * 100:.1f}%   worst {rel.max() * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
