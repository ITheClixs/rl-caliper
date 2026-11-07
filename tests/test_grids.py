"""The headline grid is chosen by identity, and ambiguity is an error."""

import pytest

from caliper.analysis.grids import HEADLINE_POOL, REPLICATE_POOL, forecast_grid


def _grid(pool, cells=192, estimator="rloo", tag=""):
    return {
        "manifest": {"config": {"pool": pool, "estimator": estimator}},
        "result": {"cells": [{"tag": tag}] * cells},
    }


def test_the_pool_decides_which_grid_is_quoted_whatever_the_load_order():
    headline, replicate = _grid(HEADLINE_POOL, tag="h"), _grid(REPLICATE_POOL, tag="r")
    for order in ([headline, replicate], [replicate, headline]):
        assert forecast_grid(records=order)["result"]["cells"][0]["tag"] == "h"
        assert forecast_grid(REPLICATE_POOL, records=order)["result"]["cells"][0]["tag"] == "r"


def test_sub_grids_and_other_estimators_are_not_candidates():
    records = [_grid(HEADLINE_POOL, cells=12), _grid(HEADLINE_POOL, estimator="grpo_std")]
    with pytest.raises(LookupError):
        forecast_grid(records=records)


def test_two_grids_with_the_same_identity_are_an_error_rather_than_a_coin_toss():
    with pytest.raises(LookupError):
        forecast_grid(records=[_grid(HEADLINE_POOL), _grid(HEADLINE_POOL)])


def test_the_anchor_is_not_among_the_seeds_it_is_graded_against():
    from caliper.analysis.grids import exclude_anchor

    scores = [9.0, 1.0, 2.0, 3.0, 4.0]
    record = {"manifest": {}, "result": {"cells": [
        {"seed_scores": scores, "measured_std": 0.0, "seeds": 5}]}}
    graded = exclude_anchor(record, n_boot=50)
    cell = graded["result"]["cells"][0]
    assert cell["seeds"] == 4
    assert abs(cell["measured_std"] - 1.2909944487358056) < 1e-12  # sd of 1, 2, 3, 4
    assert record["result"]["cells"][0]["measured_std"] == 0.0  # the stored record is untouched
