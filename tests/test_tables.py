"""The table generator picks records by identity, never by load order alone."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import tables  # noqa: E402


def _record(model, prompts, std, checksum=1.0, optimiser="sgd", steps=12, transport=False):
    return {
        "manifest": {
            "config": {
                "model": model,
                "optimiser": optimiser,
                "steps": steps,
                "variance_prompts": prompts,
            }
        },
        "result": {
            "transport": transport,
            "kernel": [0.0] * steps,
            "init_checksum": checksum,
            "predicted_std": std,
        },
    }


def _stds(cheap):
    return {key: value["predicted_std"] for key, value in cheap.items()}


def test_a_variant_from_another_model_is_never_pooled_with_the_chosen_run():
    chosen = _record("small", None, 0.004, transport=True)
    records = [
        _record("small", None, 0.0025, checksum=7.0),
        _record("small", 64, 0.0026, checksum=7.0),
        # a later run of a different model, with its own initialisation, must not take over
        _record("large", None, 0.0007, checksum=9.0, optimiser="adam"),
        _record("large", 32, 0.0207, checksum=9.0, optimiser="adam"),
    ]
    assert _stds(tables.cheap_variants(chosen, records)) == {None: 0.0025, 64: 0.0026}


def test_variants_from_a_different_initialisation_are_dropped():
    chosen = _record("small", None, 0.004, transport=True)
    records = [
        _record("small", None, 0.0050, checksum=3.0),
        _record("small", 64, 0.0026, checksum=7.0),
        _record("small", None, 0.0025, checksum=7.0),
    ]
    assert _stds(tables.cheap_variants(chosen, records)) == {64: 0.0026, None: 0.0025}


def test_a_run_without_a_recorded_initialisation_is_excluded():
    chosen = _record("small", None, 0.004, transport=True)
    records = [_record("small", None, 0.0050, checksum=None)]
    assert tables.cheap_variants(chosen, records) == {}
