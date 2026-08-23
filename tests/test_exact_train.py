"""The enumerable trainer, the pool builder, and the run bookkeeping."""

import numpy as np
import pytest

from caliper.exact.pool import accept_pool, build
from caliper.exact.train import ExactTrainer, RunConfig
from caliper.runtime import io


def test_diversity_controls_the_spread_of_accept_sets():
    rng = np.random.default_rng(0)
    policy, _ = build(3, 3, 8, (0.2, 0.8), 1.0, seed=0)
    probs = policy.sequence_probs()

    def mean_disagreement(diversity):
        pool = accept_pool(probs, 32, (0.2, 0.8), diversity, np.random.default_rng(1))
        return float(np.mean([np.abs(pool[i] - pool[j]).mean()
                              for i in range(8) for j in range(8) if i != j]))

    del rng
    assert mean_disagreement(0.02) < mean_disagreement(1.0)


def test_pool_respects_the_pass_rate_window():
    policy, accepts = build(3, 3, 24, (0.3, 0.6), 0.5, seed=1)
    probs = policy.sequence_probs()
    rates = accepts @ probs
    assert np.all(rates > 0.02)
    assert np.all(rates < 0.98)


def test_training_improves_the_exact_objective():
    policy, accepts = build(3, 3, 32, (0.1, 0.9), 1.0, seed=2)
    trainer = ExactTrainer(policy, accepts, RunConfig(steps=60, drift_target=2e-4))
    before = trainer.objective()
    history = trainer.run(np.random.default_rng(0))
    assert history["final_objective"] > before


def test_the_controller_pulls_drift_towards_its_target():
    policy, accepts = build(3, 3, 32, (0.1, 0.9), 1.0, seed=3)
    target = 1e-4
    trainer = ExactTrainer(
        policy,
        accepts,
        RunConfig(steps=120, drift_target=target, initial_step_size=20.0, measure_every=1),
    )
    history = trainer.run(np.random.default_rng(0))
    settled = np.array(history["drift"][40:])
    assert abs(np.median(np.log(settled / target))) < 1.0


def test_run_records_carry_the_code_state(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "RUNS", tmp_path)
    config = {"alpha": 1, "beta": "two"}
    path = io.save("unit_test", config, {"value": 3})
    assert path.exists()
    loaded = io.load_all("unit_test")
    assert len(loaded) == 1
    manifest = loaded[0]["manifest"]
    assert manifest["config"] == config
    assert manifest["config_hash"] == io.config_hash(config)
    assert loaded[0]["result"]["value"] == 3


def test_config_hash_is_order_independent():
    assert io.config_hash({"a": 1, "b": 2}) == io.config_hash({"b": 2, "a": 1})
    assert io.config_hash({"a": 1}) != io.config_hash({"a": 2})


def test_missing_run_directory_reads_as_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "RUNS", tmp_path)
    assert io.load_all("absent") == []


@pytest.mark.parametrize("group_size", [2, 8])
def test_exact_trainer_rejects_unknown_estimators(group_size):
    policy, accepts = build(3, 3, 8, (0.2, 0.8), 1.0, seed=4)
    with pytest.raises(KeyError):
        ExactTrainer(policy, accepts, RunConfig(estimator="nope", group_size=group_size))


def test_runs_are_returned_oldest_first(tmp_path, monkeypatch):
    """The filename is a configuration hash, so ordering has to come from the manifest."""
    monkeypatch.setattr(io, "RUNS", tmp_path)
    io.save("ordering", {"which": "first"}, {"n": 1})
    io.save("ordering", {"which": "second"}, {"n": 2})
    clocks = [r["manifest"]["wall_clock"] for r in io.load_all("ordering")]
    assert clocks == sorted(clocks)
    assert io.load_all("ordering")[-1]["manifest"]["config"]["which"] == "second"
