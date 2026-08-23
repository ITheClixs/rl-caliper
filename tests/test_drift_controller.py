"""The controller has to be unbiased in log space and bounded per step."""

import numpy as np

from caliper.control.drift import DriftController


def test_one_step_correction_when_the_drift_is_clean():
    """Drift is quadratic in the step, so the exponent 1/2 lands on target in a single update."""
    controller = DriftController(target=1e-3, step_size=0.1, max_ratio=10.0)
    realised = 4e-3  # a step twice as large as it should be
    controller.update(realised)
    assert controller.step_size == 0.05


def test_step_ratio_is_bounded():
    controller = DriftController(target=1.0, step_size=1.0, max_ratio=1.5)
    controller.update(1e-9)
    assert controller.step_size == 1.5
    controller.update(1e9)
    assert controller.step_size == 1.0


def test_non_positive_drift_leaves_the_step_alone():
    controller = DriftController(target=1e-3, step_size=0.2)
    assert controller.update(0.0) == 0.2


def test_tracks_a_noisy_quadratic_process():
    rng = np.random.default_rng(0)
    controller = DriftController(target=1e-3, step_size=1e-4)
    coefficient = 50.0
    history = []
    for _ in range(400):
        drift = 0.5 * coefficient * controller.step_size**2 * np.exp(rng.normal(0, 0.6))
        history.append(drift)
        controller.update(drift)
    settled = np.array(history[100:])
    assert abs(np.mean(np.log(settled / 1e-3))) < 0.2
