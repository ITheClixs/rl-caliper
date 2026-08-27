"""A curvature residual one run can compute about itself, before any replica exists."""

import numpy as np

from caliper.exact.curvature import curvature_residual, noise_radius
from caliper.exact.pool import build
from caliper.objectives.advantages import weight_table


def test_the_residual_is_zero_on_a_linear_field():
    """A field with no second derivative must give exactly zero, whatever the radius.

    The residual is a symmetric second difference of the mean update along one direction,
    normalised by the first difference. On an affine field the second difference cancels.
    """

    def affine(delta):
        matrix = np.arange(delta.size * delta.size).reshape(delta.size, -1) * 1e-3
        return matrix @ delta + 0.5

    direction = np.zeros(6)
    direction[0] = 1.0
    for radius in (1e-3, 1e-2, 1e-1):
        assert curvature_residual(affine, direction, radius) < 1e-9


def test_the_residual_grows_with_curvature():
    """A quadratic field bends, and the residual should see it grow with the probe radius."""

    def bent(delta):
        return delta + 3.0 * delta**2

    direction = np.zeros(4)
    direction[0] = 1.0
    small = curvature_residual(bent, direction, 1e-3)
    large = curvature_residual(bent, direction, 1e-1)
    assert large > small
    assert large > 0.05


def test_the_radius_is_the_size_of_one_update_of_noise():
    """The probe radius must be the perturbation a single update actually injects.

    Probing at an arbitrary radius answers a question no run is asking. The scale that matters
    is the step times the spread of the batch gradient about its mean.
    """
    policy, accepts = build(3, 2, 12, (0.05, 0.95), 1.0, seed=0)
    weights = weight_table("rloo", 8)
    wide = noise_radius(policy, accepts, weights, n_prompts=4, step_size=0.5)
    narrow = noise_radius(policy, accepts, weights, n_prompts=64, step_size=0.5)
    assert wide > narrow, "more prompts is less noise, so a smaller radius"
    halved = noise_radius(policy, accepts, weights, n_prompts=4, step_size=0.25)
    np.testing.assert_allclose(halved, wide / 2, rtol=1e-9)
