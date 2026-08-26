"""The diagonal sensitivity of one Adam update to the gradient it was handed."""

import numpy as np

from caliper.real.adjoint import adam_sensitivity


def _mlx_update(m, v, grad, lr, b1, b2, eps):
    """MLX's Adam, without bias correction, as a plain numpy reference."""
    m_new = b1 * m + (1 - b1) * grad
    v_new = b2 * v + (1 - b2) * grad**2
    return -lr * m_new / (np.sqrt(v_new) + eps)


def test_the_sensitivity_matches_a_numerical_derivative():
    """`B = d theta' / d g_hat` is diagonal, so a central difference checks it coordinate-wise."""
    rng = np.random.default_rng(0)
    dim = 12
    m = rng.normal(scale=1e-3, size=dim)
    v = rng.uniform(1e-8, 1e-5, size=dim)
    grad = rng.normal(scale=1e-3, size=dim)
    lr, b1, b2, eps = 3e-4, 0.9, 0.999, 1e-8

    analytic = adam_sensitivity(m, v, grad, lr, b1, b2, eps)

    step = 1e-9
    numeric = np.zeros(dim)
    for i in range(dim):
        plus, minus = grad.copy(), grad.copy()
        plus[i] += step
        minus[i] -= step
        numeric[i] = (
            _mlx_update(m, v, plus, lr, b1, b2, eps)[i]
            - _mlx_update(m, v, minus, lr, b1, b2, eps)[i]
        ) / (2 * step)
    np.testing.assert_allclose(analytic, np.abs(numeric), rtol=2e-4, atol=0)


def test_a_larger_second_moment_damps_the_sensitivity():
    """Adam's normalisation is what makes the injected term state-dependent."""
    dim = 6
    m = np.full(dim, 1e-3)
    grad = np.full(dim, 1e-3)
    small = adam_sensitivity(m, np.full(dim, 1e-9), grad, 3e-4, 0.9, 0.999, 1e-8)
    large = adam_sensitivity(m, np.full(dim, 1e-4), grad, 3e-4, 0.9, 0.999, 1e-8)
    assert np.all(large < small)


def test_the_sensitivity_survives_a_dead_coordinate_in_single_precision():
    """A coordinate with no gradient and no accumulated second moment must give a number.

    MLX hands the moments back in float32. There `max(root, 1e-30) * (root + eps)**2` underflows
    to exactly zero when both the gradient and the state are zero, and the ratio becomes 0/0.
    The limit is zero: the term through `v` carries a factor of the gradient itself.
    """
    dim = 5
    z32 = np.zeros(dim, dtype=np.float32)
    out = adam_sensitivity(z32, z32, z32, 2e-5)
    assert np.all(np.isfinite(out)), out
    # only the direct term survives, bounded by epsilon
    np.testing.assert_allclose(out, 2e-5 * 0.1 / 1e-8, rtol=1e-6)


def test_mixed_precision_agrees_with_double():
    rng = np.random.default_rng(1)
    dim = 32
    m = rng.normal(scale=1e-4, size=dim)
    v = np.abs(rng.normal(scale=1e-7, size=dim))
    g = rng.normal(scale=1e-4, size=dim)
    v[:8] = 0.0
    g[:4] = 0.0
    single = adam_sensitivity(m.astype(np.float32), v.astype(np.float32),
                              g.astype(np.float32), 2e-5)
    double = adam_sensitivity(m, v, g, 2e-5)
    assert np.all(np.isfinite(single))
    np.testing.assert_allclose(single, double, rtol=1e-3)
