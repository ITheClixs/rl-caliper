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
