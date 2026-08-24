"""The pieces of the backward pass that can be checked without training anything."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from caliper.real.adjoint import flatten, unflatten_like  # noqa: E402


@pytest.fixture
def tree():
    return {
        "layers": [
            {"lora_a": np.arange(6, dtype=np.float32).reshape(2, 3)},
            {"lora_b": np.arange(4, dtype=np.float32).reshape(4, 1)},
        ]
    }


def test_flatten_then_unflatten_is_the_identity(tree):
    rebuilt = unflatten_like(flatten(tree), tree)
    np.testing.assert_allclose(flatten(rebuilt), flatten(tree))


def test_unflatten_preserves_every_shape(tree):
    rebuilt = unflatten_like(flatten(tree) + 1.0, tree)
    assert np.asarray(rebuilt["layers"][0]["lora_a"]).shape == (2, 3)
    assert np.asarray(rebuilt["layers"][1]["lora_b"]).shape == (4, 1)


def test_a_mismatched_vector_is_rejected(tree):
    with pytest.raises(ValueError):
        unflatten_like(np.zeros(flatten(tree).size + 1), tree)


def test_flatten_orders_consistently(tree):
    """Two flattenings of the same tree must line up, or the adjoint mixes coordinates."""
    first = flatten(tree)
    second = flatten(unflatten_like(first, tree))
    np.testing.assert_array_equal(first, second)
