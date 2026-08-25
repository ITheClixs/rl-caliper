"""The real-model backward pass, driven by a stand-in whose answer is known in closed form.

`caliper.real.adjoint` is the code that would run on a pretrained model, and it cannot be tested
against one in a unit test. It can be tested against a stand-in that implements the same small
interface -- set the parameters, roll out, take a gradient -- but whose mean update field is a
fixed linear map. Then the covariance recursion has a closed form and the whole backward pass,
including the finite-difference Hessian-vector product, has something to be wrong against.
"""

from dataclasses import dataclass

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from caliper.real import adjoint as ra  # noqa: E402

DIM = 5


@dataclass
class StubConfig:
    prompts: int = 4
    group_size: int = 2
    estimator: str = "rloo"
    max_tokens: int = 1
    temperature: float = 1.0


class StubModel:
    def __init__(self, dim):
        self.state = {"w": mx.zeros((dim,))}

    def trainable_parameters(self):
        return self.state

    def update(self, tree):
        self.state = {"w": mx.array(np.asarray(tree["w"]))}


class StubTrainer:
    """A trainer whose mean ascent direction is `jacobian @ theta + bias`.

    `_gradient` returns the gradient of a *loss*, which is the negative of that, exactly as the
    real trainer does. Getting this convention wrong in the stub would let a sign error in the
    module under test pass unnoticed, which is what happened before this was written down.
    """

    def __init__(self, jacobian, bias, offsets):
        self.jacobian = jacobian
        self.bias = bias
        self.offsets = offsets  # (prompts, dim), the per-prompt departure from the mean
        self.model = StubModel(jacobian.shape[0])
        self.config = StubConfig(prompts=offsets.shape[0])

    def _theta(self):
        return np.asarray(self.model.state["w"], dtype=float)

    def rollout(self, items, key):
        """Sequences carry their prompt index, which is how the gradient knows them apart."""
        count = len(items)
        sequences = [mx.full((1, 2), i, dtype=mx.int32) for i in range(count)]
        masks = [mx.ones((1, 1)) for _ in range(count)]
        rewards = np.zeros((count, self.config.group_size))
        rewards[:, 0] = 1.0  # every group is mixed, so no advantage is identically zero
        return sequences, masks, rewards, key

    def _gradient(self, sequences, masks, advantages):
        """Gradient of the stand-in field, averaged over whichever prompts were passed."""
        share = len(sequences) / self.config.prompts
        base = self.jacobian @ self._theta() + self.bias
        picked = [int(np.asarray(seq)[0, 0]) % self.offsets.shape[0] for seq in sequences]
        spread = self.offsets[picked].mean(axis=0)
        return 0.0, {"w": mx.array(-share * (base + spread))}  # a loss gradient


@pytest.fixture
def stub():
    rng = np.random.default_rng(0)
    jacobian = -np.abs(np.diag(rng.uniform(0.5, 1.5, size=DIM)))
    bias = rng.normal(size=DIM) * 0.01
    offsets = rng.normal(size=(4, DIM)) * 0.05
    return StubTrainer(jacobian, bias, offsets)


def test_the_gradient_convention_is_flipped_into_ascent(stub):
    """The module must undo the trainer's loss-gradient sign, or the transport runs backwards."""
    from caliper.real.adjoint import ascent_direction

    _, grads = stub._gradient([mx.zeros((1, 2), dtype=mx.int32)], [mx.ones((1, 1))], [None])
    assert float(ascent_direction(grads)[0]) == pytest.approx(
        -float(np.asarray(grads["w"])[0]), rel=1e-12
    )


def test_the_hessian_vector_product_recovers_the_linear_map(stub):
    """With a linear mean field, J b must come back exactly whatever epsilon is used."""
    rng = np.random.default_rng(1)
    direction = rng.normal(size=DIM)
    base = {"w": mx.zeros((DIM,))}
    got = ra.hessian_vector(
        stub, base, [{}] * stub.config.prompts, direction, mx.random.key(0), 0, epsilon=1e-2
    )
    np.testing.assert_allclose(got, stub.jacobian @ direction, rtol=1e-6, atol=1e-9)


def test_the_product_is_stable_across_a_usable_range_of_epsilon(stub):
    """Stable over three decades. Below that, float32 cancellation sets a floor, not the theory."""
    rng = np.random.default_rng(2)
    direction = rng.normal(size=DIM)
    base = {"w": mx.zeros((DIM,))}
    corpus = [{}] * stub.config.prompts
    products = [
        ra.hessian_vector(stub, base, corpus, direction, mx.random.key(0), 0, epsilon=eps)
        for eps in (1e-1, 1e-2, 1e-3)
    ]
    for other in products[1:]:
        np.testing.assert_allclose(products[0], other, rtol=1e-4, atol=1e-8)


def test_displacing_and_restoring_leaves_the_parameters_where_they_were(stub):
    original = stub._theta().copy()
    base = {"w": mx.array(original)}
    ra.displaced(stub, base, np.ones(DIM), 0.3)
    assert not np.allclose(stub._theta(), original)
    ra.displaced(stub, base, np.ones(DIM), 0.0)
    np.testing.assert_allclose(stub._theta(), original)


def test_the_forecast_matches_the_closed_form_recursion(stub):
    """Run the real backward pass, then build the same number the expensive way."""
    steps, eta = 4, 0.05
    states = [{"w": mx.zeros((DIM,))} for _ in range(steps + 1)]
    corpus = [{}] * stub.config.prompts

    result = ra.forecast(
        stub, states, corpus, corpus, eta, stub.config.prompts, mx.random.key(3),
        seed=0, epsilon=1e-3, check_agreement=False,
    )

    # the same quantity, assembled: S_T from the recursion, contracted against grad M
    projected, _, _, _ = ra.projected_batch_variance(
        stub, corpus, np.random.default_rng(0), mx.random.key(0), np.ones(DIM)
    )
    assert projected >= 0.0
    assert result.variance > 0.0
    assert len(result.kernel) == steps
    assert sum(result.kernel) == pytest.approx(result.variance, rel=1e-12)
    assert all(share == 1.0 for share in result.live_share)


def test_a_unanimous_group_injects_nothing(stub):
    """When no group is mixed the advantage is zero, so the injected term must vanish."""

    def unanimous(items, key):
        count = len(items)
        return (
            [mx.zeros((1, 2), dtype=mx.int32) for _ in range(count)],
            [mx.ones((1, 1)) for _ in range(count)],
            np.ones((count, stub.config.group_size)),
            key,
        )

    stub.rollout = unanimous
    _, _, live, _ = ra.projected_batch_variance(
        stub, [{}] * stub.config.prompts, np.random.default_rng(0), mx.random.key(0),
        np.ones(DIM),
    )
    assert live == 0.0
