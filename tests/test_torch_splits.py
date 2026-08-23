"""The batched estimator must agree with the reference implementation term for term."""

import numpy as np
import pytest
import torch

from caliper.estimators.splits import decompose
from caliper.estimators.torch_splits import decompose_batched


@pytest.mark.parametrize("group_size,n_prompts", [(4, 8), (8, 16), (16, 32)])
@pytest.mark.parametrize("blocks", [2, 4])
@pytest.mark.parametrize("corpus", [None, 512])
def test_batched_matches_reference(group_size, n_prompts, blocks, corpus):
    rng = np.random.default_rng(0)
    population, dim = 3, 12
    cells = rng.normal(size=(blocks, 2, population, dim))
    batched = decompose_batched(torch.tensor(cells), n_prompts, group_size, corpus_size=corpus)
    for member in range(population):
        reference = decompose(
            cells[:, :, member, :], n_prompts, group_size, corpus_size=corpus
        )
        for name in ("signal", "tau_b", "tau_w", "tau_w_scaled"):
            assert float(batched[name][member]) == pytest.approx(getattr(reference, name))
