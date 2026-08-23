"""The exact-match families and their verifier."""

import numpy as np
import pytest

from caliper.real.tasks import FAMILIES, reward


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_family_produces_a_prompt_and_a_reachable_answer(name):
    rng = np.random.default_rng(0)
    items = FAMILIES[name].sample(8, rng)
    assert len(items) == 8
    for item in items:
        assert item["prompt"].strip()
        assert item["answer"].strip()
        assert reward(item["answer"], item["answer"]) == 1.0


def test_verifier_reads_only_the_first_line_and_ignores_trailing_punctuation():
    assert reward("42\nsome chatter", "42") == 1.0
    assert reward("  42.  ", "42") == 1.0
    assert reward("43", "42") == 0.0
    assert reward("", "42") == 0.0


def test_families_are_deterministic_given_a_seed():
    a = FAMILIES["count_letter"].sample(4, np.random.default_rng(7))
    b = FAMILIES["count_letter"].sample(4, np.random.default_rng(7))
    assert a == b


def test_answers_are_consistent_with_the_prompt():
    rng = np.random.default_rng(1)
    item = FAMILIES["add_two"].sample(1, rng)[0]
    numbers = [int(t) for t in item["prompt"].split("?")[0].replace("What is", "").split("+")]
    assert int(item["answer"]) == sum(numbers)

    item = FAMILIES["sort_digits"].sample(1, rng)[0]
    listed = [int(t) for t in item["prompt"].split(":")[1].split("\n")[0].split(",")]
    assert [int(t) for t in item["answer"].split()] == sorted(listed)
