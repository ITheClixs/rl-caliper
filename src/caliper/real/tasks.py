"""Short exact-match tasks for a pretrained model.

Each family generates a prompt and a single correct answer string; the verifier is exact match on
the first line of the completion. Families differ in difficulty, which is the point: the pass-rate
distribution is what drives the within-prompt noise term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WORDS = [
    "apple", "river", "table", "cloud", "stone", "brave", "chair", "green", "light", "mouse",
    "paper", "sugar", "tiger", "water", "zebra", "amber", "candle", "dragon", "forest", "island",
    "jungle", "kettle", "ladder", "meadow", "needle", "orange", "pencil", "rocket", "silver",
    "temple",
]


@dataclass(frozen=True)
class Family:
    name: str
    build: object

    def sample(self, n: int, rng: np.random.Generator) -> list[dict]:
        return [self.build(rng) for _ in range(n)]


def _sort_digits(rng):
    values = rng.integers(10, 100, size=4)
    return {
        "prompt": (
            "Sort these numbers in increasing order: "
            + ", ".join(str(int(v)) for v in values)
            + "\nAnswer with the sorted numbers separated by single spaces, and nothing else."
        ),
        "answer": " ".join(str(int(v)) for v in sorted(values)),
    }


def _last_letters(rng):
    picks = [WORDS[i] for i in rng.choice(len(WORDS), size=3, replace=False)]
    return {
        "prompt": (
            "Take the last letter of each word and join them: " + ", ".join(picks)
            + "\nAnswer with the three letters only, no spaces and nothing else."
        ),
        "answer": "".join(w[-1] for w in picks),
    }


def _reverse_list(rng):
    values = rng.integers(0, 100, size=5)
    return {
        "prompt": (
            "Reverse this list: " + " ".join(str(int(v)) for v in values)
            + "\nAnswer with the reversed list separated by single spaces, and nothing else."
        ),
        "answer": " ".join(str(int(v)) for v in values[::-1]),
    }


def _count_letter(rng):
    word = WORDS[rng.integers(len(WORDS))]
    letter = word[rng.integers(len(word))]
    return {
        "prompt": (
            f"How many times does the letter '{letter}' appear in the word '{word}'?"
            "\nAnswer with a single number and nothing else."
        ),
        "answer": str(word.count(letter)),
    }


def _add_two(rng):
    a, b = rng.integers(10, 100, size=2)
    return {
        "prompt": (
            f"What is {int(a)} + {int(b)}?\nAnswer with a single number and nothing else."
        ),
        "answer": str(int(a) + int(b)),
    }


def _nth_word(rng):
    picks = [WORDS[i] for i in rng.choice(len(WORDS), size=6, replace=False)]
    k = int(rng.integers(1, 7))
    ordinal = ["first", "second", "third", "fourth", "fifth", "sixth"][k - 1]
    return {
        "prompt": (
            f"What is the {ordinal} word in this list: " + ", ".join(picks)
            + "\nAnswer with that word only and nothing else."
        ),
        "answer": picks[k - 1],
    }


FAMILIES = {
    "sort_digits": Family("sort_digits", _sort_digits),
    "last_letters": Family("last_letters", _last_letters),
    "reverse_list": Family("reverse_list", _reverse_list),
    "count_letter": Family("count_letter", _count_letter),
    "add_two": Family("add_two", _add_two),
    "nth_word": Family("nth_word", _nth_word),
}


def reward(completion: str, answer: str) -> float:
    return float(completion.strip().split("\n")[0].strip().rstrip(".") == answer)
