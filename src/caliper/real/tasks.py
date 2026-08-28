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


def _count_in_string(rng):
    """Counting a letter in a random string, which has no memorised answer to fall back on."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    target = letters[rng.integers(6)]
    body = "".join(letters[i] for i in rng.integers(0, 6, size=14))
    return {
        "prompt": (
            f"How many times does the letter '{target}' appear in this string: {body}"
            "\nAnswer with a single number and nothing else."
        ),
        "answer": str(body.count(target)),
    }


def _add_three(rng):
    a, b, c = rng.integers(100, 1000, size=3)
    return {
        "prompt": (
            f"What is {int(a)} + {int(b)} - {int(c)}?"
            "\nAnswer with a single number and nothing else."
        ),
        "answer": str(int(a) + int(b) - int(c)),
    }


def _sort_words(rng):
    picks = [WORDS[i] for i in rng.choice(len(WORDS), size=5, replace=False)]
    return {
        "prompt": (
            "Sort these words alphabetically: " + ", ".join(picks)
            + "\nAnswer with the sorted words separated by single spaces, and nothing else."
        ),
        "answer": " ".join(sorted(picks)),
    }


FAMILIES = {
    "sort_digits": Family("sort_digits", _sort_digits),
    "last_letters": Family("last_letters", _last_letters),
    "reverse_list": Family("reverse_list", _reverse_list),
    "count_letter": Family("count_letter", _count_letter),
    "add_two": Family("add_two", _add_two),
    "nth_word": Family("nth_word", _nth_word),
    "count_in_string": Family("count_in_string", _count_in_string),
    "add_three": Family("add_three", _add_three),
    "sort_words": Family("sort_words", _sort_words),
}


def reward(completion: str, answer: str) -> float:
    return float(completion.strip().split("\n")[0].strip().rstrip(".") == answer)


# ---------------------------------------------------------------------------
# Parametric rungs. A fixed task suite only carries stochastic support for the
# models it happens to suit: at seven billion `add_two` is solved and
# `last_letters` is impossible, and neither can contribute noise. These take a
# difficulty knob so a corpus with a chosen live-group probability can be built
# for whatever policy is in front of us, rather than hoping one exists.


def _sort_words_n(words: int):
    def make(rng):
        picks = [WORDS[i] for i in rng.choice(len(WORDS), size=words, replace=False)]
        return {
            "prompt": (
                "Sort these words alphabetically: " + ", ".join(picks)
                + "\nAnswer with the sorted words separated by single spaces, and nothing else."
            ),
            "answer": " ".join(sorted(picks)),
        }
    return make


def _sort_digits_n(count: int, high: int):
    def make(rng):
        values = rng.integers(10, high, size=count)
        return {
            "prompt": (
                "Sort these numbers in increasing order: "
                + ", ".join(str(int(v)) for v in values)
                + "\nAnswer with the sorted numbers separated by single spaces, and nothing else."
            ),
            "answer": " ".join(str(int(v)) for v in sorted(values)),
        }
    return make


def _add_chain(terms: int, high: int):
    def make(rng):
        values = [int(v) for v in rng.integers(10, high, size=terms)]
        signs = [1] + [int(s) for s in rng.choice([-1, 1], size=terms - 1)]
        parts = [str(values[0])]
        total = values[0]
        for value, sign in zip(values[1:], signs[1:], strict=True):
            parts.append(("+ " if sign > 0 else "- ") + str(value))
            total += sign * value
        return {
            "prompt": (
                "What is " + " ".join(parts) + "?"
                "\nAnswer with a single number and nothing else."
            ),
            "answer": str(total),
        }
    return make


#: rungs ordered from easy to hard, so a survey can find the band a policy sits in
LADDER = {
    f"sort_words_{n}": Family(f"sort_words_{n}", _sort_words_n(n)) for n in (3, 4, 5, 6, 7, 8)
}
LADDER |= {
    f"sort_digits_{n}": Family(f"sort_digits_{n}", _sort_digits_n(n, 100))
    for n in (4, 5, 6, 7, 8)
}
LADDER |= {
    f"add_chain_{n}": Family(f"add_chain_{n}", _add_chain(n, 1000)) for n in (3, 4, 5, 6)
}
FAMILIES |= LADDER
