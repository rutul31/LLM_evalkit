"""
evalkit.datasets.truthfulqa — TruthfulQA dataset loader.

Downloads the *generation* split of TruthfulQA from HuggingFace Hub
(first run only — subsequent loads are served from the HF cache) and
maps each row to the canonical evalkit sample format::

    {
        "prompt":    <question>,
        "expected":  <best_answer>,
        "category":  <category>,
        "source":    "truthfulqa",
    }

The returned list is shuffled (deterministically seeded) and trimmed
to ``limit`` rows so you can iterate quickly during development.
"""

import random
from typing import Any

from datasets import load_dataset


def load(limit: int = 50, seed: int = 42) -> list[dict[str, Any]]:
    """
    Load TruthfulQA (generation split) and return *limit* samples.

    Parameters
    ----------
    limit : int
        Maximum number of samples to return.
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    list[dict[str, Any]]
        Shuffled, trimmed list of sample dicts.
    """
    ds = load_dataset("truthful_qa", "generation", split="validation")

    samples: list[dict[str, Any]] = []
    for row in ds:
        samples.append(
            {
                "prompt": row["question"],
                "expected": row["best_answer"],
                "category": row["category"],
                "source": "truthfulqa",
            }
        )

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:limit]
