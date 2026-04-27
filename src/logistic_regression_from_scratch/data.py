"""Data generation and splitting helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    features: list[list[float]]
    labels: list[int]
    feature_names: list[str]

    @property
    def rows(self) -> int:
        return len(self.labels)


def generate_classification_data(n_samples: int = 1000, seed: int = 42) -> Dataset:
    """Generate a reproducible 2D binary classification problem."""

    rng = random.Random(seed)
    features: list[list[float]] = []
    labels: list[int] = []
    for _ in range(n_samples):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)
        margin = 1.8 * x1 - 1.2 * x2 + 0.35 + rng.gauss(0, 0.65)
        labels.append(1 if margin >= 0 else 0)
        features.append([x1, x2])
    return Dataset(features, labels, ["x1", "x2"])


def train_test_split(dataset: Dataset, test_size: float = 0.3, seed: int = 42) -> tuple[Dataset, Dataset]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    indices = list(range(dataset.rows))
    random.Random(seed).shuffle(indices)
    test_count = max(1, int(dataset.rows * test_size))
    test_indices = set(indices[:test_count])
    train_x: list[list[float]] = []
    train_y: list[int] = []
    test_x: list[list[float]] = []
    test_y: list[int] = []
    for idx, row in enumerate(dataset.features):
        if idx in test_indices:
            test_x.append(row)
            test_y.append(dataset.labels[idx])
        else:
            train_x.append(row)
            train_y.append(dataset.labels[idx])
    return Dataset(train_x, train_y, dataset.feature_names.copy()), Dataset(test_x, test_y, dataset.feature_names.copy())
