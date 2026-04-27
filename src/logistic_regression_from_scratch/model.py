"""Gradient-descent logistic regression without external ML libraries."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1 + exp_z)


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.1
    max_iter: int = 2000
    tolerance: float = 1e-8
    l2: float = 0.0
    weights_: list[float] = field(default_factory=list)
    losses_: list[float] = field(default_factory=list)

    def fit(self, X: list[list[float]], y: list[int]) -> "LogisticRegressionGD":
        if not X:
            raise ValueError("X must contain at least one row")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        n_features = len(X[0])
        self.weights_ = [0.0 for _ in range(n_features + 1)]
        self.losses_ = []
        last_loss = float("inf")
        for _ in range(self.max_iter):
            gradients = [0.0 for _ in self.weights_]
            for row, label in zip(X, y):
                row_i = [1.0, *row]
                pred = sigmoid(dot(self.weights_, row_i))
                error = pred - label
                for idx, value in enumerate(row_i):
                    gradients[idx] += error * value
            for idx in range(len(gradients)):
                gradients[idx] /= len(X)
                if idx > 0:
                    gradients[idx] += self.l2 * self.weights_[idx]
                self.weights_[idx] -= self.learning_rate * gradients[idx]
            loss = self.loss(X, y)
            self.losses_.append(loss)
            if abs(last_loss - loss) < self.tolerance:
                break
            last_loss = loss
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        self._check_fit()
        return [sigmoid(dot(self.weights_, [1.0, *row])) for row in X]

    def predict(self, X: list[list[float]], threshold: float = 0.5) -> list[int]:
        return [1 if prob >= threshold else 0 for prob in self.predict_proba(X)]

    def loss(self, X: list[list[float]], y: list[int]) -> float:
        total = 0.0
        for row, label in zip(X, y):
            prob = min(max(sigmoid(dot(self.weights_, [1.0, *row])), 1e-12), 1 - 1e-12)
            total += -(label * math.log(prob) + (1 - label) * math.log(1 - prob))
        penalty = 0.5 * self.l2 * sum(weight * weight for weight in self.weights_[1:])
        return total / len(X) + penalty

    def _check_fit(self) -> None:
        if not self.weights_:
            raise RuntimeError("Model must be fit before prediction")


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
