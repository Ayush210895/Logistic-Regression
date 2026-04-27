#!/usr/bin/env python3
"""Run a reproducible logistic regression experiment."""

from __future__ import annotations

from pathlib import Path

from logistic_regression_from_scratch import (
    LogisticRegressionGD,
    accuracy_score,
    generate_classification_data,
    precision_recall_f1,
    train_test_split,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset = generate_classification_data(n_samples=1000, seed=42)
    train, test = train_test_split(dataset, test_size=0.3, seed=42)
    model = LogisticRegressionGD(learning_rate=0.2, max_iter=2500, tolerance=1e-9, l2=0.001).fit(
        train.features, train.labels
    )
    predictions = model.predict(test.features)
    metrics = {
        "samples": dataset.rows,
        "train_rows": train.rows,
        "test_rows": test.rows,
        "iterations": len(model.losses_),
        "final_loss": model.losses_[-1],
        "accuracy": accuracy_score(test.labels, predictions),
        **precision_recall_f1(test.labels, predictions),
    }
    output = ROOT / "outputs/experiment_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(metrics, model.weights_), encoding="utf-8")
    print(output)


def render_report(metrics: dict[str, float | int], weights: list[float]) -> str:
    lines = ["# Logistic Regression Experiment", "", "| Metric | Value |", "| --- | ---: |"]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |")
    lines.extend(["", "## Learned Weights", "", "| Term | Weight |", "| --- | ---: |"])
    for idx, weight in enumerate(weights):
        term = "intercept" if idx == 0 else f"x{idx}"
        lines.append(f"| {term} | {weight:.4f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
