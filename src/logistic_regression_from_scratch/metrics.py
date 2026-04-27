"""Classification metrics."""

from __future__ import annotations


def accuracy_score(y_true: list[int], y_pred: list[int]) -> float:
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def confusion_matrix(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    return {
        "tp": sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred)),
        "tn": sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred)),
        "fp": sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred)),
        "fn": sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred)),
    }


def precision_recall_f1(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if cm["tp"] + cm["fp"] else 0.0
    recall = cm["tp"] / (cm["tp"] + cm["fn"]) if cm["tp"] + cm["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
