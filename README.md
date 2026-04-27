# Logistic Regression From Scratch

Binary logistic regression implemented from first principles with gradient descent. The original notebook, `Lab2_LogisticRegression.ipynb`, is preserved; this repo now adds a reusable package, reproducible experiment, tests, and CI.

## Highlights

- Implements sigmoid, binary cross-entropy, gradients, L2 regularization, and prediction manually.
- Uses gradient descent without scikit-learn.
- Includes a reproducible synthetic classification experiment.
- Reports accuracy, precision, recall, F1, loss, and learned weights.

## Current Result

| Metric | Value |
| --- | ---: |
| Samples | 1,000 |
| Train rows | 700 |
| Test rows | 300 |
| Accuracy | 0.9100 |
| Precision | 0.8994 |
| Recall | 0.9471 |
| F1 | 0.9226 |

Generated report: `outputs/experiment_report.md`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/run_experiment.py
python -m unittest discover -s tests
```

## Repository Structure

```text
.
|-- Lab2_LogisticRegression.ipynb
|-- src/logistic_regression_from_scratch/
|-- scripts/run_experiment.py
|-- tests/
|-- outputs/experiment_report.md
`-- .github/workflows/tests.yml
```

## Portfolio Fit

This project is part of a broader "ML from scratch" portfolio thread. It shows the mechanics of probabilistic classification and optimization clearly, without hiding the learning loop behind a library call.
