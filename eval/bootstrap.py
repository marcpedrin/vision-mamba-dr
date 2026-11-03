"""
Bootstrap 95% confidence intervals for QWK on each test set.

    Resample the test images with replacement 1000 times (seed 42) and recompute QWK
    with eval/metrics.py each time. The interval is the 2.5th to 97.5th percentile.

    QWK only depends on the (true, predicted) pairs, so they are rebuilt from the
    confusion matrices saved in results.json. A per-image predictions CSV
    (columns true, pred) can be used instead with --predictions.

    python -m eval.bootstrap
    python -m eval.bootstrap --results results/results.json
"""

import argparse
import json

import numpy as np
import pandas as pd

from .metrics import quadratic_weighted_kappa


def pairs_from_confusion(matrix):
    """Confusion matrix (rows true, cols predicted) -> y_true, y_pred arrays."""
    y_true, y_pred = [], []
    for t, row in enumerate(matrix):
        for p, count in enumerate(row):
            y_true += [t] * int(count)
            y_pred += [p] * int(count)
    return np.array(y_true), np.array(y_pred)


def bootstrap_qwk(y_true, y_pred, n_resamples=1000, seed=42, num_classes=5):
    """Returns (qwk, low, high) with a 95% percentile interval."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores[i] = quadratic_weighted_kappa(y_true[idx], y_pred[idx], num_classes)
    point = quadratic_weighted_kappa(y_true, y_pred, num_classes)
    low, high = np.percentile(scores, [2.5, 97.5])
    return point, low, high


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bootstrap CIs for QWK")
    parser.add_argument("--results", default="results/results.json")
    parser.add_argument("--predictions", action="append", default=[],
                        help="name=path.csv with true,pred columns (overrides results.json)")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    sets = {}
    with open(args.results, encoding="utf-8") as f:
        for name, metrics in json.load(f)["results"].items():
            sets[name] = pairs_from_confusion(metrics["confusion_matrix"])
    for item in args.predictions:
        name, path = item.split("=", 1)
        table = pd.read_csv(path)
        sets[name] = (table["true"].to_numpy(), table["pred"].to_numpy())

    out = {}
    for name, (y_true, y_pred) in sets.items():
        point, low, high = bootstrap_qwk(y_true, y_pred, args.resamples, args.seed)
        out[name] = (point, low, high)
        print(f"{name:14s} n={len(y_true):5d}  QWK {point:.3f}  "
              f"95% CI [{low:.3f}, {high:.3f}]")
    return out


if __name__ == "__main__":
    main()
