"""
Check the from-scratch metrics against scikit-learn.

    sklearn is ONLY imported here to prove ours are right, never in vim/ data/ train/ eval/.
    Skipped if sklearn isnt installed.

"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

#skip the whole file if sklearn is missing
skm = pytest.importorskip("sklearn.metrics")

from eval.metrics import (
    accuracy,
    compute_all_metrics,
    confusion_matrix,
    precision_recall_f1,
    quadratic_weighted_kappa,
)


def random_labels(seed, n=500, imbalanced=False):
    """Seeded random true/pred grades."""
    rng = np.random.RandomState(seed)
    if imbalanced:
        #about the EyePACS mix
        probs = [0.73, 0.07, 0.15, 0.03, 0.02]
        y_true = rng.choice(5, size=n, p=probs)
    else:
        y_true = rng.randint(0, 5, size=n)
    #right 60% of the time
    y_pred = np.where(rng.rand(n) < 0.6, y_true, rng.randint(0, 5, size=n))
    return y_true, y_pred


#each test runs once per (seed, imbalanced)
CASES = [(0, False), (1, False), (2, False), (3, True), (4, True), (5, True)]


@pytest.mark.parametrize("seed,imbalanced", CASES)
def test_confusion_matrix_matches_sklearn(seed, imbalanced):
    y_true, y_pred = random_labels(seed, imbalanced=imbalanced)
    ours = confusion_matrix(y_true, y_pred, num_classes=5)
    theirs = skm.confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize("seed,imbalanced", CASES)
def test_accuracy_matches_sklearn(seed, imbalanced):
    y_true, y_pred = random_labels(seed, imbalanced=imbalanced)
    assert np.allclose(accuracy(y_true, y_pred), skm.accuracy_score(y_true, y_pred))


@pytest.mark.parametrize("seed,imbalanced", CASES)
def test_precision_recall_f1_match_sklearn(seed, imbalanced):
    y_true, y_pred = random_labels(seed, imbalanced=imbalanced)
    ours = precision_recall_f1(y_true, y_pred, num_classes=5)
    #zero_division=0 same as ours
    theirs = skm.precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0
    )
    for our_values, their_values in zip(ours, theirs):
        assert np.allclose(our_values, their_values)


@pytest.mark.parametrize("seed,imbalanced", CASES)
def test_qwk_matches_sklearn(seed, imbalanced):
    y_true, y_pred = random_labels(seed, imbalanced=imbalanced)
    ours = quadratic_weighted_kappa(y_true, y_pred, num_classes=5)
    theirs = skm.cohen_kappa_score(y_true, y_pred, weights="quadratic")
    assert np.allclose(ours, theirs)


def test_qwk_worked_example_from_docstring():
    #3 grades, only the last prediction changes: right, off by 1, off by 2
    #same accuracy for the last two but QWK punishes off by 2 much more
    y_true = [0, 0, 1, 2]
    assert np.isclose(quadratic_weighted_kappa(y_true, [0, 1, 1, 2], 3), 0.8)
    assert np.isclose(quadratic_weighted_kappa(y_true, [0, 1, 1, 1], 3), 3 / 7)
    assert np.isclose(quadratic_weighted_kappa(y_true, [0, 1, 1, 0], 3), -0.25)


def test_always_grade_zero_baseline():
    #always saying 0: ~73% accuracy but QWK 0
    y_true, _ = random_labels(seed=10, n=2000, imbalanced=True)
    y_pred = np.zeros_like(y_true)
    metrics = compute_all_metrics(y_true, y_pred)
    print(f"Always-0 baseline: accuracy {metrics['accuracy']:.3f}, "
          f"QWK {metrics['qwk']:.3f}")
    assert metrics["accuracy"] > 0.65
    assert metrics["qwk"] == 0.0


def test_perfect_predictions():
    y_true, _ = random_labels(seed=11)
    metrics = compute_all_metrics(y_true, y_true)
    assert metrics["accuracy"] == 1.0
    assert np.isclose(metrics["qwk"], 1.0)
    assert np.isclose(metrics["macro_f1"], 1.0)


if __name__ == "__main__":
    for seed, imbalanced in CASES:
        test_confusion_matrix_matches_sklearn(seed, imbalanced)
        test_accuracy_matches_sklearn(seed, imbalanced)
        test_precision_recall_f1_match_sklearn(seed, imbalanced)
        test_qwk_matches_sklearn(seed, imbalanced)
    test_qwk_worked_example_from_docstring()
    test_always_grade_zero_baseline()
    test_perfect_predictions()
    print("OK")
