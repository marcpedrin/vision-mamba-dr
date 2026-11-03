"""
Metrics from scratch in NumPy (tests/test_metrics.py checks them against sklearn).

    confusion matrix, accuracy, per class P/R/F1, QWK (main score, Kaggle metric)
    Accuracy is misleading here: always saying grade 0 already gets ~73%, but QWK = 0.

"""

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes=5):
    """M[i, j] = images with true grade i predicted as j. Rows truth, cols prediction."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def accuracy(y_true, y_pred):
    """Fraction predicted correctly."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, num_classes=5):
    """Per class precision, recall, f1, support. 0 when dividing by zero (like sklearn).

        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        f1        = 2PR / (P + R)
    """
    matrix = confusion_matrix(y_true, y_pred, num_classes)

    #TP = diagonal
    tp = np.diag(matrix).astype(np.float64)
    #column sums = times predicted (TP + FP)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    #row sums = times it really occurs (TP + FN)
    true_count = matrix.sum(axis=1).astype(np.float64)

    #only divide where denominator > 0, rest stays 0
    precision = np.divide(tp, predicted_count, out=np.zeros(num_classes),
                          where=predicted_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros(num_classes),
                       where=true_count > 0)
    p_plus_r = precision + recall
    f1 = np.divide(2 * precision * recall, p_plus_r, out=np.zeros(num_classes),
                   where=p_plus_r > 0)
    support = true_count.astype(np.int64)

    return precision, recall, f1, support


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    """QWK = 1 - sum(W * O) / sum(W * E)

        O = confusion matrix
        W[i, j] = (i - j)^2 / (K - 1)^2, so being off by 4 grades costs 16x off by 1
        E = what chance would give: outer(true counts, predicted counts) / N
        1 = perfect, 0 = chance, < 0 worse than chance
    """
    #O
    observed = confusion_matrix(y_true, y_pred, num_classes).astype(np.float64)
    n = observed.sum()

    #W: column minus row gives every i - j
    grades = np.arange(num_classes)
    diff = grades.reshape(-1, 1) - grades.reshape(1, -1)
    weights = diff.astype(np.float64) ** 2 / (num_classes - 1) ** 2

    #E
    true_hist = observed.sum(axis=1)
    pred_hist = observed.sum(axis=0)
    expected = np.outer(true_hist, pred_hist) / n

    observed_penalty = np.sum(weights * observed)
    expected_penalty = np.sum(weights * expected)

    #0/0 when everything is one class, call it 0
    if expected_penalty == 0:
        return 0.0

    return float(1.0 - observed_penalty / expected_penalty)


def compute_all_metrics(y_true, y_pred, num_classes=5):
    """All metrics as plain python values (so it can go straight into json/csv)."""
    precision, recall, f1, support = precision_recall_f1(y_true, y_pred, num_classes)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred, num_classes),
        #macro = plain mean, so rare grades count as much as grade 0
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, num_classes).tolist(),
    }
