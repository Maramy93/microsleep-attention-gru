"""Evaluation utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def optimize_threshold(y_true, y_proba, metric="geometric_mean"):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold, best_score = 0.5, -1
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            sensitivity = recall_score(y_true, y_pred, zero_division=0)
            specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            score = np.sqrt(sensitivity * specificity)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


def calculate_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": sensitivity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "geometric_mean": np.sqrt(sensitivity * specificity),
        "auc_roc": roc_auc_score(y_true, y_proba),
        "auc_pr": average_precision_score(y_true, y_proba),
        "confusion_matrix": cm,
    }
