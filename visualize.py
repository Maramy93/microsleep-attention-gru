"""Plotting functions for ROC, PR, and training curves."""

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def plot_auc_curves(y_true, y_proba, model_name="Attention-GRU"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{model_name} AUC-ROC={auc_roc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Sensitivity")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"{model_name} AUC-PR={auc_pr:.3f}")
    plt.xlabel("Recall / Sensitivity")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
