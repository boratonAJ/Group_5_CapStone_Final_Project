from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score


def safe_auc(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_log_loss(y_true, y_prob):
    try:
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        return float("nan")


def classification_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": safe_auc(y_true, y_prob),
        "log_loss": safe_log_loss(y_true, y_prob),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tpr": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "tnr": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "positive_rate": float(np.mean(np.asarray(y_pred) == 1)),
    }
