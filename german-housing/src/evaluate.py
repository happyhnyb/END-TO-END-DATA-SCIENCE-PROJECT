"""Small metric helpers used by the training script."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)


def regression_level_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0),
        "R2": r2_score(y_true, y_pred),
    }


def return_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    non_zero = true_sign != 0
    dir_acc = (
        float(np.mean(true_sign[non_zero] == pred_sign[non_zero]))
        if non_zero.any()
        else np.nan
    )

    return {
        "ret_MAE": mean_absolute_error(y_true, y_pred),
        "ret_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "dir_acc": dir_acc,
    }


def classification_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }
