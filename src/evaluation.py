"""Reusable model-evaluation utilities for the Heart Disease Prediction project.

These functions are model-agnostic and operate on fitted scikit-learn-compatible
estimators. They centralize prediction generation and the project's standard
classification metrics so notebooks can share one evaluation implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_predictions(model: Any, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate class predictions and, when supported, positive-class scores.

    ``predict_proba`` is preferred for ROC-AUC because it provides probability
    estimates. If unavailable, ``decision_function`` is used when supported.
    """
    y_pred = np.asarray(model.predict(X))
    y_score: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            y_score = probabilities[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = np.asarray(model.decision_function(X))
    return y_pred, y_score


def calculate_specificity(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Calculate specificity (true-negative rate) for binary classification."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    denominator = tn + fp
    return float(tn / denominator) if denominator else 0.0


def get_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    """Return Accuracy, Precision, Recall, Specificity, F1 and optional ROC-AUC."""
    metrics: Dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": calculate_specificity(y_true, y_pred),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_score))
    return metrics


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y_true: Iterable[int],
    y_score: Optional[Iterable[float]] = None,
) -> Dict[str, Any]:
    """Evaluate a fitted classifier and return predictions, metrics, and reports.

    If ``y_score`` is omitted, scores are generated from ``predict_proba`` or
    ``decision_function`` when the estimator supports either interface.
    """
    y_pred, generated_score = get_predictions(model, X)
    if y_score is None:
        y_score = generated_score
    metrics = get_classification_metrics(y_true, y_pred, y_score)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    return {
        "y_pred": y_pred,
        "y_score": None if y_score is None else np.asarray(y_score),
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def compare_model_metrics(results: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Create a model-by-metric comparison table from evaluation results."""
    return pd.DataFrame(results).T