"""Reusable plotting utilities for the Heart Disease Prediction project."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve


def plot_numerical_distributions(data: pd.DataFrame, features: Sequence[str], bins: int = 20, figsize: Tuple[int, int] = (12, 8)):
    """Plot histograms for numerical features and return the figure/axes."""
    axes = data[list(features)].hist(figsize=figsize, bins=bins, edgecolor="black")
    fig = np.asarray(axes).ravel()[0].figure
    fig.suptitle("Distributions of Numerical Features", fontsize=16)
    fig.tight_layout()
    return fig, axes


def plot_numerical_boxplots(data: pd.DataFrame, features: Sequence[str], figsize: Tuple[int, int] = (12, 10)):
    """Plot numerical-feature boxplots and return the figure/axes."""
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.boxplot(data=data, x=feature, ax=axes[i])
        axes[i].set_title(f"Potential Outliers — {feature}")
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    return fig, axes


def plot_numerical_by_target(data: pd.DataFrame, features: Sequence[str], target: str = "target", figsize: Tuple[int, int] = (12, 12)):
    """Plot numerical-feature distributions by binary target class."""
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.boxplot(data=data, x=target, y=feature, hue=target, legend=False, ax=axes[i])
        axes[i].set_title(f"{feature} by Target")
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    return fig, axes


def plot_categorical_by_target(data: pd.DataFrame, features: Sequence[str], target: str = "target", figsize: Tuple[int, int] = (14, 18)):
    """Plot categorical feature counts split by target class."""
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.countplot(data=data, x=feature, hue=target, ax=axes[i])
        axes[i].set_title(f"{feature} Distribution by Target")
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    return fig, axes


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix", figsize: Tuple[int, int] = (6, 5)):
    """Plot a binary confusion matrix and return the figure/axes."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"], ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    fig.tight_layout()
    return fig, ax


def plot_roc_curve(y_true: Iterable[int], y_score: Iterable[float], label: str = "Model"):
    """Plot a ROC curve from true labels and positive-class scores."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {label}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax, roc_auc


def plot_model_comparison(metrics: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)):
    """Plot a model-by-metric comparison table."""
    fig, ax = plt.subplots(figsize=figsize)
    metrics.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout()
    return fig, ax


def plot_feature_importance(feature_names: Sequence[str], importances: Iterable[float], top_n: int = 15, figsize: Tuple[int, int] = (9, 6)):
    """Plot the largest model feature importances."""
    values = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    values.plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {min(top_n, len(values))} Feature Importances")
    fig.tight_layout()
    return fig, ax


def plot_cv_roc_auc(fold_scores: Iterable[float], title: str = "Cross-Validation ROC-AUC by Fold", figsize: Tuple[int, int] = (8, 5)):
    """Plot fold-level ROC-AUC scores with their mean."""
    scores = np.asarray(list(fold_scores), dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(1, len(scores) + 1), scores)
    mean_score = scores.mean()
    ax.axhline(mean_score, linestyle="--", linewidth=2, label=f"Mean CV AUC = {mean_score:.4f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Cross-Validation Fold")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_categorical_percentages(data: pd.DataFrame, features: Sequence[str], target: str = "target", figsize: Tuple[int, int] = (14, 18)):
    """Plot within-target categorical percentages for each feature."""
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    axes = axes.flatten()
    for i, feature in enumerate(features):
        percentage_data = pd.crosstab(data[feature], data[target], normalize="columns") * 100
        percentage_data.columns = ["No Disease (0)", "Disease (1)"]
        percentage_data.plot(kind="bar", ax=axes[i])
        axes[i].set_title(f"{feature} Distribution Within Target Groups")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Percentage (%)")
        axes[i].tick_params(axis="x", rotation=0)
        axes[i].legend(title="Target")
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    return fig, axes


def plot_missing_values(missing_counts: pd.Series, figsize: Tuple[int, int] = (7, 4)):
    """Plot missing-value counts by predictor."""
    fig, ax = plt.subplots(figsize=figsize)
    missing_counts.plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Predictor")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Number of Missing Values")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig, ax


def plot_correlation_heatmap(correlation: pd.DataFrame, title: str = "Correlation Matrix", figsize: Tuple[int, int] = (9, 7)):
    """Plot a correlation matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
