"""Central model definitions and configurations for the Heart Disease project."""
from __future__ import annotations

from typing import Dict, Mapping

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

BASELINE_MODEL_CONFIGS = {
    "Logistic Regression": {"max_iter": 1000, "random_state": RANDOM_STATE},
    "K-Nearest Neighbors": {"n_neighbors": 5},
    "Decision Tree": {"random_state": RANDOM_STATE},
    "Gaussian Naive Bayes": {},
    "Support Vector Machine": {"kernel": "rbf", "probability": True, "random_state": RANDOM_STATE},
    "Random Forest": {"n_estimators": 200, "random_state": RANDOM_STATE},
}

XGBOOST_CONFIG = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}


def get_baseline_models() -> Dict[str, object]:
    """Return fresh instances of the six established baseline classifiers."""
    return {
        "Logistic Regression": LogisticRegression(**BASELINE_MODEL_CONFIGS["Logistic Regression"]),
        "K-Nearest Neighbors": KNeighborsClassifier(**BASELINE_MODEL_CONFIGS["K-Nearest Neighbors"]),
        "Decision Tree": DecisionTreeClassifier(**BASELINE_MODEL_CONFIGS["Decision Tree"]),
        "Gaussian Naive Bayes": GaussianNB(**BASELINE_MODEL_CONFIGS["Gaussian Naive Bayes"]),
        "Support Vector Machine": SVC(**BASELINE_MODEL_CONFIGS["Support Vector Machine"]),
        "Random Forest": RandomForestClassifier(**BASELINE_MODEL_CONFIGS["Random Forest"]),
    }


def create_xgboost_model() -> XGBClassifier:
    """Return the project's established XGBoost advanced-model configuration."""
    return XGBClassifier(**XGBOOST_CONFIG)


def get_model_configs() -> Mapping[str, Mapping[str, object]]:
    """Return a copy-like mapping of the documented model configurations."""
    return {**BASELINE_MODEL_CONFIGS, "XGBoost": XGBOOST_CONFIG.copy()}


def get_baseline_and_advanced_models() -> Dict[str, object]:
    """Return the six baselines plus XGBoost using fresh estimator instances."""
    models = get_baseline_models()
    models["XGBoost"] = create_xgboost_model()
    return models


def build_model_pipeline(model, numerical_features=None, categorical_features=None):
    """Build a leakage-safe pipeline for a supplied classifier."""
    from .preprocessing import create_model_pipeline
    return create_model_pipeline(model, numerical_features, categorical_features)


def build_model_pipelines(models: Mapping[str, object], numerical_features=None, categorical_features=None):
    """Build leakage-safe pipelines for a mapping of named classifiers."""
    return {
        name: build_model_pipeline(model, numerical_features, categorical_features)
        for name, model in models.items()
    }


def get_cv_scoring():
    """Return the project's standard cross-validation scoring configuration."""
    from sklearn.metrics import make_scorer
    from .evaluation import calculate_specificity
    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "specificity": make_scorer(calculate_specificity),
        "f1": "f1",
        "roc_auc": "roc_auc",
    }