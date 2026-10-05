"""Reusable preprocessing utilities for the Heart Disease Prediction project.

The functions in this module centralize the preprocessing workflow used by the
baseline, advanced-model, and future model-comparison notebooks.
"""
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COLUMN = "num"
MODEL_TARGET_COLUMN = "target"

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def validate_columns(
    data: Union[pd.DataFrame, Iterable[str]],
    require_target: bool = True,
) -> bool:
    """Validate that the expected Cleveland dataset columns are available.

    Extra columns are allowed so the function can be used with DataFrames that
    contain derived columns. The model feature set itself is still restricted to
    ``FEATURE_COLUMNS`` by :func:`prepare_features_target`.
    """
    columns = list(data.columns) if isinstance(data, pd.DataFrame) else list(data)
    required = FEATURE_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def load_data(data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load the processed Cleveland dataset with ``?`` represented as NaN."""
    path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path, names=EXPECTED_COLUMNS, na_values="?")
    validate_columns(df, require_target=True)
    return df


def prepare_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare predictors and convert the original 0-4 target to binary labels.

    Original ``num`` values are mapped as 0 -> 0 (absence) and 1-4 -> 1
    (presence), consistent with the project methodology.
    """
    validate_columns(df, require_target=True)
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values; target rows must be resolved before modeling.")

    unique_target = set(pd.Series(df[TARGET_COLUMN]).dropna().unique())
    if not unique_target.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"Unexpected values in {TARGET_COLUMN}: {sorted(unique_target)}")

    X = df[FEATURE_COLUMNS].copy()
    y = (df[TARGET_COLUMN] > 0).astype(int)
    y.name = MODEL_TARGET_COLUMN
    return X, y


def get_feature_groups() -> Dict[str, list]:
    """Return the project's authoritative numerical and categorical groups."""
    return {
        "numerical": NUMERICAL_FEATURES.copy(),
        "categorical": CATEGORICAL_FEATURES.copy(),
        "all": FEATURE_COLUMNS.copy(),
    }


def create_preprocessor(
    numerical_features: Optional[Iterable[str]] = None,
    categorical_features: Optional[Iterable[str]] = None,
) -> ColumnTransformer:
    """Create the shared leakage-safe preprocessing transformer.

    Numerical columns use median imputation and standardization. Categorical
    columns use most-frequent imputation and one-hot encoding. The transformer
    is deliberately returned unfitted so fitting remains inside each model
    pipeline and therefore occurs only on training data/folds.
    """
    groups = get_feature_groups()
    numerical_features = list(numerical_features or groups["numerical"])
    categorical_features = list(categorical_features or groups["categorical"])

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def create_model_pipeline(
    model,
    numerical_features: Optional[Iterable[str]] = None,
    categorical_features: Optional[Iterable[str]] = None,
) -> Pipeline:
    """Create a model pipeline containing the shared preprocessor."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(numerical_features, categorical_features),
            ),
            ("model", model),
        ]
    )
