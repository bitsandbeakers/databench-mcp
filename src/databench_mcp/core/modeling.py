"""Model registry and method handlers for run_model dispatch."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    QuantileRegressor,
    Ridge,
)
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from databench_mcp.core.findings import add_finding
from databench_mcp.db import get_connection
from databench_mcp.workspace import assert_profiled, project_path

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _min_rows_check(df: pd.DataFrame) -> None:
    if len(df) < 10:
        raise ValueError(f"need at least 10 rows, got {len(df)}")


def _prepare_xy(
    df: pd.DataFrame, features: list[str], target: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_raw = df[features]
    X = pd.get_dummies(X_raw, drop_first=True).astype(float)
    y = df[target].astype(float)
    return X.values, y.values, list(X.columns)


def _detect_task(df: pd.DataFrame, target: str, params: dict) -> str:
    if "task" in params:
        return params["task"]
    col = df[target]
    if col.dtype == object or col.dtype == bool or pd.api.types.is_bool_dtype(col):
        return "classification"
    if col.nunique() <= 10:
        return "classification"
    return "regression"


def _split(X, y, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)


def _reg_metrics(y_te, y_pred) -> dict:
    return {
        "r2": round(float(r2_score(y_te, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 4),
    }


def _clf_metrics(y_te, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "f1": round(float(f1_score(y_te, y_pred, average="weighted", zero_division=0)), 4),
    }


def _summary_reg(method: str, target: str, metrics: dict) -> str:
    return f"{method} on '{target}' — R²={metrics.get('r2', 'N/A')}, RMSE={metrics.get('rmse', 'N/A')}."


def _summary_clf(method: str, target: str, metrics: dict) -> str:
    return f"{method} on '{target}' — accuracy={metrics.get('accuracy', 'N/A')}, F1={metrics.get('f1', 'N/A')}."


# ---------------------------------------------------------------------------
# Regression methods
# ---------------------------------------------------------------------------

def _run_linear_regression(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        **_reg_metrics(y_te, y_pred),
        "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, model.coef_)},
        "intercept": round(float(model.intercept_), 6),
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": _summary_reg("Linear regression", target, metrics),
    }


def _run_lasso(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    alpha = float(params.get("alpha", 1.0))
    model = Lasso(alpha=alpha, max_iter=5000)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        **_reg_metrics(y_te, y_pred),
        "alpha": alpha,
        "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, model.coef_)},
        "nonzero_features": int(np.count_nonzero(model.coef_)),
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": _summary_reg("Lasso", target, metrics),
    }


def _run_ridge(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    alpha = float(params.get("alpha", 1.0))
    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        **_reg_metrics(y_te, y_pred),
        "alpha": alpha,
        "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, model.coef_)},
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": _summary_reg("Ridge", target, metrics),
    }


def _run_elastic_net(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    alpha = float(params.get("alpha", 1.0))
    l1_ratio = float(params.get("l1_ratio", 0.5))
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        **_reg_metrics(y_te, y_pred),
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, model.coef_)},
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": _summary_reg("Elastic Net", target, metrics),
    }


def _run_logistic_regression(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        **_clf_metrics(y_te, y_pred),
        "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, model.coef_[0])},
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": _summary_clf("Logistic regression", target, metrics),
    }


def _run_quantile_regression(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    quantile = float(params.get("quantile", 0.5))
    model = QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    pinball = float(np.mean(
        np.where(y_te >= y_pred, quantile * (y_te - y_pred), (1 - quantile) * (y_pred - y_te))
    ))
    metrics = {
        "quantile": quantile,
        "pinball_loss": round(pinball, 4),
        "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, model.coef_)},
        "intercept": round(float(model.intercept_), 6),
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": f"Quantile ({quantile}) regression on '{target}' — pinball loss={round(pinball, 4)}.",
    }


# ---------------------------------------------------------------------------
# Registry (partial — to be completed in Task 8)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable] = {
    "linear_regression": _run_linear_regression,
    "lasso": _run_lasso,
    "ridge": _run_ridge,
    "elastic_net": _run_elastic_net,
    "logistic_regression": _run_logistic_regression,
    "quantile_regression": _run_quantile_regression,
}
