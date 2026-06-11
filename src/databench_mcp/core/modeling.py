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
# Tree-based methods
# ---------------------------------------------------------------------------

def _run_decision_tree(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    task = _detect_task(df, target, params)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    max_depth = params.get("max_depth", None)
    if task == "regression":
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        base_metrics = _reg_metrics(y_te, y_pred)
        summary_fn = _summary_reg
    else:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_tr, y_tr.astype(int))
        y_pred = model.predict(X_te)
        base_metrics = _clf_metrics(y_te.astype(int), y_pred)
        summary_fn = _summary_clf
    importance = {n: round(float(v), 6) for n, v in zip(feat_names, model.feature_importances_)}
    metrics = {**base_metrics, "feature_importance": importance, "max_depth": model.get_depth()}
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": summary_fn("Decision tree", target, metrics),
    }


def _run_random_forest(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    task = _detect_task(df, target, params)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    n_estimators = int(params.get("n_estimators", 100))
    if task == "regression":
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        base_metrics = _reg_metrics(y_te, y_pred)
        summary_fn = _summary_reg
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr.astype(int))
        y_pred = model.predict(X_te)
        base_metrics = _clf_metrics(y_te.astype(int), y_pred)
        summary_fn = _summary_clf
    importance = {n: round(float(v), 6) for n, v in zip(feat_names, model.feature_importances_)}
    metrics = {**base_metrics, "feature_importance": importance, "n_estimators": n_estimators}
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": summary_fn("Random forest", target, metrics),
    }


def _run_gradient_boosting(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    task = _detect_task(df, target, params)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    n_estimators = int(params.get("n_estimators", 100))
    if task == "regression":
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        base_metrics = _reg_metrics(y_te, y_pred)
        summary_fn = _summary_reg
    else:
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_tr, y_tr.astype(int))
        y_pred = model.predict(X_te)
        base_metrics = _clf_metrics(y_te.astype(int), y_pred)
        summary_fn = _summary_clf
    importance = {n: round(float(v), 6) for n, v in zip(feat_names, model.feature_importances_)}
    metrics = {**base_metrics, "feature_importance": importance, "n_estimators": n_estimators}
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": summary_fn("Gradient boosting", target, metrics),
    }


# ---------------------------------------------------------------------------
# SHAP, permutation importance, mutual information
# ---------------------------------------------------------------------------

def _run_shap(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    """Fit internal GBM, compute SHAP values. Caller saves shap_values to .npy."""
    import shap as _shap

    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    base_model_name = params.get("base_model", "gradient_boosting")
    if base_model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    explainer = _shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    mean_abs = {n: round(float(v), 6) for n, v in zip(feat_names, np.abs(shap_vals).mean(axis=0))}
    metrics = {
        "mean_abs_shap": mean_abs,
        "base_model": base_model_name,
        "n_samples": len(X),
    }
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": f"SHAP via {base_model_name} — top driver: {max(mean_abs, key=mean_abs.get)}.",
        "shap_values": shap_vals,
    }


def _run_permutation_importance(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    from sklearn.inspection import permutation_importance as _perm_imp

    _min_rows_check(df)
    task = _detect_task(df, target, params)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    if task == "regression":
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_tr, y_tr)
    else:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_tr, y_tr.astype(int))
        y_te = y_te.astype(int)
    result_pi = _perm_imp(model, X_te, y_te, n_repeats=10, random_state=42)
    imp_mean = {n: round(float(v), 6) for n, v in zip(feat_names, result_pi.importances_mean)}
    metrics = {"importance_mean": imp_mean}
    top = max(imp_mean, key=imp_mean.get)
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": f"Permutation importance on '{target}' — top feature: {top}.",
    }


def _run_mutual_information(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    _min_rows_check(df)
    task = _detect_task(df, target, params)
    X, y, feat_names = _prepare_xy(df, features, target)
    if task == "classification":
        scores = mutual_info_classif(X, y.astype(int), random_state=42)
    else:
        scores = mutual_info_regression(X, y, random_state=42)
    mi = {n: round(float(v), 6) for n, v in zip(feat_names, scores)}
    top = max(mi, key=mi.get)
    return {
        "metrics": {"mi_scores": mi},
        "explainability": "high",
        "summary": f"Mutual information on '{target}' — top feature: {top}.",
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable] = {
    "linear_regression": _run_linear_regression,
    "lasso": _run_lasso,
    "ridge": _run_ridge,
    "elastic_net": _run_elastic_net,
    "logistic_regression": _run_logistic_regression,
    "quantile_regression": _run_quantile_regression,
    "decision_tree": _run_decision_tree,
    "random_forest": _run_random_forest,
    "gradient_boosting": _run_gradient_boosting,
    "shap": _run_shap,
    "permutation_importance": _run_permutation_importance,
    "mutual_information": _run_mutual_information,
}
