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
        "explainability": "low",
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
# Unsupervised: KMeans, PCA
# ---------------------------------------------------------------------------

def _run_kmeans(
    df: pd.DataFrame, target: str | None, features: list[str], params: dict
) -> dict[str, Any]:
    from sklearn.cluster import KMeans

    _min_rows_check(df)
    X, _, feat_names = _prepare_xy(df, features, target or features[0])
    k = int(params.get("k", 3))
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    metrics = {
        "k": k,
        "inertia": round(float(model.inertia_), 4),
        "cluster_sizes": {str(i): int((labels == i).sum()) for i in range(k)},
    }
    return {
        "metrics": metrics,
        "explainability": "low",
        "summary": f"KMeans (k={k}) — inertia={metrics['inertia']}. "
                   f"Cluster sizes: {metrics['cluster_sizes']}.",
        "cluster_labels": labels,
    }


def _run_pca(
    df: pd.DataFrame, target: str | None, features: list[str], params: dict
) -> dict[str, Any]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    _min_rows_check(df)
    X, _, feat_names = _prepare_xy(df, features, target or features[0])
    n_components = int(params.get("n_components", min(len(feat_names), 2)))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = PCA(n_components=n_components, random_state=42)
    model.fit(X_scaled)
    evr = [round(float(v), 4) for v in model.explained_variance_ratio_]
    loadings = {
        f"PC{i+1}": {n: round(float(v), 6) for n, v in zip(feat_names, model.components_[i])}
        for i in range(n_components)
    }
    metrics = {
        "n_components": n_components,
        "explained_variance_ratio": evr,
        "cumulative_variance": round(float(sum(evr)), 4),
        "loadings": loadings,
    }
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": (
            f"PCA ({n_components} components) explains "
            f"{metrics['cumulative_variance']*100:.1f}% of variance."
        ),
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
    "kmeans": _run_kmeans,
    "pca": _run_pca,
}


# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------

def run_model(
    project: str,
    table: str,
    method: str,
    target: str | None = None,
    features: list[str] | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    """Dispatch to a registered method, persist finding, return complete entry."""
    assert_profiled(project, table)

    if method not in _REGISTRY:
        raise ValueError(
            f"unknown method '{method}'. Available: {sorted(_REGISTRY)}"
        )

    with get_connection(project) as conn:
        df = conn.execute(f'SELECT * FROM "{table}"').df()

    if len(df) < 10:
        raise ValueError(f"need at least 10 rows, got {len(df)}")

    if target is not None and target not in df.columns:
        raise ValueError(f"target '{target}' not found in table '{table}'")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    resolved_features = (
        features if features is not None
        else [c for c in numeric_cols if c != target]
    )

    for f in resolved_features:
        if f not in df.columns:
            raise ValueError(f"feature '{f}' not found in table '{table}'")

    result = _REGISTRY[method](df, target, resolved_features, params or {})

    # Save SHAP companion npy if present
    shap_values = result.pop("shap_values", None)
    cluster_labels = result.pop("cluster_labels", None)

    finding = add_finding(project, {
        "method": method,
        "target": target,
        "features": resolved_features,
        "metrics": result["metrics"],
        "explainability": result["explainability"],
        "summary": result["summary"],
    })

    if shap_values is not None:
        artifacts_dir = project_path(project) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        np.save(str(artifacts_dir / f"{finding['id']}_shap.npy"), shap_values)

    if cluster_labels is not None:
        artifacts_dir = project_path(project) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        np.save(str(artifacts_dir / f"{finding['id']}_labels.npy"), cluster_labels)

    finding["finding_id"] = finding.pop("id")
    return finding
