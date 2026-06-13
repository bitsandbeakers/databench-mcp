"""Model registry and method handlers for run_model dispatch."""
from __future__ import annotations

import json
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

    # OHE categorical features, then aggregate importances back to source feature
    categorical_features: list[str] = params.get("categorical_features") or []
    for cf in categorical_features:
        if cf not in features:
            raise ValueError(f"categorical_feature '{cf}' not in features list")

    X_raw = df[features]
    X_enc = pd.get_dummies(X_raw, columns=categorical_features, drop_first=True).astype(float)
    feat_names = list(X_enc.columns)
    y = df[target].astype(float)

    X_tr, X_te, y_tr, y_te = _split(X_enc.values, y.values)
    alpha = float(params.get("alpha", 1.0))
    model = Lasso(alpha=alpha, max_iter=5000)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # Aggregate absolute coefficients for OHE dummy groups back to source feature
    raw_coefs: dict[str, float] = {n: float(c) for n, c in zip(feat_names, model.coef_)}
    aggregated_importance: dict[str, float] = {}
    for src in features:
        if src in categorical_features:
            dummy_cols = [n for n in feat_names if n.startswith(f"{src}_")]
            agg = sum(abs(raw_coefs.get(c, 0.0)) for c in dummy_cols)
            aggregated_importance[src] = round(agg, 6)
        else:
            aggregated_importance[src] = round(abs(raw_coefs.get(src, 0.0)), 6)

    metrics = {
        **_reg_metrics(y_te, y_pred),
        "alpha": alpha,
        "feature_importance": aggregated_importance,
        "coefficients": {n: round(c, 6) for n, c in raw_coefs.items()},
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
    """Fit internal GBM, compute SHAP values. Caller saves shap_values to .npy.

    params:
      base_model        : 'gradient_boosting' (default) or 'random_forest'
      shap_mode         : 'tree_path_dependent' (default) or 'interventional'
      shap_background_n : background sample size for interventional mode (default 100)
    """
    import shap as _shap

    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    base_model_name = params.get("base_model", "gradient_boosting")
    shap_mode = params.get("shap_mode", "tree_path_dependent")
    if shap_mode not in ("tree_path_dependent", "interventional"):
        raise ValueError(
            f"Unknown shap_mode '{shap_mode}'. Choose: tree_path_dependent, interventional"
        )

    if base_model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    if shap_mode == "interventional":
        bg_n = int(params.get("shap_background_n", min(100, len(X))))
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(X), size=min(bg_n, len(X)), replace=False)
        background = X[bg_idx]
        explainer = _shap.TreeExplainer(
            model, data=background, feature_perturbation="interventional"
        )
    else:
        explainer = _shap.TreeExplainer(model)

    shap_vals = explainer.shap_values(X)
    mean_abs = {n: round(float(v), 6) for n, v in zip(feat_names, np.abs(shap_vals).mean(axis=0))}
    metrics = {
        "mean_abs_shap": mean_abs,
        "base_model": base_model_name,
        "shap_mode": shap_mode,
        "n_samples": len(X),
    }
    top = max(mean_abs, key=mean_abs.get)
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": (
            f"SHAP ({shap_mode}) via {base_model_name} — top driver: {top}."
        ),
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
# Network analysis
# ---------------------------------------------------------------------------

def _validate_network_params(df: pd.DataFrame, params: dict) -> tuple[str, str, str | None]:
    source_col = params.get("source_col")
    target_col = params.get("target_col")
    if not source_col or not target_col:
        raise ValueError("network methods require 'source_col' and 'target_col' in params")
    for col in (source_col, target_col):
        if col not in df.columns:
            raise ValueError(f"column '{col}' not found in table")
    weight_col = params.get("weight_col")
    return source_col, target_col, weight_col


def _build_network_graph(df: pd.DataFrame, source_col: str, target_col: str, weight_col: str | None):
    import igraph as ig
    src = df[source_col].astype(str).tolist()
    tgt = df[target_col].astype(str).tolist()
    if weight_col:
        tuples = list(zip(src, tgt, df[weight_col].tolist()))
        G = ig.Graph.TupleList(tuples, directed=False, weights=True)
    else:
        tuples = list(zip(src, tgt))
        G = ig.Graph.TupleList(tuples, directed=False, weights=False)
    return G


def _run_network_stats(
    df: pd.DataFrame, target: str | None, features: list[str], params: dict
) -> dict[str, Any]:
    source_col, target_col, weight_col = _validate_network_params(df, params)
    G = _build_network_graph(df, source_col, target_col, weight_col)
    if G.vcount() < 2:
        raise ValueError("need at least 2 nodes to build a network")
    components = G.connected_components(mode="weak")
    node_count = G.vcount()
    edge_count = G.ecount()
    largest_cc_size = max(len(c) for c in components)
    degrees = G.degree()
    avg_degree = round(sum(degrees) / node_count, 4) if node_count else 0.0
    density = round(G.density(), 6)
    num_components = len(components)
    metrics = {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": density,
        "avg_degree": avg_degree,
        "num_components": num_components,
        "largest_component_fraction": round(largest_cc_size / node_count, 4),
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": (
            f"Network has {node_count} nodes and {edge_count} edges "
            f"(density={density:.4f}, {num_components} component(s))."
        ),
    }


def _run_network_centrality(
    df: pd.DataFrame, target: str | None, features: list[str], params: dict
) -> dict[str, Any]:
    source_col, target_col, weight_col = _validate_network_params(df, params)
    compute_betweenness = bool(params.get("betweenness", True))
    top_n = int(params.get("top_n", 10))
    G = _build_network_graph(df, source_col, target_col, weight_col)
    if G.vcount() < 2:
        raise ValueError("need at least 2 nodes to build a network")
    names = G.vs["name"]
    degrees = G.degree()
    pageranks = G.pagerank()
    centrality_full: dict[str, dict] = {
        name: {"degree": float(deg), "pagerank": round(float(pr), 6)}
        for name, deg, pr in zip(names, degrees, pageranks)
    }
    if compute_betweenness:
        betweenness = G.betweenness(directed=False)
        for name, bw in zip(names, betweenness):
            centrality_full[name]["betweenness"] = round(float(bw), 6)

    def _top(key: str) -> list[dict]:
        return sorted(
            [{"node": name, key: centrality_full[name][key]} for name in names],
            key=lambda x: x[key], reverse=True
        )[:top_n]

    metrics: dict[str, Any] = {
        "top_by_degree": _top("degree"),
        "top_by_pagerank": _top("pagerank"),
    }
    if compute_betweenness:
        metrics["top_by_betweenness"] = _top("betweenness")
    top_node = metrics["top_by_degree"][0]["node"] if metrics["top_by_degree"] else "N/A"
    top_degree = metrics["top_by_degree"][0]["degree"] if metrics["top_by_degree"] else 0
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": (
            f"Network centrality — most connected node: {top_node} "
            f"(degree={top_degree})."
        ),
        "centrality": centrality_full,
    }


def _run_network_communities(
    df: pd.DataFrame, target: str | None, features: list[str], params: dict
) -> dict[str, Any]:
    from collections import Counter
    source_col, target_col, weight_col = _validate_network_params(df, params)
    G = _build_network_graph(df, source_col, target_col, weight_col)
    if G.vcount() < 2:
        raise ValueError("need at least 2 nodes to build a network")
    names = G.vs["name"]
    communities = G.community_multilevel(
        weights="weight" if weight_col else None,
        return_levels=False,
    )
    membership = communities.membership
    modularity = round(float(G.modularity(membership)), 6)
    num_communities = len(set(membership))
    comm_sizes = Counter(membership)
    community_sizes = {str(k): v for k, v in sorted(comm_sizes.items())}
    communities_dict = {name: int(comm) for name, comm in zip(names, membership)}
    metrics = {
        "num_communities": num_communities,
        "modularity": modularity,
        "community_sizes": community_sizes,
    }
    return {
        "metrics": metrics,
        "explainability": "low",
        "summary": (
            f"Louvain detected {num_communities} communities "
            f"(modularity={modularity:.4f})."
        ),
        "communities": communities_dict,
    }


# ---------------------------------------------------------------------------
# EBM (ExplainableBoostingRegressor / Classifier)
# ---------------------------------------------------------------------------

def _run_ebm(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    """Fit an EBM (GA²M) and return R², feature importances, and per-feature shape data."""
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

    _min_rows_check(df)
    task = _detect_task(df, target, params)

    # Build X preserving dtypes — EBM handles categoricals natively via feature_types
    X_raw = df[features]
    y = df[target].astype(float if task == "regression" else int)

    # interpret uses "nominal" for categorical features (not "categorical")
    _TYPE_MAP = {"categorical": "nominal", "nominal": "nominal", "continuous": "continuous"}

    # Allow caller to pass feature_types as a list or a dict {feature: type}
    ft_param = params.get("feature_types")
    if isinstance(ft_param, dict):
        feature_types = [_TYPE_MAP.get(ft_param.get(f, "continuous"), "continuous") for f in features]
    elif isinstance(ft_param, list):
        if len(ft_param) != len(features):
            raise ValueError(
                f"feature_types length {len(ft_param)} != features length {len(features)}"
            )
        feature_types = [_TYPE_MAP.get(t, t) for t in ft_param]
    else:
        # Auto-detect: string/object columns → nominal, else continuous
        feature_types = [
            "nominal" if X_raw[f].dtype == object or str(X_raw[f].dtype) == "category"
            else "continuous"
            for f in features
        ]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw.values, y.values, test_size=0.2, random_state=42
    )

    if task == "regression":
        model = ExplainableBoostingRegressor(
            feature_names=features,
            feature_types=feature_types,
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        base_metrics = _reg_metrics(y_te, y_pred)
    else:
        model = ExplainableBoostingClassifier(
            feature_names=features,
            feature_types=feature_types,
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        base_metrics = _clf_metrics(y_te, y_pred)

    # Feature importances from global explanation
    ebm_global = model.explain_global()
    imp_raw = ebm_global.data()  # top-level data has 'names' and 'scores'
    importances = {
        n: round(float(s), 6)
        for n, s in zip(imp_raw.get("names", features), imp_raw.get("scores", []))
    }

    # Per-feature shape data: x_vals and y_vals for each main effect
    shapes: dict[str, Any] = {}
    for i, feat in enumerate(features):
        fdata = ebm_global.data(i)
        ftype = fdata.get("type", "univariate")
        feat_idx = features.index(feat)
        is_nominal = feature_types[feat_idx] == "nominal"
        if ftype == "univariate" and not is_nominal:
            shapes[feat] = {
                "type": "continuous",
                "x": [round(float(v), 4) for v in fdata.get("names", [])],
                "y": [round(float(v), 6) for v in fdata.get("scores", [])],
            }
        else:
            # Nominal/categorical — names are string category labels
            shapes[feat] = {
                "type": "categorical",
                "x": [str(v) for v in fdata.get("names", [])],
                "y": [round(float(v), 6) for v in fdata.get("scores", [])],
            }

    metrics = {
        **base_metrics,
        "feature_importance": importances,
        "n_features": len(features),
        "feature_types_used": feature_types,
    }
    top = max(importances, key=importances.get) if importances else "N/A"
    task_label = "EBM regression" if task == "regression" else "EBM classification"
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": (
            f"{task_label} on '{target}' — "
            f"{'R²=' + str(metrics.get('r2')) if task == 'regression' else 'accuracy=' + str(metrics.get('accuracy'))}. "
            f"Top driver: {top}."
        ),
        "ebm_shapes": shapes,
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
    "network_stats": _run_network_stats,
    "network_centrality": _run_network_centrality,
    "network_communities": _run_network_communities,
    "ebm": _run_ebm,
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
    communities_data = result.pop("communities", None)
    centrality_data = result.pop("centrality", None)
    ebm_shapes = result.pop("ebm_shapes", None)

    finding = add_finding(project, {
        "method": method,
        "table": table,
        "target": target,
        "features": resolved_features,
        "params": params or {},
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

    if communities_data is not None:
        artifacts_dir = project_path(project) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        (artifacts_dir / f"{finding['id']}_communities.json").write_text(
            json.dumps(communities_data)
        )

    if centrality_data is not None:
        artifacts_dir = project_path(project) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        (artifacts_dir / f"{finding['id']}_centrality.json").write_text(
            json.dumps(centrality_data)
        )

    if ebm_shapes is not None:
        artifacts_dir = project_path(project) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        (artifacts_dir / f"{finding['id']}_ebm_shapes.json").write_text(
            json.dumps(ebm_shapes)
        )
        finding["ebm_shapes"] = ebm_shapes

    finding["finding_id"] = finding.pop("id")
    return finding


# ---------------------------------------------------------------------------
# Standalone: similarity_network
# ---------------------------------------------------------------------------

def similarity_network(
    project: str,
    table: str,
    entity_col: str,
    code_col: str,
    volume_col: str,
    k: int = 8,
    min_sim: float = 0.05,
    z_threshold: float = 1.5,
    value_col: str | None = None,
) -> dict[str, Any]:
    """Build a cosine-similarity kNN graph from a long-form entity×code volume table.

    Parameters
    ----------
    project, table : workspace + DuckDB table name
    entity_col     : column identifying each entity (e.g. hospital CCN)
    code_col       : column with the procedure/product codes (e.g. DRG, APC)
    volume_col     : numeric column with the volume/count per entity×code pair
    k              : k-nearest neighbours per entity
    min_sim        : minimum cosine similarity to include an edge
    z_threshold    : peer z-score threshold for flagging outliers
    value_col      : optional numeric column (one value per entity) used to
                     compute peer-adjusted z-scores (e.g. mean_drg_premium)
    """
    import igraph as ig
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize

    assert_profiled(project, table)

    with get_connection(project) as conn:
        df = conn.execute(f'SELECT * FROM "{table}"').df()

    for col in (entity_col, code_col, volume_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in table '{table}'")
    if value_col and value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not found in table '{table}'")

    # Drop rows with null/zero volume
    df = df.dropna(subset=[volume_col])
    df = df[df[volume_col] > 0].copy()
    df[volume_col] = df[volume_col].astype(float)

    entities = sorted(df[entity_col].unique())
    codes = sorted(df[code_col].unique())
    N = len(entities)
    if N < 3:
        raise ValueError(f"Need at least 3 distinct entities, got {N}")

    ent_idx = {e: i for i, e in enumerate(entities)}
    code_idx = {c: j for j, c in enumerate(codes)}

    rows = df[entity_col].map(ent_idx).values
    cols = df[code_col].map(code_idx).values
    data = df[volume_col].values
    mat = csr_matrix((data, (rows, cols)), shape=(N, len(codes)))
    mat_norm = normalize(mat, norm="l2")

    # Cosine similarity via dot product on normalised rows
    sim_full = (mat_norm @ mat_norm.T).toarray()

    # Build kNN graph (symmetric, no self-loops)
    k_actual = min(k, N - 1)
    edge_set: dict[tuple[int, int], float] = {}
    for i in range(N):
        top_k = np.argpartition(sim_full[i], -k_actual - 1)[-k_actual - 1:]
        for j in top_k:
            if j == i:
                continue
            s = float(sim_full[i, j])
            if s > min_sim:
                key = (min(i, j), max(i, j))
                edge_set[key] = max(edge_set.get(key, 0.0), s)

    edge_list = [(i, j, w) for (i, j), w in edge_set.items()]
    G = ig.Graph(n=N, edges=[(i, j) for i, j, _ in edge_list], directed=False)
    G.es["weight"] = [w for _, _, w in edge_list]
    G.vs["name"] = [str(e) for e in entities]

    communities = G.community_multilevel(weights="weight", return_levels=False)
    membership = communities.membership
    modularity = round(float(G.modularity(membership)), 6)
    num_communities = len(set(membership))

    entity_community = {str(e): int(membership[ent_idx[e]]) for e in entities}

    # Build per-community stats
    comm_df = pd.DataFrame({"entity": entities, "community": membership})
    if value_col:
        val_map = (
            df.groupby(entity_col)[value_col].mean().to_dict()
        )
        comm_df["value"] = comm_df["entity"].map(val_map)
        comm_stats_raw = (
            comm_df.groupby("community")["value"]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        community_stats = [
            {
                "community_id": int(row["community"]),
                "size": int(row["count"]),
                "mean_value": round(float(row["mean"]), 6),
                "std_value": round(float(row["std"]) if not np.isnan(row["std"]) else 0.0, 6),
            }
            for _, row in comm_stats_raw.iterrows()
        ]
        # Peer z-score outliers
        comm_means = comm_df.groupby("community")["value"].transform("mean")
        comm_stds = comm_df.groupby("community")["value"].transform("std").clip(lower=1e-9)
        comm_df["peer_z"] = (comm_df["value"] - comm_means) / comm_stds
        outliers_df = comm_df[comm_df["peer_z"] > z_threshold].sort_values("peer_z", ascending=False)
        outliers = [
            {
                "entity": str(row["entity"]),
                "community_id": int(row["community"]),
                "value": round(float(row["value"]), 6),
                "peer_z": round(float(row["peer_z"]), 4),
            }
            for _, row in outliers_df.iterrows()
        ]
    else:
        comm_sizes = comm_df.groupby("community").size().to_dict()
        community_stats = [
            {"community_id": int(c), "size": int(s)}
            for c, s in sorted(comm_sizes.items())
        ]
        outliers = []

    return {
        "entity_col": entity_col,
        "code_col": code_col,
        "n_entities": N,
        "n_codes": len(codes),
        "n_edges": len(edge_list),
        "k": k_actual,
        "min_sim": min_sim,
        "num_communities": num_communities,
        "modularity": modularity,
        "community_stats": community_stats,
        "entity_community": entity_community,
        "outliers": outliers,
        "z_threshold": z_threshold,
        "summary": (
            f"Cosine kNN graph: {N} entities, {len(edge_list)} edges, "
            f"{num_communities} Louvain communities (modularity={modularity:.4f}). "
            + (f"{len(outliers)} outliers above z>{z_threshold}." if outliers else "")
        ),
    }
