"""Chart generation — Plotly HTML saved to workspace/<project>/charts/."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

from databench_mcp.core.findings import get_finding
from databench_mcp.db import get_connection
from databench_mcp.workspace import assert_profiled, project_path

_CHART_TYPES = {
    "histogram",
    "boxplot",
    "distribution_overlay",
    "correlation_heatmap",
    "feature_importance_bar",
    "scatter",
    "scatter_matrix",
    "cluster_scatter",
    "shap_beeswarm",
    "shap_waterfall",
    "partial_dependence",
}


def _charts_dir(project: str) -> Path:
    d = project_path(project) / "charts"
    d.mkdir(exist_ok=True)
    return d


def _save(fig, project: str, chart_type: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = _charts_dir(project) / f"{chart_type}_{ts}.html"
    fig.write_html(str(path))
    return str(path)


def _load_df(project: str, table: str, columns: list[str] | None) -> pd.DataFrame:
    if columns:
        cols_sql = ", ".join(f'"{c}"' for c in columns)
    else:
        cols_sql = "*"
    with get_connection(project) as conn:
        return conn.execute(f'SELECT {cols_sql} FROM "{table}"').df()


def create_chart(
    project: str,
    chart_type: str,
    table: str,
    columns: list[str],
    finding_id: str | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    """Generate a Plotly chart and save as standalone HTML."""
    assert_profiled(project, table)
    params = params or {}

    if chart_type not in _CHART_TYPES:
        raise ValueError(f"Unknown chart type '{chart_type}'. Available: {sorted(_CHART_TYPES)}")

    if chart_type == "histogram":
        col = columns[0]
        df = _load_df(project, table, [col])
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")

    elif chart_type == "boxplot":
        col = columns[0]
        df = _load_df(project, table, [col])
        fig = px.box(df, y=col, title=f"Box plot: {col}")

    elif chart_type == "scatter":
        x_col, y_col = columns[0], columns[1]
        df = _load_df(project, table, [x_col, y_col])
        color_col = params.get("color")
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                         title=f"{x_col} vs {y_col}")

    elif chart_type == "scatter_matrix":
        df = _load_df(project, table, columns)
        fig = px.scatter_matrix(df, dimensions=columns,
                                title="Scatter matrix: " + ", ".join(columns))

    elif chart_type == "correlation_heatmap":
        df = _load_df(project, table, columns if columns else None)
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        title="Correlation heatmap", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1)

    elif chart_type == "feature_importance_bar":
        finding = get_finding(project, finding_id)
        importance = (
            finding["metrics"].get("feature_importance")
            or finding["metrics"].get("mean_abs_shap")
            or finding["metrics"].get("importance_mean")
            or finding["metrics"].get("mi_scores")
        )
        if importance is None:
            raise ValueError(f"Finding '{finding_id}' has no feature importance data")
        feat_df = pd.DataFrame(
            sorted(importance.items(), key=lambda x: x[1], reverse=True),
            columns=["feature", "importance"],
        )
        fig = px.bar(feat_df, x="importance", y="feature", orientation="h",
                     title=f"Feature importance — {finding['method']}")

    elif chart_type == "cluster_scatter":
        if finding_id is None:
            raise ValueError("cluster_scatter requires finding_id from a kmeans run")
        import numpy as np
        finding = get_finding(project, finding_id)
        labels_path = project_path(project) / "artifacts" / f"{finding_id}_labels.npy"
        if not labels_path.exists():
            raise ValueError(f"No cluster labels found for finding '{finding_id}'")
        labels = np.load(str(labels_path))
        x_col, y_col = columns[0], columns[1]
        df = _load_df(project, table, [x_col, y_col])
        df["cluster"] = labels.astype(str)
        fig = px.scatter(df, x=x_col, y=y_col, color="cluster",
                         title=f"Cluster scatter: {x_col} vs {y_col}")

    elif chart_type == "shap_beeswarm":
        if finding_id is None:
            raise ValueError("shap_beeswarm requires finding_id from a shap run")
        import numpy as np
        finding = get_finding(project, finding_id)
        npy_path = project_path(project) / "artifacts" / f"{finding_id}_shap.npy"
        if not npy_path.exists():
            raise ValueError(f"No SHAP values found for finding '{finding_id}'")
        shap_vals = np.load(str(npy_path))
        feat_names = finding.get("features", [f"f{i}" for i in range(shap_vals.shape[1])])
        mean_abs = pd.DataFrame({
            "feature": feat_names,
            "mean_abs_shap": shap_vals.mean(axis=0) if shap_vals.ndim == 2
                             else [float(shap_vals.mean())],
        }).sort_values("mean_abs_shap", ascending=True)
        fig = px.bar(mean_abs, x="mean_abs_shap", y="feature", orientation="h",
                     title="SHAP mean absolute values (beeswarm proxy)")

    elif chart_type == "shap_waterfall":
        raise ValueError("shap_waterfall chart type is not yet implemented")

    elif chart_type in ("distribution_overlay", "partial_dependence"):
        raise ValueError(f"Chart type '{chart_type}' not yet implemented")

    else:
        raise ValueError(f"Unknown chart type '{chart_type}'")

    path = _save(fig, project, chart_type)
    return {"chart_type": chart_type, "path": path, "title": fig.layout.title.text or chart_type}
