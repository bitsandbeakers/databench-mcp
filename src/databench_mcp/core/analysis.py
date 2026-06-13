"""EDA analysis: outlier detection, distribution analysis, correlation analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from databench_mcp.db import get_connection
from databench_mcp.workspace import assert_profiled


def _load_column(project: str, table: str, column: str) -> pd.Series:
    with get_connection(project) as conn:
        df = conn.execute(f'SELECT "{column}" FROM "{table}"').df()
    return df[column].dropna()


def detect_outliers(
    project: str,
    table: str,
    column: str,
    method: str = "iqr",
    params: dict | None = None,
) -> dict[str, Any]:
    """Flag outliers in a single numeric column. Returns counts and sample rows."""
    assert_profiled(project, table)
    params = params or {}

    series = _load_column(project, table, column)
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{column}' is not numeric")

    total = len(series)
    values = series.values

    if method == "iqr":
        multiplier = float(params.get("multiplier", 1.5))
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask = (values < lower) | (values > upper)
        threshold = {"lower_fence": round(float(lower), 4), "upper_fence": round(float(upper), 4)}

    elif method == "zscore":
        threshold_val = float(params.get("threshold", 3.0))
        z = np.abs(stats.zscore(values))
        mask = z > threshold_val
        threshold = {"z_threshold": threshold_val}

    elif method == "isolation_forest":
        contamination = float(params.get("contamination", 0.05))
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(values.reshape(-1, 1))
        mask = preds == -1
        threshold = {"contamination": contamination}

    else:
        raise ValueError(f"Unknown outlier method '{method}'. Choose: iqr, zscore, isolation_forest")

    outlier_indices = np.where(mask)[0]
    outlier_count = int(mask.sum())
    sample = series.iloc[outlier_indices[:20]].tolist()

    return {
        "table": table,
        "column": column,
        "method": method,
        "outlier_count": outlier_count,
        "total_rows": total,
        "outlier_pct": round(outlier_count / total * 100, 2) if total > 0 else 0.0,
        "threshold": threshold,
        "sample_outliers": [round(float(v), 4) for v in sample],
    }


def analyze_distribution(
    project: str,
    table: str,
    column: str,
) -> dict[str, Any]:
    """Return distribution shape stats for a numeric column."""
    assert_profiled(project, table)
    series = _load_column(project, table, column)
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{column}' is not numeric")

    values = series.values.astype(float)
    n = len(values)

    skewness = float(stats.skew(values))
    kurt = float(stats.kurtosis(values))

    if n <= 5000:
        stat, p = stats.shapiro(values)
        test_name = "shapiro"
    else:
        stat, p = stats.kstest(values, "norm", args=(values.mean(), values.std()))
        test_name = "ks"

    if abs(skewness) < 0.5 and p > 0.05:
        verdict = "approximately normal"
    elif skewness > 0.5:
        verdict = "right-skewed"
    elif skewness < -0.5:
        verdict = "left-skewed"
    else:
        verdict = "heavy-tailed"

    return {
        "column": column,
        "dtype": str(series.dtype),
        "n": n,
        "mean": round(float(values.mean()), 4),
        "median": round(float(np.median(values)), 4),
        "std": round(float(values.std()), 4),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurt, 4),
        "normality_test": test_name,
        "normality_stat": round(float(stat), 6),
        "normality_p": round(float(p), 6),
        "percentiles": {
            "p5": round(float(np.percentile(values, 5)), 4),
            "p25": round(float(np.percentile(values, 25)), 4),
            "p75": round(float(np.percentile(values, 75)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "p99": round(float(np.percentile(values, 99)), 4),
        },
        "verdict": verdict,
    }


def peer_outliers(
    project: str,
    table: str,
    entity_col: str,
    value_col: str,
    group_col: str,
    z_threshold: float = 1.5,
    top_n: int = 50,
) -> dict[str, Any]:
    """Within-group z-score outlier detection for peer benchmarking.

    For each entity, compute the z-score of value_col relative to its group_col
    peers. Entities above z_threshold are returned ranked by peer_z descending.

    Returns group summary stats and the top_n outliers.
    """
    assert_profiled(project, table)

    with get_connection(project) as conn:
        df = conn.execute(f'SELECT * FROM "{table}"').df()

    for col in (entity_col, value_col, group_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in table '{table}'")

    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"value_col '{value_col}' must be numeric")

    df = df.dropna(subset=[value_col]).copy()
    if len(df) < 3:
        raise ValueError(f"Need at least 3 rows after dropping nulls, got {len(df)}")

    comm_means = df.groupby(group_col)[value_col].transform("mean")
    comm_stds = df.groupby(group_col)[value_col].transform("std").fillna(1e-9).clip(lower=1e-9)
    df["peer_z"] = (df[value_col] - comm_means) / comm_stds

    outliers_df = (
        df[df["peer_z"] > z_threshold]
        .sort_values("peer_z", ascending=False)
        .head(top_n)
    )
    outliers = [
        {
            "entity": str(row[entity_col]),
            "group": str(row[group_col]),
            "value": round(float(row[value_col]), 6),
            "peer_z": round(float(row["peer_z"]), 4),
        }
        for _, row in outliers_df.iterrows()
    ]

    group_stats_raw = (
        df.groupby(group_col)[value_col]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    group_stats = [
        {
            "group": str(row[group_col]),
            "n": int(row["count"]),
            "mean": round(float(row["mean"]), 6),
            "std": round(float(row["std"]) if not np.isnan(row["std"]) else 0.0, 6),
            "min": round(float(row["min"]), 6),
            "max": round(float(row["max"]), 6),
        }
        for _, row in group_stats_raw.iterrows()
    ]

    return {
        "entity_col": entity_col,
        "value_col": value_col,
        "group_col": group_col,
        "z_threshold": z_threshold,
        "total_rows": len(df),
        "n_outliers": len(outliers),
        "n_groups": int(df[group_col].nunique()),
        "outliers": outliers,
        "group_stats": group_stats,
        "summary": (
            f"{len(outliers)} entities in '{value_col}' score above z>{z_threshold} "
            f"within their '{group_col}' peer group "
            f"(across {int(df[group_col].nunique())} groups, {len(df)} total rows)."
        ),
    }


def analyze_correlations(
    project: str,
    table: str,
    columns: list[str] | None = None,
    method: str = "pearson",
    top_n: int = 10,
) -> dict[str, Any]:
    """Return correlation matrix and top N pairs by absolute correlation."""
    assert_profiled(project, table)
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown correlation method '{method}'. Choose: pearson, spearman")

    with get_connection(project) as conn:
        df = conn.execute(f'SELECT * FROM "{table}"').df()

    numeric_df = df.select_dtypes(include="number")
    if columns is not None:
        for c in columns:
            if c not in numeric_df.columns:
                raise ValueError(f"Column '{c}' is not numeric or not in table")
        numeric_df = numeric_df[columns]

    if len(numeric_df.columns) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation analysis")

    corr_matrix = numeric_df.corr(method=method)
    matrix_dict = {
        col: {c: round(float(v), 4) for c, v in row.items()}
        for col, row in corr_matrix.to_dict().items()
    }

    pairs = []
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if not np.isnan(r):
                pairs.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "r": round(float(r), 4),
                    "abs_r": round(abs(float(r)), 4),
                })
    pairs.sort(key=lambda x: x["abs_r"], reverse=True)

    return {
        "method": method,
        "matrix": matrix_dict,
        "top_pairs": pairs[:top_n],
    }
