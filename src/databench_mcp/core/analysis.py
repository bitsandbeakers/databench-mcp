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
