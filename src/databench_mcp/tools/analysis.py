"""FastMCP tool wrappers for analysis functions."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.analysis import analyze_correlations as _analyze_correlations
from databench_mcp.core.analysis import analyze_distribution as _analyze_distribution
from databench_mcp.core.analysis import detect_outliers as _detect_outliers
from databench_mcp.core.analysis import peer_outliers as _peer_outliers


def detect_outliers(
    project: str,
    table: str,
    column: str,
    method: str = "iqr",
    params: dict | None = None,
) -> dict[str, Any]:
    """Flag outliers in a numeric column using IQR, Z-score, or Isolation Forest."""
    return _detect_outliers(project, table, column, method, params)


def analyze_distribution(
    project: str,
    table: str,
    column: str,
) -> dict[str, Any]:
    """Return distribution shape stats: skewness, kurtosis, normality test, percentiles."""
    return _analyze_distribution(project, table, column)


def analyze_correlations(
    project: str,
    table: str,
    columns: list[str] | None = None,
    method: str = "pearson",
    top_n: int = 10,
) -> dict[str, Any]:
    """Return correlation matrix and top N pairs by absolute correlation."""
    return _analyze_correlations(project, table, columns, method, top_n)


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

    Scores each entity against peers in the same group_col bucket. Returns
    entities above z_threshold ranked by peer_z, plus per-group summary stats.
    Use after similarity_network to find steerage candidates within archetypes.
    """
    return _peer_outliers(project, table, entity_col, value_col, group_col, z_threshold, top_n)
