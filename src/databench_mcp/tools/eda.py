"""FastMCP tool wrappers for EDA functions."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.eda import add_lag as _add_lag
from databench_mcp.core.eda import add_rolling as _add_rolling
from databench_mcp.core.eda import clean_table as _clean_table
from databench_mcp.core.eda import data_quality_report as _data_quality_report
from databench_mcp.core.eda import derive_table as _derive_table
from databench_mcp.core.eda import eda_summary as _eda_summary
from databench_mcp.core.eda import enrich_table as _enrich_table
from databench_mcp.core.eda import group_summary as _group_summary
from databench_mcp.core.eda import sql_query as _sql_query


def sql_query(project: str, sql: str, limit: int = 500) -> dict[str, Any]:
    """Execute a read-only SELECT/WITH query against the project DuckDB."""
    return _sql_query(project, sql, limit)


def eda_summary(project: str) -> dict[str, Any]:
    """Return dataset summary from the project manifest (no DB query)."""
    return _eda_summary(project)


def derive_table(project: str, sql: str, table_name: str) -> dict[str, Any]:
    """Materialise a SQL SELECT as a new DuckDB table, registered as profiled=False."""
    return _derive_table(project, sql, table_name)


def group_summary(
    project: str,
    table: str,
    group_col: str,
    agg_cols: list[str],
    agg_fns: list[str] | None = None,
) -> dict[str, Any]:
    """Return grouped aggregates for a table column."""
    return _group_summary(project, table, group_col, agg_cols, agg_fns)


def clean_table(
    project: str,
    table: str,
    strategy: str,
    new_table_name: str,
    columns: list[str] | None = None,
    fill_value: float | str | None = None,
) -> dict[str, Any]:
    """Handle missing data by materialising a cleaned derived table."""
    return _clean_table(project, table, strategy, new_table_name, columns, fill_value)


def add_lag(
    project: str,
    table: str,
    col: str,
    lags: list[int],
    new_table_name: str,
    time_col: str | None = None,
) -> dict[str, Any]:
    """Add LAG feature columns to a time-series table."""
    return _add_lag(project, table, col, lags, new_table_name, time_col)


def add_rolling(
    project: str,
    table: str,
    col: str,
    window: int,
    agg_fn: str,
    new_table_name: str,
    time_col: str | None = None,
) -> dict[str, Any]:
    """Add a rolling-window aggregate column to a time-series table."""
    return _add_rolling(project, table, col, window, agg_fn, new_table_name, time_col)


def data_quality_report(project: str, table: str) -> dict[str, Any]:
    """Analyse a profiled table for data quality issues.

    Checks for extreme/high/moderate nulls, constant or near-constant columns,
    likely ID columns masquerading as features, and high coefficient-of-variation
    on numeric columns. Returns a quality score (0–1) and sorted issue list.

    Call this immediately after profile_table on any newly ingested table.
    """
    return _data_quality_report(project, table)


def enrich_table(
    project: str,
    left_table: str,
    right_table: str,
    on: str | list[str],
    new_table_name: str,
    how: str = "inner",
) -> dict[str, Any]:
    """Join two tables and materialise the result as a new derived table."""
    return _enrich_table(project, left_table, right_table, on, new_table_name, how)
