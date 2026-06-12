"""FastMCP tool wrappers for EDA functions."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.eda import derive_table as _derive_table
from databench_mcp.core.eda import eda_summary as _eda_summary
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
