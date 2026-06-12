"""EDA tools: read-only SQL query and manifest-derived dataset summary."""
from __future__ import annotations

import datetime
import decimal
import re
from typing import Any

from databench_mcp.db import get_connection
from databench_mcp.workspace import read_manifest, write_manifest

_SELECT_PATTERN = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_DEFAULT_LIMIT = 500


def _to_json_safe(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, decimal.Decimal):
        return float(val)
    if isinstance(val, (datetime.datetime, datetime.date, datetime.time)):
        return val.isoformat()
    return str(val)


def sql_query(project: str, sql: str, limit: int = _DEFAULT_LIMIT) -> dict[str, Any]:
    """Execute a read-only SELECT/WITH query and return up to `limit` rows."""
    stripped = sql.strip().rstrip(";")
    if not _SELECT_PATTERN.match(stripped):
        raise ValueError("Only SELECT or WITH queries are permitted")
    if ";" in stripped:
        raise ValueError("Multi-statement queries are not permitted")

    with get_connection(project) as conn:
        cursor = conn.execute(stripped)
        columns = [d[0] for d in cursor.description]
        schema = {d[0]: str(d[1]) for d in cursor.description}
        raw_rows = cursor.fetchmany(limit + 1)

    truncated = len(raw_rows) > limit
    rows = [
        {col: _to_json_safe(val) for col, val in zip(columns, row)}
        for row in raw_rows[:limit]
    ]

    return {
        "sql": stripped,
        "columns": columns,
        "schema": schema,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "limit": limit,
    }


def eda_summary(project: str) -> dict[str, Any]:
    """Return a dataset summary derived from the project manifest. No DB query."""
    manifest = read_manifest(project)
    datasets = manifest.get("datasets", {})

    result_datasets = []
    for name, ds in datasets.items():
        entry: dict[str, Any] = {
            "name": name,
            "rows": ds.get("row_count"),
            "cols": ds.get("col_count"),
            "profiled": ds.get("profiled", False),
            "ingested_at": ds.get("ingested_at"),
            "source": ds.get("source"),
        }
        if ds.get("profiled") and ds.get("profile"):
            entry["columns"] = list(ds["profile"].keys())
        result_datasets.append(entry)

    return {
        "project": project,
        "dataset_count": len(result_datasets),
        "datasets": result_datasets,
    }


def derive_table(project: str, sql: str, table_name: str) -> dict[str, Any]:
    """Materialise a SQL SELECT as a new DuckDB table; register in manifest as profiled=False."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
        raise ValueError(f"table_name must be a simple identifier (got {table_name!r})")
    stripped = sql.strip().rstrip(";")
    if not _SELECT_PATTERN.match(stripped):
        raise ValueError("Only SELECT or WITH queries are permitted")
    if ";" in stripped:
        raise ValueError("Multi-statement queries are not permitted")

    with get_connection(project) as conn:
        conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS ({stripped})')
        row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        col_count = len(conn.execute(f'SELECT * FROM "{table_name}" LIMIT 0').description)

    manifest = read_manifest(project)
    manifest["datasets"][table_name] = {
        "source": "derived",
        "sql": stripped,
        "profiled": False,
        "row_count": int(row_count),
        "col_count": int(col_count),
    }
    write_manifest(project, manifest)
    return {"table": table_name, "rows": int(row_count), "columns": int(col_count)}
