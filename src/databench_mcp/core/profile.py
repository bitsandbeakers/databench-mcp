"""Profile logic — runs DuckDB SUMMARIZE and stamps the manifest."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from databench_mcp.core.ingest import _safe_identifier
from databench_mcp.db import get_connection
from databench_mcp.workspace import read_manifest, write_manifest


def _float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def profile_table(project: str, table: str) -> dict[str, Any]:
    """Run DuckDB SUMMARIZE on ``table``, store results in manifest, stamp profiled=True.

    Raises ValueError if the table is not registered in the manifest (i.e. not
    yet ingested). Safe to call multiple times — idempotent.
    """
    _safe_identifier(table)
    manifest = read_manifest(project)
    if table not in manifest["datasets"]:
        raise ValueError(
            f"Table '{table}' not in manifest for project '{project}' — "
            "run ingest_file or ingest_url first"
        )

    with get_connection(project) as conn:
        cursor = conn.execute(f"SUMMARIZE SELECT * FROM {table}")
        col_names = [d[0] for d in cursor.description]
        rows = cursor.fetchall()

    profile: dict[str, dict] = {}
    for row in rows:
        record = dict(zip(col_names, row))
        col_name = record["column_name"]
        profile[col_name] = {
            "type": record.get("column_type"),
            "null_pct": _float(record.get("null_percentage")),
            "approx_unique": record.get("approx_unique"),
            "min": record.get("min"),
            "max": record.get("max"),
            "mean": _float(record.get("avg")),
            "std": _float(record.get("std")),
        }

    now = datetime.now(timezone.utc).isoformat()
    manifest["datasets"][table]["profiled"] = True
    manifest["datasets"][table]["profiled_at"] = now
    manifest["datasets"][table]["profile"] = profile
    write_manifest(project, manifest)

    return {
        "table": table,
        "profiled": True,
        "columns": len(profile),
        "profile": profile,
    }
