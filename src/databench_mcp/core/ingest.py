"""Ingest logic — loads files and URLs into the project DuckDB and registers them
in the manifest. Tools in tools/ingest.py are thin wrappers over these functions.
"""
from __future__ import annotations

import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from databench_mcp.db import get_connection
from databench_mcp.workspace import project_path, read_manifest, write_manifest

_SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".parquet", ".xlsx", ".xls"}


def _safe_identifier(name: str) -> str:
    """Raise ValueError if name is not a safe SQL identifier."""
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        raise ValueError(
            f"Invalid table name {name!r}: must start with a letter or underscore "
            "and contain only letters, digits, and underscores"
        )
    return name


def _register_table(
    project: str, table_name: str, source: str, row_count: int, col_count: int
) -> None:
    manifest = read_manifest(project)
    manifest["datasets"][table_name] = {
        "source": source,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "profiled": False,
        "profiled_at": None,
        "row_count": row_count,
        "col_count": col_count,
    }
    write_manifest(project, manifest)


def load_file(
    project: str,
    file_path: str | Path,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Load a local CSV/TSV/Parquet/Excel file into the project DuckDB.

    Returns a type-inference report: table name, row count, column count, schema.
    Registers the table in the manifest with profiled=False.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format {ext!r}. Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    if table_name is None:
        raw = file_path.stem.lower()
        table_name = re.sub(r"[^a-zA-Z0-9_]", "_", raw).lstrip("_") or "data"
        if not re.match(r"^[a-zA-Z_]", table_name):
            table_name = f"t_{table_name}"

    table_name = _safe_identifier(table_name)

    with get_connection(project) as conn:
        if ext in (".csv", ".tsv"):
            conn.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS "
                "SELECT * FROM read_csv_auto(?)",
                [str(file_path)],
            )
        elif ext == ".parquet":
            conn.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS "
                "SELECT * FROM read_parquet(?)",
                [str(file_path)],
            )
        elif ext in (".xlsx", ".xls"):
            import polars as pl
            df = pl.read_excel(file_path, engine="openpyxl")
            arrow_table = df.to_arrow()
            conn.register("_excel_tmp", arrow_table)
            conn.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _excel_tmp"
            )
            conn.unregister("_excel_tmp")

        row_count: int = conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        raw_schema = conn.execute(f"DESCRIBE {table_name}").fetchall()

    schema = [{"name": row[0], "type": row[1]} for row in raw_schema]
    _register_table(project, table_name, str(file_path), row_count, len(schema))

    return {
        "table": table_name,
        "source": str(file_path),
        "rows": row_count,
        "columns": len(schema),
        "schema": schema,
    }


def load_url(
    project: str,
    url: str,
    table_name: str,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Download a URL to raw/ and load into the project DuckDB.

    Supports CMS data.gov Socrata API (pass ``params`` for ``$limit``, ``$where``
    filters). Extension is inferred from the URL path; defaults to ``.csv``.
    """
    table_name = _safe_identifier(table_name)

    parsed = urllib.parse.urlparse(url)
    ext = Path(parsed.path).suffix.lower() or ".csv"
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported URL file format {ext!r}. Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    raw_dir = project_path(project) / "raw"
    dest = raw_dir / f"{table_name}{ext}"

    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        dest.write_bytes(response.content)

    return load_file(project, dest, table_name)
