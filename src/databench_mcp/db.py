"""DuckDB connection factory.

Each call opens a new connection to the project's .duckdb file.
Use as a context manager: ``with get_connection(project) as conn: ...``
"""
from __future__ import annotations

import duckdb

from databench_mcp.workspace import _validate_name, project_path


def get_connection(project: str) -> duckdb.DuckDBPyConnection:
    """Return an open DuckDB connection to the project database."""
    _validate_name(project)
    proj_dir = project_path(project)
    if not proj_dir.exists():
        raise FileNotFoundError(f"Project '{project}' not found")
    db_path = proj_dir / "project.duckdb"
    return duckdb.connect(str(db_path))
