"""FastMCP tool wrappers for data ingestion."""
from __future__ import annotations

from databench_mcp.core import ingest as core


def ingest_file(project: str, file_path: str, table_name: str | None = None) -> dict:
    """Load a local CSV, TSV, Parquet, or Excel file into the project database.

    Registers the table in the project manifest with profiled=False.
    Run profile_table next to unlock analysis tools.
    """
    return core.load_file(project, file_path, table_name)


def ingest_url(
    project: str,
    url: str,
    table_name: str,
    params: dict[str, str] | None = None,
) -> dict:
    """Download a URL and load it into the project database.

    Supports CMS data.cms.gov Socrata API — pass params for $limit, $where filters.
    Raw file is saved to raw/<table_name>.<ext> before loading.
    """
    return core.load_url(project, url, table_name, params)
