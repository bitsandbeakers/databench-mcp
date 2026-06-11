"""FastMCP tool wrapper for chart generation."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.viz import create_chart as _create_chart


def create_chart(
    project: str,
    chart_type: str,
    table: str,
    columns: list[str],
    finding_id: str | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    """Generate a Plotly HTML chart and save it to the project charts directory."""
    return _create_chart(project, chart_type, table, columns, finding_id, params)
