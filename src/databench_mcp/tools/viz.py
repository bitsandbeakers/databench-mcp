"""FastMCP tool wrappers for chart generation."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.viz import create_chart as _create_chart
from databench_mcp.core.viz import create_subplot as _create_subplot


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


def create_subplot(
    project: str,
    charts: list[dict],
    rows: int,
    cols: int,
    title: str | None = None,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
) -> dict[str, Any]:
    """Combine multiple charts into a single subplot grid HTML."""
    return _create_subplot(project, charts, rows, cols, title, shared_xaxes, shared_yaxes)
