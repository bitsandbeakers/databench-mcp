"""FastMCP tool wrappers for modeling and findings functions."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.findings import list_findings as _list_findings
from databench_mcp.core.modeling import run_model as _run_model


def run_model(
    project: str,
    table: str,
    method: str,
    target: str | None = None,
    features: list[str] | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    """Run a registered analysis method, save the finding, return the finding entry."""
    return _run_model(project, table, method, target, features, params)


def list_findings(
    project: str,
    method: str | None = None,
) -> dict[str, Any]:
    """List all findings for a project, optionally filtered by method."""
    return _list_findings(project, method)
