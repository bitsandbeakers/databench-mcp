"""FastMCP tool wrapper for dashboard generation."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.dashboard import build_dashboard as _build


def build_dashboard(project: str) -> dict[str, Any]:
    """Generate a standalone Dash app from the project's chart artifacts and findings."""
    return _build(project)
