"""FastMCP tool wrappers for the hypothesis tracker."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.hypothesis import add_hypothesis as _add_hypothesis
from databench_mcp.core.hypothesis import list_hypotheses as _list_hypotheses
from databench_mcp.core.hypothesis import update_hypothesis as _update_hypothesis


def hypothesis_add(
    project: str,
    statement: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Add a new hypothesis to the project with status 'proposed'."""
    return _add_hypothesis(project, statement, tags)


def hypothesis_list(
    project: str,
    status: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """List hypotheses, optionally filtered by status or tag."""
    return _list_hypotheses(project, status, tag)


def hypothesis_update(
    project: str,
    hypothesis_id: str,
    status: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Update a hypothesis status and/or append a note."""
    return _update_hypothesis(project, hypothesis_id, status, note)
