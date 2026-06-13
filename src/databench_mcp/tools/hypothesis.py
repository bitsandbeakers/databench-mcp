"""FastMCP tool wrappers for the hypothesis tracker."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.hypothesis import add_hypothesis as _add_hypothesis
from databench_mcp.core.hypothesis import list_hypotheses as _list_hypotheses
from databench_mcp.core.hypothesis import record_evidence as _record_evidence
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


def hypothesis_record_evidence(
    project: str,
    hypothesis_id: str,
    tool_name: str,
    result_summary: str,
    status_update: str | None = None,
) -> dict[str, Any]:
    """Append a structured evidence note from a tool call, optionally updating status.

    Call this immediately after run_model, eda_summary, sql_query, or any other
    tool call whose result bears on a hypothesis. Keeps the tracker current without
    requiring a separate update step.

    Parameters
    ----------
    hypothesis_id  : e.g. 'h005'
    tool_name      : name of the tool (e.g. 'run_model', 'eda_summary')
    result_summary : 1-3 sentence description of what the result showed
    status_update  : optional new status — one of supported, refuted, inconclusive,
                     tested, prioritized, proposed
    """
    return _record_evidence(project, hypothesis_id, tool_name, result_summary, status_update)
