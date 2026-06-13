"""FastMCP tool wrappers for correction-tracking functions."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.corrections import list_corrections as _list_corrections
from databench_mcp.core.corrections import log_correction as _log_correction


def log_correction(
    project: str,
    ai_action: str,
    correction: str,
    category: str,
    databench_gap: bool = False,
    gap_description: str | None = None,
) -> dict[str, Any]:
    """Record an analyst correction to the AI's approach. Call immediately when the human redirects the analysis. category must be one of: data_leakage, wrong_grain, endogeneity, domain_methodology, statistical_error, modeling_discipline, data_quality, other. Set databench_gap=True when the correction reveals a missing safety check that databench should enforce automatically."""
    return _log_correction(project, ai_action, correction, category, databench_gap, gap_description)


def list_corrections(
    project: str,
    category: str | None = None,
    databench_gaps_only: bool = False,
) -> dict[str, Any]:
    """List corrections logged for this project. Optionally filter by category or set databench_gaps_only=True to see only corrections that represent databench product gaps."""
    return _list_corrections(project, category, databench_gaps_only)
