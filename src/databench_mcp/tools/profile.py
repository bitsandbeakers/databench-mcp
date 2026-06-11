"""FastMCP tool wrapper for table profiling."""
from __future__ import annotations

from databench_mcp.core import profile as core


def profile_table(project: str, table: str) -> dict:
    """Profile a table: types, null rates, cardinality, min/max/mean.

    Stamps the manifest profiled=True, which is required before running
    any analysis tools (analyze_outliers, analyze_stats_test, etc.).
    """
    return core.profile_table(project, table)
