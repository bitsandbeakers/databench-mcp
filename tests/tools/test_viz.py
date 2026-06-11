"""Integration tests for tools/viz.py wrapper."""
from __future__ import annotations

from pathlib import Path

from databench_mcp.tools.viz import create_chart


def test_create_chart_tool_histogram(project_with_data):
    result = create_chart("test-proj", "histogram", "providers",
                          columns=["total_drug_cost"])
    assert Path(result["path"]).exists()


def test_create_chart_tool_scatter(project_with_data):
    result = create_chart("test-proj", "scatter", "providers",
                          columns=["claim_count", "total_drug_cost"])
    assert Path(result["path"]).exists()
