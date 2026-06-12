"""Integration tests for tools/viz.py wrapper."""
from __future__ import annotations

from pathlib import Path

from databench_mcp.tools.viz import create_chart, create_subplot


def test_create_chart_tool_histogram(project_with_data):
    result = create_chart("test-proj", "histogram", "providers",
                          columns=["total_drug_cost"])
    assert Path(result["path"]).exists()


def test_create_chart_tool_scatter(project_with_data):
    result = create_chart("test-proj", "scatter", "providers",
                          columns=["claim_count", "total_drug_cost"])
    assert Path(result["path"]).exists()


def test_create_subplot_basic(project_with_data):
    charts = [
        {"chart_type": "histogram", "table": "providers",
         "columns": ["total_drug_cost"], "params": {}},
        {"chart_type": "scatter", "table": "providers",
         "columns": ["claim_count", "total_drug_cost"], "params": {}},
    ]
    result = create_subplot("test-proj", charts, rows=1, cols=2, title="Test subplot")
    assert Path(result["path"]).exists()
    assert result["charts_count"] == 2
    assert result["rows"] == 1
    assert result["cols"] == 2


def test_create_subplot_overflow(project_with_data):
    import pytest
    charts = [
        {"chart_type": "histogram", "table": "providers",
         "columns": ["total_drug_cost"], "params": {}},
        {"chart_type": "histogram", "table": "providers",
         "columns": ["claim_count"], "params": {}},
        {"chart_type": "scatter", "table": "providers",
         "columns": ["claim_count", "total_drug_cost"], "params": {}},
    ]
    with pytest.raises(ValueError, match="do not fit"):
        create_subplot("test-proj", charts, rows=1, cols=2)
