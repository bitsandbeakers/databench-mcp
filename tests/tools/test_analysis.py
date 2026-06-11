"""Integration tests for tools/analysis.py wrappers."""
from __future__ import annotations

from databench_mcp.tools.analysis import analyze_correlations, analyze_distribution, detect_outliers


def test_detect_outliers_tool(project_with_data):
    result = detect_outliers("test-proj", "providers", "total_drug_cost")
    assert "outlier_count" in result
    assert result["method"] == "iqr"


def test_analyze_distribution_tool(project_with_data):
    result = analyze_distribution("test-proj", "providers", "total_drug_cost")
    assert "verdict" in result


def test_analyze_correlations_tool(project_with_data):
    result = analyze_correlations("test-proj", "providers",
                                  columns=["total_drug_cost", "claim_count"])
    assert "top_pairs" in result
