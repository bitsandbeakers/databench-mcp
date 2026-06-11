"""Integration tests for tools/modeling.py wrappers."""
from __future__ import annotations

from databench_mcp.tools.modeling import list_findings, run_model


def test_run_model_tool(project_with_data):
    result = run_model("test-proj", "providers", "lasso",
                       target="total_drug_cost", features=["claim_count"])
    assert "finding_id" in result
    assert result["method"] == "lasso"


def test_list_findings_tool(project_with_data):
    run_model("test-proj", "providers", "ridge",
              target="total_drug_cost", features=["claim_count"])
    result = list_findings("test-proj")
    assert result["count"] == 1
