"""Tests for compare_findings in core/findings.py."""
from __future__ import annotations

import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.findings import add_finding, compare_findings


@pytest.fixture
def proj_with_findings(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("cmp-proj")
    add_finding("cmp-proj", {"method": "lasso", "table": "t", "target": "y",
                              "metrics": {"r2": 0.376, "rmse": 1.2, "mae": 0.9}})
    add_finding("cmp-proj", {"method": "ebm", "table": "t", "target": "y",
                              "metrics": {"r2": 0.697, "rmse": 0.8, "mae": 0.6}})
    add_finding("cmp-proj", {"method": "lightgbm", "table": "t", "target": "y",
                              "metrics": {"r2": 0.737, "rmse": 0.75, "mae": 0.55}})
    return tmp_path


def test_compare_returns_all_requested(proj_with_findings):
    result = compare_findings("cmp-proj", ["f001", "f002", "f003"])
    assert result["finding_count"] == 3
    assert len(result["findings"]) == 3


def test_compare_filters_metrics(proj_with_findings):
    result = compare_findings("cmp-proj", ["f001", "f002"], metrics=["r2", "rmse"])
    for f in result["findings"]:
        assert set(f["metrics"].keys()) == {"r2", "rmse"}


def test_compare_ranks_by_first_metric(proj_with_findings):
    result = compare_findings("cmp-proj", ["f001", "f002", "f003"], metrics=["r2"])
    r2_values = [f["metrics"]["r2"] for f in result["findings"]]
    assert r2_values == sorted(r2_values, reverse=True)


def test_compare_rank_by_explicit(proj_with_findings):
    result = compare_findings("cmp-proj", ["f001", "f002", "f003"],
                              metrics=["r2", "rmse"], rank_by="rmse")
    rmse_values = [f["metrics"]["rmse"] for f in result["findings"]]
    assert rmse_values == sorted(rmse_values, reverse=True)
    assert result["ranked_by"] == "rmse"


def test_compare_missing_id_raises(proj_with_findings):
    with pytest.raises(ValueError, match="not found"):
        compare_findings("cmp-proj", ["f001", "f999"])


def test_compare_no_metrics_filter_returns_all(proj_with_findings):
    result = compare_findings("cmp-proj", ["f001"], metrics=None)
    assert "r2" in result["findings"][0]["metrics"]
    assert "rmse" in result["findings"][0]["metrics"]


def test_compare_returns_method_and_target(proj_with_findings):
    result = compare_findings("cmp-proj", ["f002"])
    f = result["findings"][0]
    assert f["method"] == "ebm"
    assert f["target"] == "y"
