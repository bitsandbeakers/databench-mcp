"""Tests for core/findings.py — YAML-backed findings tracker."""
from __future__ import annotations

import yaml
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.findings import add_finding, get_finding, list_findings


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")


def test_add_finding_assigns_id():
    entry = add_finding("test-proj", {
        "method": "linear_regression",
        "target": "cost",
        "features": ["a", "b"],
        "metrics": {"r2": 0.8},
        "explainability": "high",
        "summary": "Good fit.",
    })
    assert entry["id"] == "f001"
    assert entry["method"] == "linear_regression"
    assert "created_at" in entry


def test_add_finding_increments_id():
    add_finding("test-proj", {"method": "m1", "target": None, "features": [],
                               "metrics": {}, "explainability": "high", "summary": "."})
    entry = add_finding("test-proj", {"method": "m2", "target": None, "features": [],
                                       "metrics": {}, "explainability": "low", "summary": "."})
    assert entry["id"] == "f002"


def test_add_finding_persists_to_yaml():
    add_finding("test-proj", {"method": "kmeans", "target": None, "features": ["x"],
                               "metrics": {"k": 3}, "explainability": "low", "summary": "."})
    path = ws.project_path("test-proj") / "findings.yaml"
    assert path.exists()
    data = yaml.safe_load(path.read_text())
    assert data[0]["method"] == "kmeans"


def test_get_finding_returns_entry():
    add_finding("test-proj", {"method": "rf", "target": "y", "features": ["x"],
                               "metrics": {}, "explainability": "medium", "summary": "."})
    entry = get_finding("test-proj", "f001")
    assert entry["method"] == "rf"


def test_get_finding_raises_for_unknown_id():
    with pytest.raises(ValueError, match="not found"):
        get_finding("test-proj", "f999")


def test_list_findings_returns_all():
    add_finding("test-proj", {"method": "a", "target": None, "features": [],
                               "metrics": {}, "explainability": "high", "summary": "."})
    add_finding("test-proj", {"method": "b", "target": None, "features": [],
                               "metrics": {}, "explainability": "low", "summary": "."})
    result = list_findings("test-proj")
    assert result["count"] == 2
    assert len(result["findings"]) == 2


def test_list_findings_filtered_by_method():
    add_finding("test-proj", {"method": "random_forest", "target": "y", "features": [],
                               "metrics": {}, "explainability": "medium", "summary": "."})
    add_finding("test-proj", {"method": "lasso", "target": "y", "features": [],
                               "metrics": {}, "explainability": "high", "summary": "."})
    result = list_findings("test-proj", method="lasso")
    assert result["count"] == 1
    assert result["findings"][0]["method"] == "lasso"
