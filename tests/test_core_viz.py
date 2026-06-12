"""Tests for core/viz.py — Plotly HTML chart generation."""
from __future__ import annotations

import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.viz import create_chart


def test_create_chart_histogram(project_with_data):
    result = create_chart("test-proj", "histogram", "providers",
                          columns=["total_drug_cost"])
    assert result["chart_type"] == "histogram"
    assert result["path"].endswith(".html")
    from pathlib import Path
    assert Path(result["path"]).exists()
    html = Path(result["path"]).read_text(encoding="utf-8")
    assert "plotly" in html.lower()


def test_create_chart_scatter(project_with_data):
    result = create_chart("test-proj", "scatter", "providers",
                          columns=["claim_count", "total_drug_cost"])
    assert result["chart_type"] == "scatter"
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_correlation_heatmap(project_with_data):
    result = create_chart("test-proj", "correlation_heatmap", "providers",
                          columns=["total_drug_cost", "claim_count"])
    assert result["chart_type"] == "correlation_heatmap"
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_feature_importance_bar(project_with_data):
    from databench_mcp.core.modeling import run_model
    finding = run_model("test-proj", "providers", "random_forest",
                        target="total_drug_cost", features=["claim_count"])
    # NOTE: run_model returns finding_id (not id) — use finding["finding_id"]
    result = create_chart("test-proj", "feature_importance_bar", "providers",
                          columns=[], finding_id=finding["finding_id"])
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_unknown_type(project_with_data):
    with pytest.raises(ValueError, match="Unknown chart type"):
        create_chart("test-proj", "magic_chart", "providers", columns=["total_drug_cost"])


def test_create_chart_requires_profiled(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    with pytest.raises(ValueError, match="profiled"):
        create_chart("test-proj", "histogram", "providers", columns=["x"])


def test_create_chart_saves_sidecar_json(project_with_data):
    import json
    from pathlib import Path
    result = create_chart("test-proj", "scatter", "providers",
                          columns=["claim_count", "total_drug_cost"])
    html_path = Path(result["path"])
    sidecar = html_path.parent / (html_path.stem + "_params.json")
    assert sidecar.exists(), "sidecar params JSON should be saved alongside HTML"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["chart_type"] == "scatter"
    assert data["table"] == "providers"
    assert data["columns"] == ["claim_count", "total_drug_cost"]
    assert data["finding_id"] is None
    assert "params" in data
