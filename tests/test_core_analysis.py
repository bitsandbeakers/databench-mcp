"""Tests for core/analysis.py."""
from __future__ import annotations

import pytest
import databench_mcp.workspace as ws
from databench_mcp.core.analysis import detect_outliers


def test_detect_outliers_iqr(project_with_data):
    result = detect_outliers("test-proj", "providers", "total_drug_cost", method="iqr")
    assert result["method"] == "iqr"
    assert "outlier_count" in result
    assert "outlier_pct" in result
    assert 0 <= result["outlier_pct"] <= 100
    assert isinstance(result["sample_outliers"], list)


def test_detect_outliers_zscore(project_with_data):
    result = detect_outliers("test-proj", "providers", "total_drug_cost", method="zscore")
    assert result["method"] == "zscore"
    assert "threshold" in result


def test_detect_outliers_isolation_forest(project_with_data):
    result = detect_outliers("test-proj", "providers", "total_drug_cost",
                             method="isolation_forest")
    assert result["method"] == "isolation_forest"
    assert result["outlier_count"] >= 0


def test_detect_outliers_unknown_method(project_with_data):
    with pytest.raises(ValueError, match="Unknown outlier method"):
        detect_outliers("test-proj", "providers", "total_drug_cost", method="bogus")


def test_detect_outliers_requires_profiled(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    with pytest.raises(ValueError, match="profiled"):
        detect_outliers("test-proj", "providers", "total_drug_cost")
