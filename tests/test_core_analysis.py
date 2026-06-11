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


from databench_mcp.core.analysis import analyze_distribution


def test_analyze_distribution_returns_shape_stats(project_with_data):
    result = analyze_distribution("test-proj", "providers", "total_drug_cost")
    assert result["column"] == "total_drug_cost"
    assert "mean" in result
    assert "median" in result
    assert "skewness" in result
    assert "kurtosis" in result
    assert "percentiles" in result
    assert "verdict" in result
    assert result["verdict"] in (
        "approximately normal", "right-skewed", "left-skewed", "heavy-tailed"
    )


def test_analyze_distribution_includes_normality_test(project_with_data):
    result = analyze_distribution("test-proj", "providers", "total_drug_cost")
    assert "normality_stat" in result
    assert "normality_p" in result
    assert "normality_test" in result


def test_analyze_distribution_requires_profiled(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    with pytest.raises(ValueError, match="profiled"):
        analyze_distribution("test-proj", "providers", "total_drug_cost")


from databench_mcp.core.analysis import analyze_correlations


def test_analyze_correlations_pearson(project_with_data):
    result = analyze_correlations("test-proj", "providers",
                                  columns=["total_drug_cost", "claim_count"])
    assert result["method"] == "pearson"
    assert "matrix" in result
    assert "top_pairs" in result
    assert len(result["top_pairs"]) >= 1
    pair = result["top_pairs"][0]
    assert "col_a" in pair and "col_b" in pair and "r" in pair


def test_analyze_correlations_spearman(project_with_data):
    result = analyze_correlations("test-proj", "providers",
                                  columns=["total_drug_cost", "claim_count"],
                                  method="spearman")
    assert result["method"] == "spearman"


def test_analyze_correlations_defaults_to_all_numeric(project_with_data):
    result = analyze_correlations("test-proj", "providers")
    assert len(result["matrix"]) >= 2


def test_analyze_correlations_requires_profiled(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    with pytest.raises(ValueError, match="profiled"):
        analyze_correlations("test-proj", "providers")
