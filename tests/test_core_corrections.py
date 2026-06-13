"""Tests for log_correction and list_corrections in core/corrections.py."""
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.corrections import log_correction, list_corrections


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    return "test-proj"


def test_log_correction_returns_entry(proj):
    result = log_correction(
        proj,
        ai_action="Aggregated to provider level before EDA",
        correction="Must analyze at provider×DRG grain first",
        category="wrong_grain",
    )
    assert result["id"] == "c001"
    assert result["category"] == "wrong_grain"
    assert result["ai_action"] == "Aggregated to provider level before EDA"
    assert result["correction"] == "Must analyze at provider×DRG grain first"
    assert result["databench_gap"] is False
    assert result["gap_description"] is None
    assert "logged_at" in result


def test_log_correction_id_increments(proj):
    first = log_correction(proj, "action1", "fix1", "other")
    second = log_correction(proj, "action2", "fix2", "data_leakage")
    assert first["id"] == "c001"
    assert second["id"] == "c002"


def test_log_correction_persists(proj):
    log_correction(proj, "used future data", "use only past data", "data_leakage")
    result = list_corrections(proj)
    assert result["count"] == 1
    assert result["corrections"][0]["category"] == "data_leakage"


def test_list_corrections_empty(proj):
    result = list_corrections(proj)
    assert result["count"] == 0
    assert result["corrections"] == []
    assert result["project"] == proj


def test_list_corrections_filter_category(proj):
    log_correction(proj, "action1", "fix1", "wrong_grain")
    log_correction(proj, "action2", "fix2", "statistical_error")
    result = list_corrections(proj, category="wrong_grain")
    assert result["count"] == 1
    assert result["corrections"][0]["category"] == "wrong_grain"


def test_list_corrections_databench_gaps_only(proj):
    log_correction(proj, "action1", "fix1", "other", databench_gap=True, gap_description="missing tool")
    log_correction(proj, "action2", "fix2", "other", databench_gap=False)
    result = list_corrections(proj, databench_gaps_only=True)
    assert result["count"] == 1
    assert result["corrections"][0]["databench_gap"] is True


def test_invalid_category_raises(proj):
    with pytest.raises(ValueError, match="Invalid category"):
        log_correction(proj, "action", "fix", "not_a_real_category")


def test_requires_project_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        log_correction("ghost-project", "action", "fix", "other")


def test_gap_description_requires_databench_gap(proj):
    with pytest.raises(ValueError, match="gap_description requires databench_gap=True"):
        log_correction(
            proj,
            ai_action="some action",
            correction="some fix",
            category="other",
            databench_gap=False,
            gap_description="should not be allowed",
        )
