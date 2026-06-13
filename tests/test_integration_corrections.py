"""Integration tests for corrections tool wrappers."""
from __future__ import annotations

import pytest

import databench_mcp.workspace as ws
from databench_mcp.tools.corrections import list_corrections, log_correction


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("integ-proj")
    return "integ-proj"


def test_corrections_round_trip(proj):
    # 1. Log first correction — normal, no gap
    first = log_correction(
        proj,
        ai_action="Aggregated to provider level before EDA",
        correction="Must analyze at provider×DRG grain first",
        category="wrong_grain",
    )
    assert first["id"] == "c001"
    assert first["category"] == "wrong_grain"
    assert first["databench_gap"] is False
    assert first["gap_description"] is None

    # 2. Log second correction — different category, marks a databench gap
    second = log_correction(
        proj,
        ai_action="Used future data in feature engineering",
        correction="Features must only use data available before the label date",
        category="data_leakage",
        databench_gap=True,
        gap_description="databench should enforce temporal feature cutoffs automatically",
    )
    assert second["id"] == "c002"
    assert second["category"] == "data_leakage"
    assert second["databench_gap"] is True
    assert second["gap_description"] == "databench should enforce temporal feature cutoffs automatically"

    # 3. list_corrections returns both
    all_result = list_corrections(proj)
    assert all_result["count"] == 2
    ids = [c["id"] for c in all_result["corrections"]]
    assert "c001" in ids
    assert "c002" in ids

    # 4. databench_gaps_only=True returns only second
    gaps_result = list_corrections(proj, databench_gaps_only=True)
    assert gaps_result["count"] == 1
    assert gaps_result["corrections"][0]["id"] == "c002"
