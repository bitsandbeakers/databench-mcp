"""Tests for core/hypothesis.py — YAML-backed hypothesis tracker."""
from __future__ import annotations

import pytest
import yaml

import databench_mcp.workspace as ws
from databench_mcp.core.hypothesis import add_hypothesis, list_hypotheses, update_hypothesis


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")


def test_add_hypothesis_returns_entry():
    entry = add_hypothesis("test-proj", "High opioid rate correlates with cost")
    assert entry["id"] == "h001"
    assert entry["status"] == "proposed"
    assert entry["statement"] == "High opioid rate correlates with cost"
    assert entry["tags"] == []
    assert entry["notes"] == []


def test_add_hypothesis_with_tags():
    entry = add_hypothesis("test-proj", "Tag test", tags=["opioid", "cost"])
    assert entry["tags"] == ["opioid", "cost"]


def test_add_hypothesis_increments_id():
    add_hypothesis("test-proj", "First")
    entry = add_hypothesis("test-proj", "Second")
    assert entry["id"] == "h002"


def test_add_hypothesis_persists_to_yaml():
    add_hypothesis("test-proj", "Persisted statement")
    path = ws.project_path("test-proj") / "hypotheses.yaml"
    assert path.exists()
    data = yaml.safe_load(path.read_text())
    assert len(data) == 1
    assert data[0]["statement"] == "Persisted statement"


def test_list_hypotheses_returns_all():
    add_hypothesis("test-proj", "A")
    add_hypothesis("test-proj", "B")
    result = list_hypotheses("test-proj")
    assert result["count"] == 2
    assert len(result["hypotheses"]) == 2


def test_list_hypotheses_filtered_by_status():
    add_hypothesis("test-proj", "Will be promoted")
    update_hypothesis("test-proj", "h001", status="prioritized")
    add_hypothesis("test-proj", "Still proposed")
    result = list_hypotheses("test-proj", status="proposed")
    assert result["count"] == 1
    assert result["hypotheses"][0]["id"] == "h002"


def test_list_hypotheses_filtered_by_tag():
    add_hypothesis("test-proj", "Tagged", tags=["opioid"])
    add_hypothesis("test-proj", "Untagged")
    result = list_hypotheses("test-proj", tag="opioid")
    assert result["count"] == 1
    assert result["hypotheses"][0]["statement"] == "Tagged"


def test_update_hypothesis_status():
    add_hypothesis("test-proj", "Under test")
    result = update_hypothesis("test-proj", "h001", status="tested")
    assert result["status"] == "tested"
    assert "updated_at" in result


def test_update_hypothesis_appends_note():
    add_hypothesis("test-proj", "Under test")
    result = update_hypothesis("test-proj", "h001", note="p < 0.05 in sample")
    assert len(result["notes"]) == 1
    assert result["notes"][0]["text"] == "p < 0.05 in sample"
    assert "added_at" in result["notes"][0]


def test_update_hypothesis_raises_for_unknown_id():
    with pytest.raises(ValueError, match="not found"):
        update_hypothesis("test-proj", "h999", status="tested")


def test_update_hypothesis_raises_for_invalid_status():
    add_hypothesis("test-proj", "Test")
    with pytest.raises(ValueError, match="Invalid status"):
        update_hypothesis("test-proj", "h001", status="bogus")
