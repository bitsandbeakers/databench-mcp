"""Tests for tools/hypothesis.py wrappers."""
from __future__ import annotations

import pytest

import databench_mcp.workspace as ws
from databench_mcp.tools.hypothesis import hypothesis_add, hypothesis_list, hypothesis_update


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")


def test_hypothesis_add_returns_entry():
    result = hypothesis_add("test-proj", "Opioid prescribers have higher costs")
    assert result["id"] == "h001"
    assert result["status"] == "proposed"


def test_hypothesis_list_returns_count():
    hypothesis_add("test-proj", "A")
    hypothesis_add("test-proj", "B")
    result = hypothesis_list("test-proj")
    assert result["count"] == 2


def test_hypothesis_update_changes_status():
    hypothesis_add("test-proj", "Test")
    result = hypothesis_update("test-proj", "h001", status="tested")
    assert result["status"] == "tested"
