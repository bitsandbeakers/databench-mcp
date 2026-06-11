"""Tests for tools/eda.py wrappers."""
from __future__ import annotations

import pytest

import databench_mcp.workspace as ws
from databench_mcp.core import ingest as core_ingest
from databench_mcp.tools.eda import eda_summary, sql_query


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")


def test_sql_query_returns_rows(sample_csv):
    core_ingest.load_file("test-proj", sample_csv)
    result = sql_query("test-proj", "SELECT * FROM providers")
    assert result["row_count"] == 3


def test_eda_summary_returns_project():
    result = eda_summary("test-proj")
    assert result["project"] == "test-proj"
