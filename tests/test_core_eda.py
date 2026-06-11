"""Tests for core/eda.py — sql_query and eda_summary."""
from __future__ import annotations

import pytest

import databench_mcp.workspace as ws
from databench_mcp.core import ingest as core_ingest
from databench_mcp.core.eda import sql_query


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")


@pytest.fixture
def loaded_table(sample_csv):
    core_ingest.load_file("test-proj", sample_csv)
    return "providers"


# --- sql_query ---


def test_sql_query_basic(loaded_table):
    result = sql_query("test-proj", "SELECT * FROM providers")
    assert result["row_count"] == 3
    assert "npi" in result["columns"]
    assert result["truncated"] is False


def test_sql_query_filter(loaded_table):
    result = sql_query("test-proj", "SELECT npi FROM providers WHERE specialty = 'Cardiology'")
    assert result["row_count"] == 1
    assert result["rows"][0]["npi"] == 1234567890


def test_sql_query_with_cte(loaded_table):
    result = sql_query(
        "test-proj",
        "WITH cte AS (SELECT * FROM providers) SELECT * FROM cte",
    )
    assert result["row_count"] == 3


def test_sql_query_limit_truncation(loaded_table):
    result = sql_query("test-proj", "SELECT * FROM providers", limit=2)
    assert result["row_count"] == 2
    assert result["truncated"] is True


def test_sql_query_rejects_non_select(loaded_table):
    with pytest.raises(ValueError, match="Only SELECT"):
        sql_query("test-proj", "DELETE FROM providers")


def test_sql_query_rejects_semicolon(loaded_table):
    with pytest.raises(ValueError, match="Multi-statement"):
        sql_query("test-proj", "SELECT 1; DROP TABLE providers")
