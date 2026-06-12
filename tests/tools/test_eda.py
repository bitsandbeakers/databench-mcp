"""Tests for tools/eda.py wrappers."""
from __future__ import annotations

import pytest

import databench_mcp.workspace as ws
from databench_mcp.core import ingest as core_ingest
from databench_mcp.core.profile import profile_table as core_profile
from databench_mcp.tools.eda import (
    eda_summary,
    sql_query,
    group_summary,
    clean_table,
    add_lag,
    add_rolling,
    enrich_table,
)


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


def test_group_summary_tool(sample_csv):
    core_ingest.load_file("test-proj", sample_csv)
    core_profile("test-proj", "providers")
    result = group_summary("test-proj", "providers", "specialty", ["total_drug_cost"], ["mean", "count"])
    assert result["row_count"] > 0
    assert result["group_col"] == "specialty"


def test_clean_table_tool(sample_csv):
    core_ingest.load_file("test-proj", sample_csv)
    core_profile("test-proj", "providers")
    result = clean_table("test-proj", "providers", "fill_mean", "providers_clean")
    assert result["table"] == "providers_clean"
    assert result["source_table"] == "providers"


def test_add_lag_tool(sample_csv):
    core_ingest.load_file("test-proj", sample_csv)
    core_profile("test-proj", "providers")
    result = add_lag("test-proj", "providers", "total_drug_cost", [1, 2], "providers_lagged")
    assert result["table"] == "providers_lagged"
    assert len(result["new_cols"]) == 2


def test_add_rolling_tool(sample_csv):
    core_ingest.load_file("test-proj", sample_csv)
    core_profile("test-proj", "providers")
    result = add_rolling("test-proj", "providers", "total_drug_cost", 2, "mean", "providers_rolled")
    assert result["table"] == "providers_rolled"
    assert result["new_col"].endswith("_mean")


def test_enrich_table_tool(sample_csv):
    core_ingest.load_file("test-proj", sample_csv)
    core_profile("test-proj", "providers")
    # Create a second table to join with
    from databench_mcp.db import get_connection
    from databench_mcp.workspace import read_manifest, write_manifest

    with get_connection("test-proj") as conn:
        conn.execute("CREATE OR REPLACE TABLE ref_data AS SELECT DISTINCT specialty FROM providers")
    manifest = read_manifest("test-proj")
    manifest["datasets"]["ref_data"] = {
        "source": "ingest_file",
        "profiled": True,
        "profile": {"specialty": {"type": "VARCHAR", "null_pct": 0.0, "approx_unique": 3}},
        "row_count": 3,
    }
    write_manifest("test-proj", manifest)
    result = enrich_table("test-proj", "providers", "ref_data", "specialty", "providers_enriched")
    assert result["rows"] > 0
    assert result["columns"] > 0
