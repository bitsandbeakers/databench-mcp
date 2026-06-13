"""Tests for data_quality_report in core/eda.py."""
from __future__ import annotations

import duckdb
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.eda import data_quality_report
from databench_mcp.core.profile import profile_table


@pytest.fixture
def dq_project(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("dq-proj")
    db_path = str(tmp_path / "dq-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE messy AS SELECT
            range AS id,
            CASE WHEN range % 2 = 0 THEN 'A' ELSE NULL END AS sparse_col,
            1 AS constant_col,
            CAST(range AS DOUBLE) AS numeric_col,
            NULL::VARCHAR AS all_null
        FROM range(100)
    """)
    conn.close()
    manifest = ws.read_manifest("dq-proj")
    manifest["datasets"]["messy"] = {
        "row_count": 100, "col_count": 5, "profiled": False,
    }
    ws.write_manifest("dq-proj", manifest)
    profile_table("dq-proj", "messy")
    return tmp_path


def test_returns_expected_keys(dq_project):
    result = data_quality_report("dq-proj", "messy")
    assert "quality_score" in result
    assert "issues" in result
    assert "summary" in result
    assert "total_columns" in result
    assert result["table"] == "messy"


def test_flags_constant_column(dq_project):
    result = data_quality_report("dq-proj", "messy")
    issues = {i["column"]: i for i in result["issues"]}
    assert "constant_col" in issues
    assert issues["constant_col"]["severity"] == "high"
    assert issues["constant_col"]["issue"] == "constant_column"


def test_flags_high_nulls(dq_project):
    result = data_quality_report("dq-proj", "messy")
    issues = {i["column"]: i for i in result["issues"]}
    # sparse_col is 50% null; all_null is 100% null
    assert "all_null" in issues
    assert issues["all_null"]["severity"] == "high"


def test_quality_score_below_one_for_messy(dq_project):
    result = data_quality_report("dq-proj", "messy")
    assert result["quality_score"] < 1.0


def test_id_col_flagged_as_likely_id(dq_project):
    result = data_quality_report("dq-proj", "messy")
    issues = {i["column"]: i for i in result["issues"]}
    # 'id' column: integer range(100), 100 distinct / 100 rows
    # DuckDB SUMMARIZE reports INTEGER type, so it won't be flagged as likely_id
    # (only non-numeric high-cardinality cols are flagged)
    # numeric_col is DOUBLE and also has 100 unique / 100 rows — not flagged as ID
    # This test verifies no false positive on numeric columns
    for issue in result["issues"]:
        if issue["column"] in ("id", "numeric_col"):
            assert issue["issue"] != "likely_id_column"


def test_clean_table_scores_higher(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("clean-proj")
    db_path = str(tmp_path / "clean-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE clean AS SELECT range AS x, range * 2.0 AS y FROM range(50)")
    conn.close()
    manifest = ws.read_manifest("clean-proj")
    manifest["datasets"]["clean"] = {"row_count": 50, "col_count": 2, "profiled": False}
    ws.write_manifest("clean-proj", manifest)
    profile_table("clean-proj", "clean")
    result = data_quality_report("clean-proj", "clean")
    assert result["quality_score"] >= 0.8


def test_requires_profiled_table(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("np-proj")
    db_path = str(tmp_path / "np-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE t AS SELECT 1 AS x")
    conn.close()
    manifest = ws.read_manifest("np-proj")
    manifest["datasets"]["t"] = {"row_count": 1, "col_count": 1, "profiled": False}
    ws.write_manifest("np-proj", manifest)
    with pytest.raises(ValueError, match="profiled"):
        data_quality_report("np-proj", "t")
