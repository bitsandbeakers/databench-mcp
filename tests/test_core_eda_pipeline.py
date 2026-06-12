"""Tests for core/eda.py — pipeline gap functions."""
from __future__ import annotations

import duckdb
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.eda import group_summary


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("p")
    db_path = str(tmp_path / "p" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE sales AS
        SELECT 'A' AS region, 100.0 AS revenue, 10 AS units UNION ALL
        SELECT 'A', 200.0, 20 UNION ALL
        SELECT 'B', 150.0, 15 UNION ALL
        SELECT 'B', NULL,  5
    """)
    conn.close()
    manifest = ws.read_manifest("p")
    manifest["datasets"]["sales"] = {
        "profiled": True, "row_count": 4, "col_count": 3,
        "profile": {
            "region":  {"type": "VARCHAR",  "null_pct": 0,  "approx_unique": 2},
            "revenue": {"type": "DOUBLE",   "null_pct": 25, "approx_unique": 3},
            "units":   {"type": "INTEGER",  "null_pct": 0,  "approx_unique": 4},
        },
    }
    ws.write_manifest("p", manifest)
    return tmp_path


def test_group_summary_basic(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = group_summary("p", "sales", "region", ["revenue"])
    assert result["group_col"] == "region"
    assert result["row_count"] == 2
    assert any("revenue_mean" in row for row in result["rows"])


def test_group_summary_multiple_agg_fns(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = group_summary("p", "sales", "region", ["revenue"], ["mean", "count"])
    row_a = next(r for r in result["rows"] if r["region"] == "A")
    assert row_a["revenue_mean"] == 150.0
    assert row_a["revenue_count"] == 2


def test_group_summary_unknown_agg_fn(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="unknown agg_fn"):
        group_summary("p", "sales", "region", ["revenue"], ["median"])


def test_group_summary_empty_agg_cols(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="agg_cols must not be empty"):
        group_summary("p", "sales", "region", [])


def test_group_summary_unproflied_table(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("up")
    import duckdb
    db_path = str(tmp_path / "up" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE x AS SELECT 1 AS a, 2 AS b")
    conn.close()
    # No profile entry in manifest — assert_profiled should raise
    with pytest.raises(ValueError):
        group_summary("up", "x", "a", ["b"])


def test_group_summary_default_agg_fns_all_present(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = group_summary("p", "sales", "region", ["revenue"])
    row_a = next(r for r in result["rows"] if r["region"] == "A")
    # All 5 default aggregates should be present
    for fn in ["mean", "count", "min", "max", "std"]:
        assert f"revenue_{fn}" in row_a, f"missing revenue_{fn}"
