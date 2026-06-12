"""Tests for core/eda.py — derive_table."""
from __future__ import annotations

import duckdb
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.eda import derive_table


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("p")
    db_path = str(tmp_path / "p" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE edges AS SELECT 'a' AS src, 'b' AS tgt, 1.0 AS w UNION ALL SELECT 'b', 'c', 2.0")
    conn.close()
    manifest = ws.read_manifest("p")
    manifest["datasets"]["edges"] = {"profiled": True, "row_count": 2, "col_count": 3}
    ws.write_manifest("p", manifest)
    return tmp_path


def test_derive_table_creates_table(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = derive_table("p", "SELECT src, tgt FROM edges WHERE w > 1.0", "filtered_edges")
    assert result["table"] == "filtered_edges"
    assert result["rows"] == 1
    assert result["columns"] == 2


def test_derive_table_registers_in_manifest(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    derive_table("p", "SELECT * FROM edges", "copy_edges")
    manifest = ws.read_manifest("p")
    ds = manifest["datasets"]["copy_edges"]
    assert ds["source"] == "derived"
    assert ds["profiled"] is False
    assert ds["row_count"] == 2


def test_derive_table_overwrites_existing(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    derive_table("p", "SELECT * FROM edges", "copy_edges")
    result = derive_table("p", "SELECT src FROM edges", "copy_edges")
    assert result["columns"] == 1


def test_derive_table_rejects_insert(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="Only SELECT"):
        derive_table("p", "INSERT INTO edges VALUES ('x','y',0.0)", "bad")


def test_derive_table_rejects_multi_statement(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="Multi-statement"):
        derive_table("p", "SELECT 1; SELECT 2", "bad")


def test_derive_table_rejects_invalid_table_name(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="simple identifier"):
        derive_table("p", "SELECT * FROM edges", 'bad"name')
