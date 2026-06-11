"""Tests for db.py — DuckDB connection helper."""
import pytest
import databench_mcp.workspace as ws
from databench_mcp.db import get_connection


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")


def test_get_connection_returns_working_connection(tmp_path):
    with get_connection("test-proj") as conn:
        result = conn.execute("SELECT 42").fetchone()
    assert result[0] == 42


def test_get_connection_creates_duckdb_file(tmp_path):
    with get_connection("test-proj") as conn:
        conn.execute("CREATE TABLE t (x INT)")
        conn.execute("INSERT INTO t VALUES (99)")
    assert (tmp_path / "test-proj" / "project.duckdb").exists()


def test_get_connection_persists_across_connections(tmp_path):
    with get_connection("test-proj") as conn:
        conn.execute("CREATE TABLE t (x INT)")
        conn.execute("INSERT INTO t VALUES (7)")
    with get_connection("test-proj") as conn:
        result = conn.execute("SELECT x FROM t").fetchone()
    assert result[0] == 7
