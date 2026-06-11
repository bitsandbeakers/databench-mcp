"""Tests for core/profile.py — profile_table."""
import pytest
import databench_mcp.workspace as ws
from databench_mcp.core import ingest as core_ingest
from databench_mcp.core import profile as core_profile


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")


@pytest.fixture
def loaded_table(sample_csv):
    core_ingest.load_file("test-proj", sample_csv)
    return "providers"


def test_profile_table_returns_column_profiles(loaded_table):
    result = core_profile.profile_table("test-proj", loaded_table)
    assert result["table"] == loaded_table
    assert result["columns"] == 4
    assert "profile" in result
    assert "npi" in result["profile"]


def test_profile_table_stamps_manifest_profiled(loaded_table):
    core_profile.profile_table("test-proj", loaded_table)
    manifest = ws.read_manifest("test-proj")
    assert manifest["datasets"][loaded_table]["profiled"] is True
    assert manifest["datasets"][loaded_table]["profiled_at"] is not None


def test_profile_table_column_has_type_and_nulls(loaded_table):
    result = core_profile.profile_table("test-proj", loaded_table)
    col = result["profile"]["npi"]
    assert "type" in col
    assert "null_pct" in col


def test_profile_table_unlocks_assert_profiled(loaded_table):
    core_profile.profile_table("test-proj", loaded_table)
    ws.assert_profiled("test-proj", loaded_table)  # must not raise


def test_profile_table_raises_for_unknown_table():
    with pytest.raises(ValueError, match="not in manifest"):
        core_profile.profile_table("test-proj", "ghost_table")


def test_profile_table_is_idempotent(loaded_table):
    core_profile.profile_table("test-proj", loaded_table)
    result = core_profile.profile_table("test-proj", loaded_table)
    assert result["table"] == loaded_table
