"""Tests for tools/profile.py wrapper."""
import pytest
import databench_mcp.workspace as ws
from databench_mcp.core import ingest as core_ingest
from databench_mcp.tools.profile import profile_table


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")


def test_profile_table_returns_profiled_true(sample_csv):
    core_ingest.load_file("test-proj", sample_csv)
    result = profile_table("test-proj", "providers")
    assert result["profiled"] is True
    assert "profile" in result


def test_profile_table_raises_for_missing_project():
    with pytest.raises(FileNotFoundError):
        profile_table("no-such-project", "any_table")
