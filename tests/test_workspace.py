"""Tests for workspace.py — project dirs and manifest CRUD."""
import json
import pytest
import databench_mcp.workspace as ws


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)


def test_ensure_project_creates_subdirs(tmp_path):
    ws.ensure_project("test-proj")
    for subdir in ("raw", "artifacts", "recipes", "reports"):
        assert (tmp_path / "test-proj" / subdir).is_dir()


def test_ensure_project_creates_manifest(tmp_path):
    ws.ensure_project("test-proj")
    manifest_path = tmp_path / "test-proj" / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["project"] == "test-proj"
    assert "created_at" in data
    assert data["datasets"] == {}


def test_ensure_project_is_idempotent(tmp_path):
    ws.ensure_project("test-proj")
    first = json.loads((tmp_path / "test-proj" / "manifest.json").read_text())
    ws.ensure_project("test-proj")  # must not overwrite
    second = json.loads((tmp_path / "test-proj" / "manifest.json").read_text())
    assert first["created_at"] == second["created_at"]


def test_read_manifest_raises_for_missing_project():
    with pytest.raises(FileNotFoundError, match="not found"):
        ws.read_manifest("ghost")


def test_write_and_read_manifest_roundtrip(tmp_path):
    ws.ensure_project("test-proj")
    manifest = ws.read_manifest("test-proj")
    manifest["datasets"]["tbl"] = {"profiled": False, "source": "x.csv"}
    ws.write_manifest("test-proj", manifest)
    reloaded = ws.read_manifest("test-proj")
    assert reloaded["datasets"]["tbl"]["profiled"] is False


def test_write_manifest_is_atomic(tmp_path):
    ws.ensure_project("test-proj")
    manifest = ws.read_manifest("test-proj")
    manifest["datasets"]["tbl"] = {"profiled": True}
    ws.write_manifest("test-proj", manifest)
    # tmp file must be cleaned up
    assert not (tmp_path / "test-proj" / "manifest.tmp").exists()
    # data is intact
    assert ws.read_manifest("test-proj")["datasets"]["tbl"]["profiled"] is True


def test_invalid_project_name_raises():
    with pytest.raises(ValueError, match="Invalid project name"):
        ws.ensure_project("../escape")

    with pytest.raises(ValueError, match="Invalid project name"):
        ws.ensure_project("")
