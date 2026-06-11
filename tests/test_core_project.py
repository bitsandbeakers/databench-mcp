"""Tests for core/project.py — project management logic."""
import pytest
import databench_mcp.workspace as ws
from databench_mcp.core import project as core


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)


def test_create_project_returns_expected_keys(tmp_path):
    result = core.create_project("my-proj")
    assert result["project"] == "my-proj"
    assert result["status"] == "created"
    assert "path" in result
    assert "created_at" in result


def test_create_project_creates_dirs(tmp_path):
    core.create_project("my-proj")
    assert (tmp_path / "my-proj" / "raw").is_dir()


def test_create_project_is_idempotent(tmp_path):
    core.create_project("my-proj")
    result = core.create_project("my-proj")
    assert result["project"] == "my-proj"


def test_list_projects_empty_when_no_workspace(tmp_path):
    result = core.list_projects()
    assert result == {"projects": []}


def test_list_projects_after_create(tmp_path):
    core.create_project("alpha")
    core.create_project("beta")
    result = core.list_projects()
    assert result["projects"] == ["alpha", "beta"]


def test_list_projects_excludes_non_project_dirs(tmp_path):
    core.create_project("real-proj")
    (tmp_path / "not-a-project").mkdir()  # no manifest.json
    result = core.list_projects()
    assert result["projects"] == ["real-proj"]


def test_get_status_empty_datasets(tmp_path):
    core.create_project("my-proj")
    result = core.get_status("my-proj")
    assert result["project"] == "my-proj"
    assert result["dataset_count"] == 0
    assert result["datasets"] == {}


def test_get_status_missing_project():
    with pytest.raises(FileNotFoundError):
        core.get_status("ghost")
