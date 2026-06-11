"""Tests for tools/project.py wrappers — these are the functions FastMCP exposes."""
import pytest
import databench_mcp.workspace as ws
from databench_mcp.tools.project import project_create, project_list, project_status


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)


def test_project_create_returns_status_created(tmp_path):
    result = project_create("demo")
    assert result["status"] == "created"
    assert result["project"] == "demo"


def test_project_list_returns_dict_with_projects_key(tmp_path):
    project_create("one")
    result = project_list()
    assert "projects" in result
    assert "one" in result["projects"]


def test_project_status_returns_dataset_count(tmp_path):
    project_create("demo")
    result = project_status("demo")
    assert result["dataset_count"] == 0


def test_project_status_raises_for_missing_project():
    with pytest.raises(FileNotFoundError):
        project_status("no-such-project")
