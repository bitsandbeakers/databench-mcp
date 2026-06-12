import json
from pathlib import Path

import pytest


@pytest.fixture
def empty_project(tmp_path, monkeypatch):
    """Project with ingested data but no chart sidecars."""
    monkeypatch.setenv("DATABENCH_WORKSPACE", str(tmp_path))
    from databench_mcp.tools.project import project_create
    from databench_mcp.tools.ingest import ingest_file
    from databench_mcp.tools.profile import profile_table
    project_create("dash-proj")
    csv_path = tmp_path / "items.csv"
    csv_path.write_text("name,value\nAlice,10\nBob,20\n")
    ingest_file("dash-proj", str(csv_path), table_name="items")
    profile_table("dash-proj", "items")
    return "dash-proj"


def test_build_dashboard_no_charts_raises(empty_project):
    from databench_mcp.core.dashboard import build_dashboard
    with pytest.raises(ValueError, match="no charts to embed"):
        build_dashboard(empty_project)
