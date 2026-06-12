import json
from pathlib import Path

import pytest
import databench_mcp.workspace as ws


@pytest.fixture
def empty_project(tmp_path, monkeypatch):
    """Project with ingested data but no chart sidecars."""
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
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


@pytest.fixture
def two_chart_project(tmp_path, monkeypatch):
    """Two tables, one chart sidecar each (histogram on items, boxplot on scores)."""
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    from databench_mcp.tools.project import project_create
    from databench_mcp.tools.ingest import ingest_file
    from databench_mcp.tools.profile import profile_table
    from databench_mcp.workspace import project_path as _pp
    project_create("dash-proj")
    charts_dir = _pp("dash-proj") / "charts"
    charts_dir.mkdir(exist_ok=True)
    for tbl, col, csv_content, ct in [
        ("items", "value", "name,value\nAlice,10\nBob,20\n", "histogram"),
        ("scores", "score", "player,score\nX,100\nY,200\n", "boxplot"),
    ]:
        csv_path = tmp_path / f"{tbl}.csv"
        csv_path.write_text(csv_content)
        ingest_file("dash-proj", str(csv_path), table_name=tbl)
        profile_table("dash-proj", tbl)
        (charts_dir / f"{ct}_20260101T000000_params.json").write_text(
            json.dumps({"chart_type": ct, "table": tbl, "columns": [col],
                        "finding_id": None, "params": {}})
        )
    return "dash-proj"


def test_build_dashboard_generates_files(two_chart_project):
    from databench_mcp.core.dashboard import build_dashboard
    from databench_mcp.workspace import project_path
    result = build_dashboard(two_chart_project)
    dash_dir = project_path(two_chart_project) / "dashboards"
    assert Path(result["dashboard_py"]).exists()
    assert Path(result["requirements_txt"]).exists()
    assert Path(result["deploy_md"]).exists()
    assert (dash_dir / "data" / "items.parquet").exists()
    assert (dash_dir / "data" / "scores.parquet").exists()


def test_build_dashboard_tab_count(two_chart_project):
    from databench_mcp.core.dashboard import build_dashboard
    result = build_dashboard(two_chart_project)
    assert result["tabs"] == 2


def test_build_dashboard_charts_embedded(two_chart_project):
    from databench_mcp.core.dashboard import build_dashboard
    result = build_dashboard(two_chart_project)
    assert result["charts_embedded"] == 2


def test_build_dashboard_overwrites_existing(two_chart_project):
    from databench_mcp.core.dashboard import build_dashboard
    result1 = build_dashboard(two_chart_project)
    assert result1["warning"] is None
    result2 = build_dashboard(two_chart_project)
    assert result2["warning"] is not None
    assert "overwritten" in result2["warning"].lower()
