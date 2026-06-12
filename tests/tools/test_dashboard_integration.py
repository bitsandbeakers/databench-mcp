import py_compile
from pathlib import Path

import duckdb
import pytest


@pytest.fixture
def pipeline_project(tmp_path, monkeypatch):
    import databench_mcp.workspace as ws
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    from databench_mcp.tools.project import project_create
    from databench_mcp.tools.ingest import ingest_file
    from databench_mcp.tools.profile import profile_table
    from databench_mcp.core.viz import create_chart

    project_create("int-proj")

    csv1 = tmp_path / "nums.csv"
    csv1.write_text("x,y\n1,2\n3,4\n5,6\n7,8\n")
    ingest_file("int-proj", str(csv1), table_name="nums")
    profile_table("int-proj", "nums")

    csv2 = tmp_path / "cats.csv"
    csv2.write_text("label,count\nA,10\nB,20\nC,5\n")
    ingest_file("int-proj", str(csv2), table_name="cats")
    profile_table("int-proj", "cats")

    create_chart("int-proj", "histogram", "nums", columns=["x"])
    create_chart("int-proj", "boxplot", "cats", columns=["count"])

    return "int-proj"


def test_build_dashboard_tool_files(pipeline_project):
    from databench_mcp.tools.dashboard import build_dashboard

    result = build_dashboard(pipeline_project)

    assert Path(result["dashboard_py"]).exists()
    assert Path(result["requirements_txt"]).exists()
    assert Path(result["deploy_md"]).exists()
    assert len(result["tables_exported"]) > 0


def test_build_dashboard_tool_python_valid(pipeline_project):
    from databench_mcp.tools.dashboard import build_dashboard

    result = build_dashboard(pipeline_project)
    py_compile.compile(result["dashboard_py"], doraise=True)


def test_build_dashboard_tool_parquet_readable(pipeline_project):
    from databench_mcp.tools.dashboard import build_dashboard
    from databench_mcp.workspace import project_path

    result = build_dashboard(pipeline_project)
    data_dir = project_path(pipeline_project) / "dashboards" / "data"

    for table in result["tables_exported"]:
        parquet_path = str(data_dir / f"{table}.parquet")
        count = duckdb.query(f"SELECT COUNT(*) FROM '{parquet_path}'").fetchone()[0]
        assert count > 0


def test_build_dashboard_tool_counts(pipeline_project):
    from databench_mcp.tools.dashboard import build_dashboard

    result = build_dashboard(pipeline_project)
    assert result["tabs"] == 2
    assert result["charts_embedded"] >= 2
