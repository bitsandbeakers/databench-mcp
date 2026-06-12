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
    assert result1["warning"] == "no derived tables found — showing all tables"
    result2 = build_dashboard(two_chart_project)
    assert result2["warning"] is not None
    assert "overwritten" in result2["warning"].lower()


@pytest.fixture
def mixed_project(tmp_path, monkeypatch):
    """Project with both a raw table (items) and a derived table (items_clean),
    each with a chart sidecar."""
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    from databench_mcp.tools.project import project_create
    from databench_mcp.tools.ingest import ingest_file
    from databench_mcp.workspace import project_path as _pp, read_manifest, write_manifest
    project_create("dash-proj")
    charts_dir = _pp("dash-proj") / "charts"
    charts_dir.mkdir(exist_ok=True)
    # Ingest a raw table
    csv_path = tmp_path / "items.csv"
    csv_path.write_text("name,value\nAlice,10\nBob,20\n")
    ingest_file("dash-proj", str(csv_path), table_name="items")
    # Manually register a derived table (simulates clean_table output)
    import duckdb
    db_path = _pp("dash-proj") / "project.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("CREATE OR REPLACE TABLE items_clean AS SELECT * FROM items")
    manifest = read_manifest("dash-proj")
    manifest["datasets"]["items_clean"] = {
        "source": "clean_table",
        "source_table": "items",
        "strategy": "fill_mean",
        "profiled": False,
        "row_count": 2,
    }
    write_manifest("dash-proj", manifest)
    # Chart sidecars for both tables
    for tbl, col, ct in [
        ("items", "value", "histogram"),
        ("items_clean", "value", "boxplot"),
    ]:
        (charts_dir / f"{ct}_{tbl}_params.json").write_text(
            json.dumps({"chart_type": ct, "table": tbl, "columns": [col],
                        "finding_id": None, "params": {}})
        )
    return "dash-proj"


def test_build_dashboard_derived_only(mixed_project):
    """Dashboard includes only derived tables when derived tables have charts."""
    from databench_mcp.core.dashboard import build_dashboard
    result = build_dashboard(mixed_project)
    assert "items_clean" in result["tables_exported"]
    assert "items" not in result["tables_exported"]


def test_build_dashboard_derived_only_tab_count(mixed_project):
    """Derived-only filter reduces tab count to 1 (only items_clean)."""
    from databench_mcp.core.dashboard import build_dashboard
    result = build_dashboard(mixed_project)
    assert result["tabs"] == 1


@pytest.fixture
def raw_only_project(tmp_path, monkeypatch):
    """Project with only raw tables (no derived tables have charts) — fallback mode."""
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    from databench_mcp.tools.project import project_create
    from databench_mcp.tools.ingest import ingest_file
    from databench_mcp.workspace import project_path as _pp
    project_create("dash-proj")
    charts_dir = _pp("dash-proj") / "charts"
    charts_dir.mkdir(exist_ok=True)
    csv_path = tmp_path / "items.csv"
    csv_path.write_text("name,value\nAlice,10\nBob,20\n")
    ingest_file("dash-proj", str(csv_path), table_name="items")
    (charts_dir / "histogram_items_params.json").write_text(
        json.dumps({"chart_type": "histogram", "table": "items", "columns": ["value"],
                    "finding_id": None, "params": {}})
    )
    return "dash-proj"


def test_build_dashboard_fallback_all_tables(raw_only_project):
    """Fallback: when only raw tables have charts, all tables shown with warning."""
    from databench_mcp.core.dashboard import build_dashboard
    result = build_dashboard(raw_only_project)
    assert "items" in result["tables_exported"]
    assert result["warning"] is not None
    assert "no derived tables" in result["warning"]


def test_filterable_cols_detection(tmp_path, monkeypatch):
    """Low-cardinality VARCHAR column generates dropdown + callback in dashboard.py."""
    import json
    import duckdb
    import databench_mcp.workspace as ws
    from databench_mcp.core.dashboard import build_dashboard

    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("fc")

    conn = duckdb.connect(str(tmp_path / "fc" / "project.duckdb"))
    conn.execute("""
        CREATE TABLE enriched AS
        SELECT 'APAC' AS region, 100.0 AS value UNION ALL
        SELECT 'EMEA', 200.0
    """)
    conn.close()

    manifest = ws.read_manifest("fc")
    manifest["datasets"]["enriched"] = {
        "source": "enrich_table",
        "profiled": True, "row_count": 2, "col_count": 2,
        "profile": {
            "region": {"type": "VARCHAR", "null_pct": 0, "approx_unique": 2},
            "value":  {"type": "DOUBLE",  "null_pct": 0, "approx_unique": 2},
        },
    }
    ws.write_manifest("fc", manifest)

    charts_dir = tmp_path / "fc" / "charts"
    charts_dir.mkdir()
    (charts_dir / "enriched_hist_params.json").write_text(json.dumps(
        {"chart_type": "histogram", "table": "enriched", "columns": ["value"]}
    ))

    build_dashboard("fc")
    src = (tmp_path / "fc" / "dashboards" / "dashboard.py").read_text()
    assert 'filter-enriched-region' in src
    assert 'update_enriched_charts' in src


def test_no_filterable_cols_static(tmp_path, monkeypatch):
    """Numeric-only derived table gets no callback (static layout)."""
    import json
    import duckdb
    import databench_mcp.workspace as ws
    from databench_mcp.core.dashboard import build_dashboard

    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("nf")

    conn = duckdb.connect(str(tmp_path / "nf" / "project.duckdb"))
    conn.execute("CREATE TABLE metrics AS SELECT 1.0 AS a, 2.0 AS b")
    conn.close()

    manifest = ws.read_manifest("nf")
    manifest["datasets"]["metrics"] = {
        "source": "add_rolling",
        "profiled": True, "row_count": 1, "col_count": 2,
        "profile": {
            "a": {"type": "DOUBLE", "null_pct": 0, "approx_unique": 1},
            "b": {"type": "DOUBLE", "null_pct": 0, "approx_unique": 1},
        },
    }
    ws.write_manifest("nf", manifest)

    charts_dir = tmp_path / "nf" / "charts"
    charts_dir.mkdir()
    (charts_dir / "metrics_hist_params.json").write_text(json.dumps(
        {"chart_type": "histogram", "table": "metrics", "columns": ["a"]}
    ))

    build_dashboard("nf")
    src = (tmp_path / "nf" / "dashboards" / "dashboard.py").read_text()
    assert 'update_metrics_charts' not in src
    assert 'filter-metrics' not in src
