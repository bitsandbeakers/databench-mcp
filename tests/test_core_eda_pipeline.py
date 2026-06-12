"""Tests for core/eda.py — pipeline gap functions."""
from __future__ import annotations

import duckdb
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.eda import group_summary, clean_table, add_lag, add_rolling, enrich_table


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("p")
    db_path = str(tmp_path / "p" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE sales AS
        SELECT 'A' AS region, 100.0 AS revenue, 10 AS units UNION ALL
        SELECT 'A', 200.0, 20 UNION ALL
        SELECT 'B', 150.0, 15 UNION ALL
        SELECT 'B', NULL,  5
    """)
    conn.close()
    manifest = ws.read_manifest("p")
    manifest["datasets"]["sales"] = {
        "profiled": True, "row_count": 4, "col_count": 3,
        "profile": {
            "region":  {"type": "VARCHAR",  "null_pct": 0,  "approx_unique": 2},
            "revenue": {"type": "DOUBLE",   "null_pct": 25, "approx_unique": 3},
            "units":   {"type": "INTEGER",  "null_pct": 0,  "approx_unique": 4},
        },
    }
    ws.write_manifest("p", manifest)
    return tmp_path


def _register_regions_table(project_name):
    """Create a 'regions' table in project 'p' and register it in the manifest."""
    from databench_mcp.workspace import project_path, read_manifest, write_manifest
    import duckdb
    db_path = project_path(project_name) / "project.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("CREATE OR REPLACE TABLE regions AS SELECT * FROM (VALUES ('A', 'NE'), ('B', 'SW')) AS t(region, code)")
    manifest = read_manifest(project_name)
    manifest["datasets"]["regions"] = {
        "source": "ingest_file",
        "profiled": True,
        "profile": {
            "region": {"type": "VARCHAR", "null_pct": 0.0, "approx_unique": 2},
            "code": {"type": "VARCHAR", "null_pct": 0.0, "approx_unique": 2},
        },
        "row_count": 2,
    }
    write_manifest(project_name, manifest)


def test_group_summary_basic(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = group_summary("p", "sales", "region", ["revenue"])
    assert result["group_col"] == "region"
    assert result["row_count"] == 2
    assert "revenue_mean" in result["rows"][0]


def test_group_summary_multiple_agg_fns(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = group_summary("p", "sales", "region", ["revenue"], ["mean", "count"])
    row_a = next(r for r in result["rows"] if r["region"] == "A")
    assert row_a["revenue_mean"] == 150.0
    assert row_a["revenue_count"] == 2


def test_group_summary_unknown_agg_fn(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="unknown agg_fn"):
        group_summary("p", "sales", "region", ["revenue"], ["median"])


def test_group_summary_empty_agg_cols(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="agg_cols must not be empty"):
        group_summary("p", "sales", "region", [])


def test_group_summary_unprofiled_table(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("up")
    db_path = str(tmp_path / "up" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE x AS SELECT 1 AS a, 2 AS b")
    conn.close()
    # No profile entry in manifest — assert_profiled should raise
    with pytest.raises(ValueError):
        group_summary("up", "x", "a", ["b"])


def test_group_summary_default_agg_fns_all_present(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = group_summary("p", "sales", "region", ["revenue"])
    row_a = next(r for r in result["rows"] if r["region"] == "A")
    # All 5 default aggregates should be present
    for fn in ["mean", "count", "min", "max", "std"]:
        assert f"revenue_{fn}" in row_a, f"missing revenue_{fn}"


@pytest.fixture
def project_nulls(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("q")
    db_path = str(tmp_path / "q" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE readings AS
        SELECT 1.0 AS temp, 2.0 AS humidity UNION ALL
        SELECT NULL,        3.0            UNION ALL
        SELECT 3.0,         NULL           UNION ALL
        SELECT NULL,        NULL
    """)
    conn.close()
    manifest = ws.read_manifest("q")
    manifest["datasets"]["readings"] = {
        "profiled": True, "row_count": 4, "col_count": 2,
        "profile": {
            "temp":     {"type": "DOUBLE", "null_pct": 50.0, "approx_unique": 2},
            "humidity": {"type": "DOUBLE", "null_pct": 50.0, "approx_unique": 2},
        },
    }
    ws.write_manifest("q", manifest)
    return tmp_path


def test_clean_table_drop_rows(project_nulls, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_nulls)
    result = clean_table("q", "readings", "drop_rows", "clean_readings", columns=["temp"])
    assert result["rows"] < 4
    assert result["source_table"] == "readings"
    assert result["strategy"] == "drop_rows"


def test_clean_table_fill_mean(project_nulls, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_nulls)
    clean_table("q", "readings", "fill_mean", "clean_mean")
    from databench_mcp.db import get_connection
    with get_connection("q") as conn:
        null_count = conn.execute(
            'SELECT COUNT(*) FROM "clean_mean" WHERE "temp" IS NULL'
        ).fetchone()[0]
    assert null_count == 0


def test_clean_table_fill_constant(project_nulls, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_nulls)
    clean_table("q", "readings", "fill_constant", "clean_const",
                columns=["temp"], fill_value=-999.0)
    from databench_mcp.db import get_connection
    with get_connection("q") as conn:
        vals = conn.execute(
            'SELECT "temp" FROM "clean_const" WHERE "temp" = -999.0'
        ).fetchall()
    assert len(vals) == 2


def test_clean_table_fill_constant_requires_fill_value(project_nulls, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_nulls)
    with pytest.raises(ValueError, match="fill_value required"):
        clean_table("q", "readings", "fill_constant", "bad")


def test_clean_table_unknown_strategy(project_nulls, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_nulls)
    with pytest.raises(ValueError, match="unknown strategy"):
        clean_table("q", "readings", "magic_fix", "bad")


def test_clean_table_registers_manifest(project_nulls, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_nulls)
    clean_table("q", "readings", "drop_rows", "clean_drop", columns=["temp"])
    manifest = ws.read_manifest("q")
    ds = manifest["datasets"]["clean_drop"]
    assert ds["source"] == "clean_table"
    assert ds["source_table"] == "readings"
    assert ds["strategy"] == "drop_rows"


def test_clean_table_drop_cols_raises_when_all_dropped(project_nulls, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_nulls)
    project_name = "q"
    table = "readings"
    # get all columns from the table
    db_path = project_nulls / project_name / "project.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        result = conn.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"
        ).fetchall()
    all_cols = [row[0] for row in result]
    with pytest.raises(ValueError, match="strategy would drop all columns"):
        clean_table(project_name, table, "drop_cols", "shouldfail", columns=all_cols)


# --- add_lag tests ---

def test_add_lag_basic(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = add_lag("p", "sales", "revenue", [1, 3], "sales_lagged")
    assert result["table"] == "sales_lagged"
    assert result["source_table"] == "sales"
    assert result["rows"] == 4
    assert "revenue_lag_1" in result["new_cols"]
    assert "revenue_lag_3" in result["new_cols"]


def test_add_lag_columns_present_in_db(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    add_lag("p", "sales", "revenue", [2], "sales_lagged2")
    db_path = project / "p" / "project.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        cols = [r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'sales_lagged2'"
        ).fetchall()]
    assert "revenue_lag_2" in cols
    assert "revenue" in cols


def test_add_lag_with_time_col(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = add_lag("p", "sales", "revenue", [1], "sales_lagged_tc", time_col="units")
    assert result["rows"] == 4


def test_add_lag_registers_manifest(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    add_lag("p", "sales", "revenue", [1, 2], "sales_lagged_m")
    manifest = ws.read_manifest("p")
    ds = manifest["datasets"]["sales_lagged_m"]
    assert ds["source"] == "add_lag"
    assert ds["source_table"] == "sales"
    assert ds["col"] == "revenue"
    assert ds["lags"] == [1, 2]


def test_add_lag_invalid_lags_empty(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="lags must be"):
        add_lag("p", "sales", "revenue", [], "should_fail")


def test_add_lag_invalid_lags_nonpositive(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="lags must be"):
        add_lag("p", "sales", "revenue", [0, 1], "should_fail")


def test_add_lag_invalid_col_identifier(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="col must be"):
        add_lag("p", "sales", 're"venue', [1], "should_fail")


def test_add_lag_invalid_time_col_identifier(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="time_col must be"):
        add_lag("p", "sales", "revenue", [1], "should_fail", time_col='u"nits')


# --- add_rolling tests ---

def test_add_rolling_basic(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = add_rolling("p", "sales", "revenue", 2, "mean", "sales_rolled")
    assert result["table"] == "sales_rolled"
    assert result["source_table"] == "sales"
    assert result["rows"] == 4
    assert result["new_col"] == "revenue_rolling_2_mean"


def test_add_rolling_column_in_db(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    add_rolling("p", "sales", "revenue", 3, "sum", "sales_rolled2")
    db_path = project / "p" / "project.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        cols = [r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'sales_rolled2'"
        ).fetchall()]
    assert "revenue_rolling_3_sum" in cols
    assert "revenue" in cols


def test_add_rolling_with_time_col(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = add_rolling("p", "sales", "revenue", 2, "min", "sales_rolled3", time_col="units")
    assert result["rows"] == 4


def test_add_rolling_registers_manifest(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    add_rolling("p", "sales", "revenue", 2, "max", "sales_rolled_m")
    from databench_mcp.workspace import read_manifest
    manifest = read_manifest("p")
    ds = manifest["datasets"]["sales_rolled_m"]
    assert ds["source"] == "add_rolling"
    assert ds["source_table"] == "sales"
    assert ds["col"] == "revenue"
    assert ds["window"] == 2
    assert ds["agg_fn"] == "max"


def test_add_rolling_invalid_window(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="window must be"):
        add_rolling("p", "sales", "revenue", 0, "mean", "should_fail")


def test_add_rolling_unknown_agg_fn(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="unknown agg_fn"):
        add_rolling("p", "sales", "revenue", 2, "median", "should_fail")


def test_add_rolling_invalid_col_identifier(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="col must be"):
        add_rolling("p", "sales", 're"venue', 2, "mean", "should_fail")


def test_add_rolling_warns_without_time_col(project, monkeypatch):
    import warnings
    import databench_mcp.workspace as ws_inner
    monkeypatch.setattr(ws_inner, "WORKSPACE_ROOT", project)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        add_rolling("p", "sales", "revenue", 2, "mean", "sales_rolled_warn")
    assert any("time_col" in str(w.message) for w in caught)


def test_add_rolling_values_not_all_null(project, monkeypatch):
    import databench_mcp.workspace as ws_inner
    monkeypatch.setattr(ws_inner, "WORKSPACE_ROOT", project)
    add_rolling("p", "sales", "revenue", 1, "mean", "sales_rolled_vals")
    from databench_mcp.workspace import project_path
    db_path = project_path("p") / "project.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        non_nulls = conn.execute("SELECT COUNT(*) FROM sales_rolled_vals WHERE revenue_rolling_1_mean IS NOT NULL").fetchone()[0]
    # 3 of 4 source rows have non-null revenue; window=1 AVG preserves that pattern
    assert non_nulls == 3


def test_add_rolling_manifest_has_new_col(project, monkeypatch):
    import databench_mcp.workspace as ws_inner
    monkeypatch.setattr(ws_inner, "WORKSPACE_ROOT", project)
    add_rolling("p", "sales", "revenue", 3, "std", "sales_rolled_nc")
    from databench_mcp.workspace import read_manifest
    manifest = read_manifest("p")
    ds = manifest["datasets"]["sales_rolled_nc"]
    assert ds["new_col"] == "revenue_rolling_3_std"


# --- enrich_table tests ---

def test_enrich_table_inner(project, monkeypatch):
    import databench_mcp.workspace as ws
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    _register_regions_table("p")
    result = enrich_table("p", "sales", "regions", "region", "sales_enriched")
    assert result["table"] == "sales_enriched"
    assert result["left_table"] == "sales"
    assert result["right_table"] == "regions"
    assert result["how"] == "inner"
    assert "region" in result["on"]
    assert result["rows"] > 0

def test_enrich_table_left_has_more_rows(project, monkeypatch):
    import databench_mcp.workspace as ws
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    _register_regions_table("p")
    result_inner = enrich_table("p", "sales", "regions", "region", "sales_inner")
    result_left = enrich_table("p", "sales", "regions", "region", "sales_left", how="left")
    assert result_left["rows"] >= result_inner["rows"]

def test_enrich_table_columns_in_db(project, monkeypatch):
    import databench_mcp.workspace as ws
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    _register_regions_table("p")
    enrich_table("p", "sales", "regions", "region", "sales_enriched2")
    import duckdb
    from databench_mcp.workspace import project_path
    db_path = project_path("p") / "project.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        cols = [r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'sales_enriched2'").fetchall()]
    assert "code" in cols
    assert "revenue" in cols

def test_enrich_table_registers_manifest(project, monkeypatch):
    import databench_mcp.workspace as ws
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    _register_regions_table("p")
    enrich_table("p", "sales", "regions", "region", "sales_enriched3")
    from databench_mcp.workspace import read_manifest
    manifest = read_manifest("p")
    ds = manifest["datasets"]["sales_enriched3"]
    assert ds["source"] == "enrich_table"
    assert ds["left_table"] == "sales"
    assert ds["right_table"] == "regions"
    assert ds["on"] == ["region"]
    assert ds["how"] == "inner"

def test_enrich_table_unknown_how(project, monkeypatch):
    import databench_mcp.workspace as ws
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="unknown join type"):
        enrich_table("p", "sales", "sales", "region", "should_fail", how="cross")

def test_enrich_table_empty_on(project, monkeypatch):
    import databench_mcp.workspace as ws
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="at least one join column"):
        enrich_table("p", "sales", "sales", [], "should_fail")

def test_enrich_table_on_string_normalised(project, monkeypatch):
    import databench_mcp.workspace as ws
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    _register_regions_table("p")
    result = enrich_table("p", "sales", "regions", "region", "sales_enriched4")
    assert isinstance(result["on"], list)
    assert result["on"] == ["region"]


def test_enrich_table_unprofiled_right_table(project, monkeypatch):
    import databench_mcp.workspace as ws
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    # Register a table in the DB but NOT in the manifest (so assert_profiled raises)
    import duckdb
    from databench_mcp.workspace import project_path
    db_path = project_path("p") / "project.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("CREATE OR REPLACE TABLE unregistered AS SELECT 1 AS x")
    with pytest.raises(ValueError):
        enrich_table("p", "sales", "unregistered", "region", "should_fail")
