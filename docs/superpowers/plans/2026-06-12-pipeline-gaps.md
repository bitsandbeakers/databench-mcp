# Pipeline Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 new MCP tools (`group_summary`, `clean_table`, `add_lag`, `add_rolling`, `enrich_table`) and interactive dashboard enhancements (derived-tables-only filter + column filter dropdowns).

**Architecture:** Three-layer pattern: `core/eda.py` (logic) → `tools/eda.py` (MCP wrappers) → `server.py` (registration). Dashboard enhancements in `core/dashboard.py`. All materialising functions register derived tables in the manifest with `source` set to the function name.

**Tech Stack:** DuckDB (window functions, CTEs, `EXCLUDE` column syntax), FastMCP, Dash 2.17, pytest.

---

## File Map

| File | Change |
|---|---|
| `src/databench_mcp/core/eda.py` | Add 5 new functions + 3 private helpers |
| `src/databench_mcp/tools/eda.py` | Replace with wrappers for all 8 functions |
| `src/databench_mcp/server.py` | Add 5 imports + 5 `mcp.tool()` calls; `EXPECTED_TOOL_COUNT` 23 → 28 |
| `src/databench_mcp/core/dashboard.py` | Add `_DERIVED_SOURCES`, 2 helper functions, update `build_dashboard` and `_generate_dashboard_py` |
| `tests/test_core_eda_pipeline.py` | New — smoke tests for all 5 core functions |
| `tests/tools/test_eda_pipeline.py` | New — integration tests for tool wrappers |
| `tests/test_core_dashboard.py` | Add 4 new tests |

---

### Task 1: `group_summary`

**Files:**
- Modify: `src/databench_mcp/core/eda.py`
- Create: `tests/test_core_eda_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core_eda_pipeline.py`:

```python
"""Tests for core/eda.py — pipeline gap functions."""
from __future__ import annotations

import duckdb
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.eda import group_summary


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


def test_group_summary_basic(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = group_summary("p", "sales", "region", ["revenue"])
    assert result["group_col"] == "region"
    assert result["row_count"] == 2
    assert any("revenue_mean" in row for row in result["rows"])


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
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_core_eda_pipeline.py -v
```
Expected: `ImportError: cannot import name 'group_summary'`

- [ ] **Step 3: Implement `group_summary` in `src/databench_mcp/core/eda.py`**

Add after `_to_json_safe` (before `sql_query`):

```python
_AGG_FN_MAP = {
    "mean": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
    "std": "STDDEV",
}


def group_summary(
    project: str,
    table: str,
    group_col: str,
    agg_cols: list[str],
    agg_fns: list[str] | None = None,
) -> dict[str, Any]:
    """Return grouped aggregates for the given table."""
    from databench_mcp.workspace import assert_profiled

    if agg_fns is None:
        agg_fns = list(_AGG_FN_MAP.keys())
    if not agg_cols:
        raise ValueError("agg_cols must not be empty")
    unknown = [f for f in agg_fns if f not in _AGG_FN_MAP]
    if unknown:
        raise ValueError(f"unknown agg_fn {unknown[0]!r}")
    assert_profiled(project, table)

    agg_parts = ", ".join(
        f'{_AGG_FN_MAP[fn]}("{col}") AS "{col}_{fn}"'
        for col in agg_cols
        for fn in agg_fns
    )
    sql = (
        f'SELECT "{group_col}", {agg_parts} '
        f'FROM "{table}" '
        f'GROUP BY "{group_col}" '
        f'ORDER BY "{group_col}"'
    )
    with get_connection(project) as conn:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description]
        raw_rows = cursor.fetchall()

    rows = [
        {col: _to_json_safe(val) for col, val in zip(columns, row)}
        for row in raw_rows
    ]
    return {
        "table": table,
        "group_col": group_col,
        "agg_cols": agg_cols,
        "agg_fns": agg_fns,
        "rows": rows,
        "row_count": len(rows),
    }
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_core_eda_pipeline.py -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_core_eda_pipeline.py src/databench_mcp/core/eda.py
git commit -m "feat: add group_summary core function"
```

---

### Task 2: `clean_table`

**Files:**
- Modify: `src/databench_mcp/core/eda.py`
- Modify: `tests/test_core_eda_pipeline.py`

- [ ] **Step 1: Write failing tests**

Update the import line at the top of `tests/test_core_eda_pipeline.py`:

```python
from databench_mcp.core.eda import group_summary, clean_table
```

Add a new fixture and tests after the existing ones:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_core_eda_pipeline.py::test_clean_table_drop_rows -v
```
Expected: `ImportError: cannot import name 'clean_table'`

- [ ] **Step 3: Implement `clean_table` in `src/databench_mcp/core/eda.py`**

Add after `group_summary` (and its `_AGG_FN_MAP`):

```python
_CLEAN_STRATEGIES = {
    "drop_rows", "drop_cols",
    "fill_mean", "fill_median", "fill_mode", "fill_constant",
    "fill_forward", "fill_backward",
}
_NUMERIC_TYPES = (
    "DOUBLE", "FLOAT", "INT", "REAL", "DECIMAL",
    "NUMERIC", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT",
)


def _fill_select(table: str, affected: list[str], select_parts: list[str]) -> str:
    """SELECT * EXCLUDE affected cols, then override with select_parts expressions."""
    if not affected:
        return f'SELECT * FROM "{table}"'
    exclude = ", ".join(f'"{c}"' for c in affected)
    overrides = ", ".join(select_parts)
    return f'SELECT * EXCLUDE ({exclude}), {overrides} FROM "{table}"'


def _fill_select_rn(table: str, affected: list[str], select_parts: list[str]) -> str:
    """Same as _fill_select but via a ROW_NUMBER() CTE for forward/backward fill."""
    if not affected:
        return f'SELECT * FROM "{table}"'
    all_excludes = [f'"{c}"' for c in affected] + ['"_rn"']
    exclude = ", ".join(all_excludes)
    overrides = ", ".join(select_parts)
    return (
        f'WITH _base AS (SELECT *, ROW_NUMBER() OVER () AS _rn FROM "{table}") '
        f'SELECT * EXCLUDE ({exclude}), {overrides} FROM _base'
    )


def clean_table(
    project: str,
    table: str,
    strategy: str,
    new_table_name: str,
    columns: list[str] | None = None,
    fill_value: float | str | None = None,
) -> dict[str, Any]:
    """Materialise a cleaned version of *table* as *new_table_name*."""
    from databench_mcp.workspace import assert_profiled

    if strategy not in _CLEAN_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    if strategy == "fill_constant" and fill_value is None:
        raise ValueError("fill_value required for fill_constant")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new_table_name):
        raise ValueError(
            f"new_table_name must be a simple identifier (got {new_table_name!r})"
        )

    assert_profiled(project, table)
    manifest = read_manifest(project)
    profile = manifest["datasets"][table].get("profile", {})

    if columns is not None:
        affected = list(columns)
    elif strategy in ("drop_rows", "drop_cols"):
        affected = [c for c, info in profile.items() if info.get("null_pct", 0) > 0]
    else:
        affected = [
            c for c, info in profile.items()
            if info.get("null_pct", 0) > 0
            and any(t in info.get("type", "").upper() for t in _NUMERIC_TYPES)
        ]

    if strategy == "drop_rows":
        if affected:
            where = " AND ".join(f'"{c}" IS NOT NULL' for c in affected)
            body = f'SELECT * FROM "{table}" WHERE {where}'
        else:
            body = f'SELECT * FROM "{table}"'

    elif strategy == "drop_cols":
        with get_connection(project) as conn:
            all_cols = [
                d[0] for d in conn.execute(
                    f'SELECT * FROM "{table}" LIMIT 0'
                ).description
            ]
        keep = [c for c in all_cols if c not in affected]
        if not keep:
            raise ValueError("strategy would drop all columns")
        body = (
            "SELECT " + ", ".join(f'"{c}"' for c in keep)
            + f' FROM "{table}"'
        )

    elif strategy == "fill_mean":
        parts = [
            f'COALESCE("{c}", AVG("{c}") OVER ()) AS "{c}"' for c in affected
        ]
        body = _fill_select(table, affected, parts)

    elif strategy == "fill_median":
        parts = [
            f'COALESCE("{c}", (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP '
            f'(ORDER BY "{c}") FROM "{table}")) AS "{c}"'
            for c in affected
        ]
        body = _fill_select(table, affected, parts)

    elif strategy == "fill_mode":
        parts = [
            f'COALESCE("{c}", (SELECT mode("{c}") FROM "{table}")) AS "{c}"'
            for c in affected
        ]
        body = _fill_select(table, affected, parts)

    elif strategy == "fill_constant":
        if isinstance(fill_value, str):
            literal = "'" + fill_value.replace("'", "''") + "'"
        else:
            literal = str(fill_value)
        parts = [f'COALESCE("{c}", {literal}) AS "{c}"' for c in affected]
        body = _fill_select(table, affected, parts)

    elif strategy == "fill_forward":
        parts = [
            f'LAST_VALUE("{c}" IGNORE NULLS) OVER '
            f'(ORDER BY _rn ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS "{c}"'
            for c in affected
        ]
        body = _fill_select_rn(table, affected, parts)

    else:  # fill_backward
        parts = [
            f'FIRST_VALUE("{c}" IGNORE NULLS) OVER '
            f'(ORDER BY _rn ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS "{c}"'
            for c in affected
        ]
        body = _fill_select_rn(table, affected, parts)

    with get_connection(project) as conn:
        conn.execute(f'CREATE OR REPLACE TABLE "{new_table_name}" AS ({body})')
        row_count = conn.execute(
            f'SELECT COUNT(*) FROM "{new_table_name}"'
        ).fetchone()[0]
        col_count = len(
            conn.execute(f'SELECT * FROM "{new_table_name}" LIMIT 0').description
        )

    manifest["datasets"][new_table_name] = {
        "source": "clean_table",
        "source_table": table,
        "strategy": strategy,
        "profiled": False,
        "row_count": int(row_count),
        "col_count": int(col_count),
    }
    write_manifest(project, manifest)

    return {
        "table": new_table_name,
        "source_table": table,
        "strategy": strategy,
        "rows": int(row_count),
        "columns": int(col_count),
        "affected_cols": affected,
    }
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_core_eda_pipeline.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_core_eda_pipeline.py src/databench_mcp/core/eda.py
git commit -m "feat: add clean_table core function"
```

---

### Task 3: `add_lag`

**Files:**
- Modify: `src/databench_mcp/core/eda.py`
- Modify: `tests/test_core_eda_pipeline.py`

- [ ] **Step 1: Write failing tests**

Update the import line:

```python
from databench_mcp.core.eda import group_summary, clean_table, add_lag
```

Add a new fixture and tests:

```python
@pytest.fixture
def project_ts(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("ts")
    db_path = str(tmp_path / "ts" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE daily AS
        SELECT DATE '2024-01-01' AS dt, 100.0 AS sales UNION ALL
        SELECT DATE '2024-01-02', 120.0 UNION ALL
        SELECT DATE '2024-01-03', 90.0  UNION ALL
        SELECT DATE '2024-01-04', 110.0 UNION ALL
        SELECT DATE '2024-01-05', 130.0
    """)
    conn.close()
    manifest = ws.read_manifest("ts")
    manifest["datasets"]["daily"] = {
        "profiled": True, "row_count": 5, "col_count": 2,
        "profile": {
            "dt":    {"type": "DATE",   "null_pct": 0, "approx_unique": 5},
            "sales": {"type": "DOUBLE", "null_pct": 0, "approx_unique": 5},
        },
    }
    ws.write_manifest("ts", manifest)
    return tmp_path


def test_add_lag_basic(project_ts, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_ts)
    result = add_lag("ts", "daily", "sales", [1, 3], "daily_lag", time_col="dt")
    assert result["rows"] == 5
    assert "sales_lag_1" in result["new_cols"]
    assert "sales_lag_3" in result["new_cols"]


def test_add_lag_columns_present(project_ts, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_ts)
    add_lag("ts", "daily", "sales", [1], "daily_lag1", time_col="dt")
    from databench_mcp.db import get_connection
    with get_connection("ts") as conn:
        cols = [
            d[0] for d in conn.execute(
                'SELECT * FROM "daily_lag1" LIMIT 0'
            ).description
        ]
    assert "sales_lag_1" in cols
    assert "sales" in cols


def test_add_lag_registers_manifest(project_ts, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_ts)
    add_lag("ts", "daily", "sales", [1], "daily_lm")
    manifest = ws.read_manifest("ts")
    assert manifest["datasets"]["daily_lm"]["source"] == "add_lag"


def test_add_lag_empty_lags(project_ts, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_ts)
    with pytest.raises(ValueError, match="lags must not be empty"):
        add_lag("ts", "daily", "sales", [], "bad")


def test_add_lag_nonpositive(project_ts, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_ts)
    with pytest.raises(ValueError, match="positive"):
        add_lag("ts", "daily", "sales", [0], "bad")
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_core_eda_pipeline.py::test_add_lag_basic -v
```
Expected: `ImportError: cannot import name 'add_lag'`

- [ ] **Step 3: Implement `add_lag` in `src/databench_mcp/core/eda.py`**

Add after `clean_table` (and its helpers):

```python
def add_lag(
    project: str,
    table: str,
    col: str,
    lags: list[int],
    new_table_name: str,
    time_col: str | None = None,
) -> dict[str, Any]:
    """Create a new table with LAG columns for *col* at the given offsets."""
    from databench_mcp.workspace import assert_profiled

    if not lags:
        raise ValueError("lags must not be empty")
    if any(n <= 0 for n in lags):
        raise ValueError("all lags must be positive integers")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new_table_name):
        raise ValueError(
            f"new_table_name must be a simple identifier (got {new_table_name!r})"
        )

    assert_profiled(project, table)

    order_clause = f'ORDER BY "{time_col}"' if time_col else ""
    lag_parts = ", ".join(
        f'LAG("{col}", {n}) OVER ({order_clause}) AS "{col}_lag_{n}"'
        for n in lags
    )
    sql = (
        f'CREATE OR REPLACE TABLE "{new_table_name}" AS '
        f'(SELECT *, {lag_parts} FROM "{table}")'
    )
    with get_connection(project) as conn:
        conn.execute(sql)
        row_count = conn.execute(
            f'SELECT COUNT(*) FROM "{new_table_name}"'
        ).fetchone()[0]

    manifest = read_manifest(project)
    manifest["datasets"][new_table_name] = {
        "source": "add_lag",
        "source_table": table,
        "col": col,
        "lags": lags,
        "profiled": False,
        "row_count": int(row_count),
    }
    write_manifest(project, manifest)

    return {
        "table": new_table_name,
        "source_table": table,
        "new_cols": [f"{col}_lag_{n}" for n in lags],
        "rows": int(row_count),
    }
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_core_eda_pipeline.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_core_eda_pipeline.py src/databench_mcp/core/eda.py
git commit -m "feat: add add_lag core function"
```

---

### Task 4: `add_rolling`

**Files:**
- Modify: `src/databench_mcp/core/eda.py`
- Modify: `tests/test_core_eda_pipeline.py`

- [ ] **Step 1: Write failing tests**

Update import:

```python
from databench_mcp.core.eda import group_summary, clean_table, add_lag, add_rolling
```

Add tests (reuses `project_ts` fixture from Task 3):

```python
def test_add_rolling_basic(project_ts, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_ts)
    result = add_rolling("ts", "daily", "sales", 3, "mean", "daily_roll", time_col="dt")
    assert result["rows"] == 5
    assert result["new_col"] == "sales_rolling_3_mean"


def test_add_rolling_col_present(project_ts, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_ts)
    add_rolling("ts", "daily", "sales", 3, "sum", "daily_sum")
    from databench_mcp.db import get_connection
    with get_connection("ts") as conn:
        cols = [
            d[0] for d in conn.execute(
                'SELECT * FROM "daily_sum" LIMIT 0'
            ).description
        ]
    assert "sales_rolling_3_sum" in cols


def test_add_rolling_registers_manifest(project_ts, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_ts)
    add_rolling("ts", "daily", "sales", 7, "max", "daily_max7")
    manifest = ws.read_manifest("ts")
    assert manifest["datasets"]["daily_max7"]["source"] == "add_rolling"


def test_add_rolling_invalid_window(project_ts, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_ts)
    with pytest.raises(ValueError, match="window must be"):
        add_rolling("ts", "daily", "sales", 0, "mean", "bad")


def test_add_rolling_invalid_agg_fn(project_ts, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_ts)
    with pytest.raises(ValueError, match="unknown agg_fn"):
        add_rolling("ts", "daily", "sales", 3, "median", "bad")
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_core_eda_pipeline.py::test_add_rolling_basic -v
```
Expected: `ImportError: cannot import name 'add_rolling'`

- [ ] **Step 3: Implement `add_rolling` in `src/databench_mcp/core/eda.py`**

Add after `add_lag`:

```python
_ROLLING_FN_MAP = {
    "mean": "AVG",
    "sum": "SUM",
    "min": "MIN",
    "max": "MAX",
    "std": "STDDEV",
}


def add_rolling(
    project: str,
    table: str,
    col: str,
    window: int,
    agg_fn: str,
    new_table_name: str,
    time_col: str | None = None,
) -> dict[str, Any]:
    """Create a new table with a rolling-window aggregate column."""
    from databench_mcp.workspace import assert_profiled

    if window <= 0:
        raise ValueError("window must be a positive integer")
    if agg_fn not in _ROLLING_FN_MAP:
        raise ValueError(f"unknown agg_fn {agg_fn!r}")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new_table_name):
        raise ValueError(
            f"new_table_name must be a simple identifier (got {new_table_name!r})"
        )

    assert_profiled(project, table)

    sql_fn = _ROLLING_FN_MAP[agg_fn]
    order_clause = f'ORDER BY "{time_col}" ' if time_col else ""
    new_col = f"{col}_rolling_{window}_{agg_fn}"
    rolling_expr = (
        f'{sql_fn}("{col}") OVER ({order_clause}'
        f'ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) AS "{new_col}"'
    )
    sql = (
        f'CREATE OR REPLACE TABLE "{new_table_name}" AS '
        f'(SELECT *, {rolling_expr} FROM "{table}")'
    )
    with get_connection(project) as conn:
        conn.execute(sql)
        row_count = conn.execute(
            f'SELECT COUNT(*) FROM "{new_table_name}"'
        ).fetchone()[0]

    manifest = read_manifest(project)
    manifest["datasets"][new_table_name] = {
        "source": "add_rolling",
        "source_table": table,
        "col": col,
        "window": window,
        "agg_fn": agg_fn,
        "profiled": False,
        "row_count": int(row_count),
    }
    write_manifest(project, manifest)

    return {
        "table": new_table_name,
        "source_table": table,
        "new_col": new_col,
        "rows": int(row_count),
    }
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_core_eda_pipeline.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_core_eda_pipeline.py src/databench_mcp/core/eda.py
git commit -m "feat: add add_rolling core function"
```

---

### Task 5: `enrich_table`

**Files:**
- Modify: `src/databench_mcp/core/eda.py`
- Modify: `tests/test_core_eda_pipeline.py`

- [ ] **Step 1: Write failing tests**

Update import:

```python
from databench_mcp.core.eda import group_summary, clean_table, add_lag, add_rolling, enrich_table
```

Add fixture and tests:

```python
@pytest.fixture
def project_join(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("j")
    db_path = str(tmp_path / "j" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE orders AS
        SELECT 1 AS order_id, 'A' AS customer_id, 100.0 AS amount UNION ALL
        SELECT 2, 'B', 200.0 UNION ALL
        SELECT 3, 'C', 150.0
    """)
    conn.execute("""
        CREATE TABLE customers AS
        SELECT 'A' AS customer_id, 'Alice' AS name UNION ALL
        SELECT 'B', 'Bob'
    """)
    conn.close()
    manifest = ws.read_manifest("j")
    manifest["datasets"]["orders"] = {
        "profiled": True, "row_count": 3, "col_count": 3,
        "profile": {
            "order_id":    {"type": "INTEGER", "null_pct": 0, "approx_unique": 3},
            "customer_id": {"type": "VARCHAR", "null_pct": 0, "approx_unique": 3},
            "amount":      {"type": "DOUBLE",  "null_pct": 0, "approx_unique": 3},
        },
    }
    manifest["datasets"]["customers"] = {
        "profiled": True, "row_count": 2, "col_count": 2,
        "profile": {
            "customer_id": {"type": "VARCHAR", "null_pct": 0, "approx_unique": 2},
            "name":        {"type": "VARCHAR", "null_pct": 0, "approx_unique": 2},
        },
    }
    ws.write_manifest("j", manifest)
    return tmp_path


def test_enrich_table_inner(project_join, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_join)
    result = enrich_table("j", "orders", "customers", "customer_id", "enriched", how="inner")
    assert result["rows"] == 2  # only A and B match
    assert result["columns"] >= 3


def test_enrich_table_left(project_join, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_join)
    result = enrich_table(
        "j", "orders", "customers", "customer_id", "enriched_left", how="left"
    )
    assert result["rows"] == 3  # all orders preserved


def test_enrich_table_registers_manifest(project_join, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_join)
    enrich_table("j", "orders", "customers", "customer_id", "enr")
    manifest = ws.read_manifest("j")
    ds = manifest["datasets"]["enr"]
    assert ds["source"] == "enrich_table"
    assert ds["left_table"] == "orders"
    assert ds["right_table"] == "customers"


def test_enrich_table_unknown_how(project_join, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_join)
    with pytest.raises(ValueError, match="unknown how"):
        enrich_table("j", "orders", "customers", "customer_id", "bad", how="cross")


def test_enrich_table_empty_on(project_join, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_join)
    with pytest.raises(ValueError, match="on must specify"):
        enrich_table("j", "orders", "customers", [], "bad")
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_core_eda_pipeline.py::test_enrich_table_inner -v
```
Expected: `ImportError: cannot import name 'enrich_table'`

- [ ] **Step 3: Implement `enrich_table` in `src/databench_mcp/core/eda.py`**

Add after `add_rolling` and `_ROLLING_FN_MAP`:

```python
_HOW_MAP = {
    "inner": "INNER JOIN",
    "left":  "LEFT JOIN",
    "right": "RIGHT JOIN",
    "full":  "FULL OUTER JOIN",
}


def enrich_table(
    project: str,
    left_table: str,
    right_table: str,
    on: str | list[str],
    new_table_name: str,
    how: str = "inner",
) -> dict[str, Any]:
    """Materialise a JOIN of two tables as a new derived table."""
    from databench_mcp.workspace import assert_profiled

    if how not in _HOW_MAP:
        raise ValueError(f"unknown how {how!r}")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new_table_name):
        raise ValueError(
            f"new_table_name must be a simple identifier (got {new_table_name!r})"
        )

    on_cols = [on] if isinstance(on, str) else list(on)
    if not on_cols:
        raise ValueError("on must specify at least one join column")

    assert_profiled(project, left_table)
    assert_profiled(project, right_table)

    join_kw = _HOW_MAP[how]
    using_clause = "USING (" + ", ".join(f'"{c}"' for c in on_cols) + ")"
    sql = (
        f'CREATE OR REPLACE TABLE "{new_table_name}" AS ('
        f'SELECT * FROM "{left_table}" '
        f'{join_kw} "{right_table}" {using_clause})'
    )
    with get_connection(project) as conn:
        conn.execute(sql)
        row_count = conn.execute(
            f'SELECT COUNT(*) FROM "{new_table_name}"'
        ).fetchone()[0]
        col_count = len(
            conn.execute(f'SELECT * FROM "{new_table_name}" LIMIT 0').description
        )

    manifest = read_manifest(project)
    manifest["datasets"][new_table_name] = {
        "source": "enrich_table",
        "left_table": left_table,
        "right_table": right_table,
        "on": on_cols,
        "how": how,
        "profiled": False,
        "row_count": int(row_count),
        "col_count": int(col_count),
    }
    write_manifest(project, manifest)

    return {
        "table": new_table_name,
        "left_table": left_table,
        "right_table": right_table,
        "how": how,
        "on": on_cols,
        "rows": int(row_count),
        "columns": int(col_count),
    }
```

- [ ] **Step 4: Run all pipeline tests**

```
pytest tests/test_core_eda_pipeline.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_core_eda_pipeline.py src/databench_mcp/core/eda.py
git commit -m "feat: add enrich_table core function"
```

---

### Task 6: Tool Wrappers, Integration Tests, Server Registration

**Files:**
- Modify: `src/databench_mcp/tools/eda.py`
- Modify: `src/databench_mcp/server.py`
- Create: `tests/tools/test_eda_pipeline.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/tools/test_eda_pipeline.py`:

```python
"""Integration tests for pipeline gap tool wrappers."""
from __future__ import annotations

import duckdb
import pytest

import databench_mcp.workspace as ws
from databench_mcp.tools.eda import clean_table, add_lag, enrich_table


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("p")
    db_path = str(tmp_path / "p" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE a AS SELECT 'x' AS k, 1.0 AS v UNION ALL SELECT 'y', NULL
    """)
    conn.execute("""
        CREATE TABLE b AS SELECT 'x' AS k, 'foo' AS label UNION ALL SELECT 'z', 'bar'
    """)
    conn.close()
    manifest = ws.read_manifest("p")
    manifest["datasets"]["a"] = {
        "profiled": True, "row_count": 2, "col_count": 2,
        "profile": {
            "k": {"type": "VARCHAR", "null_pct": 0,  "approx_unique": 2},
            "v": {"type": "DOUBLE",  "null_pct": 50, "approx_unique": 1},
        },
    }
    manifest["datasets"]["b"] = {
        "profiled": True, "row_count": 2, "col_count": 2,
        "profile": {
            "k":     {"type": "VARCHAR", "null_pct": 0, "approx_unique": 2},
            "label": {"type": "VARCHAR", "null_pct": 0, "approx_unique": 2},
        },
    }
    ws.write_manifest("p", manifest)
    return tmp_path


def test_clean_table_integration(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = clean_table("p", "a", "drop_rows", "a_clean")
    manifest = ws.read_manifest("p")
    assert manifest["datasets"]["a_clean"]["source"] == "clean_table"
    assert result["rows"] == 1


def test_add_lag_integration(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = add_lag("p", "a", "v", [1], "a_lag")
    manifest = ws.read_manifest("p")
    assert manifest["datasets"]["a_lag"]["source"] == "add_lag"
    assert "v_lag_1" in result["new_cols"]


def test_enrich_table_integration(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = enrich_table("p", "a", "b", "k", "ab_joined")
    manifest = ws.read_manifest("p")
    ds = manifest["datasets"]["ab_joined"]
    assert ds["left_table"] == "a"
    assert ds["right_table"] == "b"
    assert result["rows"] == 1  # only 'x' in both tables
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/tools/test_eda_pipeline.py -v
```
Expected: `ImportError: cannot import name 'clean_table' from 'databench_mcp.tools.eda'`

- [ ] **Step 3: Replace `src/databench_mcp/tools/eda.py`**

```python
"""FastMCP tool wrappers for EDA functions."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.eda import add_lag as _add_lag
from databench_mcp.core.eda import add_rolling as _add_rolling
from databench_mcp.core.eda import clean_table as _clean_table
from databench_mcp.core.eda import derive_table as _derive_table
from databench_mcp.core.eda import eda_summary as _eda_summary
from databench_mcp.core.eda import enrich_table as _enrich_table
from databench_mcp.core.eda import group_summary as _group_summary
from databench_mcp.core.eda import sql_query as _sql_query


def sql_query(project: str, sql: str, limit: int = 500) -> dict[str, Any]:
    """Execute a read-only SELECT/WITH query against the project DuckDB."""
    return _sql_query(project, sql, limit)


def eda_summary(project: str) -> dict[str, Any]:
    """Return dataset summary from the project manifest (no DB query)."""
    return _eda_summary(project)


def derive_table(project: str, sql: str, table_name: str) -> dict[str, Any]:
    """Materialise a SQL SELECT as a new DuckDB table, registered as profiled=False."""
    return _derive_table(project, sql, table_name)


def group_summary(
    project: str,
    table: str,
    group_col: str,
    agg_cols: list[str],
    agg_fns: list[str] | None = None,
) -> dict[str, Any]:
    """Return grouped aggregates for a table column."""
    return _group_summary(project, table, group_col, agg_cols, agg_fns)


def clean_table(
    project: str,
    table: str,
    strategy: str,
    new_table_name: str,
    columns: list[str] | None = None,
    fill_value: float | str | None = None,
) -> dict[str, Any]:
    """Clean missing data in *table* and materialise result as *new_table_name*."""
    return _clean_table(project, table, strategy, new_table_name, columns, fill_value)


def add_lag(
    project: str,
    table: str,
    col: str,
    lags: list[int],
    new_table_name: str,
    time_col: str | None = None,
) -> dict[str, Any]:
    """Add LAG feature columns to *table* and materialise as *new_table_name*."""
    return _add_lag(project, table, col, lags, new_table_name, time_col)


def add_rolling(
    project: str,
    table: str,
    col: str,
    window: int,
    agg_fn: str,
    new_table_name: str,
    time_col: str | None = None,
) -> dict[str, Any]:
    """Add a rolling-window aggregate column and materialise as *new_table_name*."""
    return _add_rolling(project, table, col, window, agg_fn, new_table_name, time_col)


def enrich_table(
    project: str,
    left_table: str,
    right_table: str,
    on: str | list[str],
    new_table_name: str,
    how: str = "inner",
) -> dict[str, Any]:
    """JOIN two tables and materialise result as *new_table_name*."""
    return _enrich_table(project, left_table, right_table, on, new_table_name, how)
```

- [ ] **Step 4: Update `src/databench_mcp/server.py`**

Replace the existing eda import line:

```python
# Before:
from databench_mcp.tools.eda import derive_table, eda_summary, sql_query

# After:
from databench_mcp.tools.eda import (
    add_lag,
    add_rolling,
    clean_table,
    derive_table,
    eda_summary,
    enrich_table,
    group_summary,
    sql_query,
)
```

Change `EXPECTED_TOOL_COUNT = 23` → `EXPECTED_TOOL_COUNT = 28`.

After `mcp.tool(derive_table)`, add:

```python
mcp.tool(group_summary)
mcp.tool(clean_table)
mcp.tool(add_lag)
mcp.tool(add_rolling)
mcp.tool(enrich_table)
```

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v
```
Expected: all PASS (tool count assertion passes with 28)

- [ ] **Step 6: Commit**

```bash
git add src/databench_mcp/tools/eda.py src/databench_mcp/server.py tests/tools/test_eda_pipeline.py
git commit -m "feat: register 5 new tools; bump EXPECTED_TOOL_COUNT 23→28"
```

---

### Task 7: Dashboard Derived-Table Filter

**Files:**
- Modify: `src/databench_mcp/core/dashboard.py`
- Modify: `tests/test_core_dashboard.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_core_dashboard.py`:

```python
def test_build_dashboard_derived_only(tmp_path, monkeypatch):
    """Only derived-source tables appear as tabs; raw ingested tables are excluded."""
    import json
    import duckdb
    import databench_mcp.workspace as ws
    from databench_mcp.core.dashboard import build_dashboard

    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("d")

    conn = duckdb.connect(str(tmp_path / "d" / "project.duckdb"))
    conn.execute("CREATE TABLE raw_data AS SELECT 1 AS x")
    conn.execute("CREATE TABLE clean_data AS SELECT 1 AS x")
    conn.close()

    manifest = ws.read_manifest("d")
    manifest["datasets"]["raw_data"] = {
        "source": "/tmp/raw_data.csv",
        "profiled": True, "row_count": 1, "col_count": 1,
        "profile": {"x": {"type": "INTEGER", "null_pct": 0, "approx_unique": 1}},
    }
    manifest["datasets"]["clean_data"] = {
        "source": "clean_table",
        "profiled": True, "row_count": 1, "col_count": 1,
        "profile": {"x": {"type": "INTEGER", "null_pct": 0, "approx_unique": 1}},
    }
    ws.write_manifest("d", manifest)

    charts_dir = tmp_path / "d" / "charts"
    charts_dir.mkdir()
    (charts_dir / "raw_data_hist_params.json").write_text(json.dumps(
        {"chart_type": "histogram", "table": "raw_data", "columns": ["x"]}
    ))
    (charts_dir / "clean_data_hist_params.json").write_text(json.dumps(
        {"chart_type": "histogram", "table": "clean_data", "columns": ["x"]}
    ))

    result = build_dashboard("d")
    assert result["tabs"] == 1
    assert "clean_data" in result["tables_exported"]
    assert "raw_data" not in result["tables_exported"]


def test_build_dashboard_fallback_all_tables(tmp_path, monkeypatch):
    """When no derived tables have charts, fall back to all tables with a warning."""
    import json
    import duckdb
    import databench_mcp.workspace as ws
    from databench_mcp.core.dashboard import build_dashboard

    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("fb")

    conn = duckdb.connect(str(tmp_path / "fb" / "project.duckdb"))
    conn.execute("CREATE TABLE raw_only AS SELECT 1 AS x")
    conn.close()

    manifest = ws.read_manifest("fb")
    manifest["datasets"]["raw_only"] = {
        "source": "/tmp/raw_only.csv",
        "profiled": True, "row_count": 1, "col_count": 1,
        "profile": {"x": {"type": "INTEGER", "null_pct": 0, "approx_unique": 1}},
    }
    ws.write_manifest("fb", manifest)

    charts_dir = tmp_path / "fb" / "charts"
    charts_dir.mkdir()
    (charts_dir / "raw_only_hist_params.json").write_text(json.dumps(
        {"chart_type": "histogram", "table": "raw_only", "columns": ["x"]}
    ))

    result = build_dashboard("fb")
    assert result["tabs"] == 1
    assert "no derived tables" in (result.get("warning") or "")
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_core_dashboard.py::test_build_dashboard_derived_only -v
```
Expected: FAIL — `raw_data` incorrectly appears in `tables_exported`

- [ ] **Step 3: Update `src/databench_mcp/core/dashboard.py`**

Add `read_manifest` to the workspace import at the top:

```python
# Before:
from databench_mcp.workspace import project_path

# After:
from databench_mcp.workspace import project_path, read_manifest
```

After the `_DEPLOY_MD` constant (before `_MAKE_FIGURE_SOURCE`), add:

```python
_DERIVED_SOURCES = {"derived", "clean_table", "add_lag", "add_rolling", "enrich_table"}
```

In `build_dashboard`, after the `if not sidecars_by_table:` check and before `dash_dir = _dashboards_dir(project)`, insert:

```python
    # Filter to derived tables only; fall back to all if none qualify
    manifest = read_manifest(project)
    datasets = manifest.get("datasets", {})
    derived_tables = {
        tbl for tbl, ds in datasets.items()
        if ds.get("source") in _DERIVED_SOURCES
    }
    derived_sidecars = {t: v for t, v in sidecars_by_table.items() if t in derived_tables}

    fallback_warning: str | None = None
    if derived_sidecars:
        active_sidecars = derived_sidecars
    else:
        active_sidecars = sidecars_by_table
        fallback_warning = "no derived tables found — showing all tables"
```

Replace the two lines `all_tables = sorted(sidecars_by_table.keys())` and `_export_tables(project, all_tables, data_dir)` with:

```python
    all_tables = sorted(active_sidecars.keys())
    _export_tables(project, all_tables, data_dir)
```

After the existing `if dashboard_py_path.exists():` block, add:

```python
    if fallback_warning:
        warning_parts.append(fallback_warning)
```

Replace `src = _generate_dashboard_py(sidecars_by_table, findings)` with:

```python
    src = _generate_dashboard_py(active_sidecars, findings)
```

Replace `charts_embedded = sum(len(v) for v in sidecars_by_table.values())` with:

```python
    charts_embedded = sum(len(v) for v in active_sidecars.values())
```

- [ ] **Step 4: Run dashboard tests**

```
pytest tests/test_core_dashboard.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/databench_mcp/core/dashboard.py tests/test_core_dashboard.py
git commit -m "feat: filter dashboard to derived tables; fallback with warning if none"
```

---

### Task 8: Dashboard Interactive Callbacks

**Files:**
- Modify: `src/databench_mcp/core/dashboard.py`
- Modify: `tests/test_core_dashboard.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_core_dashboard.py`:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_core_dashboard.py::test_filterable_cols_detection -v
```
Expected: FAIL — no callback code generated yet

- [ ] **Step 3: Add helper functions to `src/databench_mcp/core/dashboard.py`**

After `_DERIVED_SOURCES`, add:

```python
def _filterable_cols(manifest: dict, tables: list[str]) -> dict[str, list[str]]:
    """Return {table: [col, ...]} for low-cardinality VARCHAR cols in each table."""
    result: dict[str, list[str]] = {}
    datasets = manifest.get("datasets", {})
    for table in tables:
        profile = datasets.get(table, {}).get("profile", {})
        cols = [
            col for col, info in profile.items()
            if ("VARCHAR" in info.get("type", "") or "ENUM" in info.get("type", ""))
            and info.get("approx_unique", 999) <= 20
        ]
        if cols:
            result[table] = cols
    return result


def _fetch_filter_options(
    project: str,
    filterable: dict[str, list[str]],
) -> dict[str, dict[str, list]]:
    """Fetch sorted unique string values for each filterable column (at build time)."""
    result: dict[str, dict[str, list]] = {}
    for table, cols in filterable.items():
        result[table] = {}
        for col in cols:
            with get_connection(project) as conn:
                rows = conn.execute(
                    f'SELECT DISTINCT "{col}" FROM "{table}" '
                    f'WHERE "{col}" IS NOT NULL ORDER BY "{col}"'
                ).fetchall()
            result[table][col] = [r[0] for r in rows]
    return result
```

- [ ] **Step 4: Update `_generate_dashboard_py` signature and logic**

Change the function signature from:

```python
def _generate_dashboard_py(
    sidecars_by_table: dict[str, list[dict]],
    findings: list[dict],
) -> str:
```

to:

```python
def _generate_dashboard_py(
    sidecars_by_table: dict[str, list[dict]],
    findings: list[dict],
    filterable: dict[str, list[str]] | None = None,
    filter_options: dict[str, dict[str, list]] | None = None,
) -> str:
```

At the start of the function body, after the existing `all_tables` and `all_sidecars` lines, add:

```python
    if filterable is None:
        filterable = {}
    if filter_options is None:
        filter_options = {}
    filterable_tables = sorted(filterable.keys())
```

In the `header` string, change the dash import line from:

```python
        "from dash import dcc, html, dash_table\n"
```

to a conditional (replace the entire `header = (...)` block — find the `"from dash import dcc, html, dash_table\n"` line and change it):

```python
        "from dash import dcc, html, dash_table"
        + (", Input, Output" if filterable_tables else "")
        + "\n"
```

Replace the entire `footer` variable assignment (the long string that starts with `"\n# --- layout ---\n"`) with the following. This is a complete replacement — delete from `footer = (` through the closing `)` and substitute:

```python
    # Static charts for non-filterable tables
    if filterable_tables:
        static_loop = (
            "_tabs_by_table = {}\n"
            "for _c in _CHARTS:\n"
            "    if _c[\"table\"] not in " + repr(set(filterable_tables)) + ":\n"
            "        _fig = _make_figure(_c[\"chart_type\"], tables[_c[\"table\"]],"
            " _c[\"columns\"], _c.get(\"params\", {}))\n"
            "        _tabs_by_table.setdefault(_c[\"table\"], []).append("
            "dcc.Graph(figure=_fig))\n"
            "\n"
        )
    else:
        static_loop = (
            "_tabs_by_table = {}\n"
            "for _c in _CHARTS:\n"
            "    _fig = _make_figure(_c[\"chart_type\"], tables[_c[\"table\"]],"
            " _c[\"columns\"], _c.get(\"params\", {}))\n"
            "    _tabs_by_table.setdefault(_c[\"table\"], []).append("
            "dcc.Graph(figure=_fig))\n"
            "\n"
        )

    # Tab list — static tabs first, then filterable tabs with dropdowns
    tab_list = (
        "dataset_tabs = []\n"
        "for _tbl, _figs in sorted(_tabs_by_table.items()):\n"
        "    dataset_tabs.append(dcc.Tab(label=_tbl, children=[\n"
        "        html.Div(_figs, style={\"display\": \"grid\","
        " \"gridTemplateColumns\": \"1fr 1fr\", \"gap\": \"16px\"})\n"
        "    ]))\n"
        "\n"
    )
    for _tbl in filterable_tables:
        _cols = filterable[_tbl]
        _opts = filter_options.get(_tbl, {})
        dropdown_items = ""
        for _col in _cols:
            _col_opts = [{"label": v, "value": v} for v in _opts.get(_col, [])]
            dropdown_items += (
                f"        dcc.Dropdown(id=\"filter-{_tbl}-{_col}\","
                f" options={_col_opts!r},"
                f" multi=True, placeholder=\"Filter by {_col}\","
                f" style={{\"minWidth\": \"200px\"}}),\n"
            )
        tab_list += (
            f"dataset_tabs.append(dcc.Tab(label=\"{_tbl}\", children=[\n"
            f"    html.Div([\n"
            f"{dropdown_items}"
            f"    ], style={{\"display\": \"flex\", \"gap\": \"12px\","
            f" \"marginBottom\": \"16px\"}}),\n"
            f"    html.Div(id=\"charts-{_tbl}\"),\n"
            f"]))\n"
        )
    tab_list += "\n"

    # Callbacks for filterable tables
    callbacks = ""
    for _tbl in filterable_tables:
        _cols = filterable[_tbl]
        _param_names = [f"_v{i}" for i in range(len(_cols))]
        _inputs_str = ", ".join(
            f"Input(\"filter-{_tbl}-{_col}\", \"value\")" for _col in _cols
        )
        _params_str = ", ".join(_param_names)
        _vals_list = "[" + ", ".join(_param_names) + "]"
        callbacks += (
            f"@app.callback(Output(\"charts-{_tbl}\", \"children\"), [{_inputs_str}])\n"
            f"def update_{_tbl}_charts({_params_str}):\n"
            f"    _df = tables[\"{_tbl}\"].copy()\n"
            f"    for _col, _val in zip({_cols!r}, {_vals_list}):\n"
            f"        if _val:\n"
            f"            _df = _df[_df[_col].isin(_val)]\n"
            f"    return [dcc.Graph(figure=_make_figure(\n"
            f"        _c[\"chart_type\"],\n"
            f"        tables[\"{_tbl}\"] if _c[\"chart_type\"] == \"correlation_heatmap\""
            f" else _df,\n"
            f"        _c[\"columns\"], _c.get(\"params\", {{}})))\n"
            f"        for _c in _CHARTS if _c[\"table\"] == \"{_tbl}\"]\n"
            "\n"
        )

    footer = (
        "\n# --- layout ---\n"
        "_CHARTS = " + repr(all_sidecars) + "\n"
        "\n"
        "_FINDINGS = " + repr(findings_rows) + "\n"
        "\n"
        "app = dash.Dash(__name__)\n"
        "\n"
        + static_loop
        + tab_list
        + callbacks
        + "findings_tab = dcc.Tab(label=\"Findings\", children=[\n"
        "    dash_table.DataTable(\n"
        "        data=_FINDINGS,\n"
        "        columns=[{\"name\": c, \"id\": c} for c in"
        " [\"id\", \"method\", \"table\", \"summary\", \"metrics\"]],\n"
        "        style_table={\"overflowX\": \"auto\"},\n"
        "    )\n"
        "])\n"
        "\n"
        "app.layout = html.Div([\n"
        "    dcc.Tabs(children=dataset_tabs + [findings_tab])\n"
        "])\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    app.run(debug=False)\n"
    )
```

- [ ] **Step 5: Update `build_dashboard` to compute and pass filterable data**

In `build_dashboard`, after the `_export_tables` call (i.e. after `_export_tables(project, all_tables, data_dir)`), add:

```python
    filterable = _filterable_cols(manifest, all_tables)
    filter_opts = _fetch_filter_options(project, filterable)
```

Change `src = _generate_dashboard_py(active_sidecars, findings)` to:

```python
    src = _generate_dashboard_py(active_sidecars, findings, filterable, filter_opts)
```

- [ ] **Step 6: Run full test suite**

```
pytest tests/ -v
```
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/databench_mcp/core/dashboard.py tests/test_core_dashboard.py
git commit -m "feat: add interactive column filter dropdowns to generated dashboard"
```
