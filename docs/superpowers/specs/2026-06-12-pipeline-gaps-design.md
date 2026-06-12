# Pipeline Gaps — Design Spec

**Date:** 2026-06-12
**Project:** databench-mcp
**Status:** Approved

---

## Overview

Fills five pipeline gaps with five new MCP tools plus one dashboard enhancement:

**New tools:**
1. **`group_summary`** — SQL GROUP BY helper returning grouped aggregates
2. **`clean_table`** — Missing-data / imputation tool that materialises a cleaned derived table
3. **`add_lag`** — Time-series lag feature engineering
4. **`add_rolling`** — Time-series rolling-window feature engineering
5. **`enrich_table`** — Cross-table JOIN materialised as a new derived table

**Dashboard enhancement:**
6. **Interactive dashboards** — Column filter dropdowns per dataset tab; derived-tables-only mode

`EXPECTED_TOOL_COUNT` bumps 23 → 28 (five new tools; dashboard is an enhancement to `build_dashboard`).

---

## 1. `group_summary`

### Contract

```python
group_summary(
    project: str,
    table: str,
    group_col: str,
    agg_cols: list[str],
    agg_fns: list[str] = ["mean", "count", "min", "max", "std"],
) -> {
    "table": str,
    "group_col": str,
    "agg_cols": list[str],
    "agg_fns": list[str],
    "rows": list[dict],
    "row_count": int,
}
```

### Algorithm

1. `assert_profiled(project, table)`.
2. Build a single SELECT:
   ```sql
   SELECT "{group_col}",
          {comma-sep: agg_fn("{col}") AS "{col}_{agg_fn}" for each (col, fn)}
   FROM "{table}"
   GROUP BY "{group_col}"
   ORDER BY "{group_col}"
   ```
   Valid `agg_fns`: `mean` → `AVG`, `count` → `COUNT`, `min` → `MIN`, `max` → `MAX`, `std` → `STDDEV`.
3. Execute via `get_connection`; coerce results with `_to_json_safe`.
4. Return rows + metadata. No manifest writes.

### Errors

| Condition | Behaviour |
|---|---|
| Unknown `agg_fn` in list | `ValueError: unknown agg_fn 'X'` |
| `agg_cols` empty | `ValueError: agg_cols must not be empty` |
| Table not profiled | `ValueError` from `assert_profiled` |

---

## 2. `clean_table`

### Contract

```python
clean_table(
    project: str,
    table: str,
    strategy: str,
    new_table_name: str,
    columns: list[str] | None = None,   # None = all applicable columns
    fill_value: float | str | None = None,  # required for fill_constant
) -> {
    "table": str,           # new_table_name
    "source_table": str,
    "strategy": str,
    "rows": int,
    "columns": int,
    "affected_cols": list[str],
}
```

### Strategies

| Strategy | SQL approach |
|---|---|
| `drop_rows` | `WHERE col IS NOT NULL` for each target col |
| `drop_cols` | `SELECT *` minus the target cols (drops entire columns) |
| `fill_mean` | `COALESCE(col, AVG(col) OVER ())` for each target col |
| `fill_median` | `COALESCE(col, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col) OVER ())` |
| `fill_mode` | Subquery: `COALESCE(col, (SELECT mode(col) FROM "{table}"))` |
| `fill_constant` | `COALESCE(col, {fill_value})` for each target col |
| `fill_forward` | `LAST_VALUE(col IGNORE NULLS) OVER (ORDER BY rownum ROWS UNBOUNDED PRECEDING)` |
| `fill_backward` | `FIRST_VALUE(col IGNORE NULLS) OVER (ORDER BY rownum ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)` |

`fill_forward` / `fill_backward` use a `rownum` CTE: `SELECT *, ROW_NUMBER() OVER () AS _rn FROM "{table}"` then apply window over `_rn`, then drop `_rn` from final select.

`columns` defaults to all nullable numeric columns (for fill strategies) or all columns (for drop strategies). For `drop_rows`, `columns` specifies which columns' nulls trigger row removal.

### Algorithm

1. Validate `strategy` is in the allowed set; validate `new_table_name` is a safe identifier.
2. `assert_profiled(project, table)`.
3. Determine `affected_cols`: use `columns` if provided; else auto-select from profile (numeric + nullable for fills; any nullable for drops).
4. Build the SQL expression for the strategy, wrap in `CREATE OR REPLACE TABLE "{new_table_name}" AS (...)`.
5. Execute; read back row/col counts.
6. Register in manifest: `source: "clean_table"`, `source_table: table`, `strategy: strategy`.
7. Return result dict.

### Errors

| Condition | Behaviour |
|---|---|
| Unknown strategy | `ValueError` |
| `fill_constant` with `fill_value=None` | `ValueError: fill_value required for fill_constant` |
| `drop_cols` leaves zero columns | `ValueError: strategy would drop all columns` |

---

## 3. `add_lag`

### Contract

```python
add_lag(
    project: str,
    table: str,
    col: str,
    lags: list[int],
    new_table_name: str,
    time_col: str | None = None,
) -> {
    "table": str,           # new_table_name
    "source_table": str,
    "new_cols": list[str],  # e.g. ["sales_lag_1", "sales_lag_7"]
    "rows": int,
}
```

### Algorithm

1. Validate `lags` is non-empty, all positive integers. Validate `new_table_name`.
2. `assert_profiled(project, table)`.
3. Build SQL:
   ```sql
   CREATE OR REPLACE TABLE "{new_table_name}" AS (
     SELECT *,
       LAG("{col}", 1) OVER (ORDER BY "{time_col}") AS "{col}_lag_1",
       LAG("{col}", 7) OVER (ORDER BY "{time_col}") AS "{col}_lag_7",
       ...
     FROM "{table}"
   )
   ```
   If `time_col` is None, use `OVER ()` (insertion order).
4. Execute; read row count.
5. Register in manifest: `source: "add_lag"`, `source_table: table`, `col: col`, `lags: lags`.
6. Return result dict.

### Errors

| Condition | Behaviour |
|---|---|
| `lags` empty or contains non-positive int | `ValueError` |
| `col` not in table | `ValueError` from DuckDB |

---

## 4. `add_rolling`

### Contract

```python
add_rolling(
    project: str,
    table: str,
    col: str,
    window: int,
    agg_fn: str,
    new_table_name: str,
    time_col: str | None = None,
) -> {
    "table": str,           # new_table_name
    "source_table": str,
    "new_col": str,         # e.g. "sales_rolling_7_mean"
    "rows": int,
}
```

### Algorithm

1. Validate `window > 0`. Validate `agg_fn` in `{"mean", "sum", "min", "max", "std"}`.
2. `assert_profiled(project, table)`.
3. Map `agg_fn` → SQL: `mean` → `AVG`, `std` → `STDDEV`, others uppercase.
4. Build SQL:
   ```sql
   CREATE OR REPLACE TABLE "{new_table_name}" AS (
     SELECT *,
       {SQL_FN}("{col}") OVER (
         ORDER BY "{time_col}"
         ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW
       ) AS "{col}_rolling_{window}_{agg_fn}"
     FROM "{table}"
   )
   ```
   If `time_col` is None, use `OVER (ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW)`.
5. Execute; read row count.
6. Register in manifest: `source: "add_rolling"`, `source_table: table`, `col: col`, `window: window`, `agg_fn: agg_fn`.
7. Return result dict.

---

## 5. `enrich_table`

### Contract

```python
enrich_table(
    project: str,
    left_table: str,
    right_table: str,
    on: str | list[str],
    new_table_name: str,
    how: str = "inner",
) -> {
    "table": str,           # new_table_name
    "left_table": str,
    "right_table": str,
    "how": str,
    "on": list[str],
    "rows": int,
    "columns": int,
}
```

### Algorithm

1. Validate `how` in `{"inner", "left", "right", "full"}`. Validate `new_table_name`.
2. `assert_profiled(project, left_table)` and `assert_profiled(project, right_table)`.
3. Normalise `on` to `list[str]`. Build JOIN condition:
   - Single col: `ON "{left}"."{col}" = "{right}"."{col}"`
   - Multi col: `ON {AND-joined pairs}`
4. Build SQL:
   ```sql
   CREATE OR REPLACE TABLE "{new_table_name}" AS (
     SELECT * FROM "{left_table}"
     {HOW} JOIN "{right_table}"
     ON {join_condition}
   )
   ```
   `how="full"` → `FULL OUTER JOIN`.
5. Execute; read row/col counts.
6. Register in manifest: `source: "enrich_table"`, `left_table`, `right_table`, `on`, `how`.
7. Return result dict.

### Errors

| Condition | Behaviour |
|---|---|
| Unknown `how` | `ValueError` |
| Either table not profiled | `ValueError` from `assert_profiled` |
| `on` empty | `ValueError: on must specify at least one join column` |

---

## 6. Dashboard Enhancements

### 6a. Derived-Tables-Only Filter

`build_dashboard` reads the manifest and partitions tables:

- **Derived** (included): `source` not in `{"ingest_file", "ingest_url"}` — i.e. `"derived"`, `"clean_table"`, `"add_lag"`, `"add_rolling"`, `"enrich_table"`.
- **Raw** (excluded by default): `source` in `{"ingest_file", "ingest_url"}`.

If no derived tables have chart sidecars, fall back to all tables (existing behaviour) and add `warning: "no derived tables found — showing all tables"` to the result dict.

### 6b. Column Filter Dropdowns

At build time, `build_dashboard` inspects each table's profile to find **filterable columns**:
- dtype is `VARCHAR` / `TEXT` / string-like (DuckDB type contains `"VARCHAR"` or `"ENUM"`)
- unique-count (from profile `approx_unique`) ≤ 20

For each dataset tab, if ≥1 filterable column exists, generate a `dcc.Dropdown` row above the chart grid. Each dropdown:
- `id`: `f"filter-{table}-{col}"`
- `options`: list of unique string values (fetched at build time and embedded in generated code)
- `multi=True` (multi-select)
- `placeholder`: `f"Filter by {col}"`

**Charts that skip row-filtering** (computed on full data): `correlation_heatmap`. All other renderable chart types support row-level filtering.

**Generated callback structure** (one per tab):

```python
@app.callback(
    Output("tab-{table}-charts", "children"),
    [Input("filter-{table}-{col}", "value") for col in filterable_cols],
)
def update_{table}_charts(*filter_values):
    df = tables["{table}"].copy()
    for col, vals in zip({filterable_cols!r}, filter_values):
        if vals:
            df = df[df[col].isin(vals)]
    children = []
    for _c in [c for c in _CHARTS if c["table"] == "{table}"]:
        if _c["chart_type"] == "correlation_heatmap":
            _df = tables["{table}"]   # always full data
        else:
            _df = df
        children.append(dcc.Graph(figure=_make_figure(
            _c["chart_type"], _df, _c["columns"], _c.get("params", {}))))
    return children
```

If no filterable columns exist for a table, the tab uses the current static layout (no callback registered for it).

The `_generate_dashboard_py` function signature gains a `filterable: dict[str, list[str]]` parameter (table → list of filterable col names) and a `filter_options: dict[str, dict[str, list]]` parameter (table → col → sorted unique values list). Both are computed inside `build_dashboard` before calling `_generate_dashboard_py`.

---

## 7. File Map

| File | Change |
|---|---|
| `core/eda.py` | Add `group_summary`, `clean_table`, `add_lag`, `add_rolling`, `enrich_table` |
| `tools/eda.py` | Add wrappers for all 5 new functions |
| `server.py` | Register 5 new tools; bump `EXPECTED_TOOL_COUNT` 23 → 28 |
| `core/dashboard.py` | Add derived-table filter logic; add filterable-col detection; extend `_generate_dashboard_py` for callbacks |
| `tests/test_core_eda.py` | One smoke test per new function |
| `tests/tools/test_eda_integration.py` | Integration tests for new tool wrappers |
| `tests/test_core_dashboard.py` | Tests for derived-table filter; filterable-col detection; callback code generation |

---

## 8. Testing Strategy

### `test_core_eda.py` additions

- `test_group_summary_basic` — ingest CSV, profile, call `group_summary`, assert rows non-empty and aggregated col names present
- `test_clean_table_drop_rows` — ingest CSV with nulls, clean, assert row count decreased
- `test_clean_table_fill_mean` — assert no nulls in result
- `test_clean_table_fill_constant` — assert fill_value appears where nulls were
- `test_add_lag_basic` — assert new lag columns present and row count unchanged
- `test_add_rolling_basic` — assert rolling column present
- `test_enrich_table_inner` — ingest two CSVs, enrich, assert joined columns present
- `test_enrich_table_left` — assert left-join row count ≥ inner-join row count

### `tests/tools/test_eda_integration.py` additions

- `test_clean_table_integration` — clean_table creates table in manifest with correct source
- `test_add_lag_integration` — manifest has source="add_lag"
- `test_enrich_table_integration` — manifest has both source tables recorded

### `tests/test_core_dashboard.py` additions

- `test_build_dashboard_derived_only` — project with both raw and derived tables; assert dashboard tabs only include derived tables
- `test_build_dashboard_fallback_all_tables` — project with only raw tables; assert warning key present and all tables included
- `test_filterable_cols_detection` — profile a table with low-cardinality string col; assert it appears in generated callback code
- `test_no_filterable_cols_static` — profile a numeric-only table; assert no callback generated for it

---

## 9. `EXPECTED_TOOL_COUNT`

23 → **28** (adding `group_summary`, `clean_table`, `add_lag`, `add_rolling`, `enrich_table`).
