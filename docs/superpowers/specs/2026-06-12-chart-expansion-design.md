# Chart Expansion — Design Spec

**Date:** 2026-06-12
**Project:** databench-mcp
**Status:** Approved

---

## Overview

Expands the chart layer in two directions:

1. **10 new chart types** added to `create_chart` — fills gaps in the Plotly basic chart gallery.
2. **New `create_subplot` tool** — places multiple charts in a single HTML grid using `make_subplots`.

`EXPECTED_TOOL_COUNT` bumps from 22 → **23**.

---

## 1. New Chart Types

Added to `_CHART_TYPES` in `core/viz.py` and handled in `create_chart`.

| `chart_type` | `columns` | Key `params` | Plotly call |
|---|---|---|---|
| `line` | [x, y] | `color` | `px.line` |
| `bar` | [x, y] | `color`, `barmode` (group/stack/overlay) | `px.bar` |
| `horizontal_bar` | [category, value] | `color`, `barmode` | `px.bar(..., orientation='h')` |
| `pie` | [names, values] | `hole` (0.0–1.0; >0 → donut) | `px.pie` |
| `bubble` | [x, y] | `size_col`, `color` | `px.scatter(..., size=params["size_col"])` |
| `dot` | [category, value] | `color` | Cleveland dot plot: `go.Scatter` markers, cat on y-axis |
| `table` | [] or [col, ...] | `max_rows` (default 100) | `go.Table` |
| `dumbbell` | [category, value1, value2] | `color1`, `color2` | Two `go.Scatter` series + horizontal lines per row |
| `parallel_categories` | [cat1, cat2, ...] | `color_col` | `px.parallel_categories` |
| `choropleth_map` | [locations, color] | `locations_format` (iso-3/iso-2/usa-states, default iso-3), `scope` (world/usa/europe, default world) | `px.choropleth_map` |

### Notes

- **`bubble`**: `size_col` must be a column in the same table; `columns` is `[x, y]` only.
- **`dot`** (Cleveland dot plot): y-axis is categorical (string column), x-axis is numeric. Sorted by value descending.
- **`dumbbell`**: `columns[0]` is category, `columns[1]` and `columns[2]` are the two numeric endpoints. Connected by horizontal line segments. Each row renders one dumbbell.
- **`table`**: If `columns` is empty, all columns are shown. `max_rows` caps rows to avoid giant HTMLs.
- **`choropleth_map`**: Uses `px.choropleth_map` (Plotly 5.18+) with OpenStreetMap tile background — no Mapbox token required. `locations_format` maps to Plotly's `locationmode`.
- **Categorical x-axes**: Already supported in `bar`, `line`, `scatter` etc. — no new chart type needed.

---

## 2. `create_subplot` Tool

### Contract

```python
create_subplot(
    project: str,
    charts: list[dict],        # [{chart_type, table, columns, params}, ...]
    rows: int,
    cols: int,
    title: str | None = None,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
) -> {
    "path": str,          # absolute path to saved HTML
    "rows": int,
    "cols": int,
    "charts_count": int,
}
```

### Algorithm

1. Validate `len(charts) <= rows * cols`; raise `ValueError` if exceeded.
2. For each chart spec, call `assert_profiled(project, table)` and load the DataFrame via `_load_df`.
3. Build a subplot figure: `make_subplots(rows=rows, cols=cols, shared_xaxes=shared_xaxes, shared_yaxes=shared_yaxes, subplot_titles=[c.get("title") for c in charts])`.
4. For each chart, call `_render_traces(chart_type, df, columns, params)` → list of `go.BaseTraceType`.
5. Add each trace to the subplot grid cell (row=i//cols+1, col=i%cols+1).
6. Set top-level title if provided.
7. Save via `_save(fig, project, "subplot", params_dict=None)` — no sidecar written (subplots are not re-embeddable in dashboards in v1).

### Error conditions

| Condition | Behaviour |
|---|---|
| `len(charts) > rows * cols` | `ValueError: N charts do not fit in R×C grid` |
| Chart spec references unknown table | `ValueError` from `assert_profiled` |
| Unknown `chart_type` in a chart spec | `ValueError` |

---

## 3. `_render_traces` Refactor

The key internal change: extract per-type rendering out of `create_chart` into a shared helper.

```python
def _render_traces(
    chart_type: str,
    df: pd.DataFrame,
    columns: list[str],
    params: dict,
) -> list[go.BaseTraceType]:
    ...
```

Returns raw traces (not a `Figure`). `create_chart` wraps the result:

```python
fig = go.Figure(data=_render_traces(...))
fig.update_layout(title=...)
```

`create_subplot` calls `_render_traces` per chart and places traces into grid cells.

Chart types that require artifact files (`cluster_scatter`, `shap_beeswarm`, `feature_importance_bar`) and the `network_graph` type retain their existing logic inside `_render_traces` — they just return traces instead of a full Figure.

---

## 4. `dashboard.py` `_make_figure` Update

`_MAKE_FIGURE_SOURCE` in `core/dashboard.py` is the plain-string source of `_make_figure` that lands verbatim in generated `dashboard.py` files. It needs branches for all 10 new chart types.

Subplots are **not** supported in `build_dashboard` v1 — subplot HTML files have no sidecar and are excluded from `_read_sidecars`.

---

## 5. File Map

| File | Change |
|---|---|
| `core/viz.py` | Add `_render_traces()` helper; update `create_chart` to call it; add `create_subplot()`; extend `_CHART_TYPES` |
| `tools/viz.py` | Add `create_subplot` wrapper |
| `server.py` | Register `create_subplot`; bump `EXPECTED_TOOL_COUNT` 22 → 23 |
| `core/dashboard.py` | Extend `_MAKE_FIGURE_SOURCE` with new chart type branches |
| `tests/test_core_viz.py` | One smoke test per new chart type; `create_subplot` grid test |
| `tests/tools/test_viz_integration.py` | Integration test: `create_subplot` 2-chart grid |

---

## 6. Testing Strategy

### `test_core_viz.py` additions

- One test per new chart type: ingest minimal CSV, call `create_chart`, assert HTML file written and result dict contains `path`.
- `test_create_subplot_basic` — 2 charts in a 1×2 grid; assert HTML written and `charts_count == 2`.
- `test_create_subplot_overflow` — 3 charts in 1×2 grid → `ValueError`.

### `test_viz_integration.py` additions

- `test_create_subplot_integration` — ingest 2 tables, create a subplot with one chart per table, verify HTML is valid and non-empty.

---

## 7. `EXPECTED_TOOL_COUNT`

22 → **23** (adding `create_subplot`).

No new CLI subcommand needed.
