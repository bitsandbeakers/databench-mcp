# Chart Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 10 new chart types to `create_chart` and a new `create_subplot` tool that renders multiple charts in a single HTML grid.

**Architecture:** Extract the per-type rendering logic from `create_chart` into a shared `_render_figure()` helper; both `create_chart` and `create_subplot` call it. New types are added as branches in `_render_figure`. `dashboard.py`'s `_MAKE_FIGURE_SOURCE` string mirrors the same logic for generated standalone apps.

**Tech Stack:** Plotly Express / Graph Objects, `plotly.subplots.make_subplots`, pandas, FastMCP.

---

## Context

All files are under `src/databench_mcp/`. Tests run with `uv run pytest`. Workspace root is monkeypatched via `monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)` — never `setenv`. The `project_with_data` fixture (in `tests/conftest.py`) creates a `test-proj` with a `providers` table (columns: `npi int, specialty str, state str, total_drug_cost float, claim_count float, is_high_cost bool`, 60 rows).

`core/viz.py` currently has `_CHART_TYPES` (a set) and one big `create_chart` function with an if/elif block per chart type. After this plan:
- `_render_figure(chart_type, df, columns, params, finding_id=None, project=None) -> go.Figure` holds all per-type logic.
- `create_chart` loads the DataFrame, calls `_render_figure`, saves the HTML.
- `create_subplot` loads a DataFrame per chart spec, calls `_render_figure`, adds traces to a subplot grid.

---

## Task 1: Extract `_render_figure` + add basic chart types

**Files:**
- Modify: `src/databench_mcp/core/viz.py`
- Modify: `tests/test_core_viz.py`

New chart types in this task: `line`, `bar`, `horizontal_bar`, `pie`, `bubble`.

- [ ] **Step 1: Write failing tests for the 5 new chart types**

Add to `tests/test_core_viz.py` (after the last existing test):

```python
def test_create_chart_line(project_with_data):
    result = create_chart("test-proj", "line", "providers",
                          columns=["claim_count", "total_drug_cost"])
    assert result["chart_type"] == "line"
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_bar(project_with_data):
    result = create_chart("test-proj", "bar", "providers",
                          columns=["specialty", "total_drug_cost"])
    assert result["chart_type"] == "bar"
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_horizontal_bar(project_with_data):
    result = create_chart("test-proj", "horizontal_bar", "providers",
                          columns=["specialty", "total_drug_cost"])
    assert result["chart_type"] == "horizontal_bar"
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_pie(project_with_data):
    result = create_chart("test-proj", "pie", "providers",
                          columns=["specialty", "total_drug_cost"])
    assert result["chart_type"] == "pie"
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_bubble(project_with_data):
    result = create_chart("test-proj", "bubble", "providers",
                          columns=["claim_count", "total_drug_cost"],
                          params={"size_col": "claim_count"})
    assert result["chart_type"] == "bubble"
    from pathlib import Path
    assert Path(result["path"]).exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_core_viz.py::test_create_chart_line tests/test_core_viz.py::test_create_chart_bar tests/test_core_viz.py::test_create_chart_horizontal_bar tests/test_core_viz.py::test_create_chart_pie tests/test_core_viz.py::test_create_chart_bubble -v
```

Expected: FAIL with `ValueError: Unknown chart type`.

- [ ] **Step 3: Replace `core/viz.py` with the refactored version**

Replace the entire file `src/databench_mcp/core/viz.py` with the following. The key changes:
1. Add `from plotly.subplots import make_subplots` import.
2. Add 5 new types to `_CHART_TYPES`.
3. Extract all if/elif rendering into `_render_figure()`.
4. `create_chart` now calls `_render_figure` instead of having the logic inline.

```python
"""Chart generation — Plotly HTML saved to workspace/<project>/charts/."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from databench_mcp.core.findings import get_finding
from databench_mcp.core.modeling import _build_network_graph
from databench_mcp.db import get_connection
from databench_mcp.workspace import assert_profiled, project_path

_CHART_TYPES = {
    "histogram",
    "boxplot",
    "distribution_overlay",
    "correlation_heatmap",
    "feature_importance_bar",
    "scatter",
    "scatter_matrix",
    "cluster_scatter",
    "shap_beeswarm",
    "shap_waterfall",
    "partial_dependence",
    "network_graph",
    "line",
    "bar",
    "horizontal_bar",
    "pie",
    "bubble",
}

# Maps chart_type to make_subplots cell type
_SUBPLOT_TYPE: dict[str, str] = {
    "histogram": "xy", "boxplot": "xy", "scatter": "xy", "scatter_matrix": "xy",
    "correlation_heatmap": "xy", "line": "xy", "bar": "xy", "horizontal_bar": "xy",
    "bubble": "xy", "network_graph": "xy", "feature_importance_bar": "xy",
    "cluster_scatter": "xy", "shap_beeswarm": "xy",
    "pie": "pie",
}


def _charts_dir(project: str) -> Path:
    d = project_path(project) / "charts"
    d.mkdir(exist_ok=True)
    return d


def _save(fig, project: str, chart_type: str, params_dict: dict | None = None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = _charts_dir(project) / f"{chart_type}_{ts}.html"
    fig.write_html(str(path))
    if params_dict is not None:
        sidecar = _charts_dir(project) / f"{chart_type}_{ts}_params.json"
        sidecar.write_text(json.dumps(params_dict), encoding="utf-8")
    return str(path)


def _load_df(project: str, table: str, columns: list[str] | None) -> pd.DataFrame:
    if columns:
        cols_sql = ", ".join(f'"{c}"' for c in columns)
    else:
        cols_sql = "*"
    with get_connection(project) as conn:
        return conn.execute(f'SELECT {cols_sql} FROM "{table}"').df()


def _extract_filter_nodes(finding: dict, source_col: str) -> list[str]:
    """Extract node IDs from a finding for community dropdown filtering."""
    method = finding.get("method", "")
    if method == "detect_outliers":
        outliers = finding.get("metrics", {}).get("sample_outliers", [])
        return [str(row[source_col]) for row in outliers if source_col in row]
    elif method == "network_centrality":
        nodes: list[str] = []
        metrics = finding.get("metrics", {})
        for key in ("top_by_degree", "top_by_pagerank", "top_by_betweenness"):
            nodes.extend(str(e["node"]) for e in metrics.get(key, []))
        return list(set(nodes))
    else:
        raise ValueError(
            f"unsupported finding type for filter_finding_id: {method!r}"
        )


def _render_figure(
    chart_type: str,
    df: pd.DataFrame,
    columns: list[str],
    params: dict,
    finding_id: str | None = None,
    project: str | None = None,
) -> go.Figure:
    """Render a Plotly Figure from a pre-loaded DataFrame."""
    if chart_type == "histogram":
        col = columns[0]
        return px.histogram(df, x=col, title=f"Distribution of {col}")

    elif chart_type == "boxplot":
        col = columns[0]
        return px.box(df, y=col, title=f"Box plot: {col}")

    elif chart_type == "scatter":
        x_col, y_col = columns[0], columns[1]
        color_col = params.get("color")
        return px.scatter(df, x=x_col, y=y_col, color=color_col,
                          title=f"{x_col} vs {y_col}")

    elif chart_type == "scatter_matrix":
        return px.scatter_matrix(df, dimensions=columns,
                                 title="Scatter matrix: " + ", ".join(columns))

    elif chart_type == "correlation_heatmap":
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()
        return px.imshow(corr, text_auto=True, aspect="auto",
                         title="Correlation heatmap", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1)

    elif chart_type == "line":
        x_col, y_col = columns[0], columns[1]
        return px.line(df, x=x_col, y=y_col, color=params.get("color"),
                       title=f"{y_col} over {x_col}")

    elif chart_type == "bar":
        x_col, y_col = columns[0], columns[1]
        return px.bar(df, x=x_col, y=y_col, color=params.get("color"),
                      barmode=params.get("barmode", "relative"),
                      title=f"{y_col} by {x_col}")

    elif chart_type == "horizontal_bar":
        cat_col, val_col = columns[0], columns[1]
        return px.bar(df, x=val_col, y=cat_col, color=params.get("color"),
                      barmode=params.get("barmode", "relative"),
                      orientation="h", title=f"{val_col} by {cat_col}")

    elif chart_type == "pie":
        names_col, values_col = columns[0], columns[1]
        hole = float(params.get("hole", 0))
        return px.pie(df, names=names_col, values=values_col, hole=hole,
                      title=f"{values_col} by {names_col}")

    elif chart_type == "bubble":
        x_col, y_col = columns[0], columns[1]
        size_col = params.get("size_col")
        if size_col and size_col not in df.columns:
            raise ValueError(f"size_col '{size_col}' not found in DataFrame")
        return px.scatter(df, x=x_col, y=y_col, size=size_col,
                          color=params.get("color"),
                          title=f"{x_col} vs {y_col}" + (f" (size: {size_col})" if size_col else ""))

    elif chart_type == "feature_importance_bar":
        finding = get_finding(project, finding_id)
        importance = (
            finding["metrics"].get("feature_importance")
            or finding["metrics"].get("mean_abs_shap")
            or finding["metrics"].get("importance_mean")
            or finding["metrics"].get("mi_scores")
        )
        if importance is None:
            raise ValueError(f"Finding '{finding_id}' has no feature importance data")
        feat_df = pd.DataFrame(
            sorted(importance.items(), key=lambda x: x[1], reverse=True),
            columns=["feature", "importance"],
        )
        return px.bar(feat_df, x="importance", y="feature", orientation="h",
                      title=f"Feature importance — {finding['method']}")

    elif chart_type == "cluster_scatter":
        if finding_id is None:
            raise ValueError("cluster_scatter requires finding_id from a kmeans run")
        import numpy as np
        finding = get_finding(project, finding_id)
        labels_path = project_path(project) / "artifacts" / f"{finding_id}_labels.npy"
        if not labels_path.exists():
            raise ValueError(f"No cluster labels found for finding '{finding_id}'")
        labels = np.load(str(labels_path))
        x_col, y_col = columns[0], columns[1]
        df = df.copy()
        df["cluster"] = labels.astype(str)
        return px.scatter(df, x=x_col, y=y_col, color="cluster",
                          title=f"Cluster scatter: {x_col} vs {y_col}")

    elif chart_type == "shap_beeswarm":
        if finding_id is None:
            raise ValueError("shap_beeswarm requires finding_id from a shap run")
        import numpy as np
        finding = get_finding(project, finding_id)
        npy_path = project_path(project) / "artifacts" / f"{finding_id}_shap.npy"
        if not npy_path.exists():
            raise ValueError(f"No SHAP values found for finding '{finding_id}'")
        shap_vals = np.load(str(npy_path))
        feat_names = finding.get("features", [f"f{i}" for i in range(shap_vals.shape[1])])
        mean_abs = pd.DataFrame({
            "feature": feat_names,
            "mean_abs_shap": shap_vals.mean(axis=0) if shap_vals.ndim == 2
                             else [float(shap_vals.mean())],
        }).sort_values("mean_abs_shap", ascending=True)
        return px.bar(mean_abs, x="mean_abs_shap", y="feature", orientation="h",
                      title="SHAP mean absolute values (beeswarm proxy)")

    elif chart_type == "shap_waterfall":
        raise ValueError("shap_waterfall chart type is not yet implemented")

    elif chart_type in ("distribution_overlay", "partial_dependence"):
        raise ValueError(f"Chart type '{chart_type}' not yet implemented")

    elif chart_type == "network_graph":
        source_col = params.get("source_col", columns[0] if columns else None)
        target_col_name = params.get("target_col", columns[1] if len(columns) > 1 else None)
        if not source_col or not target_col_name:
            raise ValueError("network_graph requires 'source_col' and 'target_col' in params or columns")
        weight_col = params.get("weight_col")
        color_by = params.get("color_by")
        max_nodes = int(params.get("max_nodes", 500))
        layout_name = params.get("layout", "spring")
        filter_finding_id = params.get("filter_finding_id")

        if weight_col and weight_col not in df.columns:
            raise ValueError(f"weight_col '{weight_col}' not found in table")

        G = _build_network_graph(df, source_col, target_col_name, weight_col)

        original_count = G.vcount()
        if G.vcount() > max_nodes:
            degrees = G.degree()
            top_idxs = sorted(range(G.vcount()), key=lambda i: degrees[i], reverse=True)[:max_nodes]
            G = G.induced_subgraph(top_idxs)

        names = G.vs["name"]

        if layout_name == "kamada_kawai":
            layout_coords = G.layout_kamada_kawai()
        elif layout_name == "circular":
            layout_coords = G.layout_circle()
        else:
            layout_coords = G.layout_fruchterman_reingold()

        xs = [layout_coords[i][0] for i in range(G.vcount())]
        ys = [layout_coords[i][1] for i in range(G.vcount())]

        if color_by and color_by in df.columns:
            subgraph_node_set = set(names)
            color_series = (
                df[df[source_col].astype(str).isin(subgraph_node_set)]
                .groupby(source_col)[color_by]
                .mean()
            )
            color_vals = [float(color_series.get(name, 0.0)) for name in names]
        else:
            color_vals = [float(d) for d in G.degree()]

        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for edge in G.es:
            s, t = edge.source, edge.target
            edge_x.extend([xs[s], xs[t], None])
            edge_y.extend([ys[s], ys[t], None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.5, color="#aaa"), opacity=0.3,
            hoverinfo="none", showlegend=False,
        )

        title_str = f"Network graph ({G.vcount()} nodes, {G.ecount()} edges)"
        if original_count > max_nodes:
            title_str = f"Network graph (top {max_nodes} of {original_count} nodes)"

        base_layout = go.Layout(
            title=title_str, hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        if finding_id:
            comm_path = project_path(project) / "artifacts" / f"{finding_id}_communities.json"
            if not comm_path.exists():
                raise ValueError(
                    f"no community data for finding '{finding_id}'; "
                    f"run run_model(method='network_communities') first"
                )
            all_communities = json.loads(comm_path.read_text())
            node_comm = {name: all_communities.get(name, -1) for name in names}

            if filter_finding_id:
                ff = get_finding(project, filter_finding_id)
                filter_nodes = set(_extract_filter_nodes(ff, source_col))
                visible_comms = {node_comm[n] for n in filter_nodes if n in node_comm}
            else:
                visible_comms = set(node_comm.values())

            comm_to_idxs: dict[int, list[int]] = defaultdict(list)
            for i, name in enumerate(names):
                comm_to_idxs[node_comm[name]].append(i)

            all_comms_sorted = sorted(comm_to_idxs.keys())
            node_traces = []
            for comm in all_comms_sorted:
                idxs = comm_to_idxs[comm]
                node_traces.append(go.Scatter(
                    x=[xs[i] for i in idxs], y=[ys[i] for i in idxs],
                    mode="markers", name=f"Community {comm}",
                    marker=dict(
                        size=8,
                        color=[color_vals[i] for i in idxs],
                        colorscale="RdBu",
                        showscale=(comm == all_comms_sorted[0]),
                    ),
                    text=[names[i] for i in idxs], hoverinfo="text",
                ))

            n_node_traces = len(node_traces)
            all_btn = dict(
                label="All", method="update",
                args=[{"visible": [True] + [True] * n_node_traces}],
            )
            comm_btns = []
            for ci, comm in enumerate(all_comms_sorted):
                if comm in visible_comms:
                    vis = [True] + [j == ci for j in range(n_node_traces)]
                    comm_btns.append(dict(
                        label=f"Community {comm}", method="update",
                        args=[{"visible": vis}],
                    ))

            base_layout.update(
                showlegend=True,
                updatemenus=[dict(
                    buttons=[all_btn] + comm_btns,
                    direction="down", showactive=True,
                    x=0.01, xanchor="left", y=1.15, yanchor="top",
                )],
            )
            return go.Figure(data=[edge_trace] + node_traces, layout=base_layout)
        else:
            node_trace = go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=8, color=color_vals, colorscale="RdBu",
                            colorbar=dict(title=color_by or "degree"), showscale=True),
                text=names, hoverinfo="text",
            )
            base_layout.update(showlegend=False)
            return go.Figure(data=[edge_trace, node_trace], layout=base_layout)

    raise ValueError(f"Unknown chart type '{chart_type}'. Available: {sorted(_CHART_TYPES)}")


def create_chart(
    project: str,
    chart_type: str,
    table: str,
    columns: list[str],
    finding_id: str | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    """Generate a Plotly chart and save as standalone HTML."""
    assert_profiled(project, table)
    params = params or {}

    if chart_type not in _CHART_TYPES:
        raise ValueError(f"Unknown chart type '{chart_type}'. Available: {sorted(_CHART_TYPES)}")

    # Determine which columns to load from DB
    if chart_type == "network_graph":
        df = _load_df(project, table, None)
    elif chart_type == "bubble" and params.get("size_col"):
        df = _load_df(project, table, list(columns) + [params["size_col"]])
    elif not columns or chart_type == "correlation_heatmap":
        df = _load_df(project, table, None)
    else:
        df = _load_df(project, table, columns)

    fig = _render_figure(chart_type, df, columns, params, finding_id, project)

    params_dict = {
        "chart_type": chart_type,
        "table": table,
        "columns": columns,
        "finding_id": finding_id,
        "params": params,
    }
    path = _save(fig, project, chart_type, params_dict)
    return {"chart_type": chart_type, "path": path, "title": fig.layout.title.text or chart_type}
```

- [ ] **Step 4: Run all viz tests to verify no regression and new tests pass**

```
uv run pytest tests/test_core_viz.py tests/tools/test_viz.py -v
```

Expected: all passing including the 5 new tests.

- [ ] **Step 5: Commit**

```
git add src/databench_mcp/core/viz.py tests/test_core_viz.py
git commit -m "feat: extract _render_figure helper and add line/bar/horizontal_bar/pie/bubble chart types"
```

---

## Task 2: Add advanced chart types

**Files:**
- Modify: `src/databench_mcp/core/viz.py`
- Modify: `tests/test_core_viz.py`

New chart types: `dot`, `table`, `dumbbell`, `parallel_categories`, `choropleth_map`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_core_viz.py`:

```python
def test_create_chart_dot(project_with_data):
    result = create_chart("test-proj", "dot", "providers",
                          columns=["specialty", "total_drug_cost"])
    assert result["chart_type"] == "dot"
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_table(project_with_data):
    result = create_chart("test-proj", "table", "providers",
                          columns=["specialty", "total_drug_cost"])
    assert result["chart_type"] == "table"
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_dumbbell(project_with_data):
    result = create_chart("test-proj", "dumbbell", "providers",
                          columns=["specialty", "total_drug_cost", "claim_count"])
    assert result["chart_type"] == "dumbbell"
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_parallel_categories(project_with_data):
    result = create_chart("test-proj", "parallel_categories", "providers",
                          columns=["specialty", "state"])
    assert result["chart_type"] == "parallel_categories"
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_choropleth_map(project_with_data):
    import duckdb
    db_path = str(project_with_data / "test-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute(
        "CREATE TABLE state_totals AS "
        "SELECT state, AVG(total_drug_cost) AS avg_cost "
        "FROM providers GROUP BY state"
    )
    conn.close()
    manifest = ws.read_manifest("test-proj")
    manifest["datasets"]["state_totals"] = {
        "row_count": 4, "col_count": 2, "profiled": True,
        "profile": {"state": {"type": "VARCHAR"}, "avg_cost": {"type": "DOUBLE"}},
    }
    ws.write_manifest("test-proj", manifest)
    result = create_chart("test-proj", "choropleth_map", "state_totals",
                          columns=["state", "avg_cost"],
                          params={"locations_format": "usa-states", "scope": "usa"})
    assert result["chart_type"] == "choropleth_map"
    from pathlib import Path
    assert Path(result["path"]).exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_core_viz.py::test_create_chart_dot tests/test_core_viz.py::test_create_chart_table tests/test_core_viz.py::test_create_chart_dumbbell tests/test_core_viz.py::test_create_chart_parallel_categories tests/test_core_viz.py::test_create_chart_choropleth_map -v
```

Expected: FAIL with `ValueError: Unknown chart type`.

- [ ] **Step 3: Add 5 new types to `_CHART_TYPES` in `core/viz.py`**

In `src/databench_mcp/core/viz.py`, find the `_CHART_TYPES` set and add the 5 new types:

```python
_CHART_TYPES = {
    "histogram",
    "boxplot",
    "distribution_overlay",
    "correlation_heatmap",
    "feature_importance_bar",
    "scatter",
    "scatter_matrix",
    "cluster_scatter",
    "shap_beeswarm",
    "shap_waterfall",
    "partial_dependence",
    "network_graph",
    "line",
    "bar",
    "horizontal_bar",
    "pie",
    "bubble",
    "dot",
    "table",
    "dumbbell",
    "parallel_categories",
    "choropleth_map",
}
```

Also add these 5 to `_SUBPLOT_TYPE`:

```python
_SUBPLOT_TYPE: dict[str, str] = {
    "histogram": "xy", "boxplot": "xy", "scatter": "xy", "scatter_matrix": "xy",
    "correlation_heatmap": "xy", "line": "xy", "bar": "xy", "horizontal_bar": "xy",
    "bubble": "xy", "dot": "xy", "dumbbell": "xy", "network_graph": "xy",
    "feature_importance_bar": "xy", "cluster_scatter": "xy", "shap_beeswarm": "xy",
    "pie": "pie",
    "table": "table",
    "parallel_categories": "domain",
    "choropleth_map": "map",
}
```

- [ ] **Step 4: Add the 5 new chart type handlers to `_render_figure`**

In `src/databench_mcp/core/viz.py`, add these branches inside `_render_figure` after the `elif chart_type == "bubble":` block and before the `elif chart_type == "feature_importance_bar":` block:

```python
    elif chart_type == "dot":
        cat_col, val_col = columns[0], columns[1]
        sorted_df = df[[cat_col, val_col]].dropna().sort_values(val_col, ascending=True)
        fig = go.Figure(go.Scatter(
            x=sorted_df[val_col].tolist(),
            y=sorted_df[cat_col].astype(str).tolist(),
            mode="markers",
            marker=dict(size=10, color=params.get("color") or "steelblue"),
        ))
        fig.update_layout(title=f"{val_col} by {cat_col}",
                          yaxis=dict(type="category"))
        return fig

    elif chart_type == "table":
        max_rows = int(params.get("max_rows", 100))
        display_df = df[columns].head(max_rows) if columns else df.head(max_rows)
        fig = go.Figure(go.Table(
            header=dict(values=list(display_df.columns),
                        fill_color="paleturquoise", align="left"),
            cells=dict(values=[display_df[c].tolist() for c in display_df.columns],
                       fill_color="lavender", align="left"),
        ))
        fig.update_layout(title="Table" + (f": {', '.join(columns)}" if columns else ""))
        return fig

    elif chart_type == "dumbbell":
        cat_col, val1_col, val2_col = columns[0], columns[1], columns[2]
        color1 = params.get("color1", "steelblue")
        color2 = params.get("color2", "firebrick")
        df_s = df[[cat_col, val1_col, val2_col]].dropna().sort_values(val1_col)
        cats = df_s[cat_col].astype(str).tolist()
        vals1 = df_s[val1_col].tolist()
        vals2 = df_s[val2_col].tolist()
        line_x: list = []
        line_y: list = []
        for v1, v2, cat in zip(vals1, vals2, cats):
            line_x += [v1, v2, None]
            line_y += [cat, cat, None]
        fig = go.Figure([
            go.Scatter(x=line_x, y=line_y, mode="lines",
                       line=dict(color="gray", width=1),
                       hoverinfo="none", showlegend=False),
            go.Scatter(x=vals1, y=cats, mode="markers", name=val1_col,
                       marker=dict(size=10, color=color1)),
            go.Scatter(x=vals2, y=cats, mode="markers", name=val2_col,
                       marker=dict(size=10, color=color2)),
        ])
        fig.update_layout(title=f"{val1_col} vs {val2_col} by {cat_col}",
                          yaxis=dict(type="category"))
        return fig

    elif chart_type == "parallel_categories":
        color_col = params.get("color_col")
        return px.parallel_categories(
            df, dimensions=columns, color=color_col,
            title="Parallel categories: " + ", ".join(columns),
        )

    elif chart_type == "choropleth_map":
        locations_col, color_col = columns[0], columns[1]
        _loc_fmt = {"iso-3": "ISO-3", "iso-2": "ISO-2", "usa-states": "USA-states"}
        locationmode = _loc_fmt.get(params.get("locations_format", "iso-3"), "ISO-3")
        _scope_presets = {
            "world": {"zoom": 1, "center": {"lat": 0, "lon": 0}},
            "usa": {"zoom": 3, "center": {"lat": 37.09, "lon": -95.71}},
            "europe": {"zoom": 3, "center": {"lat": 54.53, "lon": 15.26}},
        }
        preset = _scope_presets.get(params.get("scope", "world"), _scope_presets["world"])
        return px.choropleth_map(
            df, locations=locations_col, color=color_col,
            locationmode=locationmode,
            zoom=preset["zoom"], center=preset["center"],
            title=f"{color_col} by {locations_col}",
        )
```

Also update `create_chart`'s df-loading block to handle the new types correctly. The only change needed is that `table` with no columns needs all columns:

```python
    # Determine which columns to load from DB
    if chart_type == "network_graph":
        df = _load_df(project, table, None)
    elif chart_type == "bubble" and params.get("size_col"):
        df = _load_df(project, table, list(columns) + [params["size_col"]])
    elif not columns or chart_type in ("correlation_heatmap", "table"):
        df = _load_df(project, table, None)
    else:
        df = _load_df(project, table, columns)
```

- [ ] **Step 5: Run tests to verify new types pass**

```
uv run pytest tests/test_core_viz.py -v
```

Expected: all passing (existing + 10 new).

- [ ] **Step 6: Commit**

```
git add src/databench_mcp/core/viz.py tests/test_core_viz.py
git commit -m "feat: add dot/table/dumbbell/parallel_categories/choropleth_map chart types"
```

---

## Task 3: `create_subplot` in `core/viz.py` and `tools/viz.py`

**Files:**
- Modify: `src/databench_mcp/core/viz.py`
- Modify: `src/databench_mcp/tools/viz.py`
- Modify: `tests/tools/test_viz.py`

- [ ] **Step 1: Write failing tests for `create_subplot`**

Add to `tests/tools/test_viz.py`:

```python
from databench_mcp.tools.viz import create_chart, create_subplot


def test_create_subplot_basic(project_with_data):
    charts = [
        {"chart_type": "histogram", "table": "providers",
         "columns": ["total_drug_cost"], "params": {}},
        {"chart_type": "scatter", "table": "providers",
         "columns": ["claim_count", "total_drug_cost"], "params": {}},
    ]
    result = create_subplot("test-proj", charts, rows=1, cols=2, title="Test subplot")
    from pathlib import Path
    assert Path(result["path"]).exists()
    assert result["charts_count"] == 2
    assert result["rows"] == 1
    assert result["cols"] == 2


def test_create_subplot_overflow(project_with_data):
    import pytest
    charts = [
        {"chart_type": "histogram", "table": "providers",
         "columns": ["total_drug_cost"], "params": {}},
        {"chart_type": "histogram", "table": "providers",
         "columns": ["claim_count"], "params": {}},
        {"chart_type": "scatter", "table": "providers",
         "columns": ["claim_count", "total_drug_cost"], "params": {}},
    ]
    with pytest.raises(ValueError, match="do not fit"):
        create_subplot("test-proj", charts, rows=1, cols=2)
```

- [ ] **Step 2: Run to verify they fail**

```
uv run pytest tests/tools/test_viz.py::test_create_subplot_basic tests/tools/test_viz.py::test_create_subplot_overflow -v
```

Expected: FAIL with `ImportError` or `AttributeError` (function not defined).

- [ ] **Step 3: Add `create_subplot` to `core/viz.py`**

Add the following function to `src/databench_mcp/core/viz.py` after `create_chart`:

```python
def create_subplot(
    project: str,
    charts: list[dict],
    rows: int,
    cols: int,
    title: str | None = None,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
) -> dict[str, Any]:
    """Combine multiple charts into a single subplot grid HTML."""
    if len(charts) > rows * cols:
        raise ValueError(
            f"{len(charts)} charts do not fit in {rows}×{cols} grid "
            f"(max {rows * cols})"
        )

    specs = [
        [{"type": _SUBPLOT_TYPE.get(
            charts[r * cols + c]["chart_type"] if r * cols + c < len(charts) else "xy",
            "xy"
         )}
         for c in range(cols)]
        for r in range(rows)
    ]
    subplot_titles = [c.get("title", c.get("chart_type", "")) for c in charts]
    subplot_titles += [""] * (rows * cols - len(charts))

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
    )

    for i, chart_spec in enumerate(charts):
        ct = chart_spec["chart_type"]
        tbl = chart_spec["table"]
        chcols = chart_spec.get("columns", [])
        chparams = chart_spec.get("params", {})

        assert_profiled(project, tbl)

        if ct == "network_graph":
            df = _load_df(project, tbl, None)
        elif ct == "bubble" and chparams.get("size_col"):
            df = _load_df(project, tbl, list(chcols) + [chparams["size_col"]])
        elif not chcols or ct in ("correlation_heatmap", "table"):
            df = _load_df(project, tbl, None)
        else:
            df = _load_df(project, tbl, chcols)

        sub_fig = _render_figure(ct, df, chcols, chparams)
        r, c = i // cols + 1, i % cols + 1
        for trace in sub_fig.data:
            fig.add_trace(trace, row=r, col=c)

    if title:
        fig.update_layout(title_text=title)

    path = _save(fig, project, "subplot", params_dict=None)
    return {"path": path, "rows": rows, "cols": cols, "charts_count": len(charts)}
```

- [ ] **Step 4: Add `create_subplot` wrapper to `tools/viz.py`**

Replace `src/databench_mcp/tools/viz.py` with:

```python
"""FastMCP tool wrappers for chart generation."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.viz import create_chart as _create_chart
from databench_mcp.core.viz import create_subplot as _create_subplot


def create_chart(
    project: str,
    chart_type: str,
    table: str,
    columns: list[str],
    finding_id: str | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    """Generate a Plotly HTML chart and save it to the project charts directory."""
    return _create_chart(project, chart_type, table, columns, finding_id, params)


def create_subplot(
    project: str,
    charts: list[dict],
    rows: int,
    cols: int,
    title: str | None = None,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
) -> dict[str, Any]:
    """Combine multiple charts into a single subplot grid HTML."""
    return _create_subplot(project, charts, rows, cols, title, shared_xaxes, shared_yaxes)
```

- [ ] **Step 5: Run tests to verify `create_subplot` passes**

```
uv run pytest tests/tools/test_viz.py -v
```

Expected: all 4 tests passing.

- [ ] **Step 6: Commit**

```
git add src/databench_mcp/core/viz.py src/databench_mcp/tools/viz.py tests/tools/test_viz.py
git commit -m "feat: add create_subplot tool"
```

---

## Task 4: Wire `server.py` + update `dashboard.py`

**Files:**
- Modify: `src/databench_mcp/server.py`
- Modify: `src/databench_mcp/core/dashboard.py`

- [ ] **Step 1: Register `create_subplot` in `server.py` and bump tool count**

In `src/databench_mcp/server.py`, add the import and registration:

```python
from databench_mcp.tools.viz import create_chart, create_subplot
```

(Replace the existing `from databench_mcp.tools.viz import create_chart` line.)

Then add `mcp.tool(create_subplot)` directly after `mcp.tool(create_chart)`:

```python
mcp.tool(create_chart)
mcp.tool(create_subplot)
```

Change `EXPECTED_TOOL_COUNT = 22` to `EXPECTED_TOOL_COUNT = 23`.

- [ ] **Step 2: Verify tool count is correct**

```
uv run python -c "from databench_mcp.server import _assert_tool_count; _assert_tool_count(); print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Update `_RENDERABLE` in `core/dashboard.py`**

In `src/databench_mcp/core/dashboard.py`, find `_read_sidecars` and replace the `_RENDERABLE` set:

Old:
```python
    _RENDERABLE = {"histogram", "boxplot", "scatter", "scatter_matrix",
                   "correlation_heatmap", "network_graph"}
```

New:
```python
    _RENDERABLE = {
        "histogram", "boxplot", "scatter", "scatter_matrix",
        "correlation_heatmap", "network_graph",
        "line", "bar", "horizontal_bar", "pie", "bubble",
        "dot", "table", "dumbbell", "parallel_categories", "choropleth_map",
    }
```

- [ ] **Step 4: Extend `_MAKE_FIGURE_SOURCE` in `core/dashboard.py`**

In `src/databench_mcp/core/dashboard.py`, find `_MAKE_FIGURE_SOURCE`. It currently ends with:

```python
    return go.Figure(layout=go.Layout(title=f"Unsupported: {chart_type}"))

"""
```

Insert the following new branches BEFORE that final `return` line (after the `elif chart_type == "network_graph":` block):

```python
    elif chart_type == "line":
        x_col, y_col = columns[0], columns[1]
        return px.line(df, x=x_col, y=y_col, color=params.get("color"),
                       title=f"{y_col} over {x_col}")
    elif chart_type == "bar":
        x_col, y_col = columns[0], columns[1]
        return px.bar(df, x=x_col, y=y_col, color=params.get("color"),
                      barmode=params.get("barmode", "relative"),
                      title=f"{y_col} by {x_col}")
    elif chart_type == "horizontal_bar":
        cat_col, val_col = columns[0], columns[1]
        return px.bar(df, x=val_col, y=cat_col, color=params.get("color"),
                      barmode=params.get("barmode", "relative"),
                      orientation="h", title=f"{val_col} by {cat_col}")
    elif chart_type == "pie":
        names_col, values_col = columns[0], columns[1]
        hole = float(params.get("hole", 0))
        return px.pie(df, names=names_col, values=values_col, hole=hole,
                      title=f"{values_col} by {names_col}")
    elif chart_type == "bubble":
        x_col, y_col = columns[0], columns[1]
        size_col = params.get("size_col")
        return px.scatter(df, x=x_col, y=y_col, size=size_col,
                          color=params.get("color"),
                          title=f"{x_col} vs {y_col}" + (f" (size: {size_col})" if size_col else ""))
    elif chart_type == "dot":
        cat_col, val_col = columns[0], columns[1]
        sorted_df = df[[cat_col, val_col]].dropna().sort_values(val_col, ascending=True)
        fig = go.Figure(go.Scatter(
            x=sorted_df[val_col].tolist(),
            y=sorted_df[cat_col].astype(str).tolist(),
            mode="markers",
            marker=dict(size=10, color=params.get("color") or "steelblue"),
        ))
        fig.update_layout(title=f"{val_col} by {cat_col}", yaxis=dict(type="category"))
        return fig
    elif chart_type == "table":
        max_rows = int(params.get("max_rows", 100))
        display_df = df[columns].head(max_rows) if columns else df.head(max_rows)
        fig = go.Figure(go.Table(
            header=dict(values=list(display_df.columns),
                        fill_color="paleturquoise", align="left"),
            cells=dict(values=[display_df[c].tolist() for c in display_df.columns],
                       fill_color="lavender", align="left"),
        ))
        fig.update_layout(title="Table" + (": " + ", ".join(columns) if columns else ""))
        return fig
    elif chart_type == "dumbbell":
        cat_col, val1_col, val2_col = columns[0], columns[1], columns[2]
        color1 = params.get("color1", "steelblue")
        color2 = params.get("color2", "firebrick")
        df_s = df[[cat_col, val1_col, val2_col]].dropna().sort_values(val1_col)
        cats = df_s[cat_col].astype(str).tolist()
        vals1 = df_s[val1_col].tolist()
        vals2 = df_s[val2_col].tolist()
        lx, ly = [], []
        for v1, v2, cat in zip(vals1, vals2, cats):
            lx += [v1, v2, None]; ly += [cat, cat, None]
        fig = go.Figure([
            go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="gray", width=1),
                       hoverinfo="none", showlegend=False),
            go.Scatter(x=vals1, y=cats, mode="markers", name=val1_col,
                       marker=dict(size=10, color=color1)),
            go.Scatter(x=vals2, y=cats, mode="markers", name=val2_col,
                       marker=dict(size=10, color=color2)),
        ])
        fig.update_layout(title=f"{val1_col} vs {val2_col} by {cat_col}",
                          yaxis=dict(type="category"))
        return fig
    elif chart_type == "parallel_categories":
        return px.parallel_categories(
            df, dimensions=columns, color=params.get("color_col"),
            title="Parallel categories: " + ", ".join(columns),
        )
    elif chart_type == "choropleth_map":
        locations_col, color_col = columns[0], columns[1]
        _loc_fmt = {"iso-3": "ISO-3", "iso-2": "ISO-2", "usa-states": "USA-states"}
        locationmode = _loc_fmt.get(params.get("locations_format", "iso-3"), "ISO-3")
        _presets = {
            "world": {"zoom": 1, "center": {"lat": 0, "lon": 0}},
            "usa": {"zoom": 3, "center": {"lat": 37.09, "lon": -95.71}},
            "europe": {"zoom": 3, "center": {"lat": 54.53, "lon": 15.26}},
        }
        preset = _presets.get(params.get("scope", "world"), _presets["world"])
        return px.choropleth_map(
            df, locations=locations_col, color=color_col,
            locationmode=locationmode,
            zoom=preset["zoom"], center=preset["center"],
            title=f"{color_col} by {locations_col}",
        )
```

**Important:** The `_MAKE_FIGURE_SOURCE` is a plain Python string, NOT an f-string. The `{col}`, `{x_col}`, etc. inside it are literal characters — they become valid f-string placeholders in the generated `dashboard.py` at runtime. Do not convert `_MAKE_FIGURE_SOURCE` to an f-string.

- [ ] **Step 5: Run the full test suite**

```
uv run pytest tests/ -q
```

Expected: all passing (target: 183+, depending on new test count).

- [ ] **Step 6: Commit**

```
git add src/databench_mcp/server.py src/databench_mcp/core/dashboard.py
git commit -m "feat: register create_subplot, bump tool count to 23, update dashboard chart support"
```
