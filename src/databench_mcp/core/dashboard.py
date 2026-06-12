"""Dashboard generation — builds a standalone Dash app from chart sidecars."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from databench_mcp.core.findings import _read_findings as _load_findings
from databench_mcp.db import get_connection
from databench_mcp.workspace import project_path, read_manifest


_DERIVED_SOURCES = {"derived", "clean_table", "add_lag", "add_rolling", "enrich_table"}


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


_REQUIREMENTS = (
    "dash==2.17.1\n"
    "plotly==5.22.0\n"
    "pandas==2.2.2\n"
    "pyarrow==16.1.0\n"
)

_DEPLOY_MD = """\
# Deploy to Dash Community Cloud

1. Install the Dash CLI: `pip install dash`
2. Log in: `dash login`
3. From this directory: `dash deploy`

The app will be live at https://your-username.pythonanywhere.com/<project-name>
"""

# Raw string — NOT an f-string. Curly braces like {col} are intentional:
# they become f-string placeholders in the *generated* dashboard.py.
_MAKE_FIGURE_SOURCE = """\
def _make_figure(chart_type, df, columns, params):
    if chart_type == "histogram":
        col = columns[0]
        return px.histogram(df, x=col, title=f"Distribution of {col}")
    elif chart_type == "boxplot":
        col = columns[0]
        return px.box(df, y=col, title=f"Box plot: {col}")
    elif chart_type == "scatter":
        x_col, y_col = columns[0], columns[1]
        return px.scatter(df, x=x_col, y=y_col, color=params.get("color"),
                          title=f"{x_col} vs {y_col}")
    elif chart_type == "scatter_matrix":
        return px.scatter_matrix(df, dimensions=columns,
                                 title="Scatter matrix: " + ", ".join(columns))
    elif chart_type == "correlation_heatmap":
        corr = df.select_dtypes(include="number").corr()
        return px.imshow(corr, text_auto=True, aspect="auto",
                         title="Correlation heatmap",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    elif chart_type == "network_graph":
        try:
            import igraph as ig
        except ImportError:
            return go.Figure(layout=go.Layout(title="network_graph requires python-igraph"))
        src = params.get("source_col", columns[0] if columns else None)
        tgt = params.get("target_col", columns[1] if len(columns) > 1 else None)
        if not src or not tgt:
            return go.Figure(layout=go.Layout(title="network_graph: missing source/target"))
        G = ig.Graph.TupleList(zip(df[src].astype(str), df[tgt].astype(str)), directed=False)
        layout_name = params.get("layout", "spring")
        coords = (G.layout_kamada_kawai() if layout_name == "kamada_kawai"
                  else G.layout_circle() if layout_name == "circular"
                  else G.layout_fruchterman_reingold())
        xs = [coords[i][0] for i in range(G.vcount())]
        ys = [coords[i][1] for i in range(G.vcount())]
        names = G.vs["name"]
        degrees = [float(d) for d in G.degree()]
        ex, ey = [], []
        for e in G.es:
            ex += [xs[e.source], xs[e.target], None]
            ey += [ys[e.source], ys[e.target], None]
        return go.Figure(
            data=[
                go.Scatter(x=ex, y=ey, mode="lines",
                           line=dict(width=0.5, color="#aaa"), opacity=0.3,
                           hoverinfo="none", showlegend=False),
                go.Scatter(x=xs, y=ys, mode="markers",
                           marker=dict(size=8, color=degrees,
                                       colorscale="RdBu", showscale=True),
                           text=names, hoverinfo="text"),
            ],
            layout=go.Layout(
                title=f"Network graph ({G.vcount()} nodes, {G.ecount()} edges)",
                hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                showlegend=False,
            )
        )
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
        return px.choropleth(
            df, locations=locations_col, color=color_col,
            locationmode=locationmode,
            scope=params.get("scope", "world"),
            title=f"{color_col} by {locations_col}",
        )
    return go.Figure(layout=go.Layout(title=f"Unsupported: {chart_type}"))

"""


def _dashboards_dir(project: str) -> Path:
    d = project_path(project) / "dashboards"
    d.mkdir(exist_ok=True)
    return d


def _read_sidecars(project: str) -> tuple[dict[str, list[dict]], list[dict]]:
    """Return (renderable_by_table, skipped_list).

    Skipped: finding-dependent charts (require .npy / _communities.json artifacts).
    """
    charts_dir = project_path(project) / "charts"
    if not charts_dir.exists():
        return {}, []

    _RENDERABLE = {
        "histogram", "boxplot", "scatter", "scatter_matrix",
        "correlation_heatmap", "network_graph",
        "line", "bar", "horizontal_bar", "pie", "bubble",
        "dot", "table", "dumbbell", "parallel_categories", "choropleth_map",
    }
    _SKIP_ALWAYS = {"cluster_scatter", "shap_beeswarm", "feature_importance_bar",
                    "shap_waterfall", "distribution_overlay", "partial_dependence"}

    renderable: dict[str, list[dict]] = {}
    skipped: list[dict] = []

    for f in sorted(charts_dir.glob("*_params.json")):
        sidecar = json.loads(f.read_text(encoding="utf-8"))
        ct = sidecar.get("chart_type", "")

        if ct in _SKIP_ALWAYS or ct not in _RENDERABLE:
            skipped.append(sidecar)
            continue
        if ct == "network_graph" and sidecar.get("finding_id"):
            skipped.append(sidecar)
            continue

        renderable.setdefault(sidecar.get("table", ""), []).append(sidecar)

    return renderable, skipped


def _export_tables(project: str, tables: list[str], data_dir: Path) -> None:
    """Export each table from DuckDB to parquet using DuckDB's native COPY."""
    with get_connection(project) as conn:
        for table in tables:
            path = data_dir / f"{table}.parquet"
            conn.execute(f'COPY "{table}" TO {path.as_posix()!r} (FORMAT PARQUET)')


def _format_metrics(metrics: dict) -> str:
    if not metrics:
        return ""
    return "; ".join(
        f"{k}={v}" for k, v in metrics.items()
        if isinstance(v, (int, float, str, bool))
    )


def _generate_dashboard_py(
    sidecars_by_table: dict[str, list[dict]],
    findings: list[dict],
    filterable: dict[str, list[str]] | None = None,
    filter_options: dict[str, dict[str, list]] | None = None,
) -> str:
    all_tables = sorted(sidecars_by_table.keys())
    all_sidecars = [s for t in all_tables for s in sidecars_by_table[t]]
    if filterable is None:
        filterable = {}
    if filter_options is None:
        filter_options = {}
    filterable_tables = sorted(filterable.keys())

    findings_rows = [
        {
            "id": f.get("id", ""),
            "method": f.get("method", ""),
            "table": f.get("table", ""),
            "summary": f.get("summary", ""),
            "metrics": _format_metrics(f.get("metrics", {})),
        }
        for f in findings
    ]

    tables_entries = "\n".join(
        f'    {t!r}: pd.read_parquet(DATA_DIR / {(t + ".parquet")!r}),'
        for t in all_tables
    )

    header = (
        "# Generated by databench-mcp build_dashboard — do not edit manually.\n"
        "# Deploy: pip install dash plotly pandas pyarrow && python dashboard.py  OR  dash deploy\n"
        "import dash\n"
        "from dash import dcc, html, dash_table"
        + (", Input, Output" if filterable_tables else "")
        + "\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "\n"
        'DATA_DIR = Path(__file__).parent / "data"\n'
        "\n"
        "# --- data loading ---\n"
        "tables = {\n"
        + tables_entries + "\n"
        + "}\n"
        "\n"
        "# --- chart helper ---\n"
    )

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

    return header + _MAKE_FIGURE_SOURCE + footer


def build_dashboard(project: str) -> dict[str, Any]:
    """Generate a standalone Dash app from the project's chart artifacts and findings."""
    sidecars_by_table, skipped = _read_sidecars(project)

    warning_parts: list[str] = []

    if not sidecars_by_table:
        raise ValueError("no charts to embed — run create_chart first")

    # Filter to derived tables only; fall back to all if no derived tables have charts
    manifest = read_manifest(project)

    def _is_derived(table: str) -> bool:
        ds = manifest.get("datasets", {}).get(table, {})
        return ds.get("source", "") in _DERIVED_SOURCES

    derived_tables_with_charts = {t for t in sidecars_by_table if _is_derived(t)}

    if derived_tables_with_charts:
        sidecars_by_table = {t: v for t, v in sidecars_by_table.items()
                             if t in derived_tables_with_charts}
    else:
        warning_parts.append("no derived tables found — showing all tables")

    dash_dir = _dashboards_dir(project)
    data_dir = dash_dir / "data"
    data_dir.mkdir(exist_ok=True)

    all_tables = sorted(sidecars_by_table.keys())
    _export_tables(project, all_tables, data_dir)

    filterable = _filterable_cols(manifest, all_tables)
    filter_opts = _fetch_filter_options(project, filterable)

    findings = _load_findings(project)
    dashboard_py_path = dash_dir / "dashboard.py"

    if dashboard_py_path.exists():
        warning_parts.append("dashboard.py already existed and was overwritten")

    src = _generate_dashboard_py(sidecars_by_table, findings, filterable, filter_opts)
    dashboard_py_path.write_text(src, encoding="utf-8")

    req_path = dash_dir / "requirements.txt"
    req_path.write_text(_REQUIREMENTS, encoding="utf-8")

    deploy_path = dash_dir / "DEPLOY.md"
    deploy_path.write_text(_DEPLOY_MD, encoding="utf-8")

    charts_embedded = sum(len(v) for v in sidecars_by_table.values())

    if skipped:
        warning_parts.append(
            f"skipped {len(skipped)} chart(s) requiring binary artifacts: "
            + ", ".join(s.get("chart_type", "?") for s in skipped)
        )

    return {
        "dashboard_py": str(dashboard_py_path),
        "requirements_txt": str(req_path),
        "deploy_md": str(deploy_path),
        "tabs": len(sidecars_by_table),
        "charts_embedded": charts_embedded,
        "tables_exported": all_tables,
        "warning": "; ".join(warning_parts) if warning_parts else None,
    }
