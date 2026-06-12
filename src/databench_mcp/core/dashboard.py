"""Dashboard generation — builds a standalone Dash app from chart sidecars."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from databench_mcp.db import get_connection
from databench_mcp.workspace import project_path


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

    _RENDERABLE = {"histogram", "boxplot", "scatter", "scatter_matrix",
                   "correlation_heatmap", "network_graph"}
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


def _load_findings(project: str) -> list[dict]:
    path = project_path(project) / "findings.yaml"
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


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
) -> str:
    all_tables = sorted(sidecars_by_table.keys())
    all_sidecars = [s for t in all_tables for s in sidecars_by_table[t]]

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
        "from dash import dcc, html, dash_table\n"
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

    footer = (
        "\n# --- layout ---\n"
        "_CHARTS = " + repr(all_sidecars) + "\n"
        "\n"
        "_FINDINGS = " + repr(findings_rows) + "\n"
        "\n"
        "app = dash.Dash(__name__)\n"
        "\n"
        "_tabs_by_table = {}\n"
        "for _c in _CHARTS:\n"
        '    _fig = _make_figure(_c["chart_type"], tables[_c["table"]],'
        ' _c["columns"], _c.get("params", {}))\n'
        '    _tabs_by_table.setdefault(_c["table"], []).append(dcc.Graph(figure=_fig))\n'
        "\n"
        "dataset_tabs = [\n"
        "    dcc.Tab(label=_tbl, children=[\n"
        '        html.Div(_figs, style={"display": "grid",'
        ' "gridTemplateColumns": "1fr 1fr", "gap": "16px"})\n'
        "    ])\n"
        "    for _tbl, _figs in sorted(_tabs_by_table.items())\n"
        "]\n"
        "\n"
        'findings_tab = dcc.Tab(label="Findings", children=[\n'
        "    dash_table.DataTable(\n"
        "        data=_FINDINGS,\n"
        '        columns=[{"name": c, "id": c} for c in'
        ' ["id", "method", "table", "summary", "metrics"]],\n'
        '        style_table={"overflowX": "auto"},\n'
        "    )\n"
        "])\n"
        "\n"
        "app.layout = html.Div([\n"
        "    dcc.Tabs(children=dataset_tabs + [findings_tab])\n"
        "])\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    app.run(debug=False)\n"
    )

    return header + _MAKE_FIGURE_SOURCE + footer


def build_dashboard(project: str) -> dict[str, Any]:
    """Generate a standalone Dash app from the project's chart artifacts and findings."""
    sidecars_by_table, skipped = _read_sidecars(project)

    if not sidecars_by_table:
        raise ValueError("no charts to embed — run create_chart first")

    dash_dir = _dashboards_dir(project)
    data_dir = dash_dir / "data"
    data_dir.mkdir(exist_ok=True)

    all_tables = sorted(sidecars_by_table.keys())
    _export_tables(project, all_tables, data_dir)

    findings = _load_findings(project)
    dashboard_py_path = dash_dir / "dashboard.py"

    warning_parts: list[str] = []
    if dashboard_py_path.exists():
        warning_parts.append("dashboard.py already existed and was overwritten")

    src = _generate_dashboard_py(sidecars_by_table, findings)
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
