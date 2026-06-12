"""Chart generation — Plotly HTML saved to workspace/<project>/charts/."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from databench_mcp.core.findings import get_finding
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
}


def _charts_dir(project: str) -> Path:
    d = project_path(project) / "charts"
    d.mkdir(exist_ok=True)
    return d


def _save(fig, project: str, chart_type: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = _charts_dir(project) / f"{chart_type}_{ts}.html"
    fig.write_html(str(path))
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

    if chart_type == "histogram":
        col = columns[0]
        df = _load_df(project, table, [col])
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")

    elif chart_type == "boxplot":
        col = columns[0]
        df = _load_df(project, table, [col])
        fig = px.box(df, y=col, title=f"Box plot: {col}")

    elif chart_type == "scatter":
        x_col, y_col = columns[0], columns[1]
        df = _load_df(project, table, [x_col, y_col])
        color_col = params.get("color")
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                         title=f"{x_col} vs {y_col}")

    elif chart_type == "scatter_matrix":
        df = _load_df(project, table, columns)
        fig = px.scatter_matrix(df, dimensions=columns,
                                title="Scatter matrix: " + ", ".join(columns))

    elif chart_type == "correlation_heatmap":
        df = _load_df(project, table, columns if columns else None)
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        title="Correlation heatmap", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1)

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
        fig = px.bar(feat_df, x="importance", y="feature", orientation="h",
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
        df = _load_df(project, table, [x_col, y_col])
        df["cluster"] = labels.astype(str)
        fig = px.scatter(df, x=x_col, y=y_col, color="cluster",
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
        fig = px.bar(mean_abs, x="mean_abs_shap", y="feature", orientation="h",
                     title="SHAP mean absolute values (beeswarm proxy)")

    elif chart_type == "shap_waterfall":
        raise ValueError("shap_waterfall chart type is not yet implemented")

    elif chart_type in ("distribution_overlay", "partial_dependence"):
        raise ValueError(f"Chart type '{chart_type}' not yet implemented")

    elif chart_type == "network_graph":
        import igraph as ig
        import json as _json
        from collections import defaultdict

        source_col = params.get("source_col", columns[0] if columns else None)
        target_col_name = params.get("target_col", columns[1] if len(columns) > 1 else None)
        if not source_col or not target_col_name:
            raise ValueError("network_graph requires 'source_col' and 'target_col' in params or columns")
        weight_col = params.get("weight_col")
        color_by = params.get("color_by")
        max_nodes = int(params.get("max_nodes", 500))
        layout_name = params.get("layout", "spring")
        filter_finding_id = params.get("filter_finding_id")

        df = _load_df(project, table, None)

        src = df[source_col].astype(str).tolist()
        tgt = df[target_col_name].astype(str).tolist()
        if weight_col:
            tuples = list(zip(src, tgt, df[weight_col].tolist()))
            G = ig.Graph.TupleList(tuples, directed=False, weights=True)
        else:
            G = ig.Graph.TupleList(list(zip(src, tgt)), directed=False, weights=False)

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
            color_series = df.groupby(source_col)[color_by].mean()
            color_vals = [float(color_series.get(name, 0.0)) for name in names]
        else:
            color_vals = [float(d) for d in G.degree()]

        edge_x: list = []
        edge_y: list = []
        for edge in G.es:
            s, t = edge.source, edge.target
            edge_x += [xs[s], xs[t], None]
            edge_y += [ys[s], ys[t], None]

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
            all_communities = _json.loads(comm_path.read_text())
            node_comm = {name: all_communities.get(name, -1) for name in names}

            if filter_finding_id:
                ff = get_finding(project, filter_finding_id)
                filter_nodes = set(_extract_filter_nodes(ff, source_col))
                visible_comms = {node_comm[n] for n in filter_nodes if n in node_comm}
            else:
                visible_comms = set(node_comm.values())

            comm_to_idxs: dict = defaultdict(list)
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
            fig = go.Figure(data=[edge_trace] + node_traces, layout=base_layout)
        else:
            node_trace = go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=8, color=color_vals, colorscale="RdBu",
                            colorbar=dict(title=color_by or "degree"), showscale=True),
                text=names, hoverinfo="text",
            )
            base_layout.update(showlegend=False)
            fig = go.Figure(data=[edge_trace, node_trace], layout=base_layout)

    else:
        raise ValueError(f"Unknown chart type '{chart_type}'")

    path = _save(fig, project, chart_type)
    return {"chart_type": chart_type, "path": path, "title": fig.layout.title.text or chart_type}
