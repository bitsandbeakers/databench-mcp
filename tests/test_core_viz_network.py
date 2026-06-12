"""Tests for network_graph chart type in core/viz.py."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.modeling import run_model
from databench_mcp.core.viz import create_chart


@pytest.fixture
def project_with_edges(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("viz-proj")
    import igraph as ig
    G = ig.Graph.Barabasi(n=30, m=2, directed=False)
    edge_df = pd.DataFrame(
        [(str(e.source), str(e.target), 1.0) for e in G.es],
        columns=["source", "target", "weight"],
    )
    db_path = str(tmp_path / "viz-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE edges AS SELECT * FROM edge_df")
    conn.close()
    manifest = ws.read_manifest("viz-proj")
    manifest["datasets"]["edges"] = {
        "profiled": True,
        "row_count": len(edge_df),
        "col_count": 3,
        "profile": {"source": {}, "target": {}, "weight": {}},
    }
    ws.write_manifest("viz-proj", manifest)
    return tmp_path


def test_network_graph_no_finding(project_with_edges, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_with_edges)
    result = create_chart(
        "viz-proj", "network_graph", "edges",
        columns=["source", "target"],
        params={"source_col": "source", "target_col": "target", "max_nodes": 50},
    )
    assert result["chart_type"] == "network_graph"
    html = Path(result["path"]).read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    # When no finding_id, the figure JSON should not contain updatemenus array.
    # The Plotly.js bundle contains the bare word, so check for the data-JSON pattern.
    assert '"updatemenus":[' not in html  # no dropdown without finding_id


def test_network_graph_with_communities_has_dropdown(project_with_edges, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_with_edges)
    finding = run_model(
        "viz-proj", "edges", "network_communities",
        features=["source", "target"],
        params={"source_col": "source", "target_col": "target"},
    )
    result = create_chart(
        "viz-proj", "network_graph", "edges",
        columns=["source", "target"],
        finding_id=finding["finding_id"],
        params={"source_col": "source", "target_col": "target", "max_nodes": 50},
    )
    html = Path(result["path"]).read_text(encoding="utf-8")
    assert '"updatemenus":[' in html


def test_network_graph_unknown_finding_raises(project_with_edges, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_with_edges)
    with pytest.raises(ValueError, match="no community data"):
        create_chart(
            "viz-proj", "network_graph", "edges",
            columns=["source", "target"],
            finding_id="f999",
            params={"source_col": "source", "target_col": "target"},
        )


def test_network_graph_filter_finding_id_limits_dropdown(project_with_edges, monkeypatch):
    """filter_finding_id narrows the dropdown to communities containing filter nodes."""
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_with_edges)
    comm_finding = run_model(
        "viz-proj", "edges", "network_communities",
        features=["source", "target"],
        params={"source_col": "source", "target_col": "target"},
    )
    cent_finding = run_model(
        "viz-proj", "edges", "network_centrality",
        features=["source", "target"],
        params={"source_col": "source", "target_col": "target"},
    )
    result = create_chart(
        "viz-proj", "network_graph", "edges",
        columns=["source", "target"],
        finding_id=comm_finding["finding_id"],
        params={
            "source_col": "source",
            "target_col": "target",
            "max_nodes": 50,
            "filter_finding_id": cent_finding["finding_id"],
        },
    )
    html = Path(result["path"]).read_text(encoding="utf-8")
    # Dropdown is still present (communities finding still drives it)
    assert '"updatemenus":[' in html
    # Total community buttons ≤ total communities (filter may restrict to a subset).
    # Use label-specific pattern to avoid matching trace name fields.
    total_comms = comm_finding["metrics"]["num_communities"]
    import re
    button_labels = re.findall(r'"label":"(Community \d+)"', html)
    assert 1 <= len(button_labels) <= total_comms


def test_network_graph_requires_profiled(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("q")
    manifest = ws.read_manifest("q")
    manifest["datasets"]["edges"] = {"profiled": False}
    ws.write_manifest("q", manifest)
    with pytest.raises(ValueError, match="profiled"):
        create_chart("q", "network_graph", "edges", columns=["source", "target"])
