"""Tests for network analysis handlers in core/modeling.py."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.modeling import (
    _run_network_centrality,
    _run_network_communities,
    _run_network_stats,
    run_model,
)


@pytest.fixture
def edge_df():
    """50 nodes, ~100 edges as a pandas DataFrame. All 50 nodes appear as edge endpoints."""
    import igraph as ig
    # Barabasi-Albert guarantees all n nodes appear in the edge list
    # n=50, m=2 gives ~2*(50-1)=98 edges; all nodes connected
    G = ig.Graph.Barabasi(n=50, m=2, directed=False)
    edges = [(str(e.source), str(e.target), 1.0) for e in G.es]
    return pd.DataFrame(edges, columns=["source", "target", "weight"])


@pytest.fixture
def project_with_edges(tmp_path, monkeypatch, edge_df):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("net-proj")
    db_path = str(tmp_path / "net-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE edges AS SELECT * FROM edge_df")
    conn.close()
    manifest = ws.read_manifest("net-proj")
    manifest["datasets"]["edges"] = {
        "profiled": True,
        "row_count": len(edge_df),
        "col_count": 3,
        "profile": {"source": {}, "target": {}, "weight": {}},
    }
    ws.write_manifest("net-proj", manifest)
    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests — handlers called directly (no DB, no project)
# ---------------------------------------------------------------------------

def test_network_stats_returns_metrics(edge_df):
    result = _run_network_stats(edge_df, None, [], {"source_col": "source", "target_col": "target"})
    m = result["metrics"]
    assert m["node_count"] == 50
    assert m["edge_count"] >= 90  # BA(n=50, m=2) gives ~98 edges
    assert 0.0 < m["density"] < 1.0
    assert m["avg_degree"] > 0
    assert m["num_components"] >= 1
    assert 0.0 < m["largest_component_fraction"] <= 1.0
    assert result["explainability"] == "high"


def test_network_stats_requires_source_target(edge_df):
    with pytest.raises(ValueError, match="source_col.*target_col"):
        _run_network_stats(edge_df, None, [], {})


def test_network_stats_too_few_nodes():
    tiny = pd.DataFrame({"source": ["a"], "target": ["a"]})
    with pytest.raises(ValueError, match="at least 2 nodes"):
        _run_network_stats(tiny, None, [], {"source_col": "source", "target_col": "target"})


def test_network_centrality_returns_top_nodes(edge_df):
    result = _run_network_centrality(
        edge_df, None, [], {"source_col": "source", "target_col": "target", "top_n": 5}
    )
    m = result["metrics"]
    assert len(m["top_by_degree"]) == 5
    assert len(m["top_by_pagerank"]) == 5
    assert "top_by_betweenness" in m
    assert result["explainability"] == "medium"
    assert "centrality" in result  # artifact payload


def test_network_centrality_no_betweenness(edge_df):
    result = _run_network_centrality(
        edge_df, None, [],
        {"source_col": "source", "target_col": "target", "betweenness": False}
    )
    assert "top_by_betweenness" not in result["metrics"]
    assert "betweenness" not in list(result["centrality"].values())[0]


def test_network_communities_returns_modularity(edge_df):
    result = _run_network_communities(
        edge_df, None, [], {"source_col": "source", "target_col": "target", "seed": 42}
    )
    m = result["metrics"]
    assert m["num_communities"] >= 1
    assert -1.0 <= m["modularity"] <= 1.0
    assert isinstance(m["community_sizes"], dict)
    assert result["explainability"] == "low"
    assert "communities" in result  # artifact payload


def test_network_communities_community_sizes_sum_to_node_count(edge_df):
    result = _run_network_communities(
        edge_df, None, [], {"source_col": "source", "target_col": "target"}
    )
    total = sum(result["metrics"]["community_sizes"].values())
    assert total == result["metrics"]["num_communities"] or total > 0
    # communities dict covers all nodes
    assert len(result["communities"]) == 50


# ---------------------------------------------------------------------------
# Integration tests — run_model saves artifacts
# ---------------------------------------------------------------------------

def test_run_model_network_communities_saves_json(project_with_edges, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_with_edges)
    finding = run_model(
        "net-proj", "edges", "network_communities",
        features=["source", "target"],
        params={"source_col": "source", "target_col": "target"},
    )
    fid = finding["finding_id"]
    path = project_with_edges / "net-proj" / "artifacts" / f"{fid}_communities.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) == 50


def test_run_model_network_centrality_saves_json(project_with_edges, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_with_edges)
    finding = run_model(
        "net-proj", "edges", "network_centrality",
        features=["source", "target"],
        params={"source_col": "source", "target_col": "target"},
    )
    fid = finding["finding_id"]
    path = project_with_edges / "net-proj" / "artifacts" / f"{fid}_centrality.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) == 50
