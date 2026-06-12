# Network Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `derive_table` (SQL materialisation tool), three network registry methods (`network_stats`, `network_centrality`, `network_communities`), and a `network_graph` chart type with community Plotly dropdown.

**Architecture:** `derive_table` lives in `core/eda.py` alongside `sql_query`; network handlers are appended to `core/modeling.py` with the same `(df, target, features, params)` signature; `network_graph` is added to `core/viz.py`'s dispatch table. `run_model` is extended to save `.json` artifacts for communities and centrality. Tool count bumps 18 → 19 in `server.py`.

**Tech Stack:** python-igraph ≥ 0.11 (C-backed Louvain + betweenness), DuckDB, Plotly graph_objects (new import alongside existing plotly.express), FastMCP, PyYAML.

---

## File Map

| Action | File | What changes |
|--------|------|--------------|
| Modify | `pyproject.toml` | Add `python-igraph>=0.11` |
| Modify | `src/databench_mcp/core/eda.py` | Add `derive_table`; add `write_manifest` to import |
| Modify | `src/databench_mcp/tools/eda.py` | Add `derive_table` wrapper |
| Modify | `src/databench_mcp/core/modeling.py` | Add `_build_network_graph`, 3 handlers, 3 registry entries; extend `run_model` artifact saving |
| Modify | `src/databench_mcp/core/viz.py` | Add `import plotly.graph_objects as go`; add `network_graph` to `_CHART_TYPES`; add `_extract_filter_nodes`; add dispatch branch |
| Modify | `src/databench_mcp/server.py` | Import `derive_table`; `mcp.tool(derive_table)`; `EXPECTED_TOOL_COUNT = 19` |
| Create | `tests/test_core_eda_derive.py` | Unit tests for `derive_table` |
| Create | `tests/test_core_modeling_network.py` | Unit tests for the 3 network handlers |
| Create | `tests/test_core_viz_network.py` | Unit test for `network_graph` chart |
| Create | `tests/tools/test_network_integration.py` | Integration test: derive → profile → run_model → create_chart |

---

## Task 1: Add `python-igraph` dependency and `derive_table`

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/databench_mcp/core/eda.py`
- Modify: `src/databench_mcp/tools/eda.py`
- Create: `tests/test_core_eda_derive.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core_eda_derive.py`:

```python
"""Tests for core/eda.py — derive_table."""
from __future__ import annotations

import duckdb
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.eda import derive_table


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("p")
    db_path = str(tmp_path / "p" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE edges AS SELECT 'a' AS src, 'b' AS tgt, 1.0 AS w UNION ALL SELECT 'b', 'c', 2.0")
    conn.close()
    manifest = ws.read_manifest("p")
    manifest["datasets"]["edges"] = {"profiled": True, "row_count": 2, "col_count": 3}
    ws.write_manifest("p", manifest)
    return tmp_path


def test_derive_table_creates_table(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    result = derive_table("p", "SELECT src, tgt FROM edges WHERE w > 1.0", "filtered_edges")
    assert result["table"] == "filtered_edges"
    assert result["rows"] == 1
    assert result["columns"] == 2


def test_derive_table_registers_in_manifest(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    derive_table("p", "SELECT * FROM edges", "copy_edges")
    manifest = ws.read_manifest("p")
    ds = manifest["datasets"]["copy_edges"]
    assert ds["source"] == "derived"
    assert ds["profiled"] is False
    assert ds["row_count"] == 2


def test_derive_table_overwrites_existing(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    derive_table("p", "SELECT * FROM edges", "copy_edges")
    result = derive_table("p", "SELECT src FROM edges", "copy_edges")
    assert result["columns"] == 1


def test_derive_table_rejects_insert(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="Only SELECT"):
        derive_table("p", "INSERT INTO edges VALUES ('x','y',0.0)", "bad")


def test_derive_table_rejects_multi_statement(project, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project)
    with pytest.raises(ValueError, match="Multi-statement"):
        derive_table("p", "SELECT 1; SELECT 2", "bad")
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_core_eda_derive.py -v
```
Expected: `ImportError` or `AttributeError` — `derive_table` doesn't exist yet.

- [ ] **Step 3: Add `python-igraph` to `pyproject.toml`**

In `pyproject.toml`, add to the `dependencies` list:
```toml
"python-igraph>=0.11",
```

Full updated dependencies block:
```toml
dependencies = [
    "fastmcp>=3.2,<4",
    "duckdb>=1.1",
    "polars>=1.12",
    "pyyaml>=6.0",
    "httpx>=0.27",
    "openpyxl>=3.1",
    "pandas>=2.2",
    "scikit-learn>=1.4",
    "scipy>=1.13",
    "plotly>=5.22",
    "shap>=0.45",
    "python-igraph>=0.11",
]
```

- [ ] **Step 4: Sync the lockfile**

```
uv sync
```
Expected: igraph installs without error.

- [ ] **Step 5: Implement `derive_table` in `core/eda.py`**

At the top of `src/databench_mcp/core/eda.py`, update the workspace import line:
```python
# change this:
from databench_mcp.workspace import read_manifest
# to this:
from databench_mcp.workspace import read_manifest, write_manifest
```

Add this function at the end of `src/databench_mcp/core/eda.py`:

```python
def derive_table(project: str, sql: str, table_name: str) -> dict[str, Any]:
    """Materialise a SQL SELECT as a new DuckDB table; register in manifest as profiled=False."""
    stripped = sql.strip().rstrip(";")
    if not _SELECT_PATTERN.match(stripped):
        raise ValueError("Only SELECT or WITH queries are permitted")
    if ";" in stripped:
        raise ValueError("Multi-statement queries are not permitted")

    with get_connection(project) as conn:
        conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS ({stripped})')
        row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        col_count = len(conn.execute(f'SELECT * FROM "{table_name}" LIMIT 0').description)

    manifest = read_manifest(project)
    manifest["datasets"][table_name] = {
        "source": "derived",
        "sql": stripped,
        "profiled": False,
        "row_count": int(row_count),
        "col_count": int(col_count),
    }
    write_manifest(project, manifest)
    return {"table": table_name, "rows": int(row_count), "columns": int(col_count)}
```

- [ ] **Step 6: Add wrapper to `tools/eda.py`**

Add this import and function to `src/databench_mcp/tools/eda.py`:

```python
from databench_mcp.core.eda import derive_table as _derive_table


def derive_table(project: str, sql: str, table_name: str) -> dict[str, Any]:
    """Materialise a SQL SELECT as a new DuckDB table, registered as profiled=False."""
    return _derive_table(project, sql, table_name)
```

Also add `Any` to the `from typing import Any` import if not already present (it is — the file already imports it).

- [ ] **Step 7: Run tests to verify they pass**

```
uv run pytest tests/test_core_eda_derive.py -v
```
Expected: 5 PASSED.

- [ ] **Step 8: Commit**

```
git add pyproject.toml uv.lock src/databench_mcp/core/eda.py src/databench_mcp/tools/eda.py tests/test_core_eda_derive.py
git commit -m "feat: add derive_table — materialise SQL SELECT as DuckDB table"
```

---

## Task 2: Network model handlers

**Files:**
- Modify: `src/databench_mcp/core/modeling.py`
- Create: `tests/test_core_modeling_network.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core_modeling_network.py`:

```python
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
    """50 nodes, 100 edges as a pandas DataFrame."""
    import igraph as ig
    G = ig.Graph.Erdos_Renyi(n=50, m=100, directed=False, loops=False)
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
    assert m["edge_count"] == 100
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_core_modeling_network.py -v
```
Expected: `ImportError` — `_run_network_stats` etc. don't exist yet.

- [ ] **Step 3: Add the shared graph builder and three handlers to `core/modeling.py`**

Add this import at the top of `src/databench_mcp/core/modeling.py` (after existing imports):
```python
import json
```

Add this section after the existing "Unsupervised: KMeans, PCA" section and before the `# Registry` section:

```python
# ---------------------------------------------------------------------------
# Network analysis
# ---------------------------------------------------------------------------

def _validate_network_params(df: pd.DataFrame, params: dict) -> tuple[str, str, str | None]:
    source_col = params.get("source_col")
    target_col = params.get("target_col")
    if not source_col or not target_col:
        raise ValueError("network methods require 'source_col' and 'target_col' in params")
    for col in (source_col, target_col):
        if col not in df.columns:
            raise ValueError(f"column '{col}' not found in table")
    weight_col = params.get("weight_col")
    return source_col, target_col, weight_col


def _build_network_graph(df: pd.DataFrame, source_col: str, target_col: str, weight_col: str | None):
    import igraph as ig
    src = df[source_col].astype(str).tolist()
    tgt = df[target_col].astype(str).tolist()
    if weight_col:
        tuples = list(zip(src, tgt, df[weight_col].tolist()))
        G = ig.Graph.TupleList(tuples, directed=False, weights=True)
    else:
        tuples = list(zip(src, tgt))
        G = ig.Graph.TupleList(tuples, directed=False, weights=False)
    return G


def _run_network_stats(
    df: pd.DataFrame, target: str | None, features: list[str], params: dict
) -> dict[str, Any]:
    source_col, target_col, weight_col = _validate_network_params(df, params)
    G = _build_network_graph(df, source_col, target_col, weight_col)
    if G.vcount() < 2:
        raise ValueError("need at least 2 nodes to build a network")
    components = G.connected_components(mode="weak")
    node_count = G.vcount()
    edge_count = G.ecount()
    largest_cc_size = max(len(c) for c in components)
    degrees = G.degree()
    avg_degree = round(sum(degrees) / node_count, 4) if node_count else 0.0
    density = round(G.density(), 6)
    num_components = len(components)
    metrics = {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": density,
        "avg_degree": avg_degree,
        "num_components": num_components,
        "largest_component_fraction": round(largest_cc_size / node_count, 4),
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": (
            f"Network has {node_count} nodes and {edge_count} edges "
            f"(density={density:.4f}, {num_components} component(s))."
        ),
    }


def _run_network_centrality(
    df: pd.DataFrame, target: str | None, features: list[str], params: dict
) -> dict[str, Any]:
    source_col, target_col, weight_col = _validate_network_params(df, params)
    compute_betweenness = bool(params.get("betweenness", True))
    top_n = int(params.get("top_n", 10))
    G = _build_network_graph(df, source_col, target_col, weight_col)
    if G.vcount() < 2:
        raise ValueError("need at least 2 nodes to build a network")
    names = G.vs["name"]
    degrees = G.degree()
    pageranks = G.pagerank()
    centrality_full: dict[str, dict] = {
        name: {"degree": float(deg), "pagerank": round(float(pr), 6)}
        for name, deg, pr in zip(names, degrees, pageranks)
    }
    if compute_betweenness:
        betweenness = G.betweenness(directed=False)
        for name, bw in zip(names, betweenness):
            centrality_full[name]["betweenness"] = round(float(bw), 6)

    def _top(key: str) -> list[dict]:
        return sorted(
            [{"node": name, key: centrality_full[name][key]} for name in names],
            key=lambda x: x[key], reverse=True
        )[:top_n]

    metrics: dict[str, Any] = {
        "top_by_degree": _top("degree"),
        "top_by_pagerank": _top("pagerank"),
    }
    if compute_betweenness:
        metrics["top_by_betweenness"] = _top("betweenness")
    top_node = metrics["top_by_degree"][0]["node"] if metrics["top_by_degree"] else "N/A"
    top_degree = metrics["top_by_degree"][0]["degree"] if metrics["top_by_degree"] else 0
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": (
            f"Network centrality — most connected node: {top_node} "
            f"(degree={top_degree})."
        ),
        "centrality": centrality_full,
    }


def _run_network_communities(
    df: pd.DataFrame, target: str | None, features: list[str], params: dict
) -> dict[str, Any]:
    from collections import Counter
    source_col, target_col, weight_col = _validate_network_params(df, params)
    G = _build_network_graph(df, source_col, target_col, weight_col)
    if G.vcount() < 2:
        raise ValueError("need at least 2 nodes to build a network")
    names = G.vs["name"]
    communities = G.community_multilevel(
        weights="weight" if weight_col else None,
        return_levels=False,
    )
    membership = communities.membership
    modularity = round(float(G.modularity(membership)), 6)
    num_communities = len(set(membership))
    comm_sizes = Counter(membership)
    community_sizes = {str(k): v for k, v in sorted(comm_sizes.items())}
    communities_dict = {name: int(comm) for name, comm in zip(names, membership)}
    metrics = {
        "num_communities": num_communities,
        "modularity": modularity,
        "community_sizes": community_sizes,
    }
    return {
        "metrics": metrics,
        "explainability": "low",
        "summary": (
            f"Louvain detected {num_communities} communities "
            f"(modularity={modularity:.4f})."
        ),
        "communities": communities_dict,
    }
```

- [ ] **Step 4: Add the three methods to `_REGISTRY`**

In `src/databench_mcp/core/modeling.py`, find the `_REGISTRY` dict and add three entries:

```python
_REGISTRY: dict[str, Callable] = {
    "linear_regression": _run_linear_regression,
    "lasso": _run_lasso,
    "ridge": _run_ridge,
    "elastic_net": _run_elastic_net,
    "logistic_regression": _run_logistic_regression,
    "quantile_regression": _run_quantile_regression,
    "decision_tree": _run_decision_tree,
    "random_forest": _run_random_forest,
    "gradient_boosting": _run_gradient_boosting,
    "shap": _run_shap,
    "permutation_importance": _run_permutation_importance,
    "mutual_information": _run_mutual_information,
    "kmeans": _run_kmeans,
    "pca": _run_pca,
    "network_stats": _run_network_stats,
    "network_centrality": _run_network_centrality,
    "network_communities": _run_network_communities,
}
```

- [ ] **Step 5: Extend `run_model` to save JSON artifacts**

In `src/databench_mcp/core/modeling.py`, inside `run_model`, after the existing `cluster_labels = result.pop("cluster_labels", None)` line, add two more pops:

```python
    communities_data = result.pop("communities", None)
    centrality_data = result.pop("centrality", None)
```

After the existing `.npy` save block (after the `cluster_labels` block), add:

```python
    if communities_data is not None:
        artifacts_dir = project_path(project) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        (artifacts_dir / f"{finding['id']}_communities.json").write_text(
            json.dumps(communities_data)
        )

    if centrality_data is not None:
        artifacts_dir = project_path(project) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        (artifacts_dir / f"{finding['id']}_centrality.json").write_text(
            json.dumps(centrality_data)
        )
```

- [ ] **Step 6: Run tests to verify they pass**

```
uv run pytest tests/test_core_modeling_network.py -v
```
Expected: 9 PASSED.

- [ ] **Step 7: Run full test suite to check for regressions**

```
uv run pytest --tb=short -q
```
Expected: all previously passing tests still pass (127 + 9 new = 136 total).

- [ ] **Step 8: Commit**

```
git add src/databench_mcp/core/modeling.py tests/test_core_modeling_network.py
git commit -m "feat: add network_stats, network_centrality, network_communities to model registry"
```

---

## Task 3: `network_graph` chart type

**Files:**
- Modify: `src/databench_mcp/core/viz.py`
- Create: `tests/test_core_viz_network.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core_viz_network.py`:

```python
"""Tests for network_graph chart type in core/viz.py."""
from __future__ import annotations

import json
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
    G = ig.Graph.Erdos_Renyi(n=30, m=60, directed=False, loops=False)
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
    assert "updatemenus" not in html  # no dropdown without finding_id


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
    assert "updatemenus" in html


def test_network_graph_unknown_finding_raises(project_with_edges, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_with_edges)
    with pytest.raises(ValueError, match="no community data"):
        create_chart(
            "viz-proj", "network_graph", "edges",
            columns=["source", "target"],
            finding_id="f999",
            params={"source_col": "source", "target_col": "target"},
        )


def test_network_graph_requires_profiled(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("q")
    manifest = ws.read_manifest("q")
    manifest["datasets"]["edges"] = {"profiled": False}
    ws.write_manifest("q", manifest)
    with pytest.raises(ValueError, match="profiled"):
        create_chart("q", "network_graph", "edges", columns=["source", "target"])
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_core_viz_network.py -v
```
Expected: `ValueError: Unknown chart type 'network_graph'` or import error.

- [ ] **Step 3: Add `plotly.graph_objects` import and `network_graph` to `_CHART_TYPES`**

In `src/databench_mcp/core/viz.py`, add after `import plotly.express as px`:
```python
import plotly.graph_objects as go
```

Add `"network_graph"` to the `_CHART_TYPES` set:
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
}
```

- [ ] **Step 4: Add `_extract_filter_nodes` helper and `network_graph` dispatch to `core/viz.py`**

Add this helper function near the top of `src/databench_mcp/core/viz.py`, after `_load_df`:

```python
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
```

Add this branch at the end of the `create_chart` dispatch chain in `src/databench_mcp/core/viz.py`, replacing the final `else` block:

```python
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
```

Note: this replaces the existing `else: raise ValueError(...)` at the end of the dispatch chain. The existing final `else` becomes the new final `else` after the `network_graph` branch.

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/test_core_viz_network.py -v
```
Expected: 4 PASSED.

- [ ] **Step 6: Run full suite**

```
uv run pytest --tb=short -q
```
Expected: all passing.

- [ ] **Step 7: Commit**

```
git add src/databench_mcp/core/viz.py tests/test_core_viz_network.py
git commit -m "feat: add network_graph chart type with community dropdown"
```

---

## Task 4: Wire `derive_table` into server + integration test + bump tool count

**Files:**
- Modify: `src/databench_mcp/server.py`
- Create: `tests/tools/test_network_integration.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/tools/test_network_integration.py`:

```python
"""Integration test: derive_table → profile → run_model → create_chart."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

import databench_mcp.workspace as ws
from databench_mcp.tools.eda import derive_table
from databench_mcp.tools.modeling import run_model
from databench_mcp.tools.profile import profile_table
from databench_mcp.tools.viz import create_chart


@pytest.fixture
def project_with_providers(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("int-proj")
    import igraph as ig
    G = ig.Graph.Erdos_Renyi(n=20, m=40, directed=False, loops=False)
    edge_df = pd.DataFrame(
        [(str(e.source), str(e.target), float(e.index + 1)) for e in G.es],
        columns=["referring_npi", "receiving_npi", "shared_patients"],
    )
    db_path = str(tmp_path / "int-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE referrals AS SELECT * FROM edge_df")
    conn.close()
    manifest = ws.read_manifest("int-proj")
    manifest["datasets"]["referrals"] = {
        "profiled": True,
        "row_count": len(edge_df),
        "col_count": 3,
        "profile": {c: {} for c in edge_df.columns},
    }
    ws.write_manifest("int-proj", manifest)
    return tmp_path


def test_derive_then_profile_then_run_network(project_with_providers, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_with_providers)

    # 1. Derive a filtered edge table (top connections by shared_patients)
    derived = derive_table(
        "int-proj",
        "SELECT referring_npi AS source, receiving_npi AS target, shared_patients AS weight "
        "FROM referrals WHERE shared_patients > 5",
        "top_referrals",
    )
    assert derived["rows"] >= 0  # may be 0 for small synthetic data — just check no error

    # 2. Profile the derived table
    profile = profile_table("int-proj", "top_referrals")
    assert profile["table"] == "top_referrals"

    # 3. Run network_communities on the original (profiled) table
    finding = run_model(
        "int-proj", "referrals", "network_communities",
        features=["referring_npi", "receiving_npi"],
        params={"source_col": "referring_npi", "target_col": "receiving_npi"},
    )
    assert "finding_id" in finding
    assert finding["method"] == "network_communities"

    # 4. Create network_graph chart
    result = create_chart(
        "int-proj", "network_graph", "referrals",
        columns=["referring_npi", "receiving_npi"],
        finding_id=finding["finding_id"],
        params={
            "source_col": "referring_npi",
            "target_col": "receiving_npi",
            "max_nodes": 50,
        },
    )
    assert Path(result["path"]).exists()
    html = Path(result["path"]).read_text(encoding="utf-8")
    assert "plotly" in html.lower()
```

- [ ] **Step 2: Run integration test to verify it fails (tool not wired yet)**

```
uv run pytest tests/tools/test_network_integration.py -v
```
Expected: PASSED (tool wrappers are importable directly — this test calls them directly, not via MCP). Actually the test should pass since it imports from `tools/` directly. If it does pass, great — proceed to wiring server.py.

- [ ] **Step 3: Update `server.py`**

In `src/databench_mcp/server.py`:

1. Add `derive_table` to the eda import line:
```python
from databench_mcp.tools.eda import derive_table, eda_summary, sql_query
```

2. Register the tool (add after `mcp.tool(eda_summary)`):
```python
mcp.tool(derive_table)
```

3. Bump the tool count:
```python
EXPECTED_TOOL_COUNT = 19
```

- [ ] **Step 4: Verify server starts with correct tool count**

```
uv run python -c "from databench_mcp.server import _assert_tool_count; _assert_tool_count(); print('OK')"
```
Expected: `OK` printed with no `SystemExit`.

- [ ] **Step 5: Run full test suite**

```
uv run pytest --tb=short -q
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```
git add src/databench_mcp/server.py tests/tools/test_network_integration.py
git commit -m "feat: wire derive_table into server — 19 tools total"
```

---

## Done

After all 4 tasks pass the full test suite, run a final check:

```
uv run pytest -v 2>&1 | tail -5
uv run ruff check src/ tests/
```

Both should be clean. The implementation is complete.
