# Network Analysis Expansion — Design Spec

**Date:** 2026-06-12  
**Project:** databench-mcp  
**Status:** Approved

---

## Overview

Adds network analysis capability plus a general-purpose table derivation tool. One new MCP tool (`derive_table`) materializes any SQL SELECT as a new DuckDB table, enabling edge-list construction and other data transforms before analysis. Three new `run_model` methods then let users compute graph statistics, per-node centrality, and community structure on edge-list tables. One new chart type (`network_graph`) renders a force-directed Plotly HTML graph with a community dropdown that can be filtered to only show communities containing flagged nodes.

`EXPECTED_TOOL_COUNT` bumps from 18 → **19**.

---

## 1. Architecture

### Layer separation (unchanged pattern)

```
core/eda.py       →  derive_table (added alongside existing sql_query / eda_summary)
tools/eda.py      →  derive_table wrapper (added to existing module)
core/modeling.py  →  _run_network_stats, _run_network_centrality, _run_network_communities
core/viz.py       →  "network_graph" added to _CHART_TYPES dispatch
server.py         →  mcp.tool(derive_table); EXPECTED_TOOL_COUNT = 19
```

### New dependency

```toml
# pyproject.toml addition
networkx = ">=3.2"
```

No extra community-detection package needed — `networkx.community.louvain_communities` is available since networkx 3.0. No other new dependencies; `networkx` is the only addition.

### New artifacts

Both saved to `workspace/<project>/artifacts/`:

| File | Contents |
|------|----------|
| `{finding_id}_communities.json` | `{"node_id": community_int, …}` |
| `{finding_id}_centrality.json` | `{"node_id": {"degree": float, "betweenness": float, "closeness": float, "pagerank": float}, …}` |

### Preconditions

All three methods call `assert_profiled(project, table)`. They also validate that `source_col` and `target_col` exist in the table.

---

## 2. `derive_table` Tool

### Purpose

`sql_query` is read-only. `derive_table` materializes a SQL SELECT as a new DuckDB table so the result can be profiled and used by downstream analysis tools. The canonical pre-network workflow is:

```
sql_query(...)             →  explore and develop the aggregation SQL
derive_table(...)          →  materialize the edge list as a new table
profile_table(...)         →  stamp profiled=True
run_model("network_*")     →  analyze the edge-list table
```

### Contract

```
derive_table(project, sql, table_name)
→ { "table": str, "rows": int, "columns": int }
```

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `project` | str | yes | Must exist |
| `sql` | str | yes | Must be a SELECT or WITH … SELECT. Writes rejected. |
| `table_name` | str | yes | Name for the new table. Overwrites if it already exists. |

**Implementation** lives in `core/eda.py` alongside `sql_query`. Uses the same write-blocking guard (rejects anything that isn't SELECT/WITH). Executes `CREATE OR REPLACE TABLE "{table_name}" AS ({sql})`, then records the table in the manifest with `profiled=False`:

```python
manifest["datasets"][table_name] = {
    "source": "derived",
    "sql": sql,
    "profiled": False,
    "rows": row_count,
    "columns": col_count,
}
```

**Error handling:**

| Condition | Error |
|-----------|-------|
| SQL contains writes (INSERT/UPDATE/DELETE/DROP/CREATE) | `ValueError: only SELECT queries are allowed` |
| DuckDB execution error | re-raised as `ValueError: {duckdb error message}` |
| Invalid project | `FileNotFoundError` (from `read_manifest`) |

---

## 4. Method Registry Additions

`_REGISTRY` in `core/modeling.py` gains three entries:

```python
"network_stats":       _run_network_stats,
"network_centrality":  _run_network_centrality,
"network_communities": _run_network_communities,
```

All three receive `(df, target, features, params)` per the existing handler signature. `target` is unused; graph construction parameters are passed via `params`.

`features` is technically passed through but meaningless for network methods — `run_model` would default it to all numeric columns if not supplied. Users should pass `features=[source_col, target_col]` explicitly so the persisted finding record is readable. The handlers do not validate `features`; they only validate `params["source_col"]` and `params["target_col"]`.

### Required params (all three methods)

| Key | Type | Description |
|-----|------|-------------|
| `source_col` | str | Column name for edge source nodes |
| `target_col` | str | Column name for edge target nodes |
| `weight_col` | str \| None | Optional edge weight column |

Missing `source_col` or `target_col` raises `ValueError`.

### Graph construction (shared)

```python
import networkx as nx

def _build_graph(df, source_col, target_col, weight_col=None):
    if weight_col:
        G = nx.from_pandas_edgelist(df, source=source_col, target=target_col,
                                    edge_attr=weight_col)
    else:
        G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)
    return G
```

Graph is always undirected (`Graph`, not `DiGraph`) in Phase 4.5. Directed support is a future extension.

---

## 5. Method Contracts

### `network_stats`

```
run_model(project, table, method="network_stats",
          params={"source_col": "...", "target_col": "...", "weight_col": None})
→ {
    "finding_id": str,
    "method": "network_stats",
    "metrics": {
      "node_count": int,
      "edge_count": int,
      "density": float,
      "avg_degree": float,
      "num_components": int,
      "largest_component_fraction": float   # largest CC / total nodes
    },
    "explainability": "high",
    "summary": str,
    "created_at": str
  }
```

No artifact saved.

### `network_centrality`

```
run_model(project, table, method="network_centrality",
          params={"source_col": "...", "target_col": "...",
                  "betweenness": true,    # default true; set false for large graphs
                  "top_n": 10})
→ {
    "finding_id": str,
    "method": "network_centrality",
    "metrics": {
      "top_by_degree":     [{"node": str, "degree": float}, …],   # top_n entries
      "top_by_pagerank":   [{"node": str, "pagerank": float}, …],
      "top_by_betweenness":[{"node": str, "betweenness": float}, …]  # omitted if betweenness=false
    },
    "explainability": "medium",
    "summary": str,
    "created_at": str
  }
```

Saves `{finding_id}_centrality.json` with full per-node scores (all nodes, not just top_n).

### `network_communities`

```
run_model(project, table, method="network_communities",
          params={"source_col": "...", "target_col": "...", "seed": 42})
→ {
    "finding_id": str,
    "method": "network_communities",
    "metrics": {
      "num_communities": int,
      "modularity": float,
      "community_sizes": {"0": int, "1": int, …}  # community_int (as str) → node count
    },
    "explainability": "low",
    "summary": str,
    "created_at": str
  }
```

Saves `{finding_id}_communities.json`: `{"node_id": community_int, …}`.

**Explainability ratings** (added to existing table in spec §5):

| Rating | Methods (additions) |
|--------|---------------------|
| high   | `network_stats` |
| medium | `network_centrality` |
| low    | `network_communities` |

---

## 6. `network_graph` Chart

### Signature

```python
create_chart(
    project,
    chart_type="network_graph",
    table,
    columns=[source_col, target_col],   # edge list columns
    finding_id=None,                     # network_communities finding_id
    params={
        "color_by":          str | None,  # numeric column in table; averaged per source node
        "weight_col":        str | None,
        "filter_finding_id": str | None,  # finding whose node IDs filter the community dropdown
        "max_nodes":         int,          # default 500; keep top-N by degree
        "layout":            str,          # "spring" | "kamada_kawai" | "circular"; default "spring"
    }
)
→ {"chart_type": "network_graph", "path": str, "title": str}
```

### Node color

- If `color_by` is set: aggregate that column by mean per source node from the edge table. Color scale: Plotly's `RdBu` (low = blue, high = red).
- If `color_by` is not set: color by degree (computed from the graph).

### Community dropdown (requires `finding_id`)

1. Load `{finding_id}_communities.json`.
2. If `filter_finding_id` is set:
   - Load that finding from `findings.yaml`.
   - Extract node IDs: for `detect_outliers` findings, read each row dict in `sample_outliers` and extract the value at key `source_col` (the column name passed to the current `create_chart` call). For `network_centrality` findings, read node strings from `top_by_degree` / `top_by_pagerank` lists. Other finding types raise `ValueError: unsupported finding type for filter_finding_id`.
   - Map extracted nodes → community IDs via the communities JSON.
   - Build dropdown buttons only for communities containing at least one extracted node.
3. If no `filter_finding_id`: dropdown shows all communities.
4. Dropdown is a Plotly `updatemenus` component — one button per community, plus an "All" button at the top.
5. Each button sets `visible` on the per-community node traces.

If `finding_id` is not provided, all nodes render as a single trace with no dropdown.

### Large graph handling

If the graph has more nodes than `max_nodes` (default 500): keep the top-`max_nodes` nodes by degree and their induced subgraph. Log the truncation in the chart title: `"Network graph (top 500 of 12,400 nodes)"`.

### Edge rendering

All edges share one low-opacity line trace (`opacity=0.3`, `line_width=0.5`). Edges are not split by community (would be too noisy).

---

## 7. Error Handling

| Condition | Error |
|-----------|-------|
| `source_col` or `target_col` missing from params | `ValueError: network methods require 'source_col' and 'target_col' in params` |
| Column not found in table | `ValueError: column '{c}' not found in table` |
| `network_graph` with `finding_id` but no communities artifact | `ValueError: no community data for finding '{id}'; run run_model(method='network_communities') first` |
| `filter_finding_id` references unknown finding | `ValueError: finding '{id}' not found` |
| Graph has fewer than 2 nodes | `ValueError: need at least 2 nodes to build a network` |

---

## 8. Testing Strategy

```
tests/
  test_core_eda.py                # extend existing: derive_table happy path + write-rejection
  test_core_modeling_network.py   # unit: all three handlers with synthetic edge DataFrame
  test_core_viz_network.py        # unit: network_graph chart output is valid HTML
  tools/
    test_network_integration.py   # integration: derive_table → profile → run_model → create_chart
```

**`derive_table` tests:** happy path (SELECT creates table, manifest updated with `profiled=False`), write-blocked (INSERT raises ValueError), overwrite idempotent (calling twice with same `table_name` succeeds).

**Synthetic fixture:** 50-node, 100-edge random graph built with `networkx.gnm_random_graph(50, 100, seed=42)`, exported to a DataFrame with columns `source`, `target`, `weight`.

Each method: one happy-path test, one "fewer than 2 nodes" error test.

`network_centrality`: one test with `betweenness=False` to verify the key is omitted.

`network_graph`: test with `finding_id` (communities) and `filter_finding_id` (outlier stub); verify the HTML contains `"updatemenus"`.

---

## 9. Extensibility

To add directed graph support later:
1. Add `directed=False` param to all three network methods.
2. Switch `nx.from_pandas_edgelist` → `nx.from_pandas_edgelist` with `create_using=nx.DiGraph()`.
3. Degree → in-degree + out-degree.

No changes to tool contracts or `server.py`.
