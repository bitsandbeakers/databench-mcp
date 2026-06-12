# build_dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `build_dashboard(project)` — generates a standalone Dash app from a project's chart sidecars, exports each table to parquet, and returns paths to output files.

**Architecture:** Three-layer pattern (core/dashboard.py → tools/dashboard.py → server.py). Core handles all logic; tools module is a one-line wrapper; server registers and bumps `EXPECTED_TOOL_COUNT` 21 → 22. Generated `dashboard.py` is standalone (`pip install dash plotly pandas pyarrow`) — embeds sidecar data via `repr()` and inlines `_make_figure` as a raw string constant to avoid f-string escaping issues with `{columns[0]}` etc.

**Tech Stack:** DuckDB `COPY TO PARQUET`, PyYAML (findings.yaml), Python string concatenation (code gen), `py_compile` (test validation).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/databench_mcp/workspace.py` | Modify | Add `"dashboards"` to `_SUBDIRS` |
| `src/databench_mcp/core/dashboard.py` | Create | All `build_dashboard` logic |
| `src/databench_mcp/tools/dashboard.py` | Create | MCP tool wrapper (1 function) |
| `src/databench_mcp/server.py` | Modify | Register tool + bump count 21→22 |
| `tests/test_workspace.py` | Modify | Assert dashboards dir is scaffolded |
| `tests/test_core_dashboard.py` | Create | 5 unit tests for `build_dashboard` |
| `tests/tools/test_dashboard_integration.py` | Create | 4 end-to-end tests |

---

### Task 1: Add `dashboards` workspace subdirectory

**Files:**
- Modify: `src/databench_mcp/workspace.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Write the failing test**

Open `tests/test_workspace.py` and append:

```python
def test_project_create_scaffolds_dashboards_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABENCH_WORKSPACE", str(tmp_path))
    from databench_mcp.workspace import project_create, project_path
    project_create("test-proj")
    assert (project_path("test-proj") / "dashboards").is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_workspace.py::test_project_create_scaffolds_dashboards_dir -v
```

Expected: FAIL — `AssertionError` (dashboards dir not created)

- [ ] **Step 3: Update `_SUBDIRS` in workspace.py**

Read `src/databench_mcp/workspace.py`, find the `_SUBDIRS` tuple, add `"dashboards"`:

```python
_SUBDIRS = ("raw", "artifacts", "recipes", "reports", "dashboards")
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_workspace.py::test_project_create_scaffolds_dashboards_dir -v
```

Expected: PASS

- [ ] **Step 5: Run workspace tests to check regressions**

```
pytest tests/test_workspace.py -v
```

Expected: all passing

- [ ] **Step 6: Commit**

```bash
git add src/databench_mcp/workspace.py tests/test_workspace.py
git commit -m "feat: scaffold dashboards/ subdirectory on project_create"
```

---

### Task 2: `core/dashboard.py` skeleton — helpers and error path

**Files:**
- Create: `src/databench_mcp/core/dashboard.py`
- Create: `tests/test_core_dashboard.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core_dashboard.py`:

```python
import json
from pathlib import Path

import pytest


@pytest.fixture
def empty_project(tmp_path, monkeypatch):
    """Project with ingested data but no chart sidecars."""
    monkeypatch.setenv("DATABENCH_WORKSPACE", str(tmp_path))
    from databench_mcp.tools.project import project_create
    from databench_mcp.tools.ingest import ingest_file
    from databench_mcp.tools.profile import profile_table
    project_create("dash-proj")
    csv_path = tmp_path / "items.csv"
    csv_path.write_text("name,value\nAlice,10\nBob,20\n")
    ingest_file("dash-proj", str(csv_path), table_name="items")
    profile_table("dash-proj", "items")
    return "dash-proj"


def test_build_dashboard_no_charts_raises(empty_project):
    from databench_mcp.core.dashboard import build_dashboard
    with pytest.raises(ValueError, match="no charts to embed"):
        build_dashboard(empty_project)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_core_dashboard.py::test_build_dashboard_no_charts_raises -v
```

Expected: FAIL — `ImportError` or `ModuleNotFoundError` (module not yet created)

- [ ] **Step 3: Create `src/databench_mcp/core/dashboard.py`**

```python
"""Dashboard generation — builds a standalone Dash app from chart sidecars."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from databench_mcp.db import get_connection
from databench_mcp.workspace import project_path


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
            conn.execute(f'COPY "{table}" TO {str(path)!r} (FORMAT PARQUET)')


def _load_findings(project: str) -> list[dict]:
    path = project_path(project) / "findings.yaml"
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def build_dashboard(project: str) -> dict[str, Any]:
    """Generate a standalone Dash app from the project's chart artifacts and findings."""
    sidecars_by_table, skipped = _read_sidecars(project)

    if not sidecars_by_table:
        raise ValueError("no charts to embed — run create_chart first")

    raise NotImplementedError("generation not yet implemented")
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_core_dashboard.py::test_build_dashboard_no_charts_raises -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/databench_mcp/core/dashboard.py tests/test_core_dashboard.py
git commit -m "feat: add core/dashboard.py skeleton with sidecar reader and error path"
```

---

### Task 3: Complete `core/dashboard.py` — code generation

**Files:**
- Modify: `src/databench_mcp/core/dashboard.py`
- Modify: `tests/test_core_dashboard.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core_dashboard.py`:

```python
@pytest.fixture
def two_chart_project(tmp_path, monkeypatch):
    """Two tables, one chart sidecar each (histogram on items, boxplot on scores)."""
    monkeypatch.setenv("DATABENCH_WORKSPACE", str(tmp_path))
    from databench_mcp.tools.project import project_create
    from databench_mcp.tools.ingest import ingest_file
    from databench_mcp.tools.profile import profile_table
    from databench_mcp.workspace import project_path as _pp
    project_create("dash-proj")
    charts_dir = _pp("dash-proj") / "charts"
    charts_dir.mkdir(exist_ok=True)
    for tbl, col, csv_content, ct in [
        ("items", "value", "name,value\nAlice,10\nBob,20\n", "histogram"),
        ("scores", "score", "player,score\nX,100\nY,200\n", "boxplot"),
    ]:
        csv_path = tmp_path / f"{tbl}.csv"
        csv_path.write_text(csv_content)
        ingest_file("dash-proj", str(csv_path), table_name=tbl)
        profile_table("dash-proj", tbl)
        (charts_dir / f"{ct}_20260101T000000_params.json").write_text(
            json.dumps({"chart_type": ct, "table": tbl, "columns": [col],
                        "finding_id": None, "params": {}})
        )
    return "dash-proj"


def test_build_dashboard_generates_files(two_chart_project):
    from databench_mcp.core.dashboard import build_dashboard
    from databench_mcp.workspace import project_path
    result = build_dashboard(two_chart_project)
    dash_dir = project_path(two_chart_project) / "dashboards"
    assert Path(result["dashboard_py"]).exists()
    assert Path(result["requirements_txt"]).exists()
    assert Path(result["deploy_md"]).exists()
    assert (dash_dir / "data" / "items.parquet").exists()
    assert (dash_dir / "data" / "scores.parquet").exists()


def test_build_dashboard_tab_count(two_chart_project):
    from databench_mcp.core.dashboard import build_dashboard
    result = build_dashboard(two_chart_project)
    assert result["tabs"] == 2


def test_build_dashboard_charts_embedded(two_chart_project):
    from databench_mcp.core.dashboard import build_dashboard
    result = build_dashboard(two_chart_project)
    assert result["charts_embedded"] == 2


def test_build_dashboard_overwrites_existing(two_chart_project):
    from databench_mcp.core.dashboard import build_dashboard
    result1 = build_dashboard(two_chart_project)
    assert result1["warning"] is None
    result2 = build_dashboard(two_chart_project)
    assert result2["warning"] is not None
    assert "overwritten" in result2["warning"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_core_dashboard.py -v
```

Expected: `test_build_dashboard_no_charts_raises` passes; 4 new tests fail with `NotImplementedError`

- [ ] **Step 3: Add constants to `core/dashboard.py`**

Insert after the imports (before `_dashboards_dir`):

```python
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
```

- [ ] **Step 4: Add `_format_metrics` and `_generate_dashboard_py` to `core/dashboard.py`**

Append after `_load_findings`:

```python
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
```

- [ ] **Step 5: Replace `raise NotImplementedError` with the full `build_dashboard` implementation**

Replace the stub body (keep the `_read_sidecars` call and ValueError):

```python
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
```

- [ ] **Step 6: Run all core dashboard tests**

```
pytest tests/test_core_dashboard.py -v
```

Expected: all 5 tests pass

- [ ] **Step 7: Run full suite**

```
pytest -v
```

Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add src/databench_mcp/core/dashboard.py tests/test_core_dashboard.py
git commit -m "feat: implement build_dashboard code generation in core/dashboard.py"
```

---

### Task 4: MCP tool wrapper, server registration, and integration tests

**Files:**
- Create: `src/databench_mcp/tools/dashboard.py`
- Modify: `src/databench_mcp/server.py`
- Create: `tests/tools/test_dashboard_integration.py`

- [ ] **Step 1: Write the failing integration tests**

Create `tests/tools/test_dashboard_integration.py`:

```python
import py_compile
from pathlib import Path

import duckdb
import pytest


@pytest.fixture
def pipeline_project(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABENCH_WORKSPACE", str(tmp_path))
    from databench_mcp.tools.project import project_create
    from databench_mcp.tools.ingest import ingest_file
    from databench_mcp.tools.profile import profile_table
    from databench_mcp.core.viz import create_chart

    project_create("int-proj")

    csv1 = tmp_path / "nums.csv"
    csv1.write_text("x,y\n1,2\n3,4\n5,6\n7,8\n")
    ingest_file("int-proj", str(csv1), table_name="nums")
    profile_table("int-proj", "nums")

    csv2 = tmp_path / "cats.csv"
    csv2.write_text("label,count\nA,10\nB,20\nC,5\n")
    ingest_file("int-proj", str(csv2), table_name="cats")
    profile_table("int-proj", "cats")

    create_chart("int-proj", "histogram", "nums", columns=["x"])
    create_chart("int-proj", "boxplot", "cats", columns=["count"])

    return "int-proj"


def test_build_dashboard_tool_files(pipeline_project):
    from databench_mcp.tools.dashboard import build_dashboard
    from databench_mcp.workspace import project_path

    result = build_dashboard(pipeline_project)
    dash_dir = project_path(pipeline_project) / "dashboards"

    assert Path(result["dashboard_py"]).exists()
    assert Path(result["requirements_txt"]).exists()
    assert Path(result["deploy_md"]).exists()
    assert len(result["tables_exported"]) > 0


def test_build_dashboard_tool_python_valid(pipeline_project):
    from databench_mcp.tools.dashboard import build_dashboard

    result = build_dashboard(pipeline_project)
    py_compile.compile(result["dashboard_py"], doraise=True)


def test_build_dashboard_tool_parquet_readable(pipeline_project):
    from databench_mcp.tools.dashboard import build_dashboard
    from databench_mcp.workspace import project_path

    result = build_dashboard(pipeline_project)
    data_dir = project_path(pipeline_project) / "dashboards" / "data"

    for table in result["tables_exported"]:
        parquet_path = str(data_dir / f"{table}.parquet")
        count = duckdb.query(f"SELECT COUNT(*) FROM '{parquet_path}'").fetchone()[0]
        assert count > 0


def test_build_dashboard_tool_counts(pipeline_project):
    from databench_mcp.tools.dashboard import build_dashboard

    result = build_dashboard(pipeline_project)
    assert result["tabs"] == 2
    assert result["charts_embedded"] >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/tools/test_dashboard_integration.py -v
```

Expected: FAIL — `ImportError` (`tools/dashboard.py` not yet created)

- [ ] **Step 3: Create `src/databench_mcp/tools/dashboard.py`**

```python
"""MCP tool wrapper for dashboard generation."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.dashboard import build_dashboard as _build


def build_dashboard(project: str) -> dict[str, Any]:
    """Generate a standalone Dash app from the project's chart artifacts and findings."""
    return _build(project)
```

- [ ] **Step 4: Update `src/databench_mcp/server.py`**

Add the import alongside the other tools imports:

```python
from databench_mcp.tools.dashboard import build_dashboard
```

Add the registration after `mcp.tool(run_recipe)`:

```python
mcp.tool(build_dashboard)
```

Change the tool count:

```python
EXPECTED_TOOL_COUNT = 22
```

- [ ] **Step 5: Run integration tests**

```
pytest tests/tools/test_dashboard_integration.py -v
```

Expected: all 4 tests pass

- [ ] **Step 6: Run full suite**

```
pytest -v
```

Expected: all tests pass (`EXPECTED_TOOL_COUNT=22` matches actual count of 22)

- [ ] **Step 7: Commit**

```bash
git add src/databench_mcp/tools/dashboard.py src/databench_mcp/server.py tests/tools/test_dashboard_integration.py
git commit -m "feat: register build_dashboard MCP tool — bump EXPECTED_TOOL_COUNT to 22"
```

---

## Self-Review

**Spec coverage:**
- ✅ `workspace.py` — `"dashboards"` added to `_SUBDIRS` (Task 1)
- ✅ `build_dashboard` contract — all 7 return fields (dashboard_py, requirements_txt, deploy_md, tabs, charts_embedded, tables_exported, warning) (Task 3)
- ✅ Algorithm steps 1–7 from spec (read sidecars, group by table, export parquet, generate dashboard.py, write requirements.txt + DEPLOY.md, return dict) (Task 3)
- ✅ Chart rendering constraints — `_SKIP_ALWAYS` set + `network_graph` with `finding_id` check (Task 2)
- ✅ Error: no charts → `ValueError` (Task 2)
- ✅ Error: dashboard.py already exists → overwritten + warning (Task 3)
- ✅ `_make_figure` inlined, handles all 6 renderable chart types including network_graph with igraph fallback (Task 3)
- ✅ `EXPECTED_TOOL_COUNT` 21 → 22 (Task 4)
- ✅ All 5 unit tests from spec (Task 3) + all 4 integration tests from spec (Task 4)

**Placeholder scan:** None found.

**Type consistency:**
- `_read_sidecars` returns `tuple[dict[str, list[dict]], list[dict]]` — used consistently in `build_dashboard`
- `_generate_dashboard_py` takes `dict[str, list[dict]]` and `list[dict]` — matches what `build_dashboard` passes
- `build_dashboard` return dict keys match the spec contract exactly
