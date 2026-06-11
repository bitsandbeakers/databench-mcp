# Phase 4: Augmented Analysis Tools — Design Spec

**Date:** 2026-06-11  
**Project:** databench-mcp  
**Status:** Approved

---

## Overview

Phase 4 adds six analysis tools that make databench-mcp genuinely useful: EDA/outlier detection, distribution and correlation analysis, a method-registry-backed model runner, a findings tracker, and Plotly HTML visualization. The workflow is explicitly **human-in-the-loop augmented** — Claude surfaces findings, proposes methods, the user selects, Claude executes, the user picks the best model, then visualizations tell the story.

---

## 1. Architecture

### Layer separation (unchanged pattern from Phases 1–3)

```
tools/analysis.py     →  thin MCP wrappers, validation, error surface
tools/modeling.py     →  thin MCP wrappers
tools/viz.py          →  thin MCP wrapper

core/analysis.py      →  detect_outliers, analyze_distribution, analyze_correlations
core/modeling.py      →  run_model dispatch table
core/findings.py      →  YAML-backed findings tracker
core/viz.py           →  create_chart (Plotly HTML)
```

All core functions receive a `project_path: Path` and a `db: duckdb.DuckDBPyConnection` injected by the tool wrapper (same pattern as profile.py). No core function opens its own DB connection.

### Precondition enforcement

`run_model`, `detect_outliers`, `analyze_distribution`, `analyze_correlations`, and `create_chart` all require `profiled=True` in the manifest for the target table. The check lives in the tool wrapper (same as existing analysis tools). Raises `ValueError("table '{table}' must be profiled before analysis")`.

### New dependencies

```toml
# pyproject.toml additions
scikit-learn = ">=1.4"
scipy = ">=1.13"
plotly = ">=5.22"
shap = ">=0.45"
```

---

## 2. Tool Set (6 new tools → 18 total)

| Tool | Core module | Purpose |
|------|-------------|---------|
| `detect_outliers` | `core/analysis.py` | Flag outlier rows using IQR, Z-score, or Isolation Forest |
| `analyze_distribution` | `core/analysis.py` | Shape summary: skewness, kurtosis, normality test, percentiles |
| `analyze_correlations` | `core/analysis.py` | Pearson/Spearman matrix + top N pairs by absolute correlation |
| `run_model` | `core/modeling.py` | Dispatch to any registered method; auto-saves findings |
| `list_findings` | `core/findings.py` | List/filter saved findings by project |
| `create_chart` | `core/viz.py` | Generate Plotly HTML chart; save to `charts/` in project dir |

`EXPECTED_TOOL_COUNT` bumped to **18** in `server.py`.

---

## 3. Method Registry

`core/modeling.py` uses a dispatch table — a `dict[str, Callable]` — so adding a method never requires touching control flow.

```python
_REGISTRY: dict[str, Callable] = {
    "linear_regression":      _run_linear_regression,
    "lasso":                  _run_lasso,
    "ridge":                  _run_ridge,
    "elastic_net":            _run_elastic_net,
    "logistic_regression":    _run_logistic_regression,
    "decision_tree":          _run_decision_tree,
    "random_forest":          _run_random_forest,
    "gradient_boosting":      _run_gradient_boosting,
    "shap":                   _run_shap,
    "permutation_importance": _run_permutation_importance,
    "mutual_information":     _run_mutual_information,
    "quantile_regression":    _run_quantile_regression,
    "kmeans":                 _run_kmeans,
    "pca":                    _run_pca,
}
```

Each handler receives `(df: pd.DataFrame, target: str | None, features: list[str], params: dict)` and returns a standardized findings dict (see §5).

**SHAP specifics:** `_run_shap` fits its own tree model internally (gradient_boosting by default; override with `params={"base_model": "random_forest"}`). SHAP values are large arrays — they are **not** stored in `findings.yaml`. Instead, they are saved as a companion NumPy file at `workspace/<project>/findings/<finding_id>_shap.npy`. `create_chart(chart_type="shap_beeswarm", finding_id="f001")` loads from that path.

---

## 4. Tool Contracts

### `detect_outliers`

```
detect_outliers(project, table, column, method="iqr", params={})
→ {
    "table": str, "column": str, "method": str,
    "outlier_count": int, "total_rows": int, "outlier_pct": float,
    "threshold": any,          # fence values (IQR), z threshold, or contamination
    "sample_outliers": [...]   # up to 20 representative rows
  }
```

**Methods:** `iqr`, `zscore`, `isolation_forest`  
`params` examples: `{"multiplier": 2.5}` for IQR, `{"threshold": 3.0}` for Z-score, `{"contamination": 0.05}` for Isolation Forest.

### `analyze_distribution`

```
analyze_distribution(project, table, column)
→ {
    "column": str, "dtype": str,
    "mean": float, "median": float, "std": float,
    "skewness": float, "kurtosis": float,
    "shapiro_stat": float, "shapiro_p": float,   # null if n > 5000 (uses KS test instead)
    "percentiles": {"p5": ..., "p25": ..., "p75": ..., "p95": ..., "p99": ...},
    "verdict": str   # "approximately normal" | "right-skewed" | "left-skewed" | "heavy-tailed"
  }
```

### `analyze_correlations`

```
analyze_correlations(project, table, columns=None, method="pearson", top_n=10)
→ {
    "method": str,
    "matrix": { "col_a": { "col_b": float, ... }, ... },
    "top_pairs": [
      { "col_a": str, "col_b": str, "r": float, "abs_r": float }
    ]  # top_n by abs_r, descending
  }
```

`columns` defaults to all numeric columns if omitted.

### `run_model`

```
run_model(project, table, method, target=None, features=None, params={})
→ {
    "finding_id": str,         # auto-assigned "f001", "f002", …
    "method": str,
    "target": str | null,
    "features": [str],
    "metrics": { ... },        # method-specific, see §5
    "explainability": "high" | "medium" | "low",
    "summary": str,            # 1–2 sentence plain-English finding
    "created_at": str
  }
```

`features` defaults to all numeric columns except `target` if omitted.  
Finding is auto-persisted to `findings.yaml` in the project directory.

### `list_findings`

```
list_findings(project, method=None)
→ {
    "project": str,
    "count": int,
    "findings": [ { finding_id, method, target, metrics_summary, created_at }, ... ]
  }
```

### `create_chart`

```
create_chart(project, chart_type, table, columns, finding_id=None, params={})
→ {
    "path": str,    # absolute path to saved .html file
    "chart_type": str,
    "title": str
  }
```

Chart saved to `workspace/<project>/charts/<chart_type>_<timestamp>.html`.

**Supported chart types:**
- `histogram`, `boxplot`, `distribution_overlay`
- `correlation_heatmap`
- `feature_importance_bar` (accepts RF, GB, SHAP, permutation results via `finding_id`)
- `scatter`, `scatter_matrix`
- `cluster_scatter` (post-kmeans, color by cluster label)
- `shap_beeswarm`, `shap_waterfall` (requires prior `run_model(method="shap")`; loads companion `.npy` file)
- `partial_dependence`

---

## 5. Standardized Findings Schema

All `run_model` outputs share this envelope. Method-specific data lives under `metrics`.

```yaml
# findings.yaml entry
- id: f001
  method: random_forest
  target: total_drug_cost
  features: [specialty, state, provider_type]
  metrics:
    r2: 0.74
    rmse: 12400.5
    feature_importance:
      specialty: 0.42
      state: 0.31
      provider_type: 0.27
  explainability: medium
  summary: "Random forest explains 74% of variance in total_drug_cost. Specialty is the dominant driver (42% importance), followed by state (31%)."
  created_at: "2026-06-11T22:00:00"
```

**Explainability ratings** (hardcoded per method):

| Rating | Methods |
|--------|---------|
| high | `linear_regression`, `lasso`, `ridge`, `elastic_net`, `logistic_regression`, `decision_tree`, `mutual_information` |
| medium | `random_forest`, `gradient_boosting`, `permutation_importance`, `shap`, `pca` |
| low | `isolation_forest`, `quantile_regression`, `kmeans` |

---

## 6. Findings Tracker (`core/findings.py`)

Same pattern as `core/hypothesis.py`:
- Persists to `workspace/<project>/findings.yaml`
- IDs auto-assigned (`f001`, `f002`, …)
- `load_findings(project_path)` / `save_findings(project_path, findings)`
- `add_finding(project_path, finding_dict) → str` (returns new ID)
- `get_finding(project_path, finding_id) → dict`
- `list_findings(project_path, method=None) → list[dict]`

No update/delete in Phase 4 — findings are immutable once written (re-run the model to get a new finding).

---

## 7. Error Handling

| Error condition | Response |
|-----------------|----------|
| Table not profiled | `ValueError: table must be profiled before analysis` |
| Unknown method | `ValueError: unknown method '{m}'. Available: [list]` |
| Non-numeric column passed to numeric-only method | `ValueError: column '{c}' is not numeric` |
| Target not in table | `ValueError: target '{t}' not found in table` |
| Insufficient rows for method (< 10) | `ValueError: need at least 10 rows, got {n}` |
| SHAP chart requested but no companion `.npy` found | `ValueError: no SHAP values found for finding '{id}'; run run_model(method="shap") first` |

All errors surface as clean `ValueError` strings — the tool wrapper catches and returns `{"error": str(e)}` rather than letting exceptions propagate to FastMCP.

---

## 8. Testing Strategy

Same pyramid as Phases 1–3:

```
tests/
  test_core_analysis.py      # unit: outlier detection, distribution, correlation
  test_core_modeling.py      # unit: each registry method with synthetic DataFrame
  test_core_findings.py      # unit: YAML read/write, ID assignment
  test_core_viz.py           # unit: chart output is valid HTML, file saved
  tools/
    test_analysis.py         # integration: tool wrappers → real DuckDB
    test_modeling.py         # integration: run_model end-to-end
    test_viz.py              # integration: create_chart saves file
```

Each method in the registry gets at least one happy-path test and one test for insufficient data (< 10 rows). SHAP test uses a small sklearn RandomForestRegressor to avoid long fit times.

Fixture: synthetic 200-row Medicare-like DataFrame with `npi`, `specialty`, `state`, `total_drug_cost`, `claim_count` columns — reused across all Phase 4 tests.

---

## 9. Extensibility

To add a new method in Phase 5+:
1. Add `_run_<method>` function to `core/modeling.py`
2. Add entry to `_REGISTRY`
3. Add test in `test_core_modeling.py`

No changes to `server.py`, `tools/modeling.py`, or `run_model` tool contract. The dispatch table means new methods are zero-friction additions.

---

## 10. Phase 5 Note (Dash App)

Phase 4 outputs are designed to feed a Dash app in Phase 5:
- `findings.yaml` is the data source for a findings browser
- Chart `.html` files can be embedded via `IFrame` or rendered inline
- The method registry is already structured to support a method-picker UI

No Phase 5 code in this phase. Note logged here to avoid design decisions that would require breaking changes.
