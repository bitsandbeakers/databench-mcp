# Phase 4: Analysis, Modeling & Visualization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 new tools (detect_outliers, analyze_distribution, analyze_correlations, run_model, list_findings, create_chart) that power human-in-the-loop augmented analysis on any profiled table.

**Architecture:** Core logic in `core/analysis.py`, `core/modeling.py`, `core/findings.py`, `core/viz.py`; thin tool wrappers mirror the existing Phase 3 pattern; a dispatch-table registry in `core/modeling.py` makes adding methods zero-friction; findings persist to YAML exactly like hypotheses.

**Tech Stack:** scikit-learn (models, SHAP base), scipy (distribution stats), plotly (charts), shap (SHAP values), pandas (DataFrame bridge between DuckDB and sklearn), PyYAML (findings persistence)

---

## File Map

**Create:**
- `src/databench_mcp/core/analysis.py` — detect_outliers, analyze_distribution, analyze_correlations
- `src/databench_mcp/core/findings.py` — YAML-backed findings tracker (f001, f002, …)
- `src/databench_mcp/core/modeling.py` — run_model dispatch table + all 14 method handlers
- `src/databench_mcp/core/viz.py` — create_chart (Plotly HTML saved to disk)
- `src/databench_mcp/tools/analysis.py` — thin wrappers for 3 analysis tools
- `src/databench_mcp/tools/modeling.py` — thin wrappers for run_model + list_findings
- `src/databench_mcp/tools/viz.py` — thin wrapper for create_chart
- `tests/test_core_analysis.py`
- `tests/test_core_findings.py`
- `tests/test_core_modeling.py`
- `tests/test_core_viz.py`
- `tests/tools/test_analysis.py`
- `tests/tools/test_modeling.py`
- `tests/tools/test_viz.py`

**Modify:**
- `pyproject.toml` — add scikit-learn, scipy, plotly, shap, pandas to dependencies
- `src/databench_mcp/server.py` — register 6 new tools, bump EXPECTED_TOOL_COUNT 12→18
- `tests/conftest.py` — add `medicare_df` and `project_with_data` fixtures

---

## Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependencies to pyproject.toml**

In the `dependencies` list, add after `"httpx>=0.27"`:

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
]
```

- [ ] **Step 2: Sync the virtual environment**

```
uv sync
```

Expected: packages install without error, no version conflicts.

- [ ] **Step 3: Verify imports work**

```
uv run python -c "import sklearn, scipy, plotly, shap, pandas; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```
git add pyproject.toml uv.lock
git commit -m "build: add scikit-learn, scipy, plotly, shap, pandas dependencies"
```

---

## Task 2: core/findings.py + tests

Findings are immutable once written. Same YAML pattern as `core/hypothesis.py`. IDs are `f001`, `f002`, …

**Files:**
- Create: `src/databench_mcp/core/findings.py`
- Create: `tests/test_core_findings.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core_findings.py`:

```python
"""Tests for core/findings.py — YAML-backed findings tracker."""
from __future__ import annotations

import yaml
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.findings import add_finding, get_finding, list_findings


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")


def test_add_finding_assigns_id():
    entry = add_finding("test-proj", {
        "method": "linear_regression",
        "target": "cost",
        "features": ["a", "b"],
        "metrics": {"r2": 0.8},
        "explainability": "high",
        "summary": "Good fit.",
    })
    assert entry["id"] == "f001"
    assert entry["method"] == "linear_regression"
    assert "created_at" in entry


def test_add_finding_increments_id():
    add_finding("test-proj", {"method": "m1", "target": None, "features": [],
                               "metrics": {}, "explainability": "high", "summary": "."})
    entry = add_finding("test-proj", {"method": "m2", "target": None, "features": [],
                                       "metrics": {}, "explainability": "low", "summary": "."})
    assert entry["id"] == "f002"


def test_add_finding_persists_to_yaml():
    add_finding("test-proj", {"method": "kmeans", "target": None, "features": ["x"],
                               "metrics": {"k": 3}, "explainability": "low", "summary": "."})
    path = ws.project_path("test-proj") / "findings.yaml"
    assert path.exists()
    data = yaml.safe_load(path.read_text())
    assert data[0]["method"] == "kmeans"


def test_get_finding_returns_entry():
    add_finding("test-proj", {"method": "rf", "target": "y", "features": ["x"],
                               "metrics": {}, "explainability": "medium", "summary": "."})
    entry = get_finding("test-proj", "f001")
    assert entry["method"] == "rf"


def test_get_finding_raises_for_unknown_id():
    with pytest.raises(ValueError, match="not found"):
        get_finding("test-proj", "f999")


def test_list_findings_returns_all():
    add_finding("test-proj", {"method": "a", "target": None, "features": [],
                               "metrics": {}, "explainability": "high", "summary": "."})
    add_finding("test-proj", {"method": "b", "target": None, "features": [],
                               "metrics": {}, "explainability": "low", "summary": "."})
    result = list_findings("test-proj")
    assert result["count"] == 2
    assert len(result["findings"]) == 2


def test_list_findings_filtered_by_method():
    add_finding("test-proj", {"method": "random_forest", "target": "y", "features": [],
                               "metrics": {}, "explainability": "medium", "summary": "."})
    add_finding("test-proj", {"method": "lasso", "target": "y", "features": [],
                               "metrics": {}, "explainability": "high", "summary": "."})
    result = list_findings("test-proj", method="lasso")
    assert result["count"] == 1
    assert result["findings"][0]["method"] == "lasso"
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_core_findings.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` (module not yet created).

- [ ] **Step 3: Create src/databench_mcp/core/findings.py**

```python
"""Per-project findings tracker backed by findings.yaml."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import yaml

from databench_mcp.workspace import project_path, read_manifest


def _findings_path(project: str):
    return project_path(project) / "findings.yaml"


def _read_findings(project: str) -> list[dict]:
    path = _findings_path(project)
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text()) or []


def _write_findings(project: str, findings: list[dict]) -> None:
    path = _findings_path(project)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.dump(findings, default_flow_style=False, allow_unicode=True))
    os.replace(tmp, path)


def _next_id(findings: list[dict]) -> str:
    nums = [
        int(f["id"][1:])
        for f in findings
        if f.get("id", "").startswith("f") and f["id"][1:].isdigit()
    ]
    return f"f{(max(nums) + 1):03d}" if nums else "f001"


def add_finding(project: str, data: dict[str, Any]) -> dict[str, Any]:
    """Assign ID, timestamp, save to findings.yaml, return complete entry."""
    read_manifest(project)
    findings = _read_findings(project)
    entry: dict[str, Any] = {
        "id": _next_id(findings),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    findings.append(entry)
    _write_findings(project, findings)
    return entry


def get_finding(project: str, finding_id: str) -> dict[str, Any]:
    """Return a single finding by ID. Raises ValueError if not found."""
    read_manifest(project)
    for f in _read_findings(project):
        if f.get("id") == finding_id:
            return f
    raise ValueError(f"Finding '{finding_id}' not found in project '{project}'")


def list_findings(
    project: str,
    method: str | None = None,
) -> dict[str, Any]:
    """Return findings, optionally filtered by method."""
    read_manifest(project)
    findings = _read_findings(project)
    if method is not None:
        findings = [f for f in findings if f.get("method") == method]
    return {"project": project, "count": len(findings), "findings": findings}
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_core_findings.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```
git add src/databench_mcp/core/findings.py tests/test_core_findings.py
git commit -m "feat: add core/findings.py — YAML-backed findings tracker"
```

---

## Task 3: core/analysis.py — detect_outliers + tests

**Files:**
- Create: `src/databench_mcp/core/analysis.py` (partial — outliers only)
- Create: `tests/test_core_analysis.py` (partial)

The function loads a column from DuckDB, applies the selected method, returns outlier metadata and sample rows.

- [ ] **Step 1: Add shared fixtures to tests/conftest.py**

Append to the existing `tests/conftest.py`:

```python
import numpy as np
import pandas as pd
import duckdb
import databench_mcp.workspace as ws


@pytest.fixture
def medicare_df():
    """Synthetic 60-row Medicare-like DataFrame for modeling/analysis tests."""
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame({
        "npi": range(1001, 1001 + n),
        "specialty": rng.choice(["Cardiology", "Oncology", "Primary Care"], n),
        "state": rng.choice(["CA", "TX", "NY", "FL"], n),
        "total_drug_cost": rng.exponential(scale=50000, size=n),
        "claim_count": rng.integers(10, 500, size=n).astype(float),
        "is_high_cost": (rng.random(n) > 0.5),
    })


@pytest.fixture
def project_with_data(tmp_path, monkeypatch, medicare_df):
    """Project with medicare_df loaded into DuckDB and manifest stamped profiled=True."""
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    db_path = str(tmp_path / "test-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE providers AS SELECT * FROM medicare_df")
    conn.close()
    manifest = ws.read_manifest("test-proj")
    manifest["datasets"]["providers"] = {
        "row_count": len(medicare_df),
        "col_count": len(medicare_df.columns),
        "profiled": True,
        "profile": {c: {"type": "DOUBLE"} for c in medicare_df.columns},
    }
    ws.write_manifest("test-proj", manifest)
    return tmp_path
```

- [ ] **Step 2: Write the failing outlier tests**

Create `tests/test_core_analysis.py`:

```python
"""Tests for core/analysis.py."""
from __future__ import annotations

import pytest
import databench_mcp.workspace as ws
from databench_mcp.core.analysis import detect_outliers


def test_detect_outliers_iqr(project_with_data):
    result = detect_outliers("test-proj", "providers", "total_drug_cost", method="iqr")
    assert result["method"] == "iqr"
    assert "outlier_count" in result
    assert "outlier_pct" in result
    assert 0 <= result["outlier_pct"] <= 100
    assert isinstance(result["sample_outliers"], list)


def test_detect_outliers_zscore(project_with_data):
    result = detect_outliers("test-proj", "providers", "total_drug_cost", method="zscore")
    assert result["method"] == "zscore"
    assert "threshold" in result


def test_detect_outliers_isolation_forest(project_with_data):
    result = detect_outliers("test-proj", "providers", "total_drug_cost",
                             method="isolation_forest")
    assert result["method"] == "isolation_forest"
    assert result["outlier_count"] >= 0


def test_detect_outliers_unknown_method(project_with_data):
    with pytest.raises(ValueError, match="Unknown outlier method"):
        detect_outliers("test-proj", "providers", "total_drug_cost", method="bogus")


def test_detect_outliers_requires_profiled(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    with pytest.raises(ValueError, match="profiled"):
        detect_outliers("test-proj", "providers", "total_drug_cost")
```

- [ ] **Step 3: Run to verify failure**

```
uv run pytest tests/test_core_analysis.py -v
```

Expected: `ImportError` — module not yet created.

- [ ] **Step 4: Create src/databench_mcp/core/analysis.py with detect_outliers**

```python
"""EDA analysis: outlier detection, distribution analysis, correlation analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from databench_mcp.db import get_connection
from databench_mcp.workspace import assert_profiled


def _load_column(project: str, table: str, column: str) -> pd.Series:
    with get_connection(project) as conn:
        df = conn.execute(f'SELECT "{column}" FROM "{table}"').df()
    return df[column].dropna()


def detect_outliers(
    project: str,
    table: str,
    column: str,
    method: str = "iqr",
    params: dict | None = None,
) -> dict[str, Any]:
    """Flag outliers in a single numeric column. Returns counts and sample rows."""
    assert_profiled(project, table)
    params = params or {}

    series = _load_column(project, table, column)
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{column}' is not numeric")

    total = len(series)
    values = series.values

    if method == "iqr":
        multiplier = float(params.get("multiplier", 1.5))
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask = (values < lower) | (values > upper)
        threshold = {"lower_fence": round(float(lower), 4), "upper_fence": round(float(upper), 4)}

    elif method == "zscore":
        threshold_val = float(params.get("threshold", 3.0))
        z = np.abs(stats.zscore(values))
        mask = z > threshold_val
        threshold = {"z_threshold": threshold_val}

    elif method == "isolation_forest":
        contamination = float(params.get("contamination", 0.05))
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(values.reshape(-1, 1))
        mask = preds == -1
        threshold = {"contamination": contamination}

    else:
        raise ValueError(f"Unknown outlier method '{method}'. Choose: iqr, zscore, isolation_forest")

    outlier_indices = np.where(mask)[0]
    outlier_count = int(mask.sum())
    sample = series.iloc[outlier_indices[:20]].tolist()

    return {
        "table": table,
        "column": column,
        "method": method,
        "outlier_count": outlier_count,
        "total_rows": total,
        "outlier_pct": round(outlier_count / total * 100, 2) if total > 0 else 0.0,
        "threshold": threshold,
        "sample_outliers": [round(float(v), 4) for v in sample],
    }
```

- [ ] **Step 5: Run tests**

```
uv run pytest tests/test_core_analysis.py::test_detect_outliers_iqr tests/test_core_analysis.py::test_detect_outliers_zscore tests/test_core_analysis.py::test_detect_outliers_isolation_forest tests/test_core_analysis.py::test_detect_outliers_unknown_method tests/test_core_analysis.py::test_detect_outliers_requires_profiled -v
```

Expected: all 5 PASS.

- [ ] **Step 6: Commit**

```
git add src/databench_mcp/core/analysis.py tests/test_core_analysis.py tests/conftest.py
git commit -m "feat: add detect_outliers (iqr/zscore/isolation_forest)"
```

---

## Task 4: core/analysis.py — analyze_distribution + tests

**Files:**
- Modify: `src/databench_mcp/core/analysis.py` (append function)
- Modify: `tests/test_core_analysis.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core_analysis.py`:

```python
from databench_mcp.core.analysis import analyze_distribution


def test_analyze_distribution_returns_shape_stats(project_with_data):
    result = analyze_distribution("test-proj", "providers", "total_drug_cost")
    assert result["column"] == "total_drug_cost"
    assert "mean" in result
    assert "median" in result
    assert "skewness" in result
    assert "kurtosis" in result
    assert "percentiles" in result
    assert "verdict" in result
    assert result["verdict"] in (
        "approximately normal", "right-skewed", "left-skewed", "heavy-tailed"
    )


def test_analyze_distribution_includes_normality_test(project_with_data):
    result = analyze_distribution("test-proj", "providers", "total_drug_cost")
    assert "normality_stat" in result
    assert "normality_p" in result
    assert "normality_test" in result


def test_analyze_distribution_requires_profiled(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    with pytest.raises(ValueError, match="profiled"):
        analyze_distribution("test-proj", "providers", "total_drug_cost")
```

- [ ] **Step 2: Run to verify failure**

```
uv run pytest tests/test_core_analysis.py::test_analyze_distribution_returns_shape_stats -v
```

Expected: `ImportError` for `analyze_distribution`.

- [ ] **Step 3: Append analyze_distribution to core/analysis.py**

```python
def analyze_distribution(
    project: str,
    table: str,
    column: str,
) -> dict[str, Any]:
    """Return distribution shape stats for a numeric column."""
    assert_profiled(project, table)
    series = _load_column(project, table, column)
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{column}' is not numeric")

    values = series.values.astype(float)
    n = len(values)

    skewness = float(stats.skew(values))
    kurt = float(stats.kurtosis(values))

    if n <= 5000:
        stat, p = stats.shapiro(values)
        test_name = "shapiro"
    else:
        stat, p = stats.kstest(values, "norm", args=(values.mean(), values.std()))
        test_name = "ks"

    if abs(skewness) < 0.5 and p > 0.05:
        verdict = "approximately normal"
    elif skewness > 0.5:
        verdict = "right-skewed"
    elif skewness < -0.5:
        verdict = "left-skewed"
    else:
        verdict = "heavy-tailed"

    return {
        "column": column,
        "dtype": str(series.dtype),
        "n": n,
        "mean": round(float(values.mean()), 4),
        "median": round(float(np.median(values)), 4),
        "std": round(float(values.std()), 4),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurt, 4),
        "normality_test": test_name,
        "normality_stat": round(float(stat), 6),
        "normality_p": round(float(p), 6),
        "percentiles": {
            "p5": round(float(np.percentile(values, 5)), 4),
            "p25": round(float(np.percentile(values, 25)), 4),
            "p75": round(float(np.percentile(values, 75)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "p99": round(float(np.percentile(values, 99)), 4),
        },
        "verdict": verdict,
    }
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_core_analysis.py -k "distribution" -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```
git add src/databench_mcp/core/analysis.py tests/test_core_analysis.py
git commit -m "feat: add analyze_distribution (skewness/kurtosis/normality)"
```

---

## Task 5: core/analysis.py — analyze_correlations + tests

**Files:**
- Modify: `src/databench_mcp/core/analysis.py` (append function)
- Modify: `tests/test_core_analysis.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core_analysis.py`:

```python
from databench_mcp.core.analysis import analyze_correlations


def test_analyze_correlations_pearson(project_with_data):
    result = analyze_correlations("test-proj", "providers",
                                  columns=["total_drug_cost", "claim_count"])
    assert result["method"] == "pearson"
    assert "matrix" in result
    assert "top_pairs" in result
    assert len(result["top_pairs"]) >= 1
    pair = result["top_pairs"][0]
    assert "col_a" in pair and "col_b" in pair and "r" in pair


def test_analyze_correlations_spearman(project_with_data):
    result = analyze_correlations("test-proj", "providers",
                                  columns=["total_drug_cost", "claim_count"],
                                  method="spearman")
    assert result["method"] == "spearman"


def test_analyze_correlations_defaults_to_all_numeric(project_with_data):
    result = analyze_correlations("test-proj", "providers")
    assert len(result["matrix"]) >= 2


def test_analyze_correlations_requires_profiled(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    with pytest.raises(ValueError, match="profiled"):
        analyze_correlations("test-proj", "providers")
```

- [ ] **Step 2: Run to verify failure**

```
uv run pytest tests/test_core_analysis.py::test_analyze_correlations_pearson -v
```

Expected: `ImportError` for `analyze_correlations`.

- [ ] **Step 3: Append analyze_correlations to core/analysis.py**

```python
def analyze_correlations(
    project: str,
    table: str,
    columns: list[str] | None = None,
    method: str = "pearson",
    top_n: int = 10,
) -> dict[str, Any]:
    """Return correlation matrix and top N pairs by absolute correlation."""
    assert_profiled(project, table)
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown correlation method '{method}'. Choose: pearson, spearman")

    with get_connection(project) as conn:
        df = conn.execute(f'SELECT * FROM "{table}"').df()

    numeric_df = df.select_dtypes(include="number")
    if columns is not None:
        for c in columns:
            if c not in numeric_df.columns:
                raise ValueError(f"Column '{c}' is not numeric or not in table")
        numeric_df = numeric_df[columns]

    if len(numeric_df.columns) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation analysis")

    corr_matrix = numeric_df.corr(method=method)
    matrix_dict = {
        col: {c: round(float(v), 4) for c, v in row.items()}
        for col, row in corr_matrix.to_dict().items()
    }

    pairs = []
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if not np.isnan(r):
                pairs.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "r": round(float(r), 4),
                    "abs_r": round(abs(float(r)), 4),
                })
    pairs.sort(key=lambda x: x["abs_r"], reverse=True)

    return {
        "method": method,
        "matrix": matrix_dict,
        "top_pairs": pairs[:top_n],
    }
```

- [ ] **Step 4: Run all analysis tests**

```
uv run pytest tests/test_core_analysis.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```
git add src/databench_mcp/core/analysis.py tests/test_core_analysis.py
git commit -m "feat: add analyze_correlations (pearson/spearman)"
```

---

## Task 6: core/modeling.py Part A — linear/logistic/quantile regression methods

Build the shared infrastructure and regression method handlers. Tree methods, SHAP, and unsupervised come in Tasks 7–8.

**Files:**
- Create: `src/databench_mcp/core/modeling.py` (partial)
- Create: `tests/test_core_modeling.py` (partial)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core_modeling.py`:

```python
"""Tests for core/modeling.py — dispatch table and method handlers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.modeling import (
    _run_linear_regression,
    _run_lasso,
    _run_ridge,
    _run_elastic_net,
    _run_logistic_regression,
    _run_quantile_regression,
)


@pytest.fixture
def reg_df():
    """Small regression DataFrame."""
    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(0, 1, n)
    return pd.DataFrame({
        "cost": x * 10 + rng.normal(0, 2, n),
        "feature_a": x,
        "feature_b": rng.normal(0, 1, n),
    })


@pytest.fixture
def clf_df():
    """Small classification DataFrame."""
    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(0, 1, n)
    return pd.DataFrame({
        "label": (x > 0).astype(int),
        "feature_a": x,
        "feature_b": rng.normal(0, 1, n),
    })


def test_linear_regression_returns_r2(reg_df):
    result = _run_linear_regression(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]
    assert "rmse" in result["metrics"]
    assert "coefficients" in result["metrics"]
    assert result["explainability"] == "high"


def test_lasso_returns_r2(reg_df):
    result = _run_lasso(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]
    assert result["explainability"] == "high"


def test_ridge_returns_r2(reg_df):
    result = _run_ridge(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]


def test_elastic_net_returns_r2(reg_df):
    result = _run_elastic_net(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]


def test_logistic_regression_returns_accuracy(clf_df):
    result = _run_logistic_regression(clf_df, "label", ["feature_a", "feature_b"], {})
    assert "accuracy" in result["metrics"]
    assert "f1" in result["metrics"]
    assert result["explainability"] == "high"


def test_quantile_regression_returns_quantile(reg_df):
    result = _run_quantile_regression(reg_df, "cost", ["feature_a", "feature_b"],
                                      {"quantile": 0.75})
    assert result["metrics"]["quantile"] == 0.75
    assert "coefficients" in result["metrics"]


def test_regression_method_raises_for_too_few_rows(reg_df):
    tiny = reg_df.head(5)
    with pytest.raises(ValueError, match="at least 10 rows"):
        _run_linear_regression(tiny, "cost", ["feature_a"], {})
```

- [ ] **Step 2: Run to verify failure**

```
uv run pytest tests/test_core_modeling.py -v
```

Expected: `ImportError` — module not yet created.

- [ ] **Step 3: Create src/databench_mcp/core/modeling.py with Part A**

```python
"""Model registry and method handlers for run_model dispatch."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    QuantileRegressor,
    Ridge,
)
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from databench_mcp.core.findings import add_finding
from databench_mcp.db import get_connection
from databench_mcp.workspace import assert_profiled, project_path

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _min_rows_check(df: pd.DataFrame) -> None:
    if len(df) < 10:
        raise ValueError(f"need at least 10 rows, got {len(df)}")


def _prepare_xy(
    df: pd.DataFrame, features: list[str], target: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_raw = df[features]
    X = pd.get_dummies(X_raw, drop_first=True).astype(float)
    y = df[target].astype(float)
    return X.values, y.values, list(X.columns)


def _detect_task(df: pd.DataFrame, target: str, params: dict) -> str:
    if "task" in params:
        return params["task"]
    col = df[target]
    if col.dtype == object or col.dtype == bool or pd.api.types.is_bool_dtype(col):
        return "classification"
    if col.nunique() <= 10:
        return "classification"
    return "regression"


def _split(X, y, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)


def _reg_metrics(y_te, y_pred) -> dict:
    return {
        "r2": round(float(r2_score(y_te, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 4),
    }


def _clf_metrics(y_te, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "f1": round(float(f1_score(y_te, y_pred, average="weighted", zero_division=0)), 4),
    }


def _summary_reg(method: str, target: str, metrics: dict) -> str:
    return f"{method} on '{target}' — R²={metrics.get('r2', 'N/A')}, RMSE={metrics.get('rmse', 'N/A')}."


def _summary_clf(method: str, target: str, metrics: dict) -> str:
    return f"{method} on '{target}' — accuracy={metrics.get('accuracy', 'N/A')}, F1={metrics.get('f1', 'N/A')}."


# ---------------------------------------------------------------------------
# Regression methods
# ---------------------------------------------------------------------------

def _run_linear_regression(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        **_reg_metrics(y_te, y_pred),
        "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, model.coef_)},
        "intercept": round(float(model.intercept_), 6),
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": _summary_reg("Linear regression", target, metrics),
    }


def _run_lasso(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    alpha = float(params.get("alpha", 1.0))
    model = Lasso(alpha=alpha, max_iter=5000)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        **_reg_metrics(y_te, y_pred),
        "alpha": alpha,
        "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, model.coef_)},
        "nonzero_features": int(np.count_nonzero(model.coef_)),
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": _summary_reg("Lasso", target, metrics),
    }


def _run_ridge(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    alpha = float(params.get("alpha", 1.0))
    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        **_reg_metrics(y_te, y_pred),
        "alpha": alpha,
        "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, model.coef_)},
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": _summary_reg("Ridge", target, metrics),
    }


def _run_elastic_net(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    alpha = float(params.get("alpha", 1.0))
    l1_ratio = float(params.get("l1_ratio", 0.5))
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        **_reg_metrics(y_te, y_pred),
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, model.coef_)},
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": _summary_reg("Elastic Net", target, metrics),
    }


def _run_logistic_regression(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        **_clf_metrics(y_te, y_pred),
        "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, model.coef_[0])},
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": _summary_clf("Logistic regression", target, metrics),
    }


def _run_quantile_regression(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    quantile = float(params.get("quantile", 0.5))
    model = QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    pinball = float(np.mean(
        np.where(y_te >= y_pred, quantile * (y_te - y_pred), (1 - quantile) * (y_pred - y_te))
    ))
    metrics = {
        "quantile": quantile,
        "pinball_loss": round(pinball, 4),
        "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, model.coef_)},
        "intercept": round(float(model.intercept_), 6),
    }
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": f"Quantile ({quantile}) regression on '{target}' — pinball loss={round(pinball, 4)}.",
    }


# ---------------------------------------------------------------------------
# Registry (partial — completed in Task 8)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable] = {
    "linear_regression": _run_linear_regression,
    "lasso": _run_lasso,
    "ridge": _run_ridge,
    "elastic_net": _run_elastic_net,
    "logistic_regression": _run_logistic_regression,
    "quantile_regression": _run_quantile_regression,
}
```

- [ ] **Step 4: Run Part A tests**

```
uv run pytest tests/test_core_modeling.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```
git add src/databench_mcp/core/modeling.py tests/test_core_modeling.py
git commit -m "feat: add modeling.py Part A — linear/logistic/quantile regression handlers"
```

---

## Task 7: core/modeling.py Part B — tree methods + feature importance

Add decision_tree, random_forest, gradient_boosting, shap, permutation_importance, mutual_information.

**Files:**
- Modify: `src/databench_mcp/core/modeling.py` (append handlers + update registry)
- Modify: `tests/test_core_modeling.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core_modeling.py`:

```python
from databench_mcp.core.modeling import (
    _run_decision_tree,
    _run_random_forest,
    _run_gradient_boosting,
    _run_shap,
    _run_permutation_importance,
    _run_mutual_information,
)


def test_decision_tree_regression(reg_df):
    result = _run_decision_tree(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]
    assert "feature_importance" in result["metrics"]
    assert result["explainability"] == "high"


def test_decision_tree_classification(clf_df):
    result = _run_decision_tree(clf_df, "label", ["feature_a", "feature_b"], {})
    assert "accuracy" in result["metrics"]


def test_random_forest_regression(reg_df):
    result = _run_random_forest(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]
    assert "feature_importance" in result["metrics"]
    assert result["explainability"] == "medium"


def test_gradient_boosting_regression(reg_df):
    result = _run_gradient_boosting(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]
    assert "feature_importance" in result["metrics"]


def test_shap_returns_values(reg_df):
    result = _run_shap(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "mean_abs_shap" in result["metrics"]
    assert "shap_values" in result
    assert result["shap_values"].shape[1] == 2


def test_permutation_importance_returns_scores(reg_df):
    result = _run_permutation_importance(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "importance_mean" in result["metrics"]
    assert result["explainability"] == "medium"


def test_mutual_information_returns_scores(reg_df):
    result = _run_mutual_information(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "mi_scores" in result["metrics"]
    assert result["explainability"] == "high"
```

- [ ] **Step 2: Run to verify failure**

```
uv run pytest tests/test_core_modeling.py::test_decision_tree_regression -v
```

Expected: `ImportError` for `_run_decision_tree`.

- [ ] **Step 3: Append tree methods to core/modeling.py**

Add these functions before `_REGISTRY`, then update `_REGISTRY`:

```python
# ---------------------------------------------------------------------------
# Tree-based methods
# ---------------------------------------------------------------------------

def _run_decision_tree(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    task = _detect_task(df, target, params)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    max_depth = params.get("max_depth", None)
    if task == "regression":
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        base_metrics = _reg_metrics(y_te, y_pred)
        summary_fn = _summary_reg
    else:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_tr, y_tr.astype(int))
        y_pred = model.predict(X_te)
        base_metrics = _clf_metrics(y_te.astype(int), y_pred)
        summary_fn = _summary_clf
    importance = {n: round(float(v), 6) for n, v in zip(feat_names, model.feature_importances_)}
    metrics = {**base_metrics, "feature_importance": importance, "max_depth": model.get_depth()}
    return {
        "metrics": metrics,
        "explainability": "high",
        "summary": summary_fn("Decision tree", target, metrics),
    }


def _run_random_forest(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    task = _detect_task(df, target, params)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    n_estimators = int(params.get("n_estimators", 100))
    if task == "regression":
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        base_metrics = _reg_metrics(y_te, y_pred)
        summary_fn = _summary_reg
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr.astype(int))
        y_pred = model.predict(X_te)
        base_metrics = _clf_metrics(y_te.astype(int), y_pred)
        summary_fn = _summary_clf
    importance = {n: round(float(v), 6) for n, v in zip(feat_names, model.feature_importances_)}
    metrics = {**base_metrics, "feature_importance": importance, "n_estimators": n_estimators}
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": summary_fn("Random forest", target, metrics),
    }


def _run_gradient_boosting(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    _min_rows_check(df)
    task = _detect_task(df, target, params)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    n_estimators = int(params.get("n_estimators", 100))
    if task == "regression":
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        base_metrics = _reg_metrics(y_te, y_pred)
        summary_fn = _summary_reg
    else:
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_tr, y_tr.astype(int))
        y_pred = model.predict(X_te)
        base_metrics = _clf_metrics(y_te.astype(int), y_pred)
        summary_fn = _summary_clf
    importance = {n: round(float(v), 6) for n, v in zip(feat_names, model.feature_importances_)}
    metrics = {**base_metrics, "feature_importance": importance, "n_estimators": n_estimators}
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": summary_fn("Gradient boosting", target, metrics),
    }


# ---------------------------------------------------------------------------
# SHAP, permutation importance, mutual information
# ---------------------------------------------------------------------------

def _run_shap(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    """Fit internal GBM, compute SHAP values. Caller is responsible for saving shap_values."""
    import shap as _shap

    _min_rows_check(df)
    X, y, feat_names = _prepare_xy(df, features, target)
    base_model_name = params.get("base_model", "gradient_boosting")
    if base_model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    explainer = _shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    mean_abs = {n: round(float(v), 6) for n, v in zip(feat_names, np.abs(shap_vals).mean(axis=0))}
    metrics = {
        "mean_abs_shap": mean_abs,
        "base_model": base_model_name,
        "n_samples": len(X),
    }
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": f"SHAP via {base_model_name} — top driver: {max(mean_abs, key=mean_abs.get)}.",
        "shap_values": shap_vals,
    }


def _run_permutation_importance(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    from sklearn.inspection import permutation_importance as _perm_imp

    _min_rows_check(df)
    task = _detect_task(df, target, params)
    X, y, feat_names = _prepare_xy(df, features, target)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    if task == "regression":
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_tr, y_tr)
    else:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_tr, y_tr.astype(int))
        y_te = y_te.astype(int)
    result_pi = _perm_imp(model, X_te, y_te, n_repeats=10, random_state=42)
    imp_mean = {n: round(float(v), 6) for n, v in zip(feat_names, result_pi.importances_mean)}
    metrics = {"importance_mean": imp_mean}
    top = max(imp_mean, key=imp_mean.get)
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": f"Permutation importance on '{target}' — top feature: {top}.",
    }


def _run_mutual_information(
    df: pd.DataFrame, target: str, features: list[str], params: dict
) -> dict[str, Any]:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    _min_rows_check(df)
    task = _detect_task(df, target, params)
    X, y, feat_names = _prepare_xy(df, features, target)
    if task == "classification":
        scores = mutual_info_classif(X, y.astype(int), random_state=42)
    else:
        scores = mutual_info_regression(X, y, random_state=42)
    mi = {n: round(float(v), 6) for n, v in zip(feat_names, scores)}
    top = max(mi, key=mi.get)
    return {
        "metrics": {"mi_scores": mi},
        "explainability": "high",
        "summary": f"Mutual information on '{target}' — top feature: {top}.",
    }
```

Then update `_REGISTRY` to add the new handlers:

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
}
```

- [ ] **Step 4: Run all modeling tests so far**

```
uv run pytest tests/test_core_modeling.py -v
```

Expected: all 14 tests PASS.

- [ ] **Step 5: Commit**

```
git add src/databench_mcp/core/modeling.py tests/test_core_modeling.py
git commit -m "feat: add tree methods, SHAP, permutation importance, mutual information to modeling"
```

---

## Task 8: core/modeling.py Part C — kmeans + PCA + run_model dispatch + full end-to-end tests

**Files:**
- Modify: `src/databench_mcp/core/modeling.py` (append kmeans/PCA, add run_model)
- Modify: `tests/test_core_modeling.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core_modeling.py`:

```python
import duckdb
from databench_mcp.core.modeling import _run_kmeans, _run_pca, run_model


def test_kmeans_returns_inertia(reg_df):
    result = _run_kmeans(reg_df, None, ["feature_a", "feature_b"], {"k": 3})
    assert "inertia" in result["metrics"]
    assert "k" in result["metrics"]
    assert "cluster_labels" in result
    assert result["explainability"] == "low"


def test_pca_returns_explained_variance(reg_df):
    result = _run_pca(reg_df, None, ["feature_a", "feature_b"], {"n_components": 2})
    assert "explained_variance_ratio" in result["metrics"]
    assert "loadings" in result["metrics"]
    assert result["explainability"] == "medium"


def test_run_model_end_to_end(project_with_data):
    result = run_model(
        "test-proj", "providers", "linear_regression",
        target="total_drug_cost", features=["claim_count"],
    )
    assert result["finding_id"] == "f001"
    assert result["method"] == "linear_regression"
    assert "r2" in result["metrics"]


def test_run_model_saves_finding(project_with_data):
    run_model("test-proj", "providers", "random_forest",
              target="total_drug_cost", features=["claim_count"])
    from databench_mcp.core.findings import list_findings
    result = list_findings("test-proj")
    assert result["count"] == 1
    assert result["findings"][0]["method"] == "random_forest"


def test_run_model_shap_saves_npy(project_with_data):
    result = run_model("test-proj", "providers", "shap",
                       target="total_drug_cost", features=["claim_count"])
    import numpy as np
    from databench_mcp.workspace import project_path
    npy_path = project_path("test-proj") / "artifacts" / f"{result['finding_id']}_shap.npy"
    assert npy_path.exists()
    vals = np.load(str(npy_path))
    assert vals.ndim == 2


def test_run_model_unknown_method(project_with_data):
    with pytest.raises(ValueError, match="unknown method"):
        run_model("test-proj", "providers", "magic_model", target="total_drug_cost")


def test_run_model_requires_profiled(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    with pytest.raises(ValueError, match="profiled"):
        run_model("test-proj", "providers", "linear_regression", target="total_drug_cost")
```

- [ ] **Step 2: Run to verify failure**

```
uv run pytest tests/test_core_modeling.py::test_kmeans_returns_inertia -v
```

Expected: `ImportError` for `_run_kmeans`.

- [ ] **Step 3: Append kmeans + PCA + run_model to core/modeling.py**

Add before `_REGISTRY`:

```python
# ---------------------------------------------------------------------------
# Unsupervised: KMeans, PCA
# ---------------------------------------------------------------------------

def _run_kmeans(
    df: pd.DataFrame, target: str | None, features: list[str], params: dict
) -> dict[str, Any]:
    from sklearn.cluster import KMeans

    _min_rows_check(df)
    X, _, feat_names = _prepare_xy(df, features, target or features[0])
    k = int(params.get("k", 3))
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    metrics = {
        "k": k,
        "inertia": round(float(model.inertia_), 4),
        "cluster_sizes": {str(i): int((labels == i).sum()) for i in range(k)},
    }
    return {
        "metrics": metrics,
        "explainability": "low",
        "summary": f"KMeans (k={k}) — inertia={metrics['inertia']}. "
                   f"Cluster sizes: {metrics['cluster_sizes']}.",
        "cluster_labels": labels,
    }


def _run_pca(
    df: pd.DataFrame, target: str | None, features: list[str], params: dict
) -> dict[str, Any]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    _min_rows_check(df)
    X, _, feat_names = _prepare_xy(df, features, target or features[0])
    n_components = int(params.get("n_components", min(len(feat_names), 2)))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = PCA(n_components=n_components, random_state=42)
    model.fit(X_scaled)
    evr = [round(float(v), 4) for v in model.explained_variance_ratio_]
    loadings = {
        f"PC{i+1}": {n: round(float(v), 6) for n, v in zip(feat_names, model.components_[i])}
        for i in range(n_components)
    }
    metrics = {
        "n_components": n_components,
        "explained_variance_ratio": evr,
        "cumulative_variance": round(float(sum(evr)), 4),
        "loadings": loadings,
    }
    return {
        "metrics": metrics,
        "explainability": "medium",
        "summary": f"PCA ({n_components} components) explains {metrics['cumulative_variance']*100:.1f}% of variance.",
    }
```

Update `_REGISTRY` to add kmeans and pca:

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
}
```

Add `run_model` after `_REGISTRY`:

```python
# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------

def run_model(
    project: str,
    table: str,
    method: str,
    target: str | None = None,
    features: list[str] | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    """Dispatch to a registered method, persist finding, return complete entry."""
    assert_profiled(project, table)

    if method not in _REGISTRY:
        raise ValueError(
            f"unknown method '{method}'. Available: {sorted(_REGISTRY)}"
        )

    with get_connection(project) as conn:
        df = conn.execute(f'SELECT * FROM "{table}"').df()

    if len(df) < 10:
        raise ValueError(f"need at least 10 rows, got {len(df)}")

    if target is not None and target not in df.columns:
        raise ValueError(f"target '{target}' not found in table '{table}'")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    resolved_features = features if features is not None else [c for c in numeric_cols if c != target]

    for f in resolved_features:
        if f not in df.columns:
            raise ValueError(f"feature '{f}' not found in table '{table}'")

    result = _REGISTRY[method](df, target, resolved_features, params or {})

    # Save SHAP companion npy if present
    shap_values = result.pop("shap_values", None)
    cluster_labels = result.pop("cluster_labels", None)

    finding = add_finding(project, {
        "method": method,
        "target": target,
        "features": resolved_features,
        "metrics": result["metrics"],
        "explainability": result["explainability"],
        "summary": result["summary"],
    })

    if shap_values is not None:
        import numpy as np
        artifacts_dir = project_path(project) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        np.save(str(artifacts_dir / f"{finding['id']}_shap.npy"), shap_values)

    if cluster_labels is not None:
        import numpy as np
        artifacts_dir = project_path(project) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        np.save(str(artifacts_dir / f"{finding['id']}_labels.npy"), cluster_labels)

    return finding
```

- [ ] **Step 4: Run all modeling tests**

```
uv run pytest tests/test_core_modeling.py -v
```

Expected: all 21 tests PASS.

- [ ] **Step 5: Commit**

```
git add src/databench_mcp/core/modeling.py tests/test_core_modeling.py
git commit -m "feat: add kmeans, PCA, run_model dispatch to modeling.py"
```

---

## Task 9: core/viz.py + tests

Generate Plotly HTML charts saved to `workspace/<project>/charts/`.

**Files:**
- Create: `src/databench_mcp/core/viz.py`
- Create: `tests/test_core_viz.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core_viz.py`:

```python
"""Tests for core/viz.py — Plotly HTML chart generation."""
from __future__ import annotations

import pytest
import databench_mcp.workspace as ws
from databench_mcp.core.viz import create_chart


def test_create_chart_histogram(project_with_data):
    result = create_chart("test-proj", "histogram", "providers",
                          columns=["total_drug_cost"])
    assert result["chart_type"] == "histogram"
    assert result["path"].endswith(".html")
    from pathlib import Path
    assert Path(result["path"]).exists()
    html = Path(result["path"]).read_text()
    assert "plotly" in html.lower()


def test_create_chart_scatter(project_with_data):
    result = create_chart("test-proj", "scatter", "providers",
                          columns=["claim_count", "total_drug_cost"])
    assert result["chart_type"] == "scatter"
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_correlation_heatmap(project_with_data):
    result = create_chart("test-proj", "correlation_heatmap", "providers",
                          columns=["total_drug_cost", "claim_count"])
    assert result["chart_type"] == "correlation_heatmap"
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_feature_importance_bar(project_with_data):
    from databench_mcp.core.modeling import run_model
    finding = run_model("test-proj", "providers", "random_forest",
                        target="total_drug_cost", features=["claim_count"])
    result = create_chart("test-proj", "feature_importance_bar", "providers",
                          columns=[], finding_id=finding["id"])
    from pathlib import Path
    assert Path(result["path"]).exists()


def test_create_chart_unknown_type(project_with_data):
    with pytest.raises(ValueError, match="Unknown chart type"):
        create_chart("test-proj", "magic_chart", "providers", columns=["total_drug_cost"])


def test_create_chart_requires_profiled(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    with pytest.raises(ValueError, match="profiled"):
        create_chart("test-proj", "histogram", "providers", columns=["x"])
```

- [ ] **Step 2: Run to verify failure**

```
uv run pytest tests/test_core_viz.py -v
```

Expected: `ImportError` — module not yet created.

- [ ] **Step 3: Create src/databench_mcp/core/viz.py**

```python
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
    "partial_dependence",
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


def _load_df(project: str, table: str, columns: list[str]) -> pd.DataFrame:
    cols_sql = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    with get_connection(project) as conn:
        return conn.execute(f'SELECT {cols_sql} FROM "{table}"').df()


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
        importance = finding["metrics"].get("feature_importance") or \
                     finding["metrics"].get("mean_abs_shap") or \
                     finding["metrics"].get("importance_mean") or \
                     finding["metrics"].get("mi_scores")
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
        finding = get_finding(project, finding_id)
        import numpy as np
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

    elif chart_type in ("distribution_overlay", "partial_dependence"):
        raise ValueError(f"Chart type '{chart_type}' not yet implemented")

    else:
        raise ValueError(f"Unknown chart type '{chart_type}'")

    path = _save(fig, project, chart_type)
    return {"chart_type": chart_type, "path": path, "title": fig.layout.title.text or chart_type}
```

- [ ] **Step 4: Run viz tests**

```
uv run pytest tests/test_core_viz.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```
git add src/databench_mcp/core/viz.py tests/test_core_viz.py
git commit -m "feat: add core/viz.py — Plotly HTML chart generation"
```

---

## Task 10: Tool wrappers + integration tests

Thin wrappers matching the existing pattern in `tools/eda.py`. Each wrapper calls its core function and returns the result directly.

**Files:**
- Create: `src/databench_mcp/tools/analysis.py`
- Create: `src/databench_mcp/tools/modeling.py`
- Create: `src/databench_mcp/tools/viz.py`
- Create: `tests/tools/test_analysis.py`
- Create: `tests/tools/test_modeling.py`
- Create: `tests/tools/test_viz.py`

- [ ] **Step 1: Create src/databench_mcp/tools/analysis.py**

```python
"""FastMCP tool wrappers for analysis functions."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.analysis import (
    analyze_correlations as _analyze_correlations,
    analyze_distribution as _analyze_distribution,
    detect_outliers as _detect_outliers,
)


def detect_outliers(
    project: str,
    table: str,
    column: str,
    method: str = "iqr",
    params: dict | None = None,
) -> dict[str, Any]:
    """Flag outliers in a numeric column using IQR, Z-score, or Isolation Forest."""
    return _detect_outliers(project, table, column, method, params)


def analyze_distribution(
    project: str,
    table: str,
    column: str,
) -> dict[str, Any]:
    """Return distribution shape stats: skewness, kurtosis, normality test, percentiles."""
    return _analyze_distribution(project, table, column)


def analyze_correlations(
    project: str,
    table: str,
    columns: list[str] | None = None,
    method: str = "pearson",
    top_n: int = 10,
) -> dict[str, Any]:
    """Return correlation matrix and top N pairs by absolute correlation."""
    return _analyze_correlations(project, table, columns, method, top_n)
```

- [ ] **Step 2: Create src/databench_mcp/tools/modeling.py**

```python
"""FastMCP tool wrappers for modeling and findings functions."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.findings import list_findings as _list_findings
from databench_mcp.core.modeling import run_model as _run_model


def run_model(
    project: str,
    table: str,
    method: str,
    target: str | None = None,
    features: list[str] | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    """Run a registered analysis method, save the finding, return the finding entry."""
    return _run_model(project, table, method, target, features, params)


def list_findings(
    project: str,
    method: str | None = None,
) -> dict[str, Any]:
    """List all findings for a project, optionally filtered by method."""
    return _list_findings(project, method)
```

- [ ] **Step 3: Create src/databench_mcp/tools/viz.py**

```python
"""FastMCP tool wrapper for chart generation."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.viz import create_chart as _create_chart


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
```

- [ ] **Step 4: Write integration tests**

Create `tests/tools/test_analysis.py`:

```python
"""Integration tests for tools/analysis.py wrappers."""
from __future__ import annotations

import pytest
from databench_mcp.tools.analysis import analyze_correlations, analyze_distribution, detect_outliers


def test_detect_outliers_tool(project_with_data):
    result = detect_outliers("test-proj", "providers", "total_drug_cost")
    assert "outlier_count" in result
    assert result["method"] == "iqr"


def test_analyze_distribution_tool(project_with_data):
    result = analyze_distribution("test-proj", "providers", "total_drug_cost")
    assert "verdict" in result


def test_analyze_correlations_tool(project_with_data):
    result = analyze_correlations("test-proj", "providers",
                                  columns=["total_drug_cost", "claim_count"])
    assert "top_pairs" in result
```

Create `tests/tools/test_modeling.py`:

```python
"""Integration tests for tools/modeling.py wrappers."""
from __future__ import annotations

import pytest
from databench_mcp.tools.modeling import list_findings, run_model


def test_run_model_tool(project_with_data):
    result = run_model("test-proj", "providers", "lasso",
                       target="total_drug_cost", features=["claim_count"])
    assert "finding_id" in result
    assert result["method"] == "lasso"


def test_list_findings_tool(project_with_data):
    run_model("test-proj", "providers", "ridge",
              target="total_drug_cost", features=["claim_count"])
    result = list_findings("test-proj")
    assert result["count"] == 1
```

Create `tests/tools/test_viz.py`:

```python
"""Integration tests for tools/viz.py wrapper."""
from __future__ import annotations

import pytest
from pathlib import Path
from databench_mcp.tools.viz import create_chart


def test_create_chart_tool_histogram(project_with_data):
    result = create_chart("test-proj", "histogram", "providers",
                          columns=["total_drug_cost"])
    assert Path(result["path"]).exists()


def test_create_chart_tool_scatter(project_with_data):
    result = create_chart("test-proj", "scatter", "providers",
                          columns=["claim_count", "total_drug_cost"])
    assert Path(result["path"]).exists()
```

- [ ] **Step 5: Run all tool wrapper tests**

```
uv run pytest tests/tools/test_analysis.py tests/tools/test_modeling.py tests/tools/test_viz.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```
git add src/databench_mcp/tools/analysis.py src/databench_mcp/tools/modeling.py src/databench_mcp/tools/viz.py tests/tools/test_analysis.py tests/tools/test_modeling.py tests/tools/test_viz.py
git commit -m "feat: add tool wrappers for analysis, modeling, viz"
```

---

## Task 11: Wire server.py + full test suite

**Files:**
- Modify: `src/databench_mcp/server.py`

- [ ] **Step 1: Update server.py**

Replace the entire contents of `src/databench_mcp/server.py` with:

```python
"""FastMCP server entry point.

Tools are thin orchestration wrappers; all logic lives in databench_mcp.core.
The tool count is asserted at startup so a refactor can never silently drop a tool.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import FastMCP

from databench_mcp import __version__
from databench_mcp.tools.analysis import analyze_correlations, analyze_distribution, detect_outliers
from databench_mcp.tools.eda import eda_summary, sql_query
from databench_mcp.tools.hypothesis import hypothesis_add, hypothesis_list, hypothesis_update
from databench_mcp.tools.ingest import ingest_file, ingest_url
from databench_mcp.tools.modeling import list_findings, run_model
from databench_mcp.tools.profile import profile_table
from databench_mcp.tools.project import project_create, project_list, project_status
from databench_mcp.tools.viz import create_chart

mcp = FastMCP("databench")

# Bump this in the same commit that adds or removes a tool.
EXPECTED_TOOL_COUNT = 18


async def ping() -> dict[str, Any]:
    """Liveness check — returns server name and version."""
    return {"status": "ok", "server": "databench-mcp", "version": __version__}


mcp.tool(ping)
mcp.tool(project_create)
mcp.tool(project_list)
mcp.tool(project_status)
mcp.tool(ingest_file)
mcp.tool(ingest_url)
mcp.tool(profile_table)
mcp.tool(sql_query)
mcp.tool(eda_summary)
mcp.tool(hypothesis_add)
mcp.tool(hypothesis_list)
mcp.tool(hypothesis_update)
mcp.tool(detect_outliers)
mcp.tool(analyze_distribution)
mcp.tool(analyze_correlations)
mcp.tool(run_model)
mcp.tool(list_findings)
mcp.tool(create_chart)


def _assert_tool_count() -> None:
    tools = asyncio.run(mcp.list_tools())
    names = sorted(t.name for t in tools)
    if len(names) != EXPECTED_TOOL_COUNT:
        raise SystemExit(
            f"databench-mcp tool count drift: expected {EXPECTED_TOOL_COUNT}, "
            f"found {len(names)} ({names})"
        )


def main() -> None:
    _assert_tool_count()
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the tool count assertion passes**

```
uv run python -c "from databench_mcp.server import _assert_tool_count; _assert_tool_count(); print('tool count ok')"
```

Expected: `tool count ok`

- [ ] **Step 3: Run full test suite**

```
uv run pytest -v
```

Expected: all tests PASS (73 existing + ~37 new = ~110 total). Zero failures.

- [ ] **Step 4: Run linter**

```
uv run ruff check .
```

Expected: no errors. If any, fix them before committing.

- [ ] **Step 5: Commit**

```
git add src/databench_mcp/server.py
git commit -m "feat: Phase 4 — analysis, modeling, viz tools (18 tools total)"
```

- [ ] **Step 6: Push**

```
git push origin main
```
