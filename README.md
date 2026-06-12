# databench-mcp

AI-augmented data-analysis MCP platform — guard-railed tools for ingestion, profiling, EDA, pipeline transforms, hypothesis tracking, statistical analysis, modeling, visualization, and reproducible recipe pipelines. Medicare public data as the proving ground.

## Setup

```bash
# Install
uv sync

# Run (stdio transport — used by MCP clients)
uv run databench-mcp
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "databench": {
      "command": "uv",
      "args": ["run", "databench-mcp"],
      "cwd": "/path/to/satellites/databench-mcp"
    }
  }
}
```

---

## Typical workflow

```
project_create
  → ingest_file / ingest_url
  → profile_table
  → sql_query / eda_summary / group_summary
  → clean_table / add_lag / add_rolling / enrich_table / derive_table
  → hypothesis_add
  → detect_outliers / analyze_distribution / analyze_correlations / run_model
  → create_chart / create_subplot
  → build_dashboard
  → reconstruct_recipe / run_recipe
  → hypothesis_update
```

Each project is an isolated DuckDB workspace. Tables must be profiled before analysis tools unlock.

---

## Tools (28 total)

### Project management

#### `project_create`
Create a new project workspace. Idempotent — safe to call if the project already exists.

| Param | Type | Required |
|-------|------|----------|
| `name` | string | yes |

```
project_create("medicare-2023")
→ { "project": "medicare-2023", "path": "…/workspace/medicare-2023" }
```

---

#### `project_list`
List all projects in the workspace.

```
project_list()
→ { "projects": ["medicare-2023", "cms-partd"] }
```

---

#### `project_status`
Manifest summary for a project: dataset count, profiling status, sources.

| Param | Type | Required |
|-------|------|----------|
| `name` | string | yes |

```
project_status("medicare-2023")
→ { "project": "medicare-2023", "datasets": { "providers": { "profiled": true, "rows": 1200000, … } } }
```

---

### Ingestion

#### `ingest_file`
Load a local CSV, TSV, Parquet, or Excel file into the project database.

Table name defaults to the file stem (`providers.csv` → `providers`). Run `profile_table` next to unlock analysis tools.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `file_path` | string | yes |
| `table_name` | string | no |

```
ingest_file("medicare-2023", "/data/providers.csv")
→ { "table": "providers", "rows": 1200000, "columns": 8 }
```

---

#### `ingest_url`
Download a URL and load it into the project database. Supports CMS `data.cms.gov` Socrata API — pass `params` for `$limit`, `$where` filters.

Raw file is saved to `raw/<table_name>.<ext>` before loading.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `url` | string | yes |
| `table_name` | string | yes |
| `params` | object | no |

```
ingest_url(
  "medicare-2023",
  "https://data.cms.gov/resource/3ntd-6v7v.csv",
  "part_d_2022",
  { "$limit": "50000", "$where": "specialty_description='Cardiology'" }
)
→ { "table": "part_d_2022", "rows": 48231, "columns": 21 }
```

---

### Profiling

#### `profile_table`
Profile a table: types, null rates, cardinality, min/max/mean for each column.

Stamps `profiled=True` in the manifest, which is required before running any analysis tools.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `table` | string | yes |

```
profile_table("medicare-2023", "providers")
→ {
    "table": "providers",
    "columns": 8,
    "profile": {
      "npi": { "type": "BIGINT", "null_pct": 0.0, "unique": 1200000 },
      "total_drug_cost": { "type": "DOUBLE", "null_pct": 0.02, "min": 0, "max": 4820000, "mean": 8421 },
      …
    }
  }
```

---

### EDA

#### `eda_summary`
Return a dataset overview from the project manifest — no database query. Shows row/column counts, profiling status, and column names for profiled tables.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |

```
eda_summary("medicare-2023")
→ {
    "project": "medicare-2023",
    "dataset_count": 2,
    "datasets": [
      { "name": "providers", "rows": 1200000, "cols": 8, "profiled": true, "columns": ["npi", "specialty", …] },
      { "name": "part_d_2022", "rows": 48231, "cols": 21, "profiled": false }
    ]
  }
```

---

#### `sql_query`
Execute a read-only `SELECT` or `WITH` query against the project DuckDB. Multi-statement queries and writes are rejected.

Results are truncated at `limit` rows (default 500). Check `truncated: true` in the response to know if rows were dropped.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `project` | string | yes | |
| `sql` | string | yes | |
| `limit` | integer | no | 500 |

```
sql_query(
  "medicare-2023",
  "SELECT specialty, COUNT(*) AS providers, AVG(total_drug_cost) AS avg_cost FROM providers GROUP BY specialty ORDER BY avg_cost DESC",
  limit=20
)
→ {
    "columns": ["specialty", "providers", "avg_cost"],
    "rows": [ { "specialty": "Hematology/Oncology", "providers": 4821, "avg_cost": 182400.5 }, … ],
    "row_count": 20,
    "truncated": false
  }
```

---

#### `group_summary`
Return grouped aggregates for a table column. Requires `profile_table` first.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `project` | string | yes | |
| `table` | string | yes | |
| `group_col` | string | yes | |
| `agg_cols` | string[] | yes | |
| `agg_fns` | string[] | no | `["mean", "count"]` |

Supported `agg_fns`: `mean`, `sum`, `min`, `max`, `count`, `std`.

```
group_summary("medicare-2023", "providers", "specialty", ["total_drug_cost"], ["mean", "count"])
→ {
    "group_col": "specialty",
    "agg_cols": ["total_drug_cost"],
    "row_count": 42,
    "rows": [ { "specialty": "Cardiology", "total_drug_cost_mean": 12400.0, "total_drug_cost_count": 4821 }, … ]
  }
```

---

#### `derive_table`
Materialise a SQL `SELECT` as a new DuckDB table, registered as `profiled=False` in the manifest.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `sql` | string | yes |
| `table_name` | string | yes |

```
derive_table("medicare-2023", "SELECT * FROM providers WHERE specialty = 'Cardiology'", "cardiologists")
→ { "table": "cardiologists", "rows": 4821, "columns": 8 }
```

---

### Pipeline transforms

These tools materialise a new derived table and register it in the manifest (`source` = function name). Run `profile_table` on the result to unlock analysis tools.

#### `clean_table`
Handle missing data by materialising a cleaned derived table.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `project` | string | yes | |
| `table` | string | yes | |
| `strategy` | string | yes | |
| `new_table_name` | string | yes | |
| `columns` | string[] | no | all nullable cols |
| `fill_value` | number\|string | no | |

**Strategies:** `drop_rows`, `drop_cols`, `fill_mean`, `fill_median`, `fill_mode`, `fill_constant`, `fill_forward`, `fill_backward`.

```
clean_table("medicare-2023", "providers", "fill_mean", "providers_clean")
→ { "table": "providers_clean", "source_table": "providers", "strategy": "fill_mean", "rows": 1200000, "columns": 8, "affected_cols": ["total_drug_cost"] }
```

---

#### `add_lag`
Add LAG feature columns for time-series analysis.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `table` | string | yes |
| `col` | string | yes |
| `lags` | integer[] | yes |
| `new_table_name` | string | yes |
| `time_col` | string | no |

```
add_lag("medicare-2023", "monthly_claims", "claim_count", [1, 2, 3], "claims_lagged", time_col="month")
→ { "table": "claims_lagged", "new_cols": ["claim_count_lag1", "claim_count_lag2", "claim_count_lag3"] }
```

---

#### `add_rolling`
Add a rolling-window aggregate column.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `table` | string | yes |
| `col` | string | yes |
| `window` | integer | yes |
| `agg_fn` | string | yes |
| `new_table_name` | string | yes |
| `time_col` | string | no |

Supported `agg_fn`: `mean`, `sum`, `min`, `max`, `std`.

```
add_rolling("medicare-2023", "monthly_claims", "claim_count", 3, "mean", "claims_rolled", time_col="month")
→ { "table": "claims_rolled", "new_col": "claim_count_mean_3" }
```

---

#### `enrich_table`
Join two tables and materialise the result as a new derived table.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `project` | string | yes | |
| `left_table` | string | yes | |
| `right_table` | string | yes | |
| `on` | string\|string[] | yes | |
| `new_table_name` | string | yes | |
| `how` | string | no | `"inner"` |

**Join types:** `inner`, `left`, `right`, `full`.

```
enrich_table("medicare-2023", "providers", "zip_regions", "zip_code", "providers_enriched")
→ { "table": "providers_enriched", "rows": 1180000, "columns": 10 }
```

---

### Hypothesis tracking

Hypotheses are stored in `hypotheses.yaml` inside the project directory. IDs are auto-assigned (`h001`, `h002`, …).

**Valid statuses:** `proposed` → `prioritized` → `tested` → `supported` / `refuted`

---

#### `hypothesis_add`
Add a new hypothesis with status `proposed`.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `statement` | string | yes |
| `tags` | string[] | no |

```
hypothesis_add(
  "medicare-2023",
  "High-opioid-prescribing providers cluster in rural ZIP codes",
  tags=["opioids", "geography"]
)
→ { "id": "h001", "statement": "…", "status": "proposed", "tags": ["opioids", "geography"], "notes": [] }
```

---

#### `hypothesis_list`
List hypotheses, optionally filtered by status or tag.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `status` | string | no |
| `tag` | string | no |

```
hypothesis_list("medicare-2023", status="proposed")
→ { "project": "medicare-2023", "count": 3, "hypotheses": [ … ] }
```

---

#### `hypothesis_update`
Update a hypothesis status and/or append a timestamped note. At least one of `status` or `note` is required.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `hypothesis_id` | string | yes |
| `status` | string | no |
| `note` | string | no |

```
hypothesis_update(
  "medicare-2023",
  "h001",
  status="supported",
  note="Confirmed via ZIP-level choropleth — rural ZIPs show 3.2× higher opioid rate"
)
→ { "id": "h001", "status": "supported", "notes": [ { "text": "Confirmed…", "added_at": "2026-06-11T…" } ] }
```

---

### Analysis

All analysis tools require `profile_table` to have been run first. Results are saved as findings (retrievable via `list_findings`).

#### `detect_outliers`
Flag outliers in a numeric column using IQR, Z-score, or Isolation Forest.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `project` | string | yes | |
| `table` | string | yes | |
| `column` | string | yes | |
| `method` | string | no | `"iqr"` |
| `params` | object | no | |

**Methods:** `iqr`, `zscore`, `isolation_forest`.

```
detect_outliers("medicare-2023", "providers", "total_drug_cost", method="iqr")
→ { "finding_id": "f001", "method": "iqr", "outlier_count": 3241, "threshold": { "lower": -8200, "upper": 42100 }, … }
```

---

#### `analyze_distribution`
Return distribution shape stats: skewness, kurtosis, normality test (Shapiro-Wilk), and percentiles.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `table` | string | yes |
| `column` | string | yes |

```
analyze_distribution("medicare-2023", "providers", "total_drug_cost")
→ { "finding_id": "f002", "skewness": 4.2, "kurtosis": 22.1, "normal": false, "percentiles": { "p25": 1200, "p50": 4800, "p75": 18400 } }
```

---

#### `analyze_correlations`
Return correlation matrix and top N pairs by absolute correlation.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `project` | string | yes | |
| `table` | string | yes | |
| `columns` | string[] | no | all numeric |
| `method` | string | no | `"pearson"` |
| `top_n` | integer | no | 10 |

**Methods:** `pearson`, `spearman`, `kendall`.

```
analyze_correlations("medicare-2023", "providers", method="spearman", top_n=5)
→ { "finding_id": "f003", "top_pairs": [ { "col_a": "total_drug_cost", "col_b": "claim_count", "r": 0.87 }, … ] }
```

---

#### `run_model`
Run a registered analysis method and save the finding.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `table` | string | yes |
| `method` | string | yes |
| `target` | string | no |
| `features` | string[] | no |
| `params` | object | no |

**Methods:** `linear_regression`, `logistic_regression`, `ridge`, `lasso`, `decision_tree`, `random_forest`, `gradient_boosting`, `kmeans`, `pca`, `network_graph`.

```
run_model("medicare-2023", "providers", "random_forest", target="high_cost_flag", features=["specialty", "claim_count"])
→ { "finding_id": "f004", "method": "random_forest", "metrics": { "accuracy": 0.84, "f1": 0.79 }, … }
```

---

#### `list_findings`
List all findings for a project, optionally filtered by method.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `method` | string | no |

```
list_findings("medicare-2023")
→ { "project": "medicare-2023", "count": 4, "findings": [ { "id": "f001", "method": "iqr", "table": "providers", "summary": "…" }, … ] }
```

---

### Visualization

Charts are saved as HTML files to `charts/` in the project directory, with a JSON sidecar used by `build_dashboard`.

#### `create_chart`
Generate a Plotly HTML chart and save it to the project charts directory.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |
| `chart_type` | string | yes |
| `table` | string | yes |
| `columns` | string[] | yes |
| `finding_id` | string | no |
| `params` | object | no |

**Chart types:** `histogram`, `boxplot`, `scatter`, `scatter_matrix`, `correlation_heatmap`, `line`, `bar`, `horizontal_bar`, `pie`, `bubble`, `dot`, `table`, `dumbbell`, `parallel_categories`, `choropleth_map`, `network_graph`.

```
create_chart("medicare-2023", "histogram", "providers", ["total_drug_cost"])
→ { "chart_type": "histogram", "table": "providers", "path": "…/charts/histogram_20260101T120000.html" }
```

---

#### `create_subplot`
Combine multiple charts into a single subplot grid HTML file.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `project` | string | yes | |
| `charts` | object[] | yes | |
| `rows` | integer | yes | |
| `cols` | integer | yes | |
| `title` | string | no | |
| `shared_xaxes` | boolean | no | false |
| `shared_yaxes` | boolean | no | false |

Each `charts` entry: `{ "chart_type": "…", "table": "…", "columns": […], "params": {} }`.

```
create_subplot("medicare-2023", [
  { "chart_type": "histogram", "table": "providers", "columns": ["total_drug_cost"] },
  { "chart_type": "boxplot",   "table": "providers", "columns": ["claim_count"] }
], rows=1, cols=2, title="Provider cost overview")
→ { "path": "…/charts/subplot_20260101T120000.html", "rows": 1, "cols": 2, "charts": 2 }
```

---

### Recipes

#### `reconstruct_recipe`
Generate `recipe.py` and `recipe_meta.yaml` from the project's current manifest state. The recipe captures every ingest, profile, and transform step so the pipeline is fully reproducible.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |

```
reconstruct_recipe("medicare-2023")
→ { "recipe_py": "…/recipe.py", "steps": 7 }
```

---

#### `run_recipe`
Execute `recipe.py` and diff outputs against expected values in `recipe_meta.yaml`. Returns `clean` if row counts and column names match, `changed` with a diff otherwise.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `project` | string | yes | |
| `diff_mode` | boolean | no | true |

```
run_recipe("medicare-2023")
→ { "status": "clean", "steps_run": 7, "diff": null }
```

---

### Dashboard

#### `build_dashboard`
Generate a standalone Dash app from the project's chart artifacts and findings. Exports each table to Parquet, writes `dashboard.py`, `requirements.txt`, and `DEPLOY.md` to `dashboards/`.

When a project has both raw-ingested and derived tables (output of `clean_table`, `add_lag`, etc.), only derived tables are included as tabs. Low-cardinality VARCHAR/ENUM columns (≤20 unique values) automatically get interactive dropdown filters in the generated app.

| Param | Type | Required |
|-------|------|----------|
| `project` | string | yes |

```
build_dashboard("medicare-2023")
→ {
    "dashboard_py": "…/dashboards/dashboard.py",
    "requirements_txt": "…/dashboards/requirements.txt",
    "deploy_md": "…/dashboards/DEPLOY.md",
    "tabs": 3,
    "charts_embedded": 7,
    "tables_exported": ["providers_clean", "claims_lagged", "providers_enriched"],
    "warning": null
  }
```

To run the generated app locally:
```bash
cd <project>/dashboards
pip install -r requirements.txt
python dashboard.py
```

---

## Development

```bash
uv run pytest          # run tests
uv run ruff check .    # lint
```

Tests live in `tests/`. Each core module has a matching `tests/test_core_<module>.py`; tool wrappers have `tests/tools/test_<module>.py`.
