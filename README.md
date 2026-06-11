# databench-mcp

AI-augmented data-analysis MCP platform — guard-railed tools for ingestion, profiling, EDA, hypothesis tracking, analysis, viz, and reproducible recipe pipelines. Medicare public data as the proving ground.

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
project_create → ingest_file / ingest_url → profile_table → sql_query / eda_summary → hypothesis_add → … → hypothesis_update
```

Each project is an isolated DuckDB workspace. Tables must be profiled before analysis tools unlock.

---

## Tools

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

## Development

```bash
uv run pytest          # run tests
uv run ruff check .    # lint
```

Tests live in `tests/`. Each core module has a matching `tests/test_core_<module>.py`; tool wrappers have `tests/tools/test_<module>.py`.
