# UHC Affordability — submission guide

EDA, outlier detection, and cost-driver analysis on the CMS Medicare **Inpatient** and
**Outpatient** hospital cost files (2023), framed for network affordability. Built and driven
end-to-end through a custom AI-analysis platform (**databench-mcp**) with an enforced
hypothesis → evidence process.

## Start here

- **Interactive dashboard (the whole story, BLUF-first):**
  https://bitsandbeakers.github.io/databench-mcp/uhc-affordability/
  *(fully self-contained — charts are interactive; the network's archetype focus + avoidance
  toggle work in the browser)*
- **Slides:** [`docs/slides/`](docs/slides/) — `UHC_Affordability_Deck_readahead.pptx`
  (no speaker notes). The full presenting version (with notes) is provided directly, not committed.

## Where everything is

| What | Where |
|------|-------|
| **Interactive dashboard** (story + methods + appendix) | live URL above · source `workspace/uhc_affordability/dashboards/app.py` · static export `docs/uhc-affordability/index.html` |
| **Slides** (presenting + read-ahead) | `docs/slides/` |
| **MCP server** — the analysis platform I built | `src/databench_mcp/` (+ `tests/`, 37 files) |
| **The analyst skill** — the process it enforced | `docs/databench-analyst-SKILL.md` |
| **Insight + outlier + driver analysis scripts** | `prov_drivers_triangulate.py`, `prov_ownership_explore.py`, `prov_ownership_payment_check.py`, `prov_drivers_model.py`, `prov_leakage_estimate.py`, `uhc_analysis.py`, `metrics_compare.py` |
| **Temporal robustness (2022 holdout)** | `temporal_check.py` |
| **Cost × quality (CMS Hospital Compare)** | `quality_build.py` → dashboard §06 |
| **Service-mix network / archetypes** | `workspace/uhc_affordability/dashboards/network_build.py` |
| **Static export + slide generator** | `workspace/uhc_affordability/dashboards/export_static.py`, `generate_slides.py` |
| **Findings write-up** | `provider_drivers_findings.md` |
| **Design system (Optum brand)** | `workspace/uhc_affordability/dashboards/design.md` |
| **Plans / specs** | `docs/superpowers/` |

## Data

Public **CMS** data, supplemented as needed:

- **Core:** Medicare Inpatient & Outpatient by Provider and Service, 2023
  (`data.cms.gov/provider-summary-by-type-of-service`)
- **Supplements:** CMS HCRIS cost report (ownership/margins), RUCA (urbanicity), public ZIP
  centroids (distance), CMS Hospital Compare (quality star ratings)

Raw CSVs and the working DuckDB are **not committed** (large + public + re-downloadable). The
small **derived chart tables** in `workspace/uhc_affordability/dashboards/data/` *are* included
so the dashboard app runs locally. The static dashboard needs no data at all.

## Run it

```bash
# View the dashboard — just open the live URL, or:
#   docs/uhc-affordability/index.html   (self-contained, no install)

# Run the dashboard app locally
pip install -r workspace/uhc_affordability/dashboards/requirements.txt
python workspace/uhc_affordability/dashboards/app.py        # http://127.0.0.1:8050

# Regenerate the slides
python workspace/uhc_affordability/dashboards/generate_slides.py            # with notes
python workspace/uhc_affordability/dashboards/generate_slides.py --no-notes # read-ahead

# The MCP server itself
uv sync && uv run databench-mcp
```

## How it was done (AI enablement)

The analysis ran through databench-mcp on a hypothesis-driven, CRISP-DM-style loop enforced by
the `databench-analyst` skill: **read state → form hypotheses → test each → record evidence →
repeat.** Every claim on the dashboard traces to a tool call and a logged note (18/18
hypotheses, evidence-tracked). A human-in-the-loop **correction ledger** (13 corrections) and
adversarial verification (which caught a target-leakage bug and a data artifact) are visible in
the dashboard's "How it was built" section and appendix.
