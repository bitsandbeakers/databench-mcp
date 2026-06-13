# CLAUDE.md — databench-mcp

## Analysis sessions

Any time you are running an analysis with this project (EDA, outlier detection, modeling,
hypothesis testing, visualization), you MUST invoke the `superpowers:databench-analyst`
skill at the start of the session and follow it for the entire session.

```
Skill("superpowers:databench-analyst")
```

This applies even after context compaction. If you resume a session and the skill is not
active, invoke it before taking any further analysis steps.

## Skill enforcement checklist (before any analysis tool call)

- [ ] `hypothesis_list` and `project_status` called to read current state
- [ ] Profiling complete on every table being analyzed
- [ ] EDA (distributions, correlations, outliers) done before any `run_model` call
- [ ] Evidence recorded via `hypothesis_record_evidence` after every finding
- [ ] Human check-in every 3-5 tool calls

## Grain discipline

Do NOT collapse to provider level prematurely. Analyze at the natural grain of each table
first:

- `inpatient`: provider × DRG (146k rows) — explore here before aggregating
- `outpatient`: provider × APC (117k rows) — explore here before aggregating
- `provider_model`: provider level — only after grain-level EDA is complete

## Current project

Active project: `uhc_cms_2023`
Analysis script: `uhc_analysis.py`
