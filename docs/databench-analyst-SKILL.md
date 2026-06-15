---
name: databench-analyst
description: Use when driving a data analysis session with databench-mcp tools. Enforces the hypothesis-driven analysis loop: ingest → profile → hypothesize → analyze → record evidence → repeat. Never skip evidence recording after a finding.
---

# DataBench Analyst

## Overview

Drive a structured, hypothesis-tracked analysis session using databench-mcp tools. The agent owns the analysis loop — calling tools, interpreting results, recording evidence, and deciding what to do next. The data scientist reviews findings and redirects.

**Announce at start:** "I'm using the databench-analyst skill to drive this analysis."

**Core principle:** Every tool call that produces a finding MUST be followed by `hypothesis_record_evidence`. No exceptions.

---

## The Non-Negotiables

Before doing anything else, read this list. These rules cannot be skipped.

1. **ALWAYS read project state first.** Call `hypothesis_list` and `project_status` before any analysis step.
2. **ALWAYS record evidence after a finding.** Any call to `run_model`, `eda_summary`, `sql_query`, `analyze_correlations`, `analyze_distribution`, `detect_outliers`, `peer_outliers`, `group_summary`, or `similarity_network` MUST be followed immediately by `hypothesis_record_evidence` for each relevant hypothesis.
3. **ALWAYS update hypothesis status** when evidence is conclusive. Don't leave a hypothesis at `proposed` after a model R²=0.7 confirms it.
4. **Check in with the human every 3-5 tool calls.** Show a brief narrative of what was found, what hypotheses changed, and what you're planning next. Let them redirect before continuing.
5. **Never skip profiling on a new table.** If a table hasn't been profiled, call `profile_table` before any analysis.
6. **ALWAYS call `log_correction` when the human corrects your approach.** If the human redirects the analysis, points out a methodological error, identifies data quality misinterpretation, flags an inappropriate variable, or corrects any aspect of your work, call `log_correction` IMMEDIATELY — before continuing. Do not skip this even for minor corrections. Choose the most specific category: `data_leakage`, `wrong_grain`, `endogeneity`, `domain_methodology`, `statistical_error`, `modeling_discipline`, `data_quality`, `other`. Set `databench_gap=True` when the correction reveals a missing safety check that databench-mcp should enforce automatically.

---

## The Loop

```
START
  ↓
Read project state
  → hypothesis_list(project)
  → project_status(project)
  ↓
Are there hypotheses?
  NO  → Hypothesis generation phase (see below)
  YES → Select next hypothesis to test (see decision tree)
  ↓
Run analysis
  → Call appropriate tool
  → Interpret result in 2-3 sentences
  → Call hypothesis_record_evidence for each affected hypothesis
  → Call hypothesis_update if status should change
  ↓
Check in with human (every 3-5 tool calls)
  → Show narrative: what was found, what changed, what's next
  → Wait for confirmation or redirect
  → If human redirects or corrects → call log_correction FIRST, then continue
  ↓
More open hypotheses?
  YES → Loop
  NO  → Surface final summary, close the session
```

---

## Hypothesis Generation Phase

When there are no hypotheses yet:

1. Call `profile_table` on each ingested table.
2. Call `eda_summary` on the primary table with `n_bins=20`.
3. Call `group_summary` if there's an obvious grouping variable.
4. From the profiles, generate 5-10 hypotheses covering:
   - Distribution shape (skew, bimodality, outliers)
   - Group differences (means, medians across categories)
   - Correlations between key numeric variables
   - Predictive relationships (what drives the target variable)
   - Anomalies (high-z entities, unusual patterns)
5. Call `hypothesis_add` for each. Tag by type: `eda`, `modeling`, `outlier`, `correlation`.
6. Surface the hypothesis list to the human for review before proceeding.

---

## Decision Tree: What to Analyze Next

Pick the highest-priority open hypothesis using this order:

| Hypothesis type | Tool to call |
|-----------------|--------------|
| Distribution / shape | `analyze_distribution` |
| Correlation between variables | `analyze_correlations` |
| Group differences | `group_summary` |
| Outliers / anomalies in a single variable | `detect_outliers` |
| Peer-adjusted outliers (entity vs. peers) | `peer_outliers` |
| Predictive relationship (linear) | `run_model` with `lasso` |
| Predictive relationship (non-linear, interpretable) | `run_model` with `ebm` |
| Predictive relationship (performance ceiling) | `run_model` with `lightgbm` |
| Clustering / segmentation | `run_model` with `kmeans` or `pca` |
| Service-mix / behavioral similarity | `similarity_network` |
| Ad-hoc question on the data | `sql_query` |
| Time-series pattern | `add_lag` or `add_rolling`, then model |

When multiple hypotheses are open, prefer:
1. `prioritized` status over `proposed`
2. Hypotheses with no evidence yet over ones with partial evidence
3. EDA questions before modeling (profile before predicting)

---

## After Each Tool Call

After every analysis tool call, do this in order — no skipping:

**Step 1: Interpret**
Write 2-3 sentences: what the result shows, whether it supports or refutes the hypothesis, and one specific number that matters (R², z-score, correlation coefficient, p-value, etc.).

**Step 2: Record evidence**
```
hypothesis_record_evidence(
    project=project,
    hypothesis_id="hXXX",
    tool_name="<tool you just called>",
    result_summary="<your 2-3 sentence interpretation>",
    status_update="supported" | "refuted" | "inconclusive" | None
)
```
Call this for EVERY hypothesis the result bears on — not just the primary one.

**Step 3: Update status if warranted**
If the evidence is conclusive and you didn't update status via `record_evidence`, call `hypothesis_update` explicitly.

**Step 4: Decide next step**
Consult the decision tree. Pick the next hypothesis to test. State your choice and why before calling the next tool.

---

## Human Check-In Format

Every 3-5 tool calls, surface this summary before continuing:

```
**Analysis update** (step N of ~M)

Findings so far:
- [hypothesis id + statement]: [status] — [one sentence on evidence]
- [hypothesis id + statement]: [status] — [one sentence on evidence]

Open hypotheses remaining: N
Next planned: [tool] on [hypothesis] — [why]

Continue, or redirect?
```

Keep it tight. Don't recap every number — just status changes and the most important finding.

---

## Modeling Sequence

When modeling a target variable, follow this sequence:

1. **Lasso first** — gets feature importance fast, handles collinearity, gives a baseline R²
2. **EBM second** — interprets non-linear effects and interactions while remaining auditable
3. **LightGBM third** — ceiling performance; compare to EBM to quantify the interpretability cost
4. **SHAP** — only if LightGBM meaningfully outperforms EBM and you need to explain it

Record evidence against the predictive hypothesis after each model. Update status to `supported` when R² exceeds your threshold or `refuted` if the target is unpredictable.

---

## Red Flags — Stop and Surface to Human

Stop the loop immediately and surface if:
- A finding is surprising and changes the direction of the analysis
- A model result is anomalous (R²=0.99 or R²<0.0)
- You've run 3+ models and none explain the target
- A new hypothesis emerges that wasn't in the original list
- Any tool raises an error that suggests a data quality issue

---

## Closing the Session

When all hypotheses are resolved (status not `proposed` or `prioritized`):

1. Call `hypothesis_list(project)` to get the final state.
2. Summarize: N supported, N refuted, N inconclusive — and the 2-3 most important findings in plain language.
3. Note any hypotheses that were added during analysis (beyond the original list).
4. Ask: "Analysis complete. Want to build a dashboard, run a recipe, or explore a new question?"

---

## Integration

**Tools this skill drives:**
- `project_status`, `hypothesis_list`, `hypothesis_add`, `hypothesis_update`, `hypothesis_record_evidence`
- `profile_table`, `eda_summary`, `sql_query`, `group_summary`
- `analyze_correlations`, `analyze_distribution`, `detect_outliers`, `peer_outliers`
- `run_model`, `list_findings`, `similarity_network`
- `add_lag`, `add_rolling`, `derive_table`
- `create_chart`, `create_subplot`, `build_dashboard`
- `log_correction`, `list_corrections`

**Pairs with:**
- `superpowers:brainstorming` — for scoping the analysis question before starting
- `superpowers:finishing-a-development-branch` — not applicable (analysis, not code)
