# UHC Affordability — Cost-Driver Findings (h009 + h011)

**Project:** `uhc_affordability` · **Date:** 2026-06-13
**Data:** CMS Medicare Provider Utilization (MUP) inpatient DY2023 (provider × DRG) + HCRIS FY2023 Hospital Cost Report + service-mix archetypes (h012).
**Scripts:** `prov_drivers_model.py` (Lasso baseline), `prov_drivers_triangulate.py` (triangulation).

---

## TL;DR

1. **For-profit hospitals charge ~33% more for comparable work.** Net of case mix, geography, and hospital archetype, nonprofit hospitals charge **−33%** and government hospitals **−34%** vs for-profit. Large, clean, and stable. *(Associational — see causal note.)*
2. **The dominant cost drivers, in order: geography (state) → case-mix → ownership.** This trio is identical across three independent importance methods, and the ranking is perfectly stable across 5 random seeds.
3. **Rural hospitals are the most exposed to Medicare cuts (h011, supported):** ~2× the Medicare dependence of urban hospitals *and* already-negative operating margins — no cushion to absorb a reimbursement reduction.

---

## Method (why this is leakage-free)

Modeling at **provider grain — one row per provider (n=2,933)** — instead of the 146k provider×DRG service rows. This matters:

- **No leakage by construction.** A provider appears exactly once, so a random train/test split cannot leak a provider's pricing across folds. (The service grain would require `GroupKFold` by provider, which the in-house tool doesn't support.)
- **DRG signal is preserved numerically** via a **case-mix index**: `ln_expected_charge` = discharge-weighted average of each DRG's *national median charge*. This is "what this provider's case mix would cost at national benchmark prices," so every other coefficient is read **net of case mix**.
- Other case-mix numerics: `mean_severity` (weighted MCC/CC tier), `hhi_specialization` (DRG-mix concentration), `n_drgs`, `log_volume`.
- **Drivers:** `ownership` (HCRIS Type of Control → Nonprofit / For-profit / Government), `archetype_group` (h012), `state`, `urbanicity`.
- **Leakage controls:** excluded `adj_charge_idx` and `markup` from the provider table — both are derived from the charges being predicted.

Ownership match rate 99.8%, archetype 95.7%. Ownership mix: Nonprofit 1,893 · For-profit 647 · Government 388.

---

## Driver ranking — triangulated

Per the senior importance workflow: LightGBM + **interventional TreeSHAP** (primary), permutation importance (cross-check), EBM glass-box (interpretable cross-check).

| Driver | SHAP rank | Permutation rank | EBM rank |
|--------|:---:|:---:|:---:|
| **state** (geography) | 1 | 2 | 2 |
| **case-mix** (`ln_expected_charge`) | 2 | 1 | 1 |
| **ownership** | 3 | 3 | 3 |
| n_drgs, archetype, urbanicity, severity, volume, specialization | minor | minor | minor |

- **Model fit:** LightGBM test R² = **0.652**, EBM test R² = **0.689**.
- **Seed stability:** SHAP ranks held at 1-1, 2-2, 3-3, … across 5 seeds — no rank ambiguity.
- **Lasso note:** default `alpha=1.0` is **degenerate here** (R² ≈ 0, 0 features selected — the L1 penalty over-shrinks at this target scale). CV-tuned `alpha=0.0015` → R² = 0.669. Importance ranking above uses the tree models, not raw Lasso coefficients (which are biased toward high-cardinality features per the guide).

### Ownership effect sizes (OLS, For-profit = reference)

Holding case mix, geography, and archetype constant:

| Group | vs For-profit | % |
|-------|:---:|:---:|
| Nonprofit | −0.394 log pts | **−32.6%** |
| Government | −0.421 log pts | **−34.4%** |
| Urbanicity: Micropolitan | −0.103 | −9.8% |
| Urbanicity: Small town | −0.207 | −18.7% |
| Urbanicity: Rural | −0.137 | −12.8% |

Negative = charges **less** than for-profit. Non-metro discount confirms h010.

---

## h011 — Rural Medicare-funding exposure (supported)

From HCRIS FY2023 (6,103 hospitals), grouped by Rural/Urban × facility type:

| Group | Medicare day share | Median operating margin | % with negative op margin | Median total margin |
|-------|:---:|:---:|:---:|:---:|
| Rural CAH | **48.3%** | −6.17% | 67.5% | 4.51% |
| Rural PPS | 27.6% | −5.32% | 61.5% | **2.86%** (p25 −4.0) |
| Urban CAH | 43.0% | −1.48% | 53.3% | 5.24% |
| Urban PPS | 24.3% | +0.11% | 48.4% | 4.89% |

**Rural hospitals — especially Critical Access Hospitals — combine the highest Medicare dependence (~half of inpatient days) with negative operating margins (2/3 already underwater on patient care) and cost-based (~101%) reimbursement.** A cut to Medicare reimbursement removes their largest revenue source from the facilities least able to absorb it. CAHs are cost-reimbursed and were absent from the MUP charge data — the cost report is the only place this exposure is visible.

---

## Caveats

- **Associational, not causal.** Ownership is *associated* with ~33% higher charges net of *observed* confounders. Unobserved confounding — local market concentration, patient acuity beyond DRG, area wage levels, payer mix — is not controlled. Do not read this as "converting a hospital to for-profit raises charges 33%."
- Effect sizes are linear (OLS); SHAP/EBM confirm ownership also matters non-linearly.
- Case-mix index uses national medians that include each provider's own (negligible) contribution; a leave-one-out benchmark would be marginally cleaner.
- Provider grain answers "what drives a provider's charge level net of case mix." The complementary "what share of *service-level* variance does DRG explain" is answered separately by the eta² decomposition (DRG ≈ 58% of charge variance).

---

## Ownership effect — deep dive

Script: `prov_ownership_explore.py`. Binary for-profit (n=647) vs non-for-profit (n=2,286).

**The premium survives every control** (cumulative, OLS):

| Model | For-profit premium |
|-------|:---:|
| Raw | +62.5% |
| + case-mix | +55.2% |
| + archetype | +55.9% |
| + urbanicity | +52.9% |
| + state (full) | **+46.3%** |

It barely erodes — geography and case mix explain only a small part. ~46% is an ownership-level pricing difference.

**Concentration (case-mix + state controlled):**

| Archetype | For-profit premium | n_fp |
|-----------|:---:|:---:|
| Generalist | +64.6% | 358 |
| Rehab-Specialty | +50.4% | 60 |
| Academic-Tertiary | +46.7% | 61 |
| Rural-SmallAcute | +22.8% | 17 |
| Surgical-Specialty | **−11.0%** | 83 |

- Premium concentrates in **community generalist** hospitals — the substitutable, steerable services.
- Physician-owned **surgical-specialty** for-profits charge *less* (competitive niche).
- **Non-metro +56.9% vs metro +41.2%** — larger premium where competition is thinner (pricing power).

**Chain effect:** HCA-branded for-profits sit **+113%** above case-mix expectation vs +30% for other for-profits (n=60; name-detection undercounts HCA, so understated). Top for-profit over-chargers (full-model residual) match h008/h013: Carepoint (NJ ×2), Regional Med Ctr San Jose (CA, HCA), Merit Health (MS ×4, CHS), Gadsden/Brookwood (AL), Desert Springs (NV).

This corroborates h008 (for-profit chains cluster as outliers) and h013 (steerage targets) with quantified effect sizes. *Note:* clean chain attribution needs the CMS ownership/chain enrollment file — name-detection is a lower bound.

## Causal analysis — recommended next step (not run)

The affordability thesis is implicitly interventional ("steer volume to lower-cost providers to reduce spend"). To move from association to effect:

- **Question:** Does for-profit ownership *cause* higher charges?
- **Treatment:** for-profit vs not. **Outcome:** provider log charge index.
- **Confounders:** case-mix index, state/market, urbanicity, size, archetype, area wage index, market concentration (HHI of local hospital market).
- **Method:** **Double / Debiased ML** (Chernozhukov et al.; `econml` `LinearDML` or `doubleml`) — residualize outcome and treatment on confounders with flexible ML, cross-fitted, for a valid effect estimate with CIs. **Causal forest** (`econml CausalForestDML`) for heterogeneity (which markets show the largest effect).
- **Identification assumptions to state:** unconfoundedness given the covariate set, positivity (for-profit and non-profit hospitals exist across the covariate space), no interference.
- **Status:** framed only, per decision. Would require leaving the in-house tool (no DML/causal-forest support) for a custom `econml`/`doubleml` script.

---

## Reproducibility

- `prov_drivers_model.py` — builds `prov_drivers` (provider-grain case-mix table) and the Lasso baseline.
- `prov_drivers_triangulate.py` — LightGBM+SHAP, permutation, EBM, seed stability, OLS effect sizes.
- Both read `workspace/uhc_affordability/project.duckdb` directly (no MCP dependency).
- Pending: record evidence on h009 (reframed) + add h015 (ownership effect) to the hypothesis tracker once the databench MCP server reconnects.
