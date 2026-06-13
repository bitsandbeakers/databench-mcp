"""
Provider-grain cost-driver TRIANGULATION (leakage-free, one row per provider).

Senior importance workflow (per the 2025-26 guide), at provider grain:
  1. LightGBM + interventional TreeSHAP   -> primary ranking
  2. Permutation importance (held-out)     -> performance-based cross-check
  3. EBM glass-box (native categoricals)   -> interpretable cross-check + R2
  4. Seed stability of the SHAP ranking
  5. Unscaled ownership effect sizes (log points -> %), For-profit reference

Reads project.duckdb directly.
"""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import lightgbm as lgb
import shap
from interpret.glassbox import ExplainableBoostingRegressor

DB = r"C:\Users\cody\nox-code\satellites\databench-mcp\workspace\uhc_affordability\project.duckdb"

SQL = r"""
WITH natl AS (
  SELECT DRG_Cd, MEDIAN(Avg_Submtd_Cvrd_Chrg) AS natl_med
  FROM inpatient_enr
  WHERE Avg_Submtd_Cvrd_Chrg>0 AND Avg_Mdcr_Pymt_Amt>0 AND Tot_Dschrgs>0 AND Rndrng_Prvdr_RUCA<>99
  GROUP BY DRG_Cd
),
svc AS (
  SELECT
    PRINTF('%06d', TRY_CAST(i.Rndrng_Prvdr_CCN AS INTEGER)) AS ccn,
    i.DRG_Cd, i.Tot_Dschrgs::DOUBLE AS disch, i.Avg_Submtd_Cvrd_Chrg AS charge, n.natl_med,
    CASE WHEN i.DRG_Desc ILIKE '%WITHOUT%' OR i.DRG_Desc ILIKE '%W/O%' THEN 0
         WHEN i.DRG_Desc ILIKE '%MCC%' THEN 2
         WHEN i.DRG_Desc ILIKE '% CC%' THEN 1 ELSE 0 END AS sev,
    i.Rndrng_Prvdr_State_Abrvtn AS state,
    CASE WHEN i.Rndrng_Prvdr_RUCA<4 THEN 'Metro' WHEN i.Rndrng_Prvdr_RUCA<7 THEN 'Micro'
         WHEN i.Rndrng_Prvdr_RUCA<10 THEN 'SmallTown' ELSE 'Rural' END AS urbanicity
  FROM inpatient_enr i JOIN natl n USING (DRG_Cd)
  WHERE i.Avg_Submtd_Cvrd_Chrg>0 AND i.Avg_Mdcr_Pymt_Amt>0 AND i.Tot_Dschrgs>0 AND i.Rndrng_Prvdr_RUCA<>99
),
own AS (
  SELECT ccn, MAX(ownership) AS ownership FROM (
    SELECT PRINTF('%06d', TRY_CAST("Provider CCN" AS INTEGER)) AS ccn,
      CASE WHEN "Type of Control" IN (1,2) THEN 'Nonprofit'
           WHEN "Type of Control" IN (3,4,5,6) THEN 'ForProfit' ELSE 'Government' END AS ownership
    FROM cost_report_2023 WHERE TRY_CAST("Provider CCN" AS INTEGER) IS NOT NULL
  ) GROUP BY ccn
),
arch AS (
  SELECT PRINTF('%06d', TRY_CAST(ccn AS INTEGER)) AS ccn, MAX(archetype_group) AS archetype_group
  FROM archetype_outlier_input WHERE TRY_CAST(ccn AS INTEGER) IS NOT NULL GROUP BY ccn
)
SELECT s.ccn,
  COUNT(DISTINCT s.DRG_Cd) AS n_drgs,
  LN(SUM(s.disch)) AS log_volume,
  SUM(s.sev*s.disch)/SUM(s.disch) AS mean_severity,
  LN(SUM(s.charge*s.disch)/SUM(s.disch)) AS ln_actual_charge,
  LN(SUM(s.natl_med*s.disch)/SUM(s.disch)) AS ln_expected_charge,
  SUM(s.disch*s.disch)/POWER(SUM(s.disch),2) AS hhi_specialization,
  ANY_VALUE(s.state) AS state, ANY_VALUE(s.urbanicity) AS urbanicity,
  COALESCE(o.ownership,'Unknown') AS ownership,
  COALESCE(a.archetype_group,'Unknown') AS archetype_group
FROM svc s LEFT JOIN own o USING (ccn) LEFT JOIN arch a USING (ccn)
GROUP BY s.ccn, o.ownership, a.archetype_group
"""

con = duckdb.connect(DB, read_only=True)
df = con.execute(SQL).df()
con.close()

target = "ln_actual_charge"
num_feats = ["ln_expected_charge", "mean_severity", "hhi_specialization", "n_drgs", "log_volume"]
cat_feats = ["ownership", "archetype_group", "state", "urbanicity"]
for c in cat_feats:
    df[c] = df[c].astype("category")

X = pd.get_dummies(df[num_feats + cat_feats], columns=cat_feats, drop_first=True).astype(float)
y = df[target].astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

def to_src(col):
    for c in cat_feats:
        if col.startswith(c + "_"):
            return c
    return col

def agg_src(values, cols):
    return pd.Series(values, index=cols).groupby(to_src).sum().sort_values(ascending=False)

print("=" * 70)
print(f"Provider-grain triangulation  |  {df.shape[0]} providers  |  target={target}")
print("=" * 70)

# ---- 1. LightGBM + interventional TreeSHAP ----
gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                        random_state=42, n_jobs=-1, verbose=-1)
gbm.fit(Xtr, ytr)
r2_gbm = r2_score(yte, gbm.predict(Xte))
bg = shap.sample(Xtr, 200, random_state=42)
expl = shap.TreeExplainer(gbm, data=bg, feature_perturbation="interventional")
sv = expl.shap_values(Xte, check_additivity=False)
shap_imp = agg_src(np.abs(sv).mean(axis=0), X.columns)
print(f"\n[1] LightGBM  test R2 = {r2_gbm:.3f}")
print("    Interventional SHAP (mean|SHAP|, aggregated to source):")
for k, v in shap_imp.items():
    print(f"      {k:<22} {v:.4f}")

# ---- 2. Permutation importance (held-out) ----
pi = permutation_importance(gbm, Xte, yte, n_repeats=15, random_state=42, n_jobs=-1)
perm_imp = agg_src(pi.importances_mean, X.columns)
print("\n[2] Permutation importance (held-out, drop in R2):")
for k, v in perm_imp.items():
    print(f"      {k:<22} {v:.4f}")

# ---- 3. EBM glass-box (native categoricals) ----
Xr = df[num_feats + cat_feats]
Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, y, test_size=0.2, random_state=42)
ebm = ExplainableBoostingRegressor(interactions=0, random_state=42)
ebm.fit(Xrtr, yrtr)
r2_ebm = r2_score(yrte, ebm.predict(Xrte))
ebm_imp = (pd.Series(ebm.term_importances(), index=ebm.term_names_)
           .sort_values(ascending=False))
print(f"\n[3] EBM glass-box  test R2 = {r2_ebm:.3f}")
print("    Term importances:")
for k, v in ebm_imp.items():
    print(f"      {k:<22} {v:.4f}")

# ---- 4. Seed stability of SHAP ranking ----
ranks = {}
for seed in [1, 7, 21, 42, 99]:
    g = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          random_state=seed, n_jobs=-1, verbose=-1).fit(Xtr, ytr)
    e = shap.TreeExplainer(g, data=bg, feature_perturbation="interventional")
    imp = agg_src(np.abs(e.shap_values(Xte, check_additivity=False)).mean(axis=0), X.columns)
    for rank, feat in enumerate(imp.index, 1):
        ranks.setdefault(feat, []).append(rank)
print("\n[4] SHAP rank stability across 5 seeds (min-max rank):")
for feat in shap_imp.index:
    rr = ranks[feat]
    print(f"      {feat:<22} rank {min(rr)}-{max(rr)}")

# ---- 5. Unscaled ownership effect sizes (For-profit reference) ----
lin = LinearRegression().fit(Xtr, ytr)
coef = pd.Series(lin.coef_, index=X.columns)
print("\n[5] Effect sizes vs FOR-PROFIT reference (OLS, log points -> %):")
for c in ["ownership_Government", "ownership_Nonprofit",
          "urbanicity_Micro", "urbanicity_SmallTown", "urbanicity_Rural"]:
    if c in coef.index:
        b = coef[c]
        print(f"      {c:<24} {b:+.3f} log pts  ->  {np.exp(b)-1:+.1%}")
print("      (negative = charges LESS than for-profit, holding case mix/geography/archetype)")
