"""
Provider-grain cost-driver model (h009 driver-ranking, leakage-free).

Represents each provider's DRG mix as numeric case-mix features (CMI-style),
then ranks drivers of a provider's charge level NET of case mix.

Grain: one row per provider -> a random split has zero provider leakage,
so no GroupKFold needed and no service-grain row explosion / EBM hang.

Reads project.duckdb directly (databench MCP not required).
"""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
    i.DRG_Cd,
    i.Tot_Dschrgs::DOUBLE AS disch,
    i.Avg_Submtd_Cvrd_Chrg AS charge,
    n.natl_med,
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
SELECT
  s.ccn,
  COUNT(DISTINCT s.DRG_Cd)                              AS n_drgs,
  LN(SUM(s.disch))                                      AS log_volume,
  SUM(s.sev*s.disch)/SUM(s.disch)                       AS mean_severity,
  LN(SUM(s.charge*s.disch)/SUM(s.disch))                AS ln_actual_charge,
  LN(SUM(s.natl_med*s.disch)/SUM(s.disch))              AS ln_expected_charge,
  SUM(s.disch*s.disch)/POWER(SUM(s.disch),2)            AS hhi_specialization,
  ANY_VALUE(s.state)                                    AS state,
  ANY_VALUE(s.urbanicity)                               AS urbanicity,
  COALESCE(o.ownership,'Unknown')                       AS ownership,
  COALESCE(a.archetype_group,'Unknown')                 AS archetype_group
FROM svc s
LEFT JOIN own o USING (ccn)
LEFT JOIN arch a USING (ccn)
GROUP BY s.ccn, o.ownership, a.archetype_group
"""

con = duckdb.connect(DB, read_only=True)
df = con.execute(SQL).df()
con.close()

print("=" * 64)
print(f"prov_drivers: {df.shape[0]} providers x {df.shape[1]} cols")
print("=" * 64)
print("\nMatch rates (LEFT JOIN coverage):")
print(f"  ownership known : {(df.ownership!='Unknown').mean():.1%}")
print(f"  archetype known : {(df.archetype_group!='Unknown').mean():.1%}")
print("\nownership distribution:")
print(df.ownership.value_counts().to_string())
print("\nnumeric summary:")
print(df[["n_drgs","log_volume","mean_severity","ln_actual_charge",
          "ln_expected_charge","hhi_specialization"]].describe().round(3).to_string())

# correlation of the case-mix control with the target (sanity)
r = np.corrcoef(df.ln_expected_charge, df.ln_actual_charge)[0, 1]
print(f"\ncorr(ln_expected_charge, ln_actual_charge) = {r:.3f}  (case-mix control should be strong)")

# ---- Lasso baseline (default alpha=1.0), provider grain, random split ----
target = "ln_actual_charge"
num_feats = ["ln_expected_charge", "mean_severity", "hhi_specialization", "n_drgs", "log_volume"]
cat_feats = ["ownership", "archetype_group", "state", "urbanicity"]

X = pd.get_dummies(df[num_feats + cat_feats], columns=cat_feats, drop_first=True).astype(float)
y = df[target].astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = make_pipeline(StandardScaler(), Lasso(alpha=1.0, max_iter=10000))
pipe.fit(Xtr, ytr)
r2_def = r2_score(yte, pipe.predict(Xte))
nz_def = int(np.count_nonzero(pipe[-1].coef_))
print("\n" + "=" * 64)
print(f"LASSO  default alpha=1.0  ->  test R2 = {r2_def:.3f}   nonzero feats = {nz_def}/{X.shape[1]}")
print("=" * 64)

# diagnostic only: what a CV-tuned alpha would give (not the headline)
cv = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=10000, n_jobs=-1)).fit(Xtr, ytr)
r2_cv = r2_score(yte, cv.predict(Xte))
print(f"[diagnostic] LassoCV alpha={cv[-1].alpha_:.4g}  ->  test R2 = {r2_cv:.3f}")

# aggregate |coef| back to source feature (from the CV model, which is informative)
coef = pd.Series(cv[-1].coef_, index=X.columns)
def src(col):
    for c in cat_feats:
        if col.startswith(c + "_"):
            return c
    return col
imp = coef.abs().groupby(src).sum().sort_values(ascending=False)
print("\nAggregated importance (sum |coef| per source feature, CV model):")
print(imp.round(3).to_string())

# signed coefs for the interpretable drivers
print("\nSelected signed coefficients (case-mix controls + ownership/urbanicity):")
show = [c for c in X.columns if c.startswith(("ownership_", "urbanicity_"))] + num_feats
for c in show:
    if c in coef.index and abs(coef[c]) > 1e-6:
        print(f"  {c:<32} {coef[c]:+.3f}")
