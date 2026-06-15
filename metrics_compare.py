"""
Re-evaluate the provider-grain cost model with appropriate, decision-relevant metrics.

Replaces the single-split, R2-only comparison with 5-fold CV reporting, per model:
  - log-scale: R2, RMSE, MAE   (what we modelled)
  - dollar-scale: median abs error, median abs % error  (what stakeholders feel)
with mean +/- SD across folds so small gaps aren't over-read.

Same SQL / features / target as prov_drivers_triangulate.py (one row per provider).
"""
from __future__ import annotations
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
from interpret.glassbox import ExplainableBoostingRegressor

DB = str(Path(__file__).resolve().parent / "workspace" / "uhc_affordability" / "project.duckdb")

SQL = r"""
WITH natl AS (
  SELECT DRG_Cd, MEDIAN(Avg_Submtd_Cvrd_Chrg) AS natl_med
  FROM inpatient_enr
  WHERE Avg_Submtd_Cvrd_Chrg>0 AND Avg_Mdcr_Pymt_Amt>0 AND Tot_Dschrgs>0 AND Rndrng_Prvdr_RUCA<>99
  GROUP BY DRG_Cd ),
svc AS (
  SELECT PRINTF('%06d', TRY_CAST(i.Rndrng_Prvdr_CCN AS INTEGER)) AS ccn,
    i.DRG_Cd, i.Tot_Dschrgs::DOUBLE AS disch, i.Avg_Submtd_Cvrd_Chrg AS charge, n.natl_med,
    CASE WHEN i.DRG_Desc ILIKE '%WITHOUT%' OR i.DRG_Desc ILIKE '%W/O%' THEN 0
         WHEN i.DRG_Desc ILIKE '%MCC%' THEN 2
         WHEN i.DRG_Desc ILIKE '% CC%' THEN 1 ELSE 0 END AS sev,
    i.Rndrng_Prvdr_State_Abrvtn AS state,
    CASE WHEN i.Rndrng_Prvdr_RUCA<4 THEN 'Metro' WHEN i.Rndrng_Prvdr_RUCA<7 THEN 'Micro'
         WHEN i.Rndrng_Prvdr_RUCA<10 THEN 'SmallTown' ELSE 'Rural' END AS urbanicity
  FROM inpatient_enr i JOIN natl n USING (DRG_Cd)
  WHERE i.Avg_Submtd_Cvrd_Chrg>0 AND i.Avg_Mdcr_Pymt_Amt>0 AND i.Tot_Dschrgs>0 AND i.Rndrng_Prvdr_RUCA<>99 ),
own AS (
  SELECT ccn, MAX(ownership) AS ownership FROM (
    SELECT PRINTF('%06d', TRY_CAST("Provider CCN" AS INTEGER)) AS ccn,
      CASE WHEN "Type of Control" IN (1,2) THEN 'Nonprofit'
           WHEN "Type of Control" IN (3,4,5,6) THEN 'ForProfit' ELSE 'Government' END AS ownership
    FROM cost_report_2023 WHERE TRY_CAST("Provider CCN" AS INTEGER) IS NOT NULL ) GROUP BY ccn ),
arch AS (
  SELECT PRINTF('%06d', TRY_CAST(ccn AS INTEGER)) AS ccn, MAX(archetype_group) AS archetype_group
  FROM archetype_outlier_input WHERE TRY_CAST(ccn AS INTEGER) IS NOT NULL GROUP BY ccn )
SELECT s.ccn,
  COUNT(DISTINCT s.DRG_Cd) AS n_drgs, LN(SUM(s.disch)) AS log_volume,
  SUM(s.sev*s.disch)/SUM(s.disch) AS mean_severity,
  LN(SUM(s.charge*s.disch)/SUM(s.disch)) AS ln_actual_charge,
  LN(SUM(s.natl_med*s.disch)/SUM(s.disch)) AS ln_expected_charge,
  SUM(s.disch*s.disch)/POWER(SUM(s.disch),2) AS hhi_specialization,
  ANY_VALUE(s.state) AS state, ANY_VALUE(s.urbanicity) AS urbanicity,
  COALESCE(o.ownership,'Unknown') AS ownership, COALESCE(a.archetype_group,'Unknown') AS archetype_group
FROM svc s LEFT JOIN own o USING (ccn) LEFT JOIN arch a USING (ccn)
GROUP BY s.ccn, o.ownership, a.archetype_group
"""

con = duckdb.connect(DB, read_only=True)
df = con.execute(SQL).df()
con.close()

target = "ln_actual_charge"
num = ["ln_expected_charge", "mean_severity", "hhi_specialization", "n_drgs", "log_volume"]
cat = ["ownership", "archetype_group", "state", "urbanicity"]
y = df[target].astype(float).to_numpy()
Xohe = pd.get_dummies(df[num + cat], columns=cat, drop_first=True).astype(float)
Xraw = df[num + cat].copy()
for c in cat:
    Xraw[c] = Xraw[c].astype("category")

def dollar_metrics(y_log, p_log):
    a, p = np.exp(y_log), np.exp(p_log)
    return np.median(np.abs(p - a)), np.median(np.abs(p - a) / a) * 100

def evaluate(make, use_raw):
    R2, RMSE, MAE, DMdAE, DMdAPE = [], [], [], [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    Xsrc = Xraw if use_raw else Xohe
    for tr, te in kf.split(Xsrc):
        Xtr = Xsrc.iloc[tr]; Xte = Xsrc.iloc[te]
        m = make().fit(Xtr, y[tr])
        p = m.predict(Xte)
        R2.append(r2_score(y[te], p))
        RMSE.append(mean_squared_error(y[te], p) ** 0.5)
        MAE.append(mean_absolute_error(y[te], p))
        d1, d2 = dollar_metrics(y[te], p)
        DMdAE.append(d1); DMdAPE.append(d2)
    f = lambda v: f"{np.mean(v):.3f}±{np.std(v):.3f}"
    g = lambda v: f"${np.mean(v):,.0f}±{np.std(v):,.0f}"
    return dict(R2=f(R2), RMSE_log=f(RMSE), MAE_log=f(MAE),
                Dollar_MdAE=g(DMdAE), Dollar_MdAPE=f"{np.mean(DMdAPE):.0f}%±{np.std(DMdAPE):.0f}")

models = {
    "Lasso (LassoCV)": (lambda: LassoCV(cv=3, n_jobs=-1, random_state=42), False),
    "LightGBM":        (lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                                  num_leaves=31, random_state=42, n_jobs=-1, verbose=-1), False),
    "EBM (glass-box)": (lambda: ExplainableBoostingRegressor(interactions=0, random_state=42), True),
}

print(f"providers={len(df)}  target={target}  (5-fold CV, mean±SD)\n")
hdr = ["model", "R2(log)", "RMSE(log)", "MAE(log)", "Dollar MdAE", "Dollar MdAPE"]
print("  ".join(f"{h:<16}" for h in hdr))
for name, (make, raw) in models.items():
    r = evaluate(make, raw)
    print("  ".join(f"{v:<16}" for v in
          [name, r["R2"], r["RMSE_log"], r["MAE_log"], r["Dollar_MdAE"], r["Dollar_MdAPE"]]))
