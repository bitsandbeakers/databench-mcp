"""
Explore the for-profit ownership effect (provider grain, leakage-free).

  1. Nested decomposition  — how much of the raw for-profit premium survives
     each control? (confounding probe: is it ownership or geography/case-mix?)
  2. Heterogeneity         — premium by archetype and by metro/non-metro.
  3. Chain probe           — HCA-branded vs other for-profits.
  4. Over-chargers         — top for-profit providers above model expectation
                             (actionable steerage targets; ties to h013).
"""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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
    i.Rndrng_Prvdr_Org_Name AS org_name,
    i.Rndrng_Prvdr_State_Abrvtn AS state,
    CASE WHEN i.Rndrng_Prvdr_RUCA<4 THEN 'Metro' ELSE 'NonMetro' END AS metro
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
SELECT s.ccn, ANY_VALUE(s.org_name) AS org_name,
  COUNT(DISTINCT s.DRG_Cd) AS n_drgs,
  LN(SUM(s.disch)) AS log_volume,
  SUM(s.sev*s.disch)/SUM(s.disch) AS mean_severity,
  LN(SUM(s.charge*s.disch)/SUM(s.disch)) AS ln_actual_charge,
  LN(SUM(s.natl_med*s.disch)/SUM(s.disch)) AS ln_expected_charge,
  SUM(s.disch*s.disch)/POWER(SUM(s.disch),2) AS hhi_specialization,
  ANY_VALUE(s.state) AS state, ANY_VALUE(s.metro) AS metro,
  COALESCE(o.ownership,'Unknown') AS ownership,
  COALESCE(a.archetype_group,'Unknown') AS archetype_group
FROM svc s LEFT JOIN own o USING (ccn) LEFT JOIN arch a USING (ccn)
GROUP BY s.ccn, o.ownership, a.archetype_group
"""

con = duckdb.connect(DB, read_only=True)
df = con.execute(SQL).df()
con.close()
df["for_profit"] = (df.ownership == "ForProfit").astype(float)
y = df["ln_actual_charge"].astype(float)

NUM = ["ln_expected_charge", "mean_severity", "hhi_specialization", "n_drgs", "log_volume"]

def design(cols):
    cats = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    return pd.get_dummies(df[cols], columns=cats, drop_first=True).astype(float)

def fp_premium(controls, data=df, yv=y):
    cols = ["for_profit"] + controls
    cats = [c for c in cols if not pd.api.types.is_numeric_dtype(data[c])]
    X = pd.get_dummies(data[cols], columns=cats, drop_first=True).astype(float)
    m = LinearRegression().fit(X, yv)
    return pd.Series(m.coef_, index=X.columns)["for_profit"]

print("=" * 70)
print(f"For-profit ownership effect  |  {len(df)} providers  "
      f"|  {int(df.for_profit.sum())} for-profit")
print("=" * 70)

# ---- 1. Nested decomposition ----
print("\n[1] Does the for-profit premium survive controls? (cumulative)")
stages = [
    ("raw (no controls)", []),
    ("+ case-mix", NUM),
    ("+ archetype", NUM + ["archetype_group"]),
    ("+ urbanicity", NUM + ["archetype_group", "metro"]),
    ("+ state (full)", NUM + ["archetype_group", "metro", "state"]),
]
for label, ctrl in stages:
    b = fp_premium(ctrl)
    print(f"    {label:<22} for-profit premium = {np.exp(b)-1:+6.1%}")

# ---- 2. Heterogeneity by archetype ----
print("\n[2] For-profit premium by archetype (case-mix + state controlled, n_fp>=10):")
ctrl = NUM + ["state"]
rows = []
for g, sub in df.groupby("archetype_group"):
    nfp = int(sub.for_profit.sum())
    if nfp >= 10 and sub.for_profit.nunique() == 2:
        b = fp_premium(ctrl, data=sub, yv=sub["ln_actual_charge"].astype(float))
        rows.append((g, nfp, len(sub), np.exp(b) - 1))
for g, nfp, n, prem in sorted(rows, key=lambda r: -r[3]):
    print(f"    {g:<26} prem {prem:+6.1%}   (n_fp={nfp}, n={n})")

# ---- 3. Metro vs non-metro ----
print("\n[3] For-profit premium: metro vs non-metro (case-mix + state controlled):")
for g, sub in df.groupby("metro"):
    if sub.for_profit.nunique() == 2:
        b = fp_premium(NUM + ["state"], data=sub, yv=sub["ln_actual_charge"].astype(float))
        print(f"    {g:<10} prem {np.exp(b)-1:+6.1%}   (n_fp={int(sub.for_profit.sum())}, n={len(sub)})")

# ---- 4. HCA-branded vs other for-profit (case-mix adjusted residual) ----
m_cm = LinearRegression().fit(df[NUM].astype(float), y)
df["resid_cm"] = y - m_cm.predict(df[NUM].astype(float))
fp = df[df.for_profit == 1].copy()
fp["hca"] = fp.org_name.str.upper().str.contains("HCA", na=False)
print("\n[4] HCA-branded vs other for-profit (residual above case-mix expectation):")
for g, sub in fp.groupby("hca"):
    lbl = "HCA-branded" if g else "Other for-profit"
    print(f"    {lbl:<18} mean resid {sub.resid_cm.mean():+.3f} log pts "
          f"({np.exp(sub.resid_cm.mean())-1:+.1%})   n={len(sub)}")

# ---- 5. Top for-profit over-chargers (full-model residual) ----
Xfull = design(NUM + ["archetype_group", "metro", "state"])
df["resid_full"] = y - LinearRegression().fit(Xfull, y).predict(Xfull)
top = (df[df.for_profit == 1]
       .sort_values("resid_full", ascending=False)
       .head(15)[["org_name", "state", "archetype_group", "resid_full"]])
print("\n[5] Top 15 for-profit over-chargers (above full-model expectation):")
for _, r in top.iterrows():
    print(f"    {r.org_name[:34]:<34} {r.state:<3} {r.archetype_group[:16]:<16} "
          f"{np.exp(r.resid_full)-1:+.0%}")
