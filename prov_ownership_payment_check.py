"""
Falsification test: is the for-profit premium PRICING behavior or real cost?

Same provider-grain model, two targets:
  - ln charge   (what the hospital bills)        -> expect large for-profit premium
  - ln payment  (what Medicare actually pays, formula-set) -> expect ~0

If for-profit shows a big charge premium but ~0 payment premium, the effect is
charging behavior, not higher underlying cost of care.
"""
from __future__ import annotations
import duckdb, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

DB = r"C:\Users\cody\nox-code\satellites\databench-mcp\workspace\uhc_affordability\project.duckdb"
SQL = r"""
WITH natl AS (
  SELECT DRG_Cd, MEDIAN(Avg_Submtd_Cvrd_Chrg) AS natl_chg, MEDIAN(Avg_Mdcr_Pymt_Amt) AS natl_pay
  FROM inpatient_enr
  WHERE Avg_Submtd_Cvrd_Chrg>0 AND Avg_Mdcr_Pymt_Amt>0 AND Tot_Dschrgs>0 AND Rndrng_Prvdr_RUCA<>99
  GROUP BY DRG_Cd
),
svc AS (
  SELECT PRINTF('%06d', TRY_CAST(i.Rndrng_Prvdr_CCN AS INTEGER)) AS ccn,
    i.Tot_Dschrgs::DOUBLE AS disch, i.Avg_Submtd_Cvrd_Chrg AS chg, i.Avg_Mdcr_Pymt_Amt AS pay,
    n.natl_chg, n.natl_pay,
    CASE WHEN i.DRG_Desc ILIKE '%WITHOUT%' OR i.DRG_Desc ILIKE '%W/O%' THEN 0
         WHEN i.DRG_Desc ILIKE '%MCC%' THEN 2 WHEN i.DRG_Desc ILIKE '% CC%' THEN 1 ELSE 0 END AS sev,
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
    FROM cost_report_2023 WHERE TRY_CAST("Provider CCN" AS INTEGER) IS NOT NULL) GROUP BY ccn),
arch AS (
  SELECT PRINTF('%06d', TRY_CAST(ccn AS INTEGER)) AS ccn, MAX(archetype_group) AS archetype_group
  FROM archetype_outlier_input WHERE TRY_CAST(ccn AS INTEGER) IS NOT NULL GROUP BY ccn)
SELECT s.ccn,
  LN(SUM(s.disch)) AS log_volume,
  SUM(s.sev*s.disch)/SUM(s.disch) AS mean_severity,
  SUM(s.disch*s.disch)/POWER(SUM(s.disch),2) AS hhi_specialization,
  LN(SUM(s.chg*s.disch)/SUM(s.disch)) AS ln_charge,
  LN(SUM(s.pay*s.disch)/SUM(s.disch)) AS ln_payment,
  LN(SUM(s.natl_chg*s.disch)/SUM(s.disch)) AS ln_exp_charge,
  LN(SUM(s.natl_pay*s.disch)/SUM(s.disch)) AS ln_exp_payment,
  ANY_VALUE(s.state) AS state, ANY_VALUE(s.metro) AS metro,
  COALESCE(o.ownership,'Unknown') AS ownership, COALESCE(a.archetype_group,'Unknown') AS archetype_group
FROM svc s LEFT JOIN own o USING (ccn) LEFT JOIN arch a USING (ccn)
GROUP BY s.ccn, o.ownership, a.archetype_group
"""
con = duckdb.connect(DB, read_only=True); df = con.execute(SQL).df(); con.close()
df["for_profit"] = (df.ownership == "ForProfit").astype(float)

def fp_premium(target, expected):
    cols = ["for_profit", expected, "mean_severity", "hhi_specialization", "log_volume",
            "archetype_group", "metro", "state"]
    cats = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    X = pd.get_dummies(df[cols], columns=cats, drop_first=True).astype(float)
    m = LinearRegression().fit(X, df[target].astype(float))
    return pd.Series(m.coef_, index=X.columns)["for_profit"]

print("=" * 64)
print("Falsification: for-profit premium on CHARGE vs MEDICARE PAYMENT")
print("(full controls: own case-mix expected + severity/spec/vol + archetype + metro + state)")
print("=" * 64)
bc = fp_premium("ln_charge", "ln_exp_charge")
bp = fp_premium("ln_payment", "ln_exp_payment")
print(f"\n  Submitted CHARGE   for-profit premium = {np.exp(bc)-1:+6.1%}")
print(f"  Medicare PAYMENT   for-profit premium = {np.exp(bp)-1:+6.1%}")
print("\n  -> If charge premium is large but payment premium ~0, the effect is")
print("     CHARGING BEHAVIOR, not higher underlying cost of care.")
