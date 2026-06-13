"""
Dollar sizing of the for-profit charge premium (the "so what").

Measured directly from 2023 Medicare inpatient volume:
  billed = submitted charge x discharges, per provider x DRG.

Two excess measures at for-profit hospitals:
  A. vs case-mix benchmark   = billed above the national-median charge per DRG
  B. ownership-attributable  = the portion explained by the +46% for-profit
                               premium (counterfactual: priced like non-for-profits)

Then steerage scenarios + a per-$1B unit so it scales to any commercial book.
Caveats printed at the end — this is Medicare volume, charge-anchored framing.
"""
from __future__ import annotations
import duckdb, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

DB = r"C:\Users\cody\nox-code\satellites\databench-mcp\workspace\uhc_affordability\project.duckdb"

# ---- service-grain billed + ownership + national-median benchmark ----
SVC = r"""
WITH natl AS (
  SELECT DRG_Cd, MEDIAN(Avg_Submtd_Cvrd_Chrg) AS natl_med
  FROM inpatient_enr
  WHERE Avg_Submtd_Cvrd_Chrg>0 AND Avg_Mdcr_Pymt_Amt>0 AND Tot_Dschrgs>0 AND Rndrng_Prvdr_RUCA<>99
  GROUP BY DRG_Cd
),
own AS (
  SELECT ccn, MAX(ownership) AS ownership FROM (
    SELECT PRINTF('%06d', TRY_CAST("Provider CCN" AS INTEGER)) AS ccn,
      CASE WHEN "Type of Control" IN (1,2) THEN 'Nonprofit'
           WHEN "Type of Control" IN (3,4,5,6) THEN 'ForProfit' ELSE 'Government' END AS ownership
    FROM cost_report_2023 WHERE TRY_CAST("Provider CCN" AS INTEGER) IS NOT NULL) GROUP BY ccn)
SELECT
  PRINTF('%06d', TRY_CAST(i.Rndrng_Prvdr_CCN AS INTEGER)) AS ccn,
  COALESCE(o.ownership,'Unknown') AS ownership,
  i.Tot_Dschrgs::DOUBLE AS disch,
  i.Avg_Submtd_Cvrd_Chrg AS charge,
  n.natl_med
FROM inpatient_enr i JOIN natl n USING (DRG_Cd)
LEFT JOIN own o ON o.ccn = PRINTF('%06d', TRY_CAST(i.Rndrng_Prvdr_CCN AS INTEGER))
WHERE i.Avg_Submtd_Cvrd_Chrg>0 AND i.Avg_Mdcr_Pymt_Amt>0 AND i.Tot_Dschrgs>0 AND i.Rndrng_Prvdr_RUCA<>99
"""
con = duckdb.connect(DB, read_only=True)
s = con.execute(SVC).df()
con.close()

s["billed"] = s.charge * s.disch
s["billed_bench"] = s.natl_med * s.disch
s["excess_vs_bench"] = (s.billed - s.billed_bench).clip(lower=0)

total_billed = s.billed.sum()
fp = s[s.ownership == "ForProfit"]
fp_billed = fp.billed.sum()

PREMIUM = 0.463            # for-profit vs rest, full controls (from prov_drivers analysis)
factor = 1 + PREMIUM
own_attrib_excess = fp_billed * (1 - 1 / factor)   # remove the premium -> counterfactual

def money(x):
    return f"${x/1e9:6.2f}B" if abs(x) >= 1e9 else f"${x/1e6:6.0f}M"

print("=" * 66)
print("DOLLAR SIZING - for-profit charge premium (2023 Medicare inpatient)")
print("=" * 66)
print(f"\nTotal submitted charges (billed) , all hospitals : {money(total_billed)}")
print(f"  For-profit hospitals                           : {money(fp_billed)}  "
      f"({fp_billed/total_billed:.0%} of billed)")
print(f"  For-profit discharges                          : {fp.disch.sum()/1e6:.2f}M")

print(f"\n[A] Excess billed at for-profits vs case-mix benchmark (national median):")
print(f"      {money(fp.excess_vs_bench.sum())}  "
      f"({fp.excess_vs_bench.sum()/fp_billed:.0%} of for-profit billed)")

print(f"\n[B] Ownership-ATTRIBUTABLE excess (the +{PREMIUM:.0%} premium removed):")
print(f"      counterfactual = for-profit billed / {factor:.3f}")
print(f"      excess = {money(own_attrib_excess)}  "
      f"({1-1/factor:.0%} of for-profit billed, "
      f"{own_attrib_excess/total_billed:.1%} of ALL inpatient billed)")

print("\nSteerage scenarios (share of that excess recovered by redirecting volume):")
for frac in (0.25, 0.50, 1.00):
    print(f"      {frac:>4.0%} steered  ->  {money(own_attrib_excess*frac)} saved")

print("\nScalable unit (for any charge-anchored commercial book):")
print(f"      For-profit hospitals bill ~{PREMIUM:.0%} above comparable peers, i.e.")
print(f"      ~{1-1/factor:.1%} of every dollar billed at a for-profit is premium.")
print(f"      -> per $1.0B of charge-anchored inpatient spend at for-profits: "
      f"{money(1e9*(1-1/factor))} excess.")

print("\nCAVEATS")
print("  - Volume is 2023 MEDICARE inpatient; Medicare pays DRG rates, not charges,")
print("    so these are charges a CHARGE-ANCHORED commercial payer would overpay,")
print("    not Medicare outlays. Commercial volume is not in this data.")
print("  - Premium applied as a uniform average (+46%); per-provider varies.")
print("  - Excess vs benchmark [A] includes non-ownership markup; [B] isolates the")
print("    for-profit-attributable share via the controlled model estimate.")
