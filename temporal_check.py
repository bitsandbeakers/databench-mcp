"""
Out-of-year holdout (h017): re-run the headline 2023 findings on 2022 MUP data with
IDENTICAL SQL and report hold / doesn't-hold. Statements only, no charts.

2022 raw CSVs are Latin-1 (the 2023 files were pre-converted to UTF-8) -> re-encode first.
2023 raw tables + HCRIS ownership are read from project.duckdb (read-only).
"""
from pathlib import Path
import duckdb, numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent / "workspace" / "uhc_affordability"
RAW = BASE / "raw"
DB = BASE / "project.duckdb"

# --- re-encode 2022 files latin-1 -> utf-8 ---
for f in ["inpatient_2022", "outpatient_2022"]:
    src = RAW / f"{f}.csv"
    dst = RAW / f"{f}_utf8.csv"
    if not dst.exists():
        dst.write_text(src.read_text(encoding="latin-1"), encoding="utf-8")

con = duckdb.connect()
con.execute(f"ATTACH '{str(DB).replace(chr(92), '/')}' AS p (READ_ONLY)")
con.execute("CREATE TABLE ip22 AS SELECT * FROM read_csv_auto(?)", [str(RAW / "inpatient_2022_utf8.csv")])
con.execute("CREATE TABLE op22 AS SELECT * FROM read_csv_auto(?)", [str(RAW / "outpatient_2022_utf8.csv")])
con.execute("CREATE VIEW ip23 AS SELECT * FROM p.inpatient_2023")
con.execute("CREATE VIEW op23 AS SELECT * FROM p.outpatient_2023")
print("OP2023 cols:", [c[0] for c in con.execute("DESCRIBE op23").fetchall()][:20])

def one(sql):
    return con.execute(sql).fetchone()[0]

def metrics(ip, op):
    m = {}
    m["skew_ip_charge"] = one(f"SELECT skewness(Avg_Submtd_Cvrd_Chrg) FROM {ip} WHERE Avg_Submtd_Cvrd_Chrg>0")
    m["disp_p90p10"] = one(f"""WITH d AS (SELECT DRG_Cd,
        quantile_cont(Avg_Submtd_Cvrd_Chrg,0.9)/nullif(quantile_cont(Avg_Submtd_Cvrd_Chrg,0.1),0) r
        FROM {ip} WHERE Avg_Submtd_Cvrd_Chrg>0 GROUP BY DRG_Cd HAVING count(*)>=50)
        SELECT median(r) FROM d""")
    m["markup_ip"] = one(f"SELECT median(Avg_Submtd_Cvrd_Chrg/nullif(Avg_Mdcr_Pymt_Amt,0)) FROM {ip} WHERE Avg_Mdcr_Pymt_Amt>0")
    m["markup_op"] = one(f"SELECT median(Avg_Tot_Sbmtd_Chrgs/nullif(Avg_Mdcr_Pymt_Amt,0)) FROM {op} WHERE Avg_Mdcr_Pymt_Amt>0")
    # urbanicity: case-mix-adjusted metro vs non-metro
    urb = con.execute(f"""WITH natl AS (SELECT DRG_Cd, median(Avg_Submtd_Cvrd_Chrg) m FROM {ip}
            WHERE Avg_Submtd_Cvrd_Chrg>0 AND Rndrng_Prvdr_RUCA<>99 GROUP BY DRG_Cd),
        r AS (SELECT CASE WHEN i.Rndrng_Prvdr_RUCA<4 THEN 'Metro' ELSE 'NonMetro' END urb,
                     i.Avg_Submtd_Cvrd_Chrg/n.m idx
              FROM {ip} i JOIN natl n USING(DRG_Cd)
              WHERE i.Avg_Submtd_Cvrd_Chrg>0 AND i.Rndrng_Prvdr_RUCA<>99)
        SELECT urb, median(idx) FROM r GROUP BY urb""").fetchall()
    d = dict(urb)
    m["nonmetro_vs_metro_pct"] = (d["NonMetro"] / d["Metro"] - 1) * 100
    return m

def state_index(ip):
    rows = con.execute(f"""WITH natl AS (SELECT DRG_Cd, median(Avg_Submtd_Cvrd_Chrg) m FROM {ip}
            WHERE Avg_Submtd_Cvrd_Chrg>0 GROUP BY DRG_Cd),
        r AS (SELECT i.Rndrng_Prvdr_State_Abrvtn st, i.Avg_Submtd_Cvrd_Chrg/n.m idx, i.Rndrng_Prvdr_CCN ccn
              FROM {ip} i JOIN natl n USING(DRG_Cd) WHERE i.Avg_Submtd_Cvrd_Chrg>0)
        SELECT st, median(idx) idx FROM r GROUP BY st HAVING count(distinct ccn)>=5""").fetchall()
    return {s: v for s, v in rows}

def provider_ip_idx(ip):
    return con.execute(f"""WITH natl AS (SELECT DRG_Cd, median(Avg_Submtd_Cvrd_Chrg) m FROM {ip}
            WHERE Avg_Submtd_Cvrd_Chrg>0 GROUP BY DRG_Cd)
        SELECT PRINTF('%06d',TRY_CAST(i.Rndrng_Prvdr_CCN AS INT)) ccn,
               median(i.Avg_Submtd_Cvrd_Chrg/n.m) idx
        FROM {ip} i JOIN natl n USING(DRG_Cd) WHERE i.Avg_Submtd_Cvrd_Chrg>0
        GROUP BY 1""").df()

def provider_op_idx(op):
    return con.execute(f"""WITH natl AS (SELECT APC_Cd, median(Avg_Tot_Sbmtd_Chrgs) m FROM {op}
            WHERE Avg_Tot_Sbmtd_Chrgs>0 GROUP BY APC_Cd)
        SELECT PRINTF('%06d',TRY_CAST(o.Rndrng_Prvdr_CCN AS INT)) ccn,
               median(o.Avg_Tot_Sbmtd_Chrgs/n.m) idx
        FROM {op} o JOIN natl n USING(APC_Cd) WHERE o.Avg_Tot_Sbmtd_Chrgs>0
        GROUP BY 1""").df()

def ipop_spearman(ip, op):
    a = provider_ip_idx(ip).rename(columns={"idx": "ip"})
    b = provider_op_idx(op).rename(columns={"idx": "op"})
    m = a.merge(b, on="ccn")
    return spearmanr(m["ip"], m["op"]).correlation, len(m)

def forprofit_premium(ip):
    ipx = provider_ip_idx(ip)
    own = con.execute("""SELECT PRINTF('%06d',TRY_CAST("Provider CCN" AS INT)) ccn,
        CASE WHEN "Type of Control" IN (1,2) THEN 'Nonprofit'
             WHEN "Type of Control" IN (3,4,5,6) THEN 'ForProfit' ELSE 'Gov' END own
        FROM p.cost_report_2023 WHERE TRY_CAST("Provider CCN" AS INT) IS NOT NULL""").df()
    m = ipx.merge(own, on="ccn")
    fp = m[m.own == "ForProfit"]["idx"].median()
    npf = m[m.own == "Nonprofit"]["idx"].median()
    return (fp / npf - 1) * 100

print("\n=== headline metrics: 2022 vs 2023 ===")
m22, m23 = metrics("ip22", "op22"), metrics("ip23", "op23")
for k in m22:
    print(f"{k:24} 2022={m22[k]:8.3f}   2023={m23[k]:8.3f}")

s22, s23 = state_index("ip22"), state_index("ip23")
common = sorted(set(s22) & set(s23))
rho = spearmanr([s22[s] for s in common], [s23[s] for s in common]).correlation
print(f"\nstate cost-index ordering: Spearman(2022,2023) = {rho:.3f}  (n_states={len(common)})")
top23 = sorted(s23, key=s23.get, reverse=True)[:5]
print("  most-expensive states 2023:", top23)
print("  most-expensive states 2022:", sorted(s22, key=s22.get, reverse=True)[:5])

r22, n22 = ipop_spearman("ip22", "op22")
r23, n23 = ipop_spearman("ip23", "op23")
print(f"\nIP/OP pricing-culture Spearman: 2022={r22:.3f} (n={n22})   2023={r23:.3f} (n={n23})")

print(f"\nfor-profit premium (median adj IP charge idx, FP vs NP, 2023 ownership): "
      f"2022={forprofit_premium('ip22'):.0f}%   2023={forprofit_premium('ip23'):.0f}%")
print(f"\nrows: ip22={one('SELECT count(*) FROM ip22')}  ip23={one('SELECT count(*) FROM ip23')}  "
      f"op22={one('SELECT count(*) FROM op22')}  op23={one('SELECT count(*) FROM op23')}")
con.close()
