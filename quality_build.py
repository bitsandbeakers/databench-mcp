"""
h018 cost x quality: join CMS Hospital Compare overall star rating to the archetype-adjusted
cost index and emit two parquet chart tables the dashboard reads (dashboards/data/).

Read-only against project.duckdb. CMS Hospital Compare ('Hospital General Information',
overall star rating keyed by CCN) was ingested as hospital_quality.
"""
from pathlib import Path
import duckdb, numpy as np

BASE = Path(__file__).resolve().parent / "workspace" / "uhc_affordability"
DB = BASE / "project.duckdb"
OUT = BASE / "dashboards" / "data"

# short labels for the 11 avoidances (match org_name -> compact display)
SHORT = {
    "Capital Health Regional Medical Center": "Capital Health Regional (NJ)",
    "Carepoint Health-Christ Hospital": "Carepoint Christ (NJ)",
    "Carepoint Health - Bayonne Medical Center": "Carepoint Bayonne (NJ)",
    "Carepoint Health-Hoboken University Medical Center": "Carepoint Hoboken (NJ)",
    "Capital Health Medical Center - Hopewell": "Capital Health Hopewell (NJ)",
    "Regional Medical Center Of San Jose": "RMC San Jose (CA)",
    "Martin Luther King, Jr. Community Hospital": "MLK Jr Community (CA)",
    "North Houston Surgical Hospital Llc": "North Houston Surgical (TX)",
    "Good Samaritan Hospital": "Good Samaritan (CA)",
    "Stanford Health Care": "Stanford (CA)",
    "Presbyterian/St Luke's Medical Center": "Presby/St Luke's (CO)",
}

con = duckdb.connect(str(DB), read_only=True)

scatter = con.execute("""
SELECT s.ccn, s.org_name, s.state, s.archetype_group,
       s.adj_charge_idx AS cost_idx, s.markup,
       TRY_CAST(q."Hospital overall rating" AS INT) AS star,
       s.high_conf_outlier AS is_avoid
FROM archetype_outliers_scored s
JOIN hospital_quality q ON q."Facility ID" = PRINTF('%06d', TRY_CAST(s.ccn AS INT))
WHERE TRY_CAST(q."Hospital overall rating" AS INT) IS NOT NULL
""").df()

rng = np.random.default_rng(7)
scatter["star_j"] = scatter["star"] + rng.uniform(-0.18, 0.18, len(scatter))
scatter["short"] = scatter["org_name"].map(SHORT).fillna("")
scatter = scatter[["ccn", "short", "state", "archetype_group", "cost_idx", "markup",
                   "star", "star_j", "is_avoid"]]

bands = con.execute("""
WITH j AS (
  SELECT s.adj_charge_idx AS c, TRY_CAST(q."Hospital overall rating" AS INT) AS star
  FROM archetype_outliers_scored s
  JOIN hospital_quality q ON q."Facility ID" = PRINTF('%06d', TRY_CAST(s.ccn AS INT))
  WHERE TRY_CAST(q."Hospital overall rating" AS INT) IS NOT NULL)
SELECT CASE WHEN c<0.8 THEN '<0.8\\ncheap'
            WHEN c<1.2 THEN '0.8-1.2\\nnorm'
            WHEN c<2.0 THEN '1.2-2.0\\npricey'
            ELSE '>=2.0\\nvery pricey' END AS band,
       CASE WHEN c<0.8 THEN 1 WHEN c<1.2 THEN 2 WHEN c<2.0 THEN 3 ELSE 4 END AS ord,
       count(*) AS n, round(avg(star),2) AS avg_star,
       round(100.0*sum(CASE WHEN star>=4 THEN 1 ELSE 0 END)/count(*),0) AS pct_4_5star
FROM j GROUP BY 1,2 ORDER BY ord
""").df()
con.close()

scatter.to_parquet(OUT / "chart_h018_scatter.parquet", index=False)
bands.to_parquet(OUT / "chart_h018_bands.parquet", index=False)
print(f"scatter rows={len(scatter)} (avoid={int(scatter.is_avoid.sum())})  bands rows={len(bands)}")
print(bands.to_string(index=False))
