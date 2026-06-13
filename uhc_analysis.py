"""
UHC Affordability Interview Task — CMS Medicare EDA
Runs the full databench pipeline: ingest → profile → EDA → outliers → correlations → viz

Datasets:
  inpatient      MUP Inpatient Provider-Service, DY2023 (provider × DRG)
  outpatient     MUP Outpatient Provider-Service, DY2023 (provider × APC)
  hospital_info  Hospital General Information (provider characteristics)
  cost_report    HCRIS Hospital Cost Report FY2023 Final (cost/financial metrics)
  provider_model Provider-level enriched table for affordability modeling
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

# ── Databench tool imports ──────────────────────────────────────────────────
from databench_mcp.tools.project import project_create, project_status
from databench_mcp.tools.ingest import ingest_file
from databench_mcp.tools.profile import profile_table
from databench_mcp.tools.eda import eda_summary, sql_query, group_summary, derive_table, enrich_table
from databench_mcp.tools.analysis import detect_outliers, analyze_correlations, analyze_distribution
from databench_mcp.tools.modeling import run_model
from databench_mcp.tools.viz import create_chart
from databench_mcp.tools.hypothesis import hypothesis_add
from databench_mcp.workspace import project_path, read_manifest, write_manifest

PROJECT = "uhc_cms_2023"

# ── CMS dataset URLs ────────────────────────────────────────────────────────
INPATIENT_URL = (
    "https://data.cms.gov/sites/default/files/2025-05/"
    "ca1c9013-8c7c-4560-a4a1-28cf7e43ccc8/MUP_INP_RY25_P03_V10_DY23_PrvSvc.CSV"
)
OUTPATIENT_URL = (
    "https://data.cms.gov/sites/default/files/2025-08/"
    "bceaa5e1-e58c-4109-9f05-832fc5e6bbc8/MUP_OUT_RY25_P04_V10_DY23_Prov_Svc.csv"
)
# Stable API endpoint — no hash rotation
HOSPITAL_INFO_URL = (
    "https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0/download?format=csv"
)
# HCRIS cost report FY2023 Final — URL may rotate on CMS refresh; re-resolve via:
# GET https://data.cms.gov/api/1/metastore/schemas/dataset/items/44060663-47d8-4ced-a115-b53b4c270acb
COST_REPORT_URL = (
    "https://data.cms.gov/sites/default/files/2026-01/"
    "3c39f483-c7e0-4025-8396-4df76942e10f/CostReport_2023_Final.csv"
)


def step(label: str, result: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)
    r = {k: v for k, v in result.items() if k not in ("data", "rows", "chart_json")}
    out = json.dumps(r, indent=2, default=str)
    print(out[:3000])
    return result


def download_and_fix_encoding(url: str, dest: Path, source_encoding: str = "cp1252") -> None:
    """Download a CMS CSV and re-save as UTF-8 (CMS files often use Windows-1252)."""
    import httpx
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  (cached) {dest.name}")
        return
    print(f"  Downloading {url} ...")
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
    text = resp.content.decode(source_encoding, errors="replace")
    dest.write_text(text, encoding="utf-8")
    print(f"  Saved {dest.stat().st_size / 1e6:.1f}MB to {dest}")


def ingest_csv_skip_blanks(project: str, file_path: Path, table_name: str) -> dict:
    """Ingest a CSV that has blank lines between rows (common in HCRIS exports).

    Pre-strips blank lines to a companion _clean file, then loads via DuckDB.
    Registers the table in the workspace manifest exactly as ingest_file would.
    """
    clean_path = file_path.with_suffix(".clean.csv")
    if not clean_path.exists():
        print(f"  Stripping blank lines from {file_path.name} ...")
        with open(file_path, encoding="utf-8", errors="replace") as src, \
             open(clean_path, "w", encoding="utf-8") as dst:
            for line in src:
                if line.strip():
                    dst.write(line)
        print(f"  Wrote {clean_path.stat().st_size / 1e6:.1f}MB to {clean_path.name}")

    db_path = str(project_path(project) / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute(
        f"CREATE OR REPLACE TABLE {table_name} AS "
        f"SELECT * FROM read_csv('{clean_path}', strict_mode=false, auto_detect=true)"
    )
    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    raw_schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
    conn.close()

    schema = [{"name": r[0], "type": r[1]} for r in raw_schema]
    manifest = read_manifest(project)
    manifest["datasets"][table_name] = {
        "source": str(file_path),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "profiled": False,
        "profiled_at": None,
        "row_count": row_count,
        "col_count": len(schema),
    }
    write_manifest(project, manifest)
    return {"table": table_name, "rows": row_count, "columns": len(schema), "schema": schema}


def main() -> None:
    # ── 1. Create project ────────────────────────────────────────────────────
    step("project_create", project_create(PROJECT))

    # ── 2. Download + re-encode to UTF-8, then ingest ───────────────────────
    raw_dir = project_path(PROJECT) / "raw"
    inp_dest = raw_dir / "inpatient.csv"
    out_dest = raw_dir / "outpatient.csv"
    hosp_info_dest = raw_dir / "hospital_info.csv"
    cost_report_dest = raw_dir / "cost_report.csv"

    download_and_fix_encoding(INPATIENT_URL, inp_dest)
    download_and_fix_encoding(OUTPATIENT_URL, out_dest)
    # Hospital General Info uses UTF-8 (no re-encoding needed, but reuse helper for caching)
    download_and_fix_encoding(HOSPITAL_INFO_URL, hosp_info_dest, source_encoding="utf-8")
    download_and_fix_encoding(COST_REPORT_URL, cost_report_dest, source_encoding="utf-8")

    step("ingest_file: inpatient", ingest_file(PROJECT, str(inp_dest), "inpatient"))
    step("ingest_file: outpatient", ingest_file(PROJECT, str(out_dest), "outpatient"))
    step("ingest_file: hospital_info", ingest_file(PROJECT, str(hosp_info_dest), "hospital_info"))
    step("ingest_file: cost_report", ingest_csv_skip_blanks(PROJECT, cost_report_dest, "cost_report"))

    # ── 3. Profile all tables ────────────────────────────────────────────────
    step("profile_table: inpatient", profile_table(PROJECT, "inpatient"))
    step("profile_table: outpatient", profile_table(PROJECT, "outpatient"))
    step("profile_table: hospital_info", profile_table(PROJECT, "hospital_info"))
    step("profile_table: cost_report", profile_table(PROJECT, "cost_report"))

    # ── 4. EDA summary (project-level manifest) ──────────────────────────────
    step("eda_summary", eda_summary(PROJECT))

    # ── 4b. Slim views of the reference tables (rename columns, drop junk) ───
    step("derive_table: hospital_info_slim", derive_table(
        PROJECT,
        sql="""
        SELECT
            "Facility ID"                       AS ccn,
            "Facility Name"                     AS hospital_name,
            "Hospital Type"                     AS hospital_type,
            "Hospital Ownership"                AS ownership_type,
            "Emergency Services"                AS emergency_services,
            TRY_CAST("Hospital overall rating" AS INTEGER) AS cms_star_rating,
            "Count of Facility MORT measures"   AS mort_measure_count,
            "Count of Facility Safety measures" AS safety_measure_count,
            "Count of Facility READM measures"  AS readm_measure_count
        FROM hospital_info
        WHERE "Facility ID" IS NOT NULL
        """,
        table_name="hospital_info_slim",
    ))
    step("profile_table: hospital_info_slim", profile_table(PROJECT, "hospital_info_slim"))

    step("derive_table: cost_report_slim", derive_table(
        PROJECT,
        sql="""
        SELECT
            PRINTF('%06d', TRY_CAST("Provider CCN" AS INTEGER)) AS ccn,
            "Hospital Name"                             AS hospital_name_cr,
            "Rural Versus Urban"                        AS rural_urban,
            "Type of Control"                           AS control_type,
            "Number of Beds"                            AS num_beds,
            "FTE - Employees on Payroll"                AS fte_employees,
            "Number of Interns and Residents (FTE)"     AS intern_resident_fte,
            "Total Discharges Title XVIII"               AS medicare_discharges,
            "Total Days Title XVIII"                     AS medicare_patient_days,
            "Total Discharges (V + XVIII + XIX + Unknown)" AS total_discharges_all,
            "Total Costs"                               AS total_costs,
            "Inpatient Revenue"                         AS inpatient_revenue,
            "Outpatient Revenue"                        AS outpatient_revenue,
            "Net Patient Revenue"                       AS net_patient_revenue,
            "Cost To Charge Ratio"                      AS cost_to_charge_ratio,
            "Allowable DSH Percentage"                  AS dsh_pct,
            "Disproportionate Share Adjustment"         AS dsh_adjustment_amt,
            "Cost of Uncompensated Care"                AS uncompensated_care_cost,
            "Total Bad Debt Expense"                    AS bad_debt_expense,
            "Total Salaries (adjusted)"                 AS total_salaries,
            "Total IME Payment"                         AS ime_payment,
            "DRG Amounts After October 1"               AS drg_base_payment
        FROM cost_report
        WHERE "Provider CCN" IS NOT NULL
          AND "CCN Facility Type" IN ('STH', 'CAH', 'PH', 'RH', 'LTCH', 'CH')
        """,
        table_name="cost_report_slim",
    ))
    step("profile_table: cost_report_slim", profile_table(PROJECT, "cost_report_slim"))

    # ── 4c. Provider-level inpatient aggregate (collapse provider×DRG → provider) ──
    step("derive_table: inpatient_by_provider", derive_table(
        PROJECT,
        sql="""
        SELECT
            PRINTF('%06d', TRY_CAST(Rndrng_Prvdr_CCN AS INTEGER)) AS ccn,
            FIRST(Rndrng_Prvdr_Org_Name)                    AS hospital_name,
            FIRST(Rndrng_Prvdr_State_Abrvtn)                AS state,
            FIRST(Rndrng_Prvdr_City)                        AS city,
            ROUND(FIRST(Rndrng_Prvdr_RUCA), 1)              AS ruca,
            COUNT(DISTINCT DRG_Cd)                          AS drg_count,
            SUM(Tot_Dschrgs)                                AS total_discharges,
            ROUND(AVG(Avg_Mdcr_Pymt_Amt), 2)               AS avg_medicare_payment,
            ROUND(MEDIAN(Avg_Mdcr_Pymt_Amt), 2)            AS median_medicare_payment,
            ROUND(AVG(Avg_Tot_Pymt_Amt), 2)                AS avg_total_payment,
            ROUND(AVG(Avg_Submtd_Cvrd_Chrg), 2)            AS avg_submitted_charge,
            ROUND(AVG(Avg_Submtd_Cvrd_Chrg / NULLIF(Avg_Mdcr_Pymt_Amt, 0)), 2) AS avg_charge_ratio,
            ROUND(STDDEV(Avg_Mdcr_Pymt_Amt), 2)            AS payment_std,
            ROUND(STDDEV(Avg_Mdcr_Pymt_Amt)
                  / NULLIF(AVG(Avg_Mdcr_Pymt_Amt), 0) * 100, 1) AS payment_cv_pct
        FROM inpatient
        WHERE Rndrng_Prvdr_CCN IS NOT NULL
        GROUP BY Rndrng_Prvdr_CCN
        HAVING SUM(Tot_Dschrgs) >= 50
        """,
        table_name="inpatient_by_provider",
    ))
    step("profile_table: inpatient_by_provider", profile_table(PROJECT, "inpatient_by_provider"))

    # ── 4d. Provider-level outpatient aggregate (collapse provider×APC → provider) ──
    step("derive_table: outpatient_by_provider", derive_table(
        PROJECT,
        sql="""
        SELECT
            PRINTF('%06d', TRY_CAST(Rndrng_Prvdr_CCN AS INTEGER)) AS ccn,
            COUNT(DISTINCT APC_Cd)                          AS apc_count,
            SUM(Bene_Cnt)                                   AS op_bene_count,
            SUM(CAPC_Srvcs)                                 AS op_total_services,
            ROUND(AVG(Avg_Mdcr_Pymt_Amt), 2)               AS op_avg_medicare_payment,
            ROUND(AVG(Avg_Tot_Sbmtd_Chrgs), 2)             AS op_avg_submitted_charge,
            ROUND(AVG(Avg_Mdcr_Alowd_Amt), 2)              AS op_avg_allowed_amt,
            ROUND(AVG(Avg_Tot_Sbmtd_Chrgs / NULLIF(Avg_Mdcr_Pymt_Amt, 0)), 2)
                                                            AS op_avg_charge_ratio,
            SUM(Outlier_Srvcs)                              AS op_outlier_srvcs,
            ROUND(SUM(Outlier_Srvcs) * 100.0 / NULLIF(SUM(CAPC_Srvcs), 0), 2)
                                                            AS op_outlier_rate_pct,
            ROUND(AVG(Avg_Mdcr_Outlier_Amt), 2)            AS op_avg_outlier_amt
        FROM outpatient
        WHERE Rndrng_Prvdr_CCN IS NOT NULL
        GROUP BY Rndrng_Prvdr_CCN
        HAVING SUM(Bene_Cnt) >= 50
        """,
        table_name="outpatient_by_provider",
    ))
    step("profile_table: outpatient_by_provider", profile_table(PROJECT, "outpatient_by_provider"))

    # ── 4e. Enriched provider model — inpatient + outpatient + cost report + hospital info ──
    # Join key: ccn (PRINTF-padded 6-digit string in all four tables)
    step("derive_table: provider_model", derive_table(
        PROJECT,
        sql="""
        SELECT
            i.ccn,
            i.hospital_name,
            i.state,
            i.city,
            i.ruca,
            -- inpatient service breadth & volume
            i.drg_count,
            i.total_discharges,
            i.avg_medicare_payment          AS inp_avg_medicare_payment,
            i.median_medicare_payment       AS inp_median_medicare_payment,
            i.avg_submitted_charge          AS inp_avg_submitted_charge,
            i.avg_charge_ratio              AS inp_avg_charge_ratio,
            i.payment_cv_pct               AS inp_payment_cv_pct,
            -- outpatient service breadth & volume
            o.apc_count,
            o.op_bene_count,
            o.op_total_services,
            o.op_avg_medicare_payment,
            o.op_avg_submitted_charge,
            o.op_avg_charge_ratio,
            o.op_outlier_rate_pct,
            -- combined site-of-care mix: what share of patient volume is outpatient?
            ROUND(o.op_bene_count * 100.0
                  / NULLIF(i.total_discharges + o.op_bene_count, 0), 1)
                                            AS op_volume_share_pct,
            -- cost report: financial structure
            c.rural_urban,
            c.control_type,
            c.num_beds,
            c.fte_employees,
            c.intern_resident_fte,
            c.total_costs,
            c.cost_to_charge_ratio,
            c.dsh_pct,
            c.dsh_adjustment_amt,
            c.uncompensated_care_cost,
            c.bad_debt_expense,
            c.net_patient_revenue,
            c.ime_payment,
            -- cost per discharge (from cost report — true cost, not payment)
            ROUND(c.total_costs / NULLIF(c.medicare_discharges, 0), 2)
                                            AS cost_per_medicare_discharge,
            ROUND(c.uncompensated_care_cost / NULLIF(c.net_patient_revenue, 0) * 100, 2)
                                            AS uncomp_care_pct_revenue,
            ROUND(c.ime_payment / NULLIF(c.total_costs, 0) * 100, 2)
                                            AS ime_pct_costs,
            -- hospital characteristics
            h.hospital_type,
            h.ownership_type,
            h.emergency_services,
            h.cms_star_rating
        FROM inpatient_by_provider i
        LEFT JOIN outpatient_by_provider o USING (ccn)
        LEFT JOIN cost_report_slim c USING (ccn)
        LEFT JOIN hospital_info_slim h USING (ccn)
        WHERE i.ccn IS NOT NULL
        """,
        table_name="provider_model",
    ))
    step("profile_table: provider_model", profile_table(PROJECT, "provider_model"))

    # ── 5. Derived: charge-to-payment ratio (inpatient) ─────────────────────
    step("derive_table: inp_ratios", derive_table(
        PROJECT,
        sql=(
            "SELECT *, "
            "Avg_Submtd_Cvrd_Chrg / NULLIF(Avg_Mdcr_Pymt_Amt, 0) AS charge_to_pay_ratio, "
            "Avg_Mdcr_Pymt_Amt / NULLIF(Avg_Tot_Pymt_Amt, 0) AS medicare_share "
            "FROM inpatient"
        ),
        table_name="inp_ratios",
    ))
    step("profile_table: inp_ratios", profile_table(PROJECT, "inp_ratios"))

    # ── 6. Group summaries — regional and service-level ──────────────────────
    step("group_summary: inpatient by state", group_summary(
        PROJECT, "inpatient", "Rndrng_Prvdr_State_Abrvtn",
        ["Avg_Mdcr_Pymt_Amt"],
        ["mean", "std", "count"],
    ))
    step("group_summary: outpatient by state", group_summary(
        PROJECT, "outpatient", "Rndrng_Prvdr_State_Abrvtn",
        ["Avg_Mdcr_Pymt_Amt"],
        ["mean", "std", "count"],
    ))
    step("group_summary: inpatient by RUCA", group_summary(
        PROJECT, "inpatient", "Rndrng_Prvdr_RUCA",
        ["Avg_Mdcr_Pymt_Amt", "Avg_Submtd_Cvrd_Chrg"],
        ["mean", "count"],
    ))

    # ── 7. SQL analytics — Q1: Services with highest cost variation ──────────
    step("sql: top 20 DRGs by cost CV%", sql_query(
        PROJECT,
        """
        SELECT
            DRG_Cd,
            FIRST(DRG_Desc)                                                     AS drg_desc,
            COUNT(DISTINCT Rndrng_Prvdr_CCN)                                    AS providers,
            ROUND(AVG(Avg_Mdcr_Pymt_Amt), 2)                                   AS mean_payment,
            ROUND(STDDEV(Avg_Mdcr_Pymt_Amt), 2)                                AS std_payment,
            ROUND(STDDEV(Avg_Mdcr_Pymt_Amt) / NULLIF(AVG(Avg_Mdcr_Pymt_Amt), 0) * 100, 1) AS cv_pct,
            ROUND(MAX(Avg_Mdcr_Pymt_Amt) / NULLIF(MIN(Avg_Mdcr_Pymt_Amt), 0), 1) AS max_min_ratio
        FROM inpatient
        GROUP BY DRG_Cd
        HAVING COUNT(DISTINCT Rndrng_Prvdr_CCN) >= 20
        ORDER BY cv_pct DESC
        LIMIT 20
        """,
    ))

    step("sql: top 20 APCs by cost CV%", sql_query(
        PROJECT,
        """
        SELECT
            APC_Cd,
            FIRST(APC_Desc)                                                     AS apc_desc,
            COUNT(DISTINCT Rndrng_Prvdr_CCN)                                    AS providers,
            ROUND(AVG(Avg_Mdcr_Pymt_Amt), 2)                                   AS mean_payment,
            ROUND(STDDEV(Avg_Mdcr_Pymt_Amt), 2)                                AS std_payment,
            ROUND(STDDEV(Avg_Mdcr_Pymt_Amt) / NULLIF(AVG(Avg_Mdcr_Pymt_Amt), 0) * 100, 1) AS cv_pct
        FROM outpatient
        GROUP BY APC_Cd
        HAVING COUNT(DISTINCT Rndrng_Prvdr_CCN) >= 20
        ORDER BY cv_pct DESC
        LIMIT 20
        """,
    ))

    # ── 8. SQL analytics — Q2: Regional cost patterns ───────────────────────
    step("sql: state-level inpatient cost summary", sql_query(
        PROJECT,
        """
        SELECT
            Rndrng_Prvdr_State_Abrvtn                           AS state,
            ROUND(AVG(Avg_Mdcr_Pymt_Amt), 2)                   AS avg_medicare_payment,
            ROUND(MEDIAN(Avg_Mdcr_Pymt_Amt), 2)                AS median_medicare_payment,
            ROUND(AVG(Avg_Submtd_Cvrd_Chrg), 2)                AS avg_submitted_charge,
            COUNT(DISTINCT Rndrng_Prvdr_CCN)                    AS providers,
            SUM(Tot_Dschrgs)                                    AS total_discharges,
            ROUND(AVG(Avg_Submtd_Cvrd_Chrg / NULLIF(Avg_Mdcr_Pymt_Amt,0)), 1) AS avg_charge_ratio
        FROM inpatient
        WHERE LENGTH(Rndrng_Prvdr_State_Abrvtn) = 2
        GROUP BY Rndrng_Prvdr_State_Abrvtn
        ORDER BY avg_medicare_payment DESC
        """,
    ))

    step("sql: state-level outpatient cost summary", sql_query(
        PROJECT,
        """
        SELECT
            Rndrng_Prvdr_State_Abrvtn                           AS state,
            ROUND(AVG(Avg_Mdcr_Pymt_Amt), 2)                   AS avg_medicare_payment,
            ROUND(MEDIAN(Avg_Mdcr_Pymt_Amt), 2)                AS median_medicare_payment,
            COUNT(DISTINCT Rndrng_Prvdr_CCN)                    AS providers,
            SUM(Bene_Cnt)                                       AS total_beneficiaries
        FROM outpatient
        WHERE LENGTH(Rndrng_Prvdr_State_Abrvtn) = 2
        GROUP BY Rndrng_Prvdr_State_Abrvtn
        ORDER BY avg_medicare_payment DESC
        """,
    ))

    # ── 9. Outlier detection ─────────────────────────────────────────────────
    step("detect_outliers: inpatient Avg_Mdcr_Pymt_Amt", detect_outliers(
        PROJECT, "inpatient", "Avg_Mdcr_Pymt_Amt",
        method="iqr", params={"threshold": 3.0},
    ))
    step("detect_outliers: outpatient Avg_Mdcr_Pymt_Amt", detect_outliers(
        PROJECT, "outpatient", "Avg_Mdcr_Pymt_Amt",
        method="iqr", params={"threshold": 3.0},
    ))
    step("detect_outliers: charge-to-pay ratio", detect_outliers(
        PROJECT, "inp_ratios", "charge_to_pay_ratio",
        method="iqr", params={"threshold": 3.0},
    ))

    # ── 10. SQL: Outlier providers by DRG peer comparison ────────────────────
    step("sql: inpatient providers with z-score > 3 vs DRG peers", sql_query(
        PROJECT,
        """
        WITH drg_stats AS (
            SELECT
                DRG_Cd,
                AVG(Avg_Mdcr_Pymt_Amt)      AS drg_mean,
                STDDEV(Avg_Mdcr_Pymt_Amt)   AS drg_std
            FROM inpatient
            GROUP BY DRG_Cd
            HAVING COUNT(*) >= 5
        )
        SELECT
            i.Rndrng_Prvdr_CCN,
            i.Rndrng_Prvdr_Org_Name         AS hospital,
            i.Rndrng_Prvdr_City             AS city,
            i.Rndrng_Prvdr_State_Abrvtn     AS state,
            ROUND(i.Rndrng_Prvdr_RUCA, 0)   AS ruca,
            i.DRG_Cd,
            LEFT(i.DRG_Desc, 50)            AS drg_desc,
            ROUND(i.Avg_Mdcr_Pymt_Amt, 0)   AS payment,
            ROUND(d.drg_mean, 0)            AS drg_mean,
            ROUND((i.Avg_Mdcr_Pymt_Amt - d.drg_mean) / NULLIF(d.drg_std, 0), 2) AS z_score
        FROM inpatient i
        JOIN drg_stats d USING (DRG_Cd)
        WHERE ABS((i.Avg_Mdcr_Pymt_Amt - d.drg_mean) / NULLIF(d.drg_std, 0)) > 3
        ORDER BY ABS((i.Avg_Mdcr_Pymt_Amt - d.drg_mean) / NULLIF(d.drg_std, 0)) DESC
        LIMIT 50
        """,
    ))

    # ── 11. Correlation analysis ──────────────────────────────────────────────
    step("analyze_correlations: inpatient", analyze_correlations(
        PROJECT, "inpatient",
        ["Tot_Dschrgs", "Avg_Submtd_Cvrd_Chrg", "Avg_Tot_Pymt_Amt", "Avg_Mdcr_Pymt_Amt"],
    ))
    step("analyze_correlations: outpatient", analyze_correlations(
        PROJECT, "outpatient",
        ["Bene_Cnt", "CAPC_Srvcs", "Avg_Tot_Sbmtd_Chrgs",
         "Avg_Mdcr_Alowd_Amt", "Avg_Mdcr_Pymt_Amt"],
    ))

    # ── 12. Distribution analysis ─────────────────────────────────────────────
    step("analyze_distribution: inpatient Avg_Mdcr_Pymt_Amt", analyze_distribution(
        PROJECT, "inpatient", "Avg_Mdcr_Pymt_Amt",
    ))
    step("analyze_distribution: charge_to_pay_ratio", analyze_distribution(
        PROJECT, "inp_ratios", "charge_to_pay_ratio",
    ))

    # ── 13. BONUS: Predictors of cost (gradient-boosted regression) ───────────
    step("run_model: random forest on inpatient payment", run_model(
        PROJECT,
        table="inpatient",
        target="Avg_Mdcr_Pymt_Amt",
        method="random_forest",
        features=["Tot_Dschrgs", "Avg_Submtd_Cvrd_Chrg", "Rndrng_Prvdr_RUCA"],
    ))

    # ── 14. Derived aggregates for charts ─────────────────────────────────────
    step("derive_table: inpatient_by_state", derive_table(
        PROJECT,
        sql=(
            "SELECT Rndrng_Prvdr_State_Abrvtn AS state, "
            "ROUND(AVG(Avg_Mdcr_Pymt_Amt), 2) AS avg_medicare_payment, "
            "COUNT(DISTINCT Rndrng_Prvdr_CCN) AS providers "
            "FROM inpatient WHERE LENGTH(Rndrng_Prvdr_State_Abrvtn) = 2 "
            "GROUP BY Rndrng_Prvdr_State_Abrvtn"
        ),
        table_name="inpatient_by_state",
    ))
    step("profile_table: inpatient_by_state", profile_table(PROJECT, "inpatient_by_state"))

    step("derive_table: top_drg_by_payment", derive_table(
        PROJECT,
        sql=(
            "SELECT DRG_Cd, FIRST(DRG_Desc) AS drg_desc, "
            "ROUND(AVG(Avg_Mdcr_Pymt_Amt), 2) AS avg_payment, "
            "COUNT(DISTINCT Rndrng_Prvdr_CCN) AS providers "
            "FROM inpatient GROUP BY DRG_Cd "
            "HAVING COUNT(DISTINCT Rndrng_Prvdr_CCN) >= 50 "
            "ORDER BY avg_payment DESC LIMIT 25"
        ),
        table_name="top_drg_by_payment",
    ))
    step("profile_table: top_drg_by_payment", profile_table(PROJECT, "top_drg_by_payment"))

    # ── 15. Visualizations ────────────────────────────────────────────────────
    step("create_chart: choropleth — avg inpatient payment by state", create_chart(
        PROJECT,
        chart_type="choropleth_map",
        table="inpatient_by_state",
        columns=["state", "avg_medicare_payment"],
        params={"locations_format": "usa-states", "scope": "usa"},
    ))

    step("create_chart: horizontal_bar — top DRGs by avg payment", create_chart(
        PROJECT,
        chart_type="horizontal_bar",
        table="top_drg_by_payment",
        columns=["DRG_Cd", "avg_payment"],
    ))

    step("create_chart: scatter — submitted charge vs Medicare payment", create_chart(
        PROJECT,
        chart_type="scatter",
        table="inpatient",
        columns=["Avg_Submtd_Cvrd_Chrg", "Avg_Mdcr_Pymt_Amt"],
    ))

    step("create_chart: scatter — RUCA vs Medicare payment", create_chart(
        PROJECT,
        chart_type="scatter",
        table="inpatient",
        columns=["Rndrng_Prvdr_RUCA", "Avg_Mdcr_Pymt_Amt"],
    ))

    step("create_chart: histogram — inpatient payment distribution", create_chart(
        PROJECT,
        chart_type="histogram",
        table="inpatient",
        columns=["Avg_Mdcr_Pymt_Amt"],
    ))

    step("create_chart: feature importance — cost predictors", create_chart(
        PROJECT,
        chart_type="feature_importance_bar",
        table="inpatient",
        columns=[],
        finding_id="f001",
    ))

    # ── 15. Log hypotheses for tracking ──────────────────────────────────────
    hypothesis_add(
        PROJECT,
        "High cost variation exists across DRGs/APCs; CV > 100% expected for complex procedures",
        "analytical",
    )
    hypothesis_add(
        PROJECT,
        "Rural providers (RUCA > 4) show lower average Medicare payments than urban providers",
        "analytical",
    )
    hypothesis_add(
        PROJECT,
        "Submitted charge-to-payment ratio > 10x identifies hospitals with aggressive chargemasters",
        "analytical",
    )
    hypothesis_add(
        PROJECT,
        "Submitted charges are the dominant predictor of Medicare payment amounts",
        "modeling",
    )

    # ── Final status ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  PIPELINE COMPLETE — project:", PROJECT)
    status = project_status(PROJECT)
    print(json.dumps(status, indent=2, default=str))
    print("=" * 60)


if __name__ == "__main__":
    main()
