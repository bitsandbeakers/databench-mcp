"""EDA tools: read-only SQL query and manifest-derived dataset summary."""
from __future__ import annotations

import datetime
import decimal
import math
import re
import warnings
from typing import Any

from databench_mcp.db import get_connection
from databench_mcp.workspace import read_manifest, write_manifest

_SELECT_PATTERN = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_DEFAULT_LIMIT = 500


def _to_json_safe(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, decimal.Decimal):
        return float(val)
    if isinstance(val, (datetime.datetime, datetime.date, datetime.time)):
        return val.isoformat()
    return str(val)


_AGG_FN_MAP = {
    "mean": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
    "std": "STDDEV",
}


def group_summary(
    project: str,
    table: str,
    group_col: str,
    agg_cols: list[str],
    agg_fns: list[str] | None = None,
) -> dict[str, Any]:
    """Return grouped aggregates for the given table."""
    from databench_mcp.workspace import assert_profiled

    if agg_fns is None:
        agg_fns = list(_AGG_FN_MAP.keys())
    if not agg_cols:
        raise ValueError("agg_cols must not be empty")
    unknown = [f for f in agg_fns if f not in _AGG_FN_MAP]
    if unknown:
        raise ValueError(f"unknown agg_fn {unknown[0]!r}")
    assert_profiled(project, table)

    agg_parts = ", ".join(
        f'{_AGG_FN_MAP[fn]}("{col}") AS "{col}_{fn}"'
        for col in agg_cols
        for fn in agg_fns
    )
    sql = (
        f'SELECT "{group_col}", {agg_parts} '
        f'FROM "{table}" '
        f'GROUP BY "{group_col}" '
        f'ORDER BY "{group_col}"'
    )
    with get_connection(project) as conn:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description]
        raw_rows = cursor.fetchall()

    rows = [
        {col: _to_json_safe(val) for col, val in zip(columns, row)}
        for row in raw_rows
    ]
    return {
        "table": table,
        "group_col": group_col,
        "agg_cols": agg_cols,
        "agg_fns": agg_fns,
        "rows": rows,
        "row_count": len(rows),
    }


_CLEAN_STRATEGIES = {
    "drop_rows", "drop_cols",
    "fill_mean", "fill_median", "fill_mode", "fill_constant",
    "fill_forward", "fill_backward",
}
_NUMERIC_TYPES = (
    "DOUBLE", "FLOAT", "INT", "REAL", "DECIMAL",
    "NUMERIC", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT",
)


def _fill_select(table: str, affected: list[str], select_parts: list[str]) -> str:
    """SELECT * EXCLUDE affected cols, then override with select_parts expressions."""
    if not affected:
        return f'SELECT * FROM "{table}"'
    exclude = ", ".join(f'"{c}"' for c in affected)
    overrides = ", ".join(select_parts)
    return f'SELECT * EXCLUDE ({exclude}), {overrides} FROM "{table}"'


def _fill_select_rn(table: str, affected: list[str], select_parts: list[str]) -> str:
    """Same as _fill_select but via a ROW_NUMBER() CTE for forward/backward fill."""
    if not affected:
        return f'SELECT * FROM "{table}"'
    all_excludes = [f'"{c}"' for c in affected] + ['"_rn"']
    exclude = ", ".join(all_excludes)
    overrides = ", ".join(select_parts)
    return (
        f'WITH _base AS (SELECT *, ROW_NUMBER() OVER () AS _rn FROM "{table}") '
        f'SELECT * EXCLUDE ({exclude}), {overrides} FROM _base'
    )


def clean_table(
    project: str,
    table: str,
    strategy: str,
    new_table_name: str,
    columns: list[str] | None = None,
    fill_value: float | str | None = None,
) -> dict[str, Any]:
    """Materialise a cleaned version of *table* as *new_table_name*."""
    from databench_mcp.workspace import assert_profiled

    if strategy not in _CLEAN_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    if strategy == "fill_constant" and fill_value is None:
        raise ValueError("fill_value required for fill_constant")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        raise ValueError(f"table must be a simple identifier (got {table!r})")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new_table_name):
        raise ValueError(
            f"new_table_name must be a simple identifier (got {new_table_name!r})"
        )

    assert_profiled(project, table)
    manifest = read_manifest(project)
    profile = manifest["datasets"][table].get("profile", {})

    if columns is not None:
        affected = list(columns)
    elif strategy in ("drop_rows", "drop_cols"):
        affected = [c for c, info in profile.items() if info.get("null_pct", 0) > 0]
    else:
        affected = [
            c for c, info in profile.items()
            if info.get("null_pct", 0) > 0
            and any(t in info.get("type", "").upper() for t in _NUMERIC_TYPES)
        ]

    if strategy == "drop_rows":
        if affected:
            where = " AND ".join(f'"{c}" IS NOT NULL' for c in affected)
            body = f'SELECT * FROM "{table}" WHERE {where}'
        else:
            body = f'SELECT * FROM "{table}"'

    elif strategy == "drop_cols":
        with get_connection(project) as conn:
            all_cols = [
                d[0] for d in conn.execute(
                    f'SELECT * FROM "{table}" LIMIT 0'
                ).description
            ]
        keep = [c for c in all_cols if c not in affected]
        if not keep:
            raise ValueError("strategy would drop all columns")
        body = (
            "SELECT " + ", ".join(f'"{c}"' for c in keep)
            + f' FROM "{table}"'
        )

    elif strategy == "fill_mean":
        parts = [
            f'COALESCE("{c}", AVG("{c}") OVER ()) AS "{c}"' for c in affected
        ]
        body = _fill_select(table, affected, parts)

    elif strategy == "fill_median":
        parts = [
            f'COALESCE("{c}", (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP '
            f'(ORDER BY "{c}") FROM "{table}")) AS "{c}"'
            for c in affected
        ]
        body = _fill_select(table, affected, parts)

    elif strategy == "fill_mode":
        parts = [
            f'COALESCE("{c}", (SELECT mode("{c}") FROM "{table}")) AS "{c}"'
            for c in affected
        ]
        body = _fill_select(table, affected, parts)

    elif strategy == "fill_constant":
        if isinstance(fill_value, float) and not math.isfinite(fill_value):
            raise ValueError(f"fill_value must be a finite number (got {fill_value!r})")
        if isinstance(fill_value, str):
            literal = "'" + fill_value.replace("'", "''") + "'"
        else:
            literal = str(fill_value)
        parts = [f'COALESCE("{c}", {literal}) AS "{c}"' for c in affected]
        body = _fill_select(table, affected, parts)

    elif strategy == "fill_forward":
        parts = [
            f'LAST_VALUE("{c}" IGNORE NULLS) OVER '
            f'(ORDER BY _rn ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS "{c}"'
            for c in affected
        ]
        body = _fill_select_rn(table, affected, parts)

    else:  # fill_backward
        parts = [
            f'FIRST_VALUE("{c}" IGNORE NULLS) OVER '
            f'(ORDER BY _rn ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS "{c}"'
            for c in affected
        ]
        body = _fill_select_rn(table, affected, parts)

    with get_connection(project) as conn:
        conn.execute(f'CREATE OR REPLACE TABLE "{new_table_name}" AS ({body})')
        row_count = conn.execute(
            f'SELECT COUNT(*) FROM "{new_table_name}"'
        ).fetchone()[0]
        col_count = len(
            conn.execute(f'SELECT * FROM "{new_table_name}" LIMIT 0').description
        )

    manifest["datasets"][new_table_name] = {
        "source": "clean_table",
        "source_table": table,
        "strategy": strategy,
        "profiled": False,
        "row_count": int(row_count),
        "col_count": int(col_count),
    }
    write_manifest(project, manifest)

    return {
        "table": new_table_name,
        "source_table": table,
        "strategy": strategy,
        "rows": int(row_count),
        "columns": int(col_count),
        "affected_cols": affected,
    }


def add_lag(
    project: str,
    table: str,
    col: str,
    lags: list[int],
    new_table_name: str,
    time_col: str | None = None,
) -> dict:
    """Materialise a new table with LAG columns appended for each lag value in *lags*."""
    from databench_mcp.workspace import assert_profiled

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        raise ValueError(f"table must be a simple identifier (got {table!r})")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new_table_name):
        raise ValueError(
            f"new_table_name must be a simple identifier (got {new_table_name!r})"
        )
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col):
        raise ValueError(f"col must be a simple identifier (got {col!r})")
    if time_col is not None and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", time_col):
        raise ValueError(f"time_col must be a simple identifier (got {time_col!r})")
    if not lags or not all(isinstance(n, int) and n > 0 for n in lags):
        raise ValueError("lags must be non-empty positive integers")

    assert_profiled(project, table)

    if time_col is None:
        warnings.warn(
            "add_lag: time_col not specified — lag values will be in arbitrary row order",
            UserWarning,
            stacklevel=2,
        )
    over_clause = f'ORDER BY "{time_col}"' if time_col is not None else ""
    lag_parts = ", ".join(
        f'LAG("{col}", {n}) OVER ({over_clause}) AS "{col}_lag_{n}"'
        for n in lags
    )
    body = f'SELECT *, {lag_parts} FROM "{table}"'
    sql = f'CREATE OR REPLACE TABLE "{new_table_name}" AS ({body})'

    with get_connection(project) as conn:
        conn.execute(sql)
        row_count = conn.execute(
            f'SELECT COUNT(*) FROM "{new_table_name}"'
        ).fetchone()[0]

    manifest = read_manifest(project)
    manifest["datasets"][new_table_name] = {
        "source": "add_lag",
        "source_table": table,
        "col": col,
        "lags": lags,
        "profiled": False,
        "row_count": int(row_count),
    }
    write_manifest(project, manifest)

    return {
        "table": new_table_name,
        "source_table": table,
        "new_cols": [f"{col}_lag_{n}" for n in lags],
        "rows": int(row_count),
    }


_ROLLING_AGG_MAP = {
    "mean": "AVG",
    "sum": "SUM",
    "min": "MIN",
    "max": "MAX",
    "std": "STDDEV",
}


def add_rolling(
    project: str,
    table: str,
    col: str,
    window: int,
    agg_fn: str,
    new_table_name: str,
    time_col: str | None = None,
) -> dict:
    """Materialise a new table with a rolling-window aggregate column appended."""
    from databench_mcp.workspace import assert_profiled

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        raise ValueError(f"table must be a simple identifier (got {table!r})")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new_table_name):
        raise ValueError(
            f"new_table_name must be a simple identifier (got {new_table_name!r})"
        )
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col):
        raise ValueError(f"col must be a simple identifier (got {col!r})")
    if time_col is not None and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", time_col):
        raise ValueError(f"time_col must be a simple identifier (got {time_col!r})")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
    if agg_fn not in _ROLLING_AGG_MAP:
        raise ValueError(f"unknown agg_fn {agg_fn!r}")

    assert_profiled(project, table)

    if time_col is None:
        warnings.warn(
            "add_rolling: time_col not specified — rolling values will be in arbitrary row order",
            UserWarning,
            stacklevel=2,
        )

    sql_fn = _ROLLING_AGG_MAP[agg_fn]
    new_col = f"{col}_rolling_{window}_{agg_fn}"
    order_clause = f'ORDER BY "{time_col}" ' if time_col is not None else ""
    over_clause = f"({order_clause}ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW)"
    body = (
        f'SELECT *, {sql_fn}("{col}") OVER {over_clause} AS "{new_col}" '
        f'FROM "{table}"'
    )
    sql = f'CREATE OR REPLACE TABLE "{new_table_name}" AS ({body})'

    with get_connection(project) as conn:
        conn.execute(sql)
        row_count = conn.execute(
            f'SELECT COUNT(*) FROM "{new_table_name}"'
        ).fetchone()[0]

    manifest = read_manifest(project)
    manifest["datasets"][new_table_name] = {
        "source": "add_rolling",
        "source_table": table,
        "col": col,
        "window": window,
        "agg_fn": agg_fn,
        "new_col": new_col,
        "profiled": False,
        "row_count": int(row_count),
    }
    write_manifest(project, manifest)

    return {
        "table": new_table_name,
        "source_table": table,
        "new_col": new_col,
        "rows": int(row_count),
    }



def enrich_table(
    project: str,
    left_table: str,
    right_table: str,
    on: "str | list[str]",
    new_table_name: str,
    how: str = "inner",
) -> dict:
    """Join two profiled tables and register the result."""
    # Build HOW clause (validate by dict lookup)
    how_clause_map = {
        "inner": "INNER",
        "left": "LEFT",
        "right": "RIGHT",
        "full": "FULL OUTER",
    }
    if how not in how_clause_map:
        raise ValueError(f"unknown join type {how!r}")

    # Validate table names and new_table_name
    ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for name in (new_table_name, left_table, right_table):
        if not ident_re.match(name):
            raise ValueError(f"invalid identifier: {name!r}")

    # Normalize on to list
    if isinstance(on, str):
        on_list = [on]
    else:
        on_list = list(on)

    if not on_list:
        raise ValueError("on must specify at least one join column")

    for col in on_list:
        if not ident_re.match(col):
            raise ValueError(f"invalid column identifier: {col!r}")

    # Check both tables are profiled
    from databench_mcp.workspace import assert_profiled
    assert_profiled(project, left_table)
    assert_profiled(project, right_table)

    how_clause = how_clause_map[how]

    # Build USING clause
    using_cols = ", ".join(on_list)
    using_clause = f"USING ({using_cols})"

    sql = (
        f'CREATE OR REPLACE TABLE "{new_table_name}" AS ('
        f'SELECT * FROM "{left_table}" '
        f'{how_clause} JOIN "{right_table}" '
        f'{using_clause}'
        f')' 
    )

    with get_connection(project) as conn:
        conn.execute(sql)
        row_count = conn.execute(f'SELECT COUNT(*) FROM "{new_table_name}"').fetchone()[0]
        col_count = len(conn.execute(f'SELECT * FROM "{new_table_name}" LIMIT 0').description)

    manifest = read_manifest(project)
    manifest["datasets"][new_table_name] = {
        "source": "enrich_table",
        "left_table": left_table,
        "right_table": right_table,
        "on": on_list,
        "how": how,
        "profiled": False,
        "row_count": int(row_count),
        "col_count": int(col_count),
    }
    write_manifest(project, manifest)

    return {
        "table": new_table_name,
        "left_table": left_table,
        "right_table": right_table,
        "how": how,
        "on": on_list,
        "rows": int(row_count),
        "columns": int(col_count),
    }


_NUMERIC_TYPE_TOKENS = ("DOUBLE", "FLOAT", "INT", "REAL", "DECIMAL", "NUMERIC", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT")
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_SEVERITY_WEIGHT = {"high": 3, "medium": 1, "low": 0.3}


def data_quality_report(project: str, table: str) -> dict[str, Any]:
    """Analyse a profiled table and return actionable data quality issues.

    Checks each column for: extreme/high/moderate nulls, constant or near-constant
    values, likely ID columns masquerading as features, and high coefficient-of-
    variation on numeric columns. Returns a quality score (0–1) and a sorted issue
    list (high → medium → low severity).
    """
    from databench_mcp.workspace import assert_profiled

    assert_profiled(project, table)
    manifest = read_manifest(project)
    ds = manifest["datasets"][table]
    profile = ds.get("profile", {})
    row_count = ds.get("row_count", 0) or 0

    issues: list[dict[str, Any]] = []

    for col, info in profile.items():
        null_pct = info.get("null_pct") or 0.0
        approx_unique = info.get("approx_unique") or 0
        col_type = (info.get("type") or "").upper()
        mean = info.get("mean")
        std = info.get("std")
        is_numeric = any(t in col_type for t in _NUMERIC_TYPE_TOKENS)

        # Null rate tiers
        if null_pct >= 90:
            issues.append({
                "column": col, "severity": "high", "issue": "extreme_nulls",
                "detail": f"{null_pct:.1f}% null — column is nearly empty",
            })
        elif null_pct >= 50:
            issues.append({
                "column": col, "severity": "medium", "issue": "high_nulls",
                "detail": f"{null_pct:.1f}% null — over half missing",
            })
        elif null_pct >= 20:
            issues.append({
                "column": col, "severity": "low", "issue": "moderate_nulls",
                "detail": f"{null_pct:.1f}% null",
            })

        # Constant / near-constant
        if approx_unique <= 1:
            issues.append({
                "column": col, "severity": "high", "issue": "constant_column",
                "detail": f"{approx_unique} distinct value(s) — carries no information",
            })
        elif row_count > 0 and approx_unique <= 4 and (approx_unique / row_count) < 0.01:
            issues.append({
                "column": col, "severity": "medium", "issue": "near_constant",
                "detail": f"{approx_unique} distinct values across {row_count:,} rows",
            })

        # Likely ID column (high-cardinality non-numeric)
        if not is_numeric and row_count > 0 and (approx_unique / row_count) > 0.95:
            issues.append({
                "column": col, "severity": "medium", "issue": "likely_id_column",
                "detail": f"{approx_unique:,}/{row_count:,} distinct — likely an ID, not a feature",
            })

        # High coefficient of variation (numeric)
        if is_numeric and std is not None and mean is not None and mean != 0:
            cv = abs(std / mean)
            if cv > 5:
                issues.append({
                    "column": col, "severity": "low", "issue": "high_variation",
                    "detail": f"CV={cv:.1f} — potential outlier concentration or mixed scales",
                })

    issues.sort(key=lambda x: _SEVERITY_ORDER[x["severity"]])

    # Quality score: 1.0 = no issues, decays with weighted penalty
    penalty = sum(_SEVERITY_WEIGHT[i["severity"]] for i in issues)
    max_penalty = len(profile) * _SEVERITY_WEIGHT["high"] if profile else 1
    quality_score = round(max(0.0, 1.0 - penalty / max_penalty), 3)

    counts = {s: sum(1 for i in issues if i["severity"] == s) for s in _SEVERITY_ORDER}
    if not issues:
        summary = f"No issues found across {len(profile)} columns. Quality score: 1.0."
    else:
        parts = [f"{counts[s]} {s}" for s in ("high", "medium", "low") if counts[s]]
        summary = f"{len(issues)} issue(s): {', '.join(parts)}. Quality score: {quality_score}."

    return {
        "project": project,
        "table": table,
        "total_columns": len(profile),
        "row_count": row_count,
        "issue_count": len(issues),
        "quality_score": quality_score,
        "issues": issues,
        "summary": summary,
    }


def sql_query(project: str, sql: str, limit: int = _DEFAULT_LIMIT) -> dict[str, Any]:
    """Execute a read-only SELECT/WITH query and return up to `limit` rows."""
    stripped = sql.strip().rstrip(";")
    if not _SELECT_PATTERN.match(stripped):
        raise ValueError("Only SELECT or WITH queries are permitted")
    if ";" in stripped:
        raise ValueError("Multi-statement queries are not permitted")

    with get_connection(project) as conn:
        cursor = conn.execute(stripped)
        columns = [d[0] for d in cursor.description]
        schema = {d[0]: str(d[1]) for d in cursor.description}
        raw_rows = cursor.fetchmany(limit + 1)

    truncated = len(raw_rows) > limit
    rows = [
        {col: _to_json_safe(val) for col, val in zip(columns, row)}
        for row in raw_rows[:limit]
    ]

    return {
        "sql": stripped,
        "columns": columns,
        "schema": schema,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "limit": limit,
    }


def eda_summary(project: str) -> dict[str, Any]:
    """Return a dataset summary derived from the project manifest. No DB query."""
    manifest = read_manifest(project)
    datasets = manifest.get("datasets", {})

    result_datasets = []
    for name, ds in datasets.items():
        entry: dict[str, Any] = {
            "name": name,
            "rows": ds.get("row_count"),
            "cols": ds.get("col_count"),
            "profiled": ds.get("profiled", False),
            "ingested_at": ds.get("ingested_at"),
            "source": ds.get("source"),
        }
        if ds.get("profiled") and ds.get("profile"):
            entry["columns"] = list(ds["profile"].keys())
        result_datasets.append(entry)

    return {
        "project": project,
        "dataset_count": len(result_datasets),
        "datasets": result_datasets,
    }


def derive_table(project: str, sql: str, table_name: str) -> dict[str, Any]:
    """Materialise a SQL SELECT as a new DuckDB table; register in manifest as profiled=False."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
        raise ValueError(f"table_name must be a simple identifier (got {table_name!r})")
    stripped = sql.strip().rstrip(";")
    if not _SELECT_PATTERN.match(stripped):
        raise ValueError("Only SELECT or WITH queries are permitted")
    if ";" in stripped:
        raise ValueError("Multi-statement queries are not permitted")

    with get_connection(project) as conn:
        conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS ({stripped})')
        row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        col_count = len(conn.execute(f'SELECT * FROM "{table_name}" LIMIT 0').description)

    manifest = read_manifest(project)
    manifest["datasets"][table_name] = {
        "source": "derived",
        "sql": stripped,
        "profiled": False,
        "row_count": int(row_count),
        "col_count": int(col_count),
    }
    write_manifest(project, manifest)
    return {"table": table_name, "rows": int(row_count), "columns": int(col_count)}
