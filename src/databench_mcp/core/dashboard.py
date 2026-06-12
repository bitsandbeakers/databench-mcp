"""Dashboard generation — builds a standalone Dash app from chart sidecars."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from databench_mcp.db import get_connection
from databench_mcp.workspace import project_path


def _dashboards_dir(project: str) -> Path:
    d = project_path(project) / "dashboards"
    d.mkdir(exist_ok=True)
    return d


def _read_sidecars(project: str) -> tuple[dict[str, list[dict]], list[dict]]:
    """Return (renderable_by_table, skipped_list).

    Skipped: finding-dependent charts (require .npy / _communities.json artifacts).
    """
    charts_dir = project_path(project) / "charts"
    if not charts_dir.exists():
        return {}, []

    _RENDERABLE = {"histogram", "boxplot", "scatter", "scatter_matrix",
                   "correlation_heatmap", "network_graph"}
    _SKIP_ALWAYS = {"cluster_scatter", "shap_beeswarm", "feature_importance_bar",
                    "shap_waterfall", "distribution_overlay", "partial_dependence"}

    renderable: dict[str, list[dict]] = {}
    skipped: list[dict] = []

    for f in sorted(charts_dir.glob("*_params.json")):
        sidecar = json.loads(f.read_text(encoding="utf-8"))
        ct = sidecar.get("chart_type", "")

        if ct in _SKIP_ALWAYS or ct not in _RENDERABLE:
            skipped.append(sidecar)
            continue
        if ct == "network_graph" and sidecar.get("finding_id"):
            skipped.append(sidecar)
            continue

        renderable.setdefault(sidecar.get("table", ""), []).append(sidecar)

    return renderable, skipped


def _export_tables(project: str, tables: list[str], data_dir: Path) -> None:
    """Export each table from DuckDB to parquet using DuckDB's native COPY."""
    with get_connection(project) as conn:
        for table in tables:
            path = data_dir / f"{table}.parquet"
            conn.execute(f'COPY "{table}" TO {str(path)!r} (FORMAT PARQUET)')


def _load_findings(project: str) -> list[dict]:
    path = project_path(project) / "findings.yaml"
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def build_dashboard(project: str) -> dict[str, Any]:
    """Generate a standalone Dash app from the project's chart artifacts and findings."""
    sidecars_by_table, skipped = _read_sidecars(project)

    if not sidecars_by_table:
        raise ValueError("no charts to embed — run create_chart first")

    raise NotImplementedError("generation not yet implemented")
