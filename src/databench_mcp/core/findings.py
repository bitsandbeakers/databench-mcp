"""Per-project findings tracker backed by findings.yaml."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import yaml

from databench_mcp.workspace import project_path, read_manifest


def _findings_path(project: str):
    return project_path(project) / "findings.yaml"


def _read_findings(project: str) -> list[dict]:
    path = _findings_path(project)
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text()) or []


def _write_findings(project: str, findings: list[dict]) -> None:
    path = _findings_path(project)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.dump(findings, default_flow_style=False, allow_unicode=True))
    os.replace(tmp, path)


def _next_id(findings: list[dict]) -> str:
    nums = [
        int(f["id"][1:])
        for f in findings
        if f.get("id", "").startswith("f") and f["id"][1:].isdigit()
    ]
    return f"f{(max(nums) + 1):03d}" if nums else "f001"


def add_finding(project: str, data: dict[str, Any]) -> dict[str, Any]:
    """Assign ID, timestamp, save to findings.yaml, return complete entry."""
    read_manifest(project)
    findings = _read_findings(project)
    entry: dict[str, Any] = {
        "id": _next_id(findings),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    findings.append(entry)
    _write_findings(project, findings)
    return entry


def get_finding(project: str, finding_id: str) -> dict[str, Any]:
    """Return a single finding by ID. Raises ValueError if not found."""
    read_manifest(project)
    for f in _read_findings(project):
        if f.get("id") == finding_id:
            return f
    raise ValueError(f"Finding '{finding_id}' not found in project '{project}'")


def list_findings(
    project: str,
    method: str | None = None,
) -> dict[str, Any]:
    """Return findings, optionally filtered by method."""
    read_manifest(project)
    findings = _read_findings(project)
    if method is not None:
        findings = [f for f in findings if f.get("method") == method]
    return {"project": project, "count": len(findings), "findings": findings}


def compare_findings(
    project: str,
    finding_ids: list[str],
    metrics: list[str] | None = None,
    rank_by: str | None = None,
) -> dict[str, Any]:
    """Compare metrics across multiple findings and rank by a chosen metric.

    Parameters
    ----------
    finding_ids : list of finding IDs to compare (e.g. ['f001', 'f003'])
    metrics     : specific metric keys to extract (e.g. ['r2', 'rmse']); if
                  omitted all stored metrics are returned
    rank_by     : metric to sort by descending; defaults to first item of metrics
    """
    read_manifest(project)
    all_findings = _read_findings(project)
    by_id = {f["id"]: f for f in all_findings}

    missing = [fid for fid in finding_ids if fid not in by_id]
    if missing:
        raise ValueError(f"Finding(s) not found in project '{project}': {missing}")

    rows = []
    for fid in finding_ids:
        f = by_id[fid]
        stored_metrics = f.get("metrics", {})
        row: dict[str, Any] = {
            "id": fid,
            "method": f.get("method"),
            "table": f.get("table"),
            "target": f.get("target"),
            "created_at": f.get("created_at"),
            "metrics": {m: stored_metrics.get(m) for m in metrics} if metrics else dict(stored_metrics),
        }
        rows.append(row)

    effective_rank = rank_by or (metrics[0] if metrics else None)
    if effective_rank:
        rows.sort(
            key=lambda r: (r["metrics"].get(effective_rank) is not None,
                           r["metrics"].get(effective_rank) or 0),
            reverse=True,
        )

    return {
        "project": project,
        "finding_count": len(rows),
        "ranked_by": effective_rank,
        "metrics_compared": metrics,
        "findings": rows,
    }
