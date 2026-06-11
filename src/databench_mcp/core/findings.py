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
