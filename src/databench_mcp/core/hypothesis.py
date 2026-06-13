"""Per-project hypothesis tracker backed by hypotheses.yaml."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import yaml

from databench_mcp.workspace import project_path, read_manifest

_VALID_STATUSES = ("proposed", "prioritized", "tested", "supported", "refuted", "inconclusive")


def _hypotheses_path(project: str):
    return project_path(project) / "hypotheses.yaml"


def _read_hypotheses(project: str) -> list[dict]:
    path = _hypotheses_path(project)
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text()) or []


def _write_hypotheses(project: str, hypotheses: list[dict]) -> None:
    path = _hypotheses_path(project)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.dump(hypotheses, default_flow_style=False, allow_unicode=True))
    os.replace(tmp, path)


def _next_id(hypotheses: list[dict]) -> str:
    nums = [
        int(h["id"][1:])
        for h in hypotheses
        if h.get("id", "").startswith("h") and h["id"][1:].isdigit()
    ]
    return f"h{(max(nums) + 1):03d}" if nums else "h001"


def add_hypothesis(
    project: str,
    statement: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Add a new hypothesis with status 'proposed'. Returns the new entry."""
    read_manifest(project)  # asserts project exists
    hypotheses = _read_hypotheses(project)
    entry: dict[str, Any] = {
        "id": _next_id(hypotheses),
        "statement": statement,
        "status": "proposed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tags": tags or [],
        "notes": [],
    }
    hypotheses.append(entry)
    _write_hypotheses(project, hypotheses)
    return entry


def list_hypotheses(
    project: str,
    status: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """Return hypotheses, optionally filtered by status and/or tag."""
    read_manifest(project)
    if status is not None and status not in _VALID_STATUSES:
        raise ValueError(f"Invalid status {status!r}; must be one of {_VALID_STATUSES}")
    hypotheses = _read_hypotheses(project)
    if status is not None:
        hypotheses = [h for h in hypotheses if h.get("status") == status]
    if tag is not None:
        hypotheses = [h for h in hypotheses if tag in (h.get("tags") or [])]
    return {"project": project, "count": len(hypotheses), "hypotheses": hypotheses}


def update_hypothesis(
    project: str,
    hypothesis_id: str,
    status: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Update status and/or append a note. Returns the updated entry."""
    read_manifest(project)
    if status is not None and status not in _VALID_STATUSES:
        raise ValueError(f"Invalid status {status!r}; must be one of {_VALID_STATUSES}")
    hypotheses = _read_hypotheses(project)
    for h in hypotheses:
        if h.get("id") == hypothesis_id:
            if status is not None:
                h["status"] = status
                h["updated_at"] = datetime.now(timezone.utc).isoformat()
            if note is not None:
                notes = h.get("notes") or []
                notes.append({"text": note, "added_at": datetime.now(timezone.utc).isoformat()})
                h["notes"] = notes
            _write_hypotheses(project, hypotheses)
            return h
    raise ValueError(f"Hypothesis '{hypothesis_id}' not found in project '{project}'")


def record_evidence(
    project: str,
    hypothesis_id: str,
    tool_name: str,
    result_summary: str,
    status_update: str | None = None,
) -> dict[str, Any]:
    """Append a structured evidence note to a hypothesis, optionally updating status.

    Designed to be called immediately after a tool call (run_model, eda_summary,
    sql_query, etc.) to keep the hypothesis tracker current without manual tracking.

    Parameters
    ----------
    hypothesis_id  : e.g. 'h005'
    tool_name      : name of the tool that produced the evidence (e.g. 'run_model')
    result_summary : 1-3 sentence summary of what the tool result showed
    status_update  : optional new status (supported / refuted / inconclusive / tested)
    """
    read_manifest(project)
    if status_update is not None and status_update not in _VALID_STATUSES:
        raise ValueError(f"Invalid status {status_update!r}; must be one of {_VALID_STATUSES}")
    hypotheses = _read_hypotheses(project)
    for h in hypotheses:
        if h.get("id") == hypothesis_id:
            notes = h.get("notes") or []
            note_entry = {
                "tool": tool_name,
                "summary": result_summary,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            notes.append(note_entry)
            h["notes"] = notes
            if status_update is not None:
                h["status"] = status_update
                h["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_hypotheses(project, hypotheses)
            return h
    raise ValueError(f"Hypothesis '{hypothesis_id}' not found in project '{project}'")
