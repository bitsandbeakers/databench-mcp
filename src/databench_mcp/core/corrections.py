"""Per-project correction tracker backed by corrections.yaml."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import yaml

from databench_mcp.workspace import project_path, read_manifest

_VALID_CATEGORIES = (
    "data_leakage",
    "wrong_grain",
    "endogeneity",
    "domain_methodology",
    "statistical_error",
    "modeling_discipline",
    "data_quality",
    "other",
)


def _corrections_path(project: str):
    return project_path(project) / "corrections.yaml"


def _read_corrections(project: str) -> list[dict]:
    path = _corrections_path(project)
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text()) or []


def _write_corrections(project: str, corrections: list[dict]) -> None:
    path = _corrections_path(project)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.dump(corrections, default_flow_style=False, allow_unicode=True))
    os.replace(tmp, path)


def _next_id(corrections: list[dict]) -> str:
    nums = [
        int(c["id"][1:])
        for c in corrections
        if c.get("id", "").startswith("c") and c["id"][1:].isdigit()
    ]
    return f"c{(max(nums) + 1):03d}" if nums else "c001"


def log_correction(
    project: str,
    ai_action: str,
    correction: str,
    category: str,
    databench_gap: bool = False,
    gap_description: str | None = None,
) -> dict[str, Any]:
    """Log an analyst correction to an AI analysis mistake. Returns the new entry."""
    if category not in _VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category {category!r}; must be one of {_VALID_CATEGORIES}"
        )
    if gap_description is not None and not databench_gap:
        raise ValueError("gap_description requires databench_gap=True")
    read_manifest(project)  # asserts project exists
    corrections = _read_corrections(project)
    entry: dict[str, Any] = {
        "id": _next_id(corrections),
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "ai_action": ai_action,
        "correction": correction,
        "databench_gap": databench_gap,
        "gap_description": gap_description,
    }
    corrections.append(entry)
    _write_corrections(project, corrections)
    return entry


def list_corrections(
    project: str,
    category: str | None = None,
    databench_gaps_only: bool = False,
) -> dict[str, Any]:
    """Return corrections, optionally filtered by category and/or databench_gap."""
    if category is not None and category not in _VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category {category!r}; must be one of {_VALID_CATEGORIES}"
        )
    read_manifest(project)  # asserts project exists
    corrections = _read_corrections(project)
    if category is not None:
        corrections = [c for c in corrections if c.get("category") == category]
    if databench_gaps_only:
        corrections = [c for c in corrections if c.get("databench_gap") is True]
    return {"project": project, "count": len(corrections), "corrections": corrections}
