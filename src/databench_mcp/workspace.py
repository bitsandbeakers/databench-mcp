"""Workspace filesystem layout and manifest CRUD.

Every project lives at WORKSPACE_ROOT/<name>/ with a manifest.json that is the
inter-stage contract between tools (ingestion registers tables; profiling stamps
them; analysis tools gate on profiled status via assert_profiled).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE_ROOT = Path(os.environ.get("DATABENCH_WORKSPACE", "./workspace"))

_SUBDIRS = ("raw", "artifacts", "recipes", "reports")


def _validate_name(name: str) -> None:
    if not name or Path(name).parts != (name,):
        raise ValueError(f"Invalid project name {name!r}: must be a single non-empty path component")


def project_path(name: str) -> Path:
    return WORKSPACE_ROOT / name


def ensure_project(name: str) -> Path:
    """Create project directory tree and initialise manifest if absent. Idempotent."""
    _validate_name(name)
    root = project_path(name)
    for subdir in _SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "project": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "datasets": {},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return root


def read_manifest(name: str) -> dict:
    _validate_name(name)
    path = project_path(name) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Project '{name}' not found — run project_create first")
    return json.loads(path.read_text())


def write_manifest(name: str, manifest: dict) -> None:
    """Write manifest atomically via a .tmp sibling to avoid corruption on crash."""
    target = project_path(name) / "manifest.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, target)
