"""Project management logic — called by tools/project.py wrappers."""
from __future__ import annotations

import databench_mcp.workspace as ws
from databench_mcp.workspace import (
    ensure_project,
    project_path,
    read_manifest,
)


def create_project(name: str) -> dict:
    """Create workspace dirs and manifest; idempotent."""
    root = ensure_project(name)
    manifest = read_manifest(name)
    return {
        "project": name,
        "path": str(root),
        "created_at": manifest["created_at"],
        "status": "created",
    }


def list_projects() -> dict:
    """Return sorted list of all projects found in WORKSPACE_ROOT."""
    workspace_root = ws.WORKSPACE_ROOT
    if not workspace_root.exists():
        return {"projects": []}
    projects = [
        d.name
        for d in workspace_root.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    ]
    return {"projects": sorted(projects)}


def get_status(name: str) -> dict:
    """Return manifest summary for a project."""
    manifest = read_manifest(name)
    datasets = manifest.get("datasets", {})
    return {
        "project": name,
        "dataset_count": len(datasets),
        "datasets": {
            tbl: {
                "profiled": ds.get("profiled", False),
                "row_count": ds.get("row_count"),
                "source": ds.get("source"),
            }
            for tbl, ds in datasets.items()
        },
        "created_at": manifest.get("created_at"),
    }
