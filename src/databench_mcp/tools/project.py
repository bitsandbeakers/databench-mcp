"""FastMCP tool wrappers for project management.

Each function is registered as an MCP tool in server.py. Docstrings become
tool descriptions visible to the LLM client.
"""
from __future__ import annotations

from databench_mcp.core import project as core


def project_create(name: str) -> dict:
    """Create a new databench project workspace. Idempotent — safe to call again."""
    return core.create_project(name)


def project_list() -> dict:
    """List all databench projects in the workspace."""
    return core.list_projects()


def project_status(name: str) -> dict:
    """Return manifest summary for a project: dataset count, profiling status, sources."""
    return core.get_status(name)
