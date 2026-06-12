"""FastMCP tool wrappers for recipe reconstruction and execution."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.recipes import reconstruct_recipe as _reconstruct
from databench_mcp.core.recipes import run_recipe as _run


def reconstruct_recipe(project: str) -> dict[str, Any]:
    """Generate recipe.py and recipe_meta.yaml from the project's current state."""
    return _reconstruct(project)


def run_recipe(project: str, diff_mode: bool = True) -> dict[str, Any]:
    """Execute recipe.py and diff outputs against expected. Returns status clean/changed/error."""
    return _run(project, diff_mode)
