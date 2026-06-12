"""Integration test: full pipeline → reconstruct → run → clean → run again → clean."""
from __future__ import annotations

import numpy as np
import pytest

import databench_mcp.workspace as ws
from databench_mcp.tools.recipes import reconstruct_recipe, run_recipe


@pytest.fixture
def recipe_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    rows = "\n".join(f"{i},{int(rng.integers(10,100))},{rng.random():.4f}" for i in range(n))
    path = tmp_path / "data.csv"
    path.write_text("id,count,score\n" + rows, encoding="utf-8")
    return path


@pytest.fixture
def pipeline_project(tmp_path, monkeypatch, recipe_csv):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    from databench_mcp.core.project import create_project
    from databench_mcp.core.ingest import load_file
    from databench_mcp.core.profile import profile_table
    from databench_mcp.core.modeling import run_model
    from databench_mcp.core.viz import create_chart
    create_project("proj")
    load_file("proj", str(recipe_csv), table_name="data")
    profile_table("proj", "data")
    run_model("proj", "data", "kmeans", features=["count", "score"])
    create_chart("proj", "scatter", "data", columns=["count", "score"])
    return tmp_path


def test_reconstruct_recipe_tool(pipeline_project):
    result = reconstruct_recipe("proj")
    from pathlib import Path
    assert Path(result["recipe_py"]).exists()
    assert Path(result["meta_yaml"]).exists()
    assert result["steps"] >= 4  # ingest + profile + run_model + create_chart


def test_run_recipe_tool_clean(pipeline_project):
    reconstruct_recipe("proj")
    result = run_recipe("proj")
    assert result["status"] == "clean"
    assert len(result["finding_ids"]) == 1
    assert result["error"] is None


def test_run_recipe_tool_idempotent(pipeline_project):
    reconstruct_recipe("proj")
    run_recipe("proj")           # first: baseline
    result = run_recipe("proj")  # second: compare
    assert result["status"] == "clean"


def test_run_recipe_tool_no_diff_flag(pipeline_project):
    reconstruct_recipe("proj")
    result = run_recipe("proj", diff_mode=False)
    assert result["status"] == "clean"
