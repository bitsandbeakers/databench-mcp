"""Tests for core/recipes.py — reconstruct_recipe and run_recipe."""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
import yaml

import databench_mcp.workspace as ws
from databench_mcp.core.recipes import reconstruct_recipe, run_recipe


@pytest.fixture
def recipe_csv(tmp_path):
    """50-row CSV with enough rows for run_model."""
    rng = np.random.default_rng(0)
    n = 50
    rows = "\n".join(f"{i},{int(rng.integers(10,100))},{rng.random():.4f}" for i in range(n))
    path = tmp_path / "data.csv"
    path.write_text("id,count,score\n" + rows)
    return path


@pytest.fixture
def project_for_recipes(tmp_path, monkeypatch, recipe_csv):
    """Project with ingested + profiled 'data' table."""
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    from databench_mcp.core.project import create_project
    from databench_mcp.core.ingest import load_file
    from databench_mcp.core.profile import profile_table
    create_project("p")
    load_file("p", str(recipe_csv), table_name="data")
    profile_table("p", "data")
    return tmp_path


def test_reconstruct_generates_valid_python(project_for_recipes):
    from pathlib import Path
    from databench_mcp.core.modeling import run_model
    from databench_mcp.core.viz import create_chart
    run_model("p", "data", "kmeans", features=["count", "score"])
    create_chart("p", "scatter", "data", columns=["count", "score"])

    result = reconstruct_recipe("p")

    assert Path(result["recipe_py"]).exists()
    assert Path(result["meta_yaml"]).exists()
    assert result["steps"] > 0
    assert result["warning"] is None

    # recipe.py must be importable
    spec = importlib.util.spec_from_file_location("r", result["recipe_py"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.run)

    # recipe_meta.yaml must be valid YAML with expected keys
    meta = yaml.safe_load(Path(result["meta_yaml"]).read_text())
    assert meta["project"] == "p"
    assert isinstance(meta["steps"], list)
    assert meta["last_run"] is None
    assert "tolerances" in meta


def test_reconstruct_empty_project_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    from databench_mcp.core.project import create_project
    create_project("empty")
    with pytest.raises(ValueError, match="nothing to reconstruct"):
        reconstruct_recipe("empty")


def test_reconstruct_step_kinds(project_for_recipes):
    from databench_mcp.core.modeling import run_model
    from databench_mcp.core.viz import create_chart
    run_model("p", "data", "kmeans", features=["count", "score"])
    create_chart("p", "scatter", "data", columns=["count", "score"])

    result = reconstruct_recipe("p")
    meta = yaml.safe_load(open(result["meta_yaml"]).read())
    kinds = [s["kind"] for s in meta["steps"]]
    assert "ingest_file" in kinds
    assert "profile_table" in kinds
    assert "run_model" in kinds
    assert "create_chart" in kinds


def test_reconstruct_overwrites_existing(project_for_recipes):
    from pathlib import Path
    reconstruct_recipe("p")
    result2 = reconstruct_recipe("p")
    assert result2["warning"] is not None
    assert "overwritten" in result2["warning"]


# ---------------------------------------------------------------------------
# run_recipe tests
# ---------------------------------------------------------------------------


def test_run_recipe_first_run_is_always_clean(project_for_recipes):
    from databench_mcp.core.modeling import run_model
    from databench_mcp.core.viz import create_chart
    run_model("p", "data", "kmeans", features=["count", "score"])
    create_chart("p", "scatter", "data", columns=["count", "score"])
    reconstruct_recipe("p")

    result = run_recipe("p")
    assert result["status"] == "clean"
    assert len(result["finding_ids"]) > 0
    assert result["error"] is None


def test_run_recipe_idempotent_clean(project_for_recipes):
    from databench_mcp.core.modeling import run_model
    from databench_mcp.core.viz import create_chart
    run_model("p", "data", "kmeans", features=["count", "score"])
    create_chart("p", "scatter", "data", columns=["count", "score"])
    reconstruct_recipe("p")

    run_recipe("p")           # first run: establishes baseline
    result = run_recipe("p")  # second run: should still be clean (deterministic data)
    assert result["status"] == "clean"
    assert result["changes"] == []


def test_run_recipe_detects_changed_metric(project_for_recipes):
    from pathlib import Path
    from databench_mcp.core.modeling import run_model
    from databench_mcp.core.viz import create_chart
    run_model("p", "data", "kmeans", features=["count", "score"])
    create_chart("p", "scatter", "data", columns=["count", "score"])
    reconstruct_recipe("p")
    run_recipe("p")  # establish baseline

    # Corrupt the expected inertia in recipe_meta.yaml
    meta_path = Path(project_for_recipes) / "p" / "recipes" / "recipe_meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    for step in meta["steps"]:
        if step["kind"] == "run_model" and step.get("expected"):
            if "inertia" in step["expected"].get("metrics", {}):
                step["expected"]["metrics"]["inertia"] = -999.0
    meta_path.write_text(yaml.dump(meta, default_flow_style=False, allow_unicode=True), encoding="utf-8")

    result = run_recipe("p")
    assert result["status"] == "changed"
    assert any(c["field"] == "inertia" for c in result["changes"])


def test_run_recipe_no_diff_always_clean(project_for_recipes):
    from pathlib import Path
    from databench_mcp.core.modeling import run_model
    from databench_mcp.core.viz import create_chart
    run_model("p", "data", "kmeans", features=["count", "score"])
    create_chart("p", "scatter", "data", columns=["count", "score"])
    reconstruct_recipe("p")
    run_recipe("p")  # establish baseline

    # Corrupt expected just like the previous test
    meta_path = Path(project_for_recipes) / "p" / "recipes" / "recipe_meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    for step in meta["steps"]:
        if step["kind"] == "run_model" and step.get("expected"):
            if "inertia" in step["expected"].get("metrics", {}):
                step["expected"]["metrics"]["inertia"] = -999.0
    meta_path.write_text(yaml.dump(meta, default_flow_style=False, allow_unicode=True), encoding="utf-8")

    result = run_recipe("p", diff_mode=False)
    assert result["status"] == "clean"   # diff skipped; expected updated unconditionally
    assert result["changes"] == []


def test_run_recipe_missing_recipe_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    from databench_mcp.core.project import create_project
    create_project("q")
    with pytest.raises(FileNotFoundError, match="reconstruct_recipe"):
        run_recipe("q")
