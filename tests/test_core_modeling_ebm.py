"""Tests for EBM handler and similarity_network in core/modeling.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.modeling import _run_ebm, similarity_network


# ---------------------------------------------------------------------------
# EBM handler tests
# ---------------------------------------------------------------------------

@pytest.fixture
def reg_df():
    rng = np.random.default_rng(7)
    n = 50
    x = rng.normal(0, 1, n)
    return pd.DataFrame({
        "cost": x * 5 + rng.normal(0, 1, n),
        "feature_a": x,
        "feature_b": rng.normal(0, 1, n),
        "state": rng.choice(["CA", "TX", "NY"], n),
    })


@pytest.fixture
def clf_df():
    rng = np.random.default_rng(9)
    n = 50
    x = rng.normal(0, 1, n)
    return pd.DataFrame({
        "label": (x > 0).astype(int),
        "feature_a": x,
        "feature_b": rng.normal(0, 1, n),
    })


def test_ebm_regression_returns_r2(reg_df):
    result = _run_ebm(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]
    assert "rmse" in result["metrics"]
    assert "feature_importance" in result["metrics"]
    assert result["explainability"] == "high"


def test_ebm_regression_shape_data(reg_df):
    result = _run_ebm(reg_df, "cost", ["feature_a", "feature_b"], {})
    shapes = result["ebm_shapes"]
    assert "feature_a" in shapes
    assert "x" in shapes["feature_a"]
    assert "y" in shapes["feature_a"]
    assert len(shapes["feature_a"]["x"]) > 0


def test_ebm_categorical_feature(reg_df):
    result = _run_ebm(
        reg_df, "cost", ["feature_a", "feature_b", "state"],
        {"feature_types": {"state": "categorical"}},
    )
    shapes = result["ebm_shapes"]
    assert "state" in shapes
    # categorical shape has string x values
    assert all(isinstance(v, str) for v in shapes["state"]["x"])


def test_ebm_feature_types_list(reg_df):
    result = _run_ebm(
        reg_df, "cost", ["feature_a", "feature_b"],
        {"feature_types": ["continuous", "continuous"]},
    )
    assert "r2" in result["metrics"]


def test_ebm_feature_types_length_mismatch(reg_df):
    with pytest.raises(ValueError, match="feature_types length"):
        _run_ebm(
            reg_df, "cost", ["feature_a", "feature_b"],
            {"feature_types": ["continuous"]},
        )


def test_ebm_classification(clf_df):
    result = _run_ebm(clf_df, "label", ["feature_a", "feature_b"], {"task": "classification"})
    assert "accuracy" in result["metrics"]
    assert "f1" in result["metrics"]


def test_ebm_summary_contains_target(reg_df):
    result = _run_ebm(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "cost" in result["summary"]


# ---------------------------------------------------------------------------
# similarity_network tests
# ---------------------------------------------------------------------------

@pytest.fixture
def long_volume_df():
    """Long-form entity×code volume table with 5 entities and 3 codes."""
    rows = [
        ("H001", "DRG470", 100),
        ("H001", "DRG871", 20),
        ("H002", "DRG470", 90),
        ("H002", "DRG871", 25),
        ("H003", "DRG291", 80),
        ("H003", "DRG292", 60),
        ("H004", "DRG291", 75),
        ("H004", "DRG292", 55),
        ("H005", "DRG470", 50),
        ("H005", "DRG871", 10),
        ("H005", "DRG292", 5),
    ]
    return pd.DataFrame(rows, columns=["ccn", "drg", "vol"])


@pytest.fixture
def sim_net_project(tmp_path, monkeypatch, long_volume_df):
    """Project with long_volume_df loaded into DuckDB."""
    import duckdb
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("sn-proj")
    db_path = str(tmp_path / "sn-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE volumes AS SELECT * FROM long_volume_df")
    conn.close()
    manifest = ws.read_manifest("sn-proj")
    manifest["datasets"]["volumes"] = {
        "row_count": len(long_volume_df),
        "col_count": len(long_volume_df.columns),
        "profiled": True,
        "profile": {c: {"type": "VARCHAR"} for c in long_volume_df.columns},
    }
    ws.write_manifest("sn-proj", manifest)
    return tmp_path


def test_similarity_network_basic(sim_net_project):
    result = similarity_network(
        "sn-proj", "volumes", entity_col="ccn", code_col="drg", volume_col="vol"
    )
    assert result["n_entities"] == 5
    assert result["n_edges"] > 0
    assert result["num_communities"] >= 1
    assert "entity_community" in result
    assert len(result["entity_community"]) == 5


def test_similarity_network_with_value_col(sim_net_project, long_volume_df, monkeypatch):
    import duckdb
    # Add a value column (one per entity)
    val_map = {"H001": 1.2, "H002": 0.8, "H003": 5.0, "H004": 4.9, "H005": 1.0}
    long_volume_df["premium"] = long_volume_df["ccn"].map(val_map)
    db_path = str(sim_net_project / "sn-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE OR REPLACE TABLE volumes AS SELECT * FROM long_volume_df")
    conn.close()

    result = similarity_network(
        "sn-proj", "volumes",
        entity_col="ccn", code_col="drg", volume_col="vol",
        value_col="premium", z_threshold=1.0,
    )
    assert "community_stats" in result
    assert isinstance(result["outliers"], list)


def test_similarity_network_missing_col(sim_net_project):
    with pytest.raises(ValueError, match="not found"):
        similarity_network(
            "sn-proj", "volumes",
            entity_col="ccn", code_col="drg", volume_col="nonexistent",
        )


def test_similarity_network_summary_nonempty(sim_net_project):
    result = similarity_network(
        "sn-proj", "volumes", entity_col="ccn", code_col="drg", volume_col="vol"
    )
    assert len(result["summary"]) > 10
