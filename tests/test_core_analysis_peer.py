"""Tests for peer_outliers in core/analysis.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.analysis import peer_outliers


@pytest.fixture
def peer_df():
    """Hospitals with communities and a premium value."""
    rng = np.random.default_rng(3)
    n = 60
    groups = rng.choice(["AMC", "Community", "Rural"], n)
    # Make group means different; one outlier in AMC
    base = {"AMC": 1.0, "Community": 0.3, "Rural": -0.2}
    values = np.array([base[g] + rng.normal(0, 0.2) for g in groups])
    # Inject one clear outlier in AMC
    amc_indices = [i for i, g in enumerate(groups) if g == "AMC"]
    values[amc_indices[0]] = 5.0
    return pd.DataFrame({
        "hospital_id": [f"H{i:03d}" for i in range(n)],
        "community": groups,
        "premium": values,
    })


@pytest.fixture
def peer_project(tmp_path, monkeypatch, peer_df):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("peer-proj")
    db_path = str(tmp_path / "peer-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE hospitals AS SELECT * FROM peer_df")
    conn.close()
    manifest = ws.read_manifest("peer-proj")
    manifest["datasets"]["hospitals"] = {
        "row_count": len(peer_df),
        "col_count": len(peer_df.columns),
        "profiled": True,
        "profile": {c: {"type": "DOUBLE"} for c in peer_df.columns},
    }
    ws.write_manifest("peer-proj", manifest)
    return tmp_path


def test_peer_outliers_finds_injected_outlier(peer_project):
    result = peer_outliers(
        "peer-proj", "hospitals",
        entity_col="hospital_id", value_col="premium", group_col="community",
        z_threshold=1.5,
    )
    assert result["n_outliers"] >= 1
    outlier_entities = [o["entity"] for o in result["outliers"]]
    # The injected outlier should appear
    assert any(o["peer_z"] > 3.0 for o in result["outliers"])


def test_peer_outliers_returns_group_stats(peer_project):
    result = peer_outliers(
        "peer-proj", "hospitals",
        entity_col="hospital_id", value_col="premium", group_col="community",
    )
    assert "group_stats" in result
    group_names = [g["group"] for g in result["group_stats"]]
    assert "AMC" in group_names
    assert "Community" in group_names
    assert "Rural" in group_names


def test_peer_outliers_z_threshold_zero_returns_all_positive(peer_project):
    result = peer_outliers(
        "peer-proj", "hospitals",
        entity_col="hospital_id", value_col="premium", group_col="community",
        z_threshold=0.0,
    )
    # All entities with positive z should be returned
    assert result["n_outliers"] > 0


def test_peer_outliers_missing_column(peer_project):
    with pytest.raises(ValueError, match="not found"):
        peer_outliers(
            "peer-proj", "hospitals",
            entity_col="hospital_id", value_col="nonexistent", group_col="community",
        )


def test_peer_outliers_non_numeric_value_col(peer_project):
    with pytest.raises(ValueError, match="must be numeric"):
        peer_outliers(
            "peer-proj", "hospitals",
            entity_col="hospital_id", value_col="community", group_col="hospital_id",
        )


def test_peer_outliers_summary_nonempty(peer_project):
    result = peer_outliers(
        "peer-proj", "hospitals",
        entity_col="hospital_id", value_col="premium", group_col="community",
    )
    assert "premium" in result["summary"]
    assert "community" in result["summary"]
