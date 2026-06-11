"""Shared pytest fixtures for databench-mcp tests."""
import polars as pl
import pytest


@pytest.fixture
def sample_csv(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    content = (
        "npi,specialty,total_drug_cost,opioid_prescriber_rate\n"
        "1234567890,Cardiology,12500.50,0.05\n"
        "9876543210,Internal Medicine,8900.00,0.12\n"
        "1111111111,Family Practice,5400.75,\n"
    )
    path = fixtures_dir / "providers.csv"
    path.write_text(content)
    return path


@pytest.fixture
def sample_parquet(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    df = pl.DataFrame({
        "npi": [1234567890, 9876543210, 1111111111],
        "specialty": ["Cardiology", "Internal Medicine", "Family Practice"],
        "total_drug_cost": [12500.50, 8900.00, 5400.75],
    })
    path = fixtures_dir / "providers.parquet"
    df.write_parquet(path)
    return path


import numpy as np
import pandas as pd
import duckdb
import databench_mcp.workspace as ws


@pytest.fixture
def medicare_df():
    """Synthetic 60-row Medicare-like DataFrame for modeling/analysis tests."""
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame({
        "npi": range(1001, 1001 + n),
        "specialty": rng.choice(["Cardiology", "Oncology", "Primary Care"], n),
        "state": rng.choice(["CA", "TX", "NY", "FL"], n),
        "total_drug_cost": rng.exponential(scale=50000, size=n),
        "claim_count": rng.integers(10, 500, size=n).astype(float),
        "is_high_cost": (rng.random(n) > 0.5),
    })


@pytest.fixture
def project_with_data(tmp_path, monkeypatch, medicare_df):
    """Project with medicare_df loaded into DuckDB and manifest stamped profiled=True."""
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    db_path = str(tmp_path / "test-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE providers AS SELECT * FROM medicare_df")
    conn.close()
    manifest = ws.read_manifest("test-proj")
    manifest["datasets"]["providers"] = {
        "row_count": len(medicare_df),
        "col_count": len(medicare_df.columns),
        "profiled": True,
        "profile": {c: {"type": "DOUBLE"} for c in medicare_df.columns},
    }
    ws.write_manifest("test-proj", manifest)
    return tmp_path
