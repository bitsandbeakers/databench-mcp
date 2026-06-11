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
