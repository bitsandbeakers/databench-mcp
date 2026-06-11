"""Tests for core/ingest.py — load_file."""
import pytest
import databench_mcp.workspace as ws
from databench_mcp.core import ingest as core_ingest


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")


def test_load_csv_returns_schema_and_row_count(sample_csv):
    result = core_ingest.load_file("test-proj", sample_csv)
    assert result["rows"] == 3
    assert result["columns"] == 4
    assert result["table"] == "providers"
    assert any(col["name"] == "npi" for col in result["schema"])


def test_load_csv_explicit_table_name(sample_csv):
    result = core_ingest.load_file("test-proj", sample_csv, table_name="my_table")
    assert result["table"] == "my_table"


def test_load_csv_registers_in_manifest(sample_csv):
    core_ingest.load_file("test-proj", sample_csv)
    manifest = ws.read_manifest("test-proj")
    assert "providers" in manifest["datasets"]
    assert manifest["datasets"]["providers"]["profiled"] is False
    assert manifest["datasets"]["providers"]["row_count"] == 3


def test_load_parquet_creates_table(sample_parquet):
    result = core_ingest.load_file("test-proj", sample_parquet, table_name="pq_test")
    assert result["rows"] == 3
    assert result["table"] == "pq_test"


def test_load_file_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "data.json"
    bad.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported"):
        core_ingest.load_file("test-proj", bad)


def test_load_file_invalid_table_name_raises(sample_csv):
    with pytest.raises(ValueError, match="Invalid table name"):
        core_ingest.load_file("test-proj", sample_csv, table_name="123bad-name")


def test_load_file_data_queryable_after_load(sample_csv):
    core_ingest.load_file("test-proj", sample_csv)
    from databench_mcp.db import get_connection
    with get_connection("test-proj") as conn:
        count = conn.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
    assert count == 3


def test_load_url_downloads_csv_and_registers(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch

    csv_bytes = b"npi,specialty\n111,Cardiology\n222,Neurology\n"
    mock_resp = MagicMock()
    mock_resp.content = csv_bytes
    mock_resp.raise_for_status = lambda: None

    with patch("databench_mcp.core.ingest.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        result = core_ingest.load_url(
            "test-proj", "https://data.cms.gov/resource/abc.csv", "cms_data"
        )

    assert result["rows"] == 2
    assert result["table"] == "cms_data"
    manifest = ws.read_manifest("test-proj")
    assert "cms_data" in manifest["datasets"]


def test_load_url_saves_raw_file(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch

    csv_bytes = b"a,b\n1,2\n"
    mock_resp = MagicMock()
    mock_resp.content = csv_bytes
    mock_resp.raise_for_status = lambda: None

    with patch("databench_mcp.core.ingest.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        core_ingest.load_url(
            "test-proj", "https://example.com/data.csv", "raw_test"
        )

    raw_path = tmp_path / "test-proj" / "raw" / "raw_test.csv"
    assert raw_path.exists()
    assert raw_path.read_bytes() == csv_bytes


def test_load_url_passes_params_to_httpx(tmp_path):
    from unittest.mock import MagicMock, patch

    csv_bytes = b"npi\n123\n"
    mock_resp = MagicMock()
    mock_resp.content = csv_bytes
    mock_resp.raise_for_status = lambda: None

    with patch("databench_mcp.core.ingest.httpx.Client") as mock_client_cls:
        mock_get = mock_client_cls.return_value.__enter__.return_value.get
        mock_get.return_value = mock_resp
        core_ingest.load_url(
            "test-proj",
            "https://data.cms.gov/resource/abc.csv",
            "filtered",
            params={"$limit": "100", "$where": "state='TX'"},
        )
        mock_get.assert_called_once_with(
            "https://data.cms.gov/resource/abc.csv",
            params={"$limit": "100", "$where": "state='TX'"},
        )
