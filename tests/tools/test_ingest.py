"""Tests for tools/ingest.py wrappers."""
import pytest
from unittest.mock import MagicMock, patch
import databench_mcp.workspace as ws
from databench_mcp.tools.ingest import ingest_file, ingest_url


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")


def test_ingest_file_returns_table_and_rows(sample_csv):
    result = ingest_file("test-proj", str(sample_csv))
    assert result["rows"] == 3
    assert "table" in result


def test_ingest_url_returns_table_and_rows(tmp_path):
    csv_bytes = b"npi,specialty\n111,Cardiology\n"
    mock_resp = MagicMock()
    mock_resp.content = csv_bytes
    mock_resp.raise_for_status = lambda: None

    with patch("databench_mcp.core.ingest.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        result = ingest_url("test-proj", "https://data.cms.gov/resource/x.csv", "x_table")

    assert result["rows"] == 1
    assert result["table"] == "x_table"
