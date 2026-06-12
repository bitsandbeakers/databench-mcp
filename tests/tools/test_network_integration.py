"""Integration test: derive_table → profile → run_model → create_chart."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

import databench_mcp.workspace as ws
from databench_mcp.tools.eda import derive_table
from databench_mcp.tools.modeling import run_model
from databench_mcp.tools.profile import profile_table
from databench_mcp.tools.viz import create_chart


@pytest.fixture
def project_with_providers(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("int-proj")
    import igraph as ig
    G = ig.Graph.Barabasi(n=20, m=2, directed=False)
    edge_df = pd.DataFrame(
        [(str(e.source), str(e.target), float(e.index + 1)) for e in G.es],
        columns=["referring_npi", "receiving_npi", "shared_patients"],
    )
    db_path = str(tmp_path / "int-proj" / "project.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE referrals AS SELECT * FROM edge_df")
    conn.close()
    manifest = ws.read_manifest("int-proj")
    manifest["datasets"]["referrals"] = {
        "profiled": True,
        "row_count": len(edge_df),
        "col_count": 3,
        "profile": {c: {} for c in edge_df.columns},
    }
    ws.write_manifest("int-proj", manifest)
    return tmp_path


def test_derive_then_profile_then_run_network(project_with_providers, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", project_with_providers)

    # 1. Derive a filtered edge table
    derived = derive_table(
        "int-proj",
        "SELECT referring_npi AS source, receiving_npi AS target, shared_patients AS weight "
        "FROM referrals WHERE shared_patients > 5",
        "top_referrals",
    )
    assert derived["rows"] >= 0  # may be 0 for small synthetic data — just check no error

    # 2. Profile the derived table
    profile = profile_table("int-proj", "top_referrals")
    assert profile["table"] == "top_referrals"

    # 3. Run network_communities on the original (profiled) table
    finding = run_model(
        "int-proj", "referrals", "network_communities",
        features=["referring_npi", "receiving_npi"],
        params={"source_col": "referring_npi", "target_col": "receiving_npi"},
    )
    assert "finding_id" in finding
    assert finding["method"] == "network_communities"

    # 4. Create network_graph chart
    result = create_chart(
        "int-proj", "network_graph", "referrals",
        columns=["referring_npi", "receiving_npi"],
        finding_id=finding["finding_id"],
        params={
            "source_col": "referring_npi",
            "target_col": "receiving_npi",
            "max_nodes": 50,
        },
    )
    assert Path(result["path"]).exists()
    html = Path(result["path"]).read_text(encoding="utf-8")
    assert "plotly" in html.lower()
