"""FastMCP tool wrappers for modeling and findings functions."""
from __future__ import annotations

from typing import Any

from databench_mcp.core.findings import list_findings as _list_findings
from databench_mcp.core.modeling import run_model as _run_model
from databench_mcp.core.modeling import similarity_network as _similarity_network


def run_model(
    project: str,
    table: str,
    method: str,
    target: str | None = None,
    features: list[str] | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    """Run a registered analysis method, save the finding, return the finding entry.

    Supported methods: linear_regression, lasso, ridge, elastic_net,
    logistic_regression, quantile_regression, decision_tree, random_forest,
    gradient_boosting, ebm, shap, permutation_importance, mutual_information,
    kmeans, pca, network_stats, network_centrality, network_communities.

    For ebm: pass params={'feature_types': {'state': 'categorical'}} to mark
    nominal features. Shape functions are saved as an artifact and returned
    in the finding under the 'ebm_shapes' key.
    """
    return _run_model(project, table, method, target, features, params)


def list_findings(
    project: str,
    method: str | None = None,
) -> dict[str, Any]:
    """List all findings for a project, optionally filtered by method."""
    return _list_findings(project, method)


def similarity_network(
    project: str,
    table: str,
    entity_col: str,
    code_col: str,
    volume_col: str,
    k: int = 8,
    min_sim: float = 0.05,
    z_threshold: float = 1.5,
    value_col: str | None = None,
) -> dict[str, Any]:
    """Build a cosine-similarity kNN graph from a long-form entity×code volume table.

    Pivots the table to an entity×code matrix, L2-normalises each row, computes
    cosine similarity, keeps the top-k neighbours per entity, and detects Louvain
    communities. If value_col is supplied, also returns peer-adjusted z-score
    outliers within each community.

    Parameters
    ----------
    table       : long-form DuckDB table (entity_col | code_col | volume_col)
    entity_col  : column identifying each entity (e.g. hospital CCN)
    code_col    : column with procedure/product codes (e.g. DRG, APC)
    volume_col  : numeric volume/count per entity×code pair
    k           : k-nearest neighbours per entity (default 8)
    min_sim     : minimum cosine similarity to include an edge (default 0.05)
    z_threshold : peer z-score threshold for outlier flagging (default 1.5)
    value_col   : optional numeric value column per entity for peer z-scoring
    """
    return _similarity_network(
        project, table, entity_col, code_col, volume_col,
        k=k, min_sim=min_sim, z_threshold=z_threshold, value_col=value_col,
    )
