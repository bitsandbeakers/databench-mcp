"""Tests for core/modeling.py — dispatch table and method handlers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import databench_mcp.workspace as ws
from databench_mcp.core.modeling import (
    _run_linear_regression,
    _run_lasso,
    _run_ridge,
    _run_elastic_net,
    _run_logistic_regression,
    _run_quantile_regression,
)


@pytest.fixture
def reg_df():
    """Small regression DataFrame."""
    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(0, 1, n)
    return pd.DataFrame({
        "cost": x * 10 + rng.normal(0, 2, n),
        "feature_a": x,
        "feature_b": rng.normal(0, 1, n),
    })


@pytest.fixture
def clf_df():
    """Small classification DataFrame."""
    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(0, 1, n)
    return pd.DataFrame({
        "label": (x > 0).astype(int),
        "feature_a": x,
        "feature_b": rng.normal(0, 1, n),
    })


def test_linear_regression_returns_r2(reg_df):
    result = _run_linear_regression(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]
    assert "rmse" in result["metrics"]
    assert "coefficients" in result["metrics"]
    assert result["explainability"] == "high"


def test_lasso_returns_r2(reg_df):
    result = _run_lasso(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]
    assert result["explainability"] == "high"


def test_ridge_returns_r2(reg_df):
    result = _run_ridge(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]


def test_elastic_net_returns_r2(reg_df):
    result = _run_elastic_net(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]


def test_logistic_regression_returns_accuracy(clf_df):
    result = _run_logistic_regression(clf_df, "label", ["feature_a", "feature_b"], {})
    assert "accuracy" in result["metrics"]
    assert "f1" in result["metrics"]
    assert result["explainability"] == "high"


def test_quantile_regression_returns_quantile(reg_df):
    result = _run_quantile_regression(reg_df, "cost", ["feature_a", "feature_b"],
                                      {"quantile": 0.75})
    assert result["metrics"]["quantile"] == 0.75
    assert "coefficients" in result["metrics"]


def test_regression_method_raises_for_too_few_rows(reg_df):
    tiny = reg_df.head(5)
    with pytest.raises(ValueError, match="at least 10 rows"):
        _run_linear_regression(tiny, "cost", ["feature_a"], {})


from databench_mcp.core.modeling import (
    _run_decision_tree,
    _run_random_forest,
    _run_gradient_boosting,
    _run_shap,
    _run_permutation_importance,
    _run_mutual_information,
)


def test_decision_tree_regression(reg_df):
    result = _run_decision_tree(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]
    assert "feature_importance" in result["metrics"]
    assert result["explainability"] == "high"


def test_decision_tree_classification(clf_df):
    result = _run_decision_tree(clf_df, "label", ["feature_a", "feature_b"], {})
    assert "accuracy" in result["metrics"]


def test_random_forest_regression(reg_df):
    result = _run_random_forest(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]
    assert "feature_importance" in result["metrics"]
    assert result["explainability"] == "medium"


def test_gradient_boosting_regression(reg_df):
    result = _run_gradient_boosting(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "r2" in result["metrics"]
    assert "feature_importance" in result["metrics"]


def test_shap_returns_values(reg_df):
    result = _run_shap(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "mean_abs_shap" in result["metrics"]
    assert "shap_values" in result
    assert result["shap_values"].shape[1] == 2


def test_permutation_importance_returns_scores(reg_df):
    result = _run_permutation_importance(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "importance_mean" in result["metrics"]
    assert result["explainability"] == "medium"


def test_mutual_information_returns_scores(reg_df):
    result = _run_mutual_information(reg_df, "cost", ["feature_a", "feature_b"], {})
    assert "mi_scores" in result["metrics"]
    assert result["explainability"] == "high"


from databench_mcp.core.modeling import _run_kmeans, _run_pca, run_model


def test_kmeans_returns_inertia(reg_df):
    result = _run_kmeans(reg_df, None, ["feature_a", "feature_b"], {"k": 3})
    assert "inertia" in result["metrics"]
    assert "k" in result["metrics"]
    assert "cluster_labels" in result
    assert result["explainability"] == "low"


def test_pca_returns_explained_variance(reg_df):
    result = _run_pca(reg_df, None, ["feature_a", "feature_b"], {"n_components": 2})
    assert "explained_variance_ratio" in result["metrics"]
    assert "loadings" in result["metrics"]
    assert result["explainability"] == "medium"


def test_run_model_end_to_end(project_with_data):
    result = run_model(
        "test-proj", "providers", "linear_regression",
        target="total_drug_cost", features=["claim_count"],
    )
    assert result["finding_id"] == "f001"
    assert result["method"] == "linear_regression"
    assert "r2" in result["metrics"]


def test_run_model_saves_finding(project_with_data):
    run_model("test-proj", "providers", "random_forest",
              target="total_drug_cost", features=["claim_count"])
    from databench_mcp.core.findings import list_findings
    result = list_findings("test-proj")
    assert result["count"] == 1
    assert result["findings"][0]["method"] == "random_forest"


def test_run_model_shap_saves_npy(project_with_data):
    result = run_model("test-proj", "providers", "shap",
                       target="total_drug_cost", features=["claim_count"])
    import numpy as np
    from databench_mcp.workspace import project_path
    npy_path = project_path("test-proj") / "artifacts" / f"{result['finding_id']}_shap.npy"
    assert npy_path.exists()
    vals = np.load(str(npy_path))
    assert vals.ndim == 2


def test_run_model_unknown_method(project_with_data):
    with pytest.raises(ValueError, match="unknown method"):
        run_model("test-proj", "providers", "magic_model", target="total_drug_cost")


def test_run_model_requires_profiled(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", tmp_path)
    ws.ensure_project("test-proj")
    with pytest.raises(ValueError, match="profiled"):
        run_model("test-proj", "providers", "linear_regression", target="total_drug_cost")
