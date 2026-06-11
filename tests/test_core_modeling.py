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
