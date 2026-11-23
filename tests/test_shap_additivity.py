"""SHAP additivity on a small synthetic LightGBM (runs in CI without the raw data)."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

shap = pytest.importorskip("shap")


def test_shap_values_plus_base_equal_margin():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"lead_days": rng.integers(0, 60, 2000), "age": rng.integers(0, 100, 2000),
                      "grp": pd.Categorical(rng.choice(list("abcd"), 2000))})
    logit = -2 + 0.04 * X.lead_days - 0.01 * X.age + (X.grp == "a") * 0.5
    y = (rng.random(2000) < 1 / (1 + np.exp(-logit))).astype(int)
    m = lgb.LGBMClassifier(n_estimators=50, verbose=-1, random_state=0).fit(X, y)
    booster = m.booster_
    expl = shap.TreeExplainer(booster)(X)
    margin = booster.predict(X, raw_score=True)
    assert np.max(np.abs(expl.values.sum(axis=1) + np.ravel(expl.base_values) - margin)) < 1e-6
    contrib = booster.predict(X, pred_contrib=True)
    assert np.allclose(contrib[:, :-1], expl.values, atol=1e-8)
