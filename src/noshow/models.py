"""Models, all pre-declared before any results were seen. No hyperparameter search.

prevalence  constant = training no-show rate
lead_only   training no-show rate of the row's lead-time bucket
logreg      sklearn LogisticRegression defaults (C=1, lbfgs) on standardised + one-hot features
lgbm        LightGBM defaults; n_estimators cap raised to 5000 and early-stopped on
            validation log-loss (patience 100). Seed fixed, deterministic mode.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LEAD_BINS = [-1, 0, 2, 7, 14, 30, np.inf]
LEAD_LABELS = ["0", "1-2", "3-7", "8-14", "15-30", ">30"]
SEED = 42


class Prevalence:
    name = "prevalence"

    def fit(self, X, y, **_):
        self.p_ = float(np.mean(y)); return self

    def predict_proba(self, X):
        return np.full(len(X), self.p_)


class LeadOnly:
    name = "lead_only"

    def fit(self, X, y, **_):
        b = pd.cut(X["lead_days"], LEAD_BINS, labels=LEAD_LABELS)
        self.rates_ = pd.Series(np.asarray(y), index=X.index).groupby(b, observed=False).mean().to_dict()
        return self

    def predict_proba(self, X):
        b = pd.cut(X["lead_days"], LEAD_BINS, labels=LEAD_LABELS)
        return b.map(self.rates_).astype(float).to_numpy()


class LogReg:
    name = "logreg"
    onehot = ["neighbourhood_grp", "appt_weekday", "sched_weekday"]

    def _prep(self, X):
        X = X.copy()
        if "prior_noshow_rate" in X:
            X["has_history"] = X["prior_noshow_rate"].notna().astype(int)
            X["prior_noshow_rate"] = X["prior_noshow_rate"].fillna(0.0)
        X["neighbourhood_grp"] = X["neighbourhood_grp"].astype(str)
        return X

    def fit(self, X, y, **_):
        X = self._prep(X)
        num = [c for c in X.columns if c not in self.onehot]
        self.pipe_ = Pipeline([
            ("ct", ColumnTransformer([("num", StandardScaler(), num),
                                      ("cat", OneHotEncoder(handle_unknown="ignore"), self.onehot)])),
            ("lr", LogisticRegression(max_iter=5000)),
        ]).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipe_.predict_proba(self._prep(X))[:, 1]


class LGBM:
    name = "lgbm"

    def fit(self, X, y, X_valid=None, y_valid=None):
        self.model_ = lgb.LGBMClassifier(n_estimators=5000, random_state=SEED, deterministic=True,
                                         force_row_wise=True, n_jobs=4, verbose=-1)
        self.model_.fit(X, y, eval_X=(X_valid,), eval_y=(y_valid,), eval_metric="binary_logloss",
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        self.best_iteration_ = int(self.model_.best_iteration_)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X, num_iteration=self.best_iteration_)[:, 1]
