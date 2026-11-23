"""Metrics: ROC-AUC, PR-AUC (average precision), calibration, top-k capture, clustered bootstrap CIs."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def auc_pr(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return {"n": int(len(y)), "noshows": int(y.sum()), "prevalence": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}


def bootstrap_ci(y, p, groups, n_boot=500, seed=0):
    """95% percentile CI, resampling *patients* (rows of a patient are not independent)."""
    y = np.asarray(y); p = np.asarray(p)
    codes, uniq = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    rng = np.random.default_rng(seed)
    roc, pr = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[g]:ends[g]] for g in pick])
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        roc.append(roc_auc_score(yb, pb)); pr.append(average_precision_score(yb, pb))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"roc_auc_ci95": q(roc), "pr_auc_ci95": q(pr), "n_boot": len(roc)}


def calibration(y, p, n_bins=10):
    y = np.asarray(y); p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=np.inf).fit(logit, y)  # unpenalised
    bins = pd.qcut(p, n_bins, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "b": bins}).groupby("b", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index(drop=True)
    ece = float((t["n"] * (t["mean_pred"] - t["observed"]).abs()).sum() / t["n"].sum())
    return {"brier": float(brier_score_loss(y, p)), "mean_pred": float(p.mean()), "observed_rate": float(y.mean()),
            "calibration_slope": float(lr.coef_[0, 0]), "calibration_intercept": float(lr.intercept_[0]),
            "ece_decile": ece, "reliability": t.round(5).to_dict(orient="records")}


def top_k_capture(y, p, threshold):
    """Share of no-shows whose score >= threshold (threshold fixed on validation beforehand)."""
    y = np.asarray(y); flag = np.asarray(p) >= threshold
    return {"threshold": float(threshold), "flagged_share": float(flag.mean()),
            "noshows_captured_share": float(y[flag].sum() / y.sum()),
            "precision_in_flagged": float(y[flag].mean()) if flag.any() else float("nan"),
            "lift_over_prevalence": float(y[flag].mean() / y.mean()) if flag.any() else float("nan")}
