"""Independent check of results/metrics.json from results/predictions_test.csv.
Uses a hand-written Mann-Whitney AUC and step-wise average precision (no pipeline code)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def ap(y, p):
    order = np.argsort(-p, kind="mergesort"); ys, ps = y[order], p[order]
    # group ties: precision/recall evaluated at each distinct threshold
    distinct = np.r_[np.diff(ps) != 0, True]
    tp = np.cumsum(ys)[distinct]; k = np.arange(1, len(ys) + 1)[distinct]
    prec, rec = tp / k, tp / ys.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))

r = json.load(open("results/metrics.json"))
d = pd.read_csv("results/predictions_test.csv")
y = d.no_show.to_numpy(); nsd = d.same_day.to_numpy() == 0
assert len(d) == r["splits"]["test"]["rows"] and y.sum() == r["splits"]["test"]["noshows"]
worst = 0
for m in ["prevalence", "lead_only", "logreg", "lgbm"]:
    p = d[f"p_{m}"].to_numpy()
    for name, mask in [("all_rows", np.ones(len(d), bool)), ("non_same_day", nsd)]:
        a, b = auc(y[mask], p[mask]), ap(y[mask], p[mask])
        ja, jb = r["models"][m][name]["roc_auc"], r["models"][m][name]["pr_auc"]
        worst = max(worst, abs(a - ja), abs(b - jb))
        print(f"{m:10s} {name:12s} roc {a:.4f} (json {ja:.4f})  pr {b:.4f} (json {jb:.4f})")
print(f"max abs difference vs metrics.json: {worst:.2e}")

# paired, patient-clustered bootstrap of lgbm - logreg
rng = np.random.default_rng(1); pats = d.patient_id.to_numpy(); paired = {}
codes, uniq = pd.factorize(pats); idx_by = pd.Series(np.arange(len(d))).groupby(codes).apply(np.array).to_list()
for name, mask in [("all_rows", np.ones(len(d), bool)), ("non_same_day", nsd)]:
    da, dp = [], []
    for _ in range(500):
        idx = np.concatenate([idx_by[g] for g in rng.integers(0, len(uniq), len(uniq))]); idx = idx[mask[idx]]
        yy = y[idx]; a1, a2 = d.p_lgbm.to_numpy()[idx], d.p_logreg.to_numpy()[idx]
        da.append(auc(yy, a1) - auc(yy, a2)); dp.append(ap(yy, a1) - ap(yy, a2))
    q = lambda v: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5))
    paired[name] = {"roc_auc_diff_mean": float(np.mean(da)), "roc_auc_diff_ci95": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
                     "pr_auc_diff_mean": float(np.mean(dp)), "pr_auc_diff_ci95": [float(np.percentile(dp, 2.5)), float(np.percentile(dp, 97.5))], "n_boot": 500}
    print(f"lgbm - logreg {name:12s} dROC {q(da)[0]:+.4f} [{q(da)[1]:+.4f}, {q(da)[2]:+.4f}]  dPR {q(dp)[0]:+.4f} [{q(dp)[1]:+.4f}, {q(dp)[2]:+.4f}]")

json.dump({"note": "Paired patient-clustered bootstrap (500, seed 1) of LightGBM minus logistic regression on test, "
                   "from saved predictions.", **paired, "max_abs_diff_vs_metrics_json": worst},
          open("results/paired_diff_test.json", "w"), indent=2)
print("wrote results/paired_diff_test.json")
