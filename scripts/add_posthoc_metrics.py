"""POST-HOC subgroup metrics on SAME-DAY rows, from saved predictions only.

This subgroup was not part of the pre-declared evaluation; it was examined
after the final run. No model is trained or re-scored. Writes
results/posthoc_same_day.json.

Usage: python scripts/add_posthoc_metrics.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from noshow.evaluate import auc_pr, bootstrap_ci  # noqa: E402


def paired_diff(d, a, b, n_boot=500, seed=1):
    y = d.no_show.to_numpy(); pa, pb = d[a].to_numpy(), d[b].to_numpy()
    codes, uniq = pd.factorize(d.patient_id)
    idx_by = pd.Series(np.arange(len(d))).groupby(codes).apply(np.array).to_list()
    rng = np.random.default_rng(seed); droc, dpr = [], []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[g] for g in rng.integers(0, len(uniq), len(uniq))])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        droc.append(roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx]))
        dpr.append(average_precision_score(yy, pa[idx]) - average_precision_score(yy, pb[idx]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"roc_auc_diff_ci95": q(droc), "pr_auc_diff_ci95": q(dpr), "n_boot": len(droc)}


def main():
    out = {"note": "POST-HOC subgroup (same-day bookings), computed after the final run from saved predictions; "
                   "not part of the pre-declared evaluation. Hypothesis-generating only."}
    for s in ["valid", "test"]:
        d = pd.read_csv(ROOT / f"results/predictions_{s}.csv")
        d = d[d.same_day == 1].reset_index(drop=True)
        res = {}
        for m in ["lead_only", "logreg", "lgbm"]:
            r = auc_pr(d.no_show, d[f"p_{m}"])
            if m != "lead_only":
                r.update(bootstrap_ci(d.no_show, d[f"p_{m}"], d.patient_id))
            res[m] = r
        res["lgbm_minus_logreg"] = paired_diff(d, "p_lgbm", "p_logreg")
        out[s] = res
        print(f"{s}: same-day n={res['lgbm']['n']:,} no-shows={res['lgbm']['noshows']} prev={res['lgbm']['prevalence']:.4f}")
        for m in ["lead_only", "logreg", "lgbm"]:
            r = res[m]; ci = f" [{r['roc_auc_ci95'][0]:.3f}-{r['roc_auc_ci95'][1]:.3f}]" if "roc_auc_ci95" in r else ""
            print(f"   {m:10s} roc={r['roc_auc']:.4f}{ci} pr={r['pr_auc']:.4f}")
        dd = res["lgbm_minus_logreg"]
        print(f"   lgbm-logreg dROC CI {dd['roc_auc_diff_ci95'][0]:+.4f}..{dd['roc_auc_diff_ci95'][1]:+.4f}  "
              f"dPR CI {dd['pr_auc_diff_ci95'][0]:+.4f}..{dd['pr_auc_diff_ci95'][1]:+.4f}")
    (ROOT / "results/posthoc_same_day.json").write_text(json.dumps(out, indent=2))
    print("wrote results/posthoc_same_day.json")


if __name__ == "__main__":
    main()
