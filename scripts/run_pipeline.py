"""End-to-end pipeline.

  python scripts/run_pipeline.py data/raw/KaggleV2-May-2016.csv            # dev: scores VALIDATION only
  python scripts/run_pipeline.py data/raw/KaggleV2-May-2016.csv --final    # one-time TEST evaluation

Dev and final run identical code; the only difference is which split is scored.
--final refuses to run if results/metrics.json already exists (test set is touched once);
every final run is appended to results/test_runs.log.
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noshow.clean import clean, label_disagreeing_groups, load_raw  # noqa: E402
from noshow.evaluate import auc_pr, bootstrap_ci, calibration, top_k_capture  # noqa: E402
from noshow.features import BASE_FEATURES, HISTORY_FEATURES, SPLITS, make_dataset  # noqa: E402
from noshow.models import LGBM, LeadOnly, LogReg, Prevalence  # noqa: E402

TOP_SHARE = 0.20


def fit_models(X, y, split, features, kinds):
    tr, va = split == "train", split == "valid"
    out = {}
    for cls in kinds:
        m = cls()
        m.fit(X.loc[tr, features], y[tr], X_valid=X.loc[va, features], y_valid=y[va])
        out[m.name] = m
    return out


def headline(y, p, same_day, groups, ci=True):
    """Metrics on ALL rows and on NON-SAME-DAY rows, side by side."""
    nsd = same_day == 0
    res = {"all_rows": auc_pr(y, p), "non_same_day": auc_pr(y[nsd], p[nsd])}
    if ci:
        res["all_rows"].update(bootstrap_ci(y, p, groups))
        res["non_same_day"].update(bootstrap_ci(y[nsd], p[nsd], groups[nsd]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    results_dir = ROOT / "results"; results_dir.mkdir(exist_ok=True)
    out_path = results_dir / ("metrics.json" if args.final else "dev_metrics.json")
    if args.final and out_path.exists():
        sys.exit("results/metrics.json exists: the test set has already been evaluated. Refusing to re-run.")
    eval_split = "test" if args.final else "valid"
    print(f"[mode] {'FINAL - scoring TEST' if args.final else 'DEV - scoring VALIDATION only'}")

    raw_bytes = Path(args.raw_csv).read_bytes()
    raw = load_raw(args.raw_csv)
    df, log = clean(raw)
    print("\n[cleaning log]\n" + log.to_text())
    print(f"label-disagreeing (patient, scheduled ts, appointment date) groups remaining: {label_disagreeing_groups(df)}")

    X, y, split, grouper = make_dataset(df)
    y = y.to_numpy(); split = split.to_numpy()
    same_day = X["same_day"].to_numpy(); pid = df["patient_id"].to_numpy()
    features = BASE_FEATURES + HISTORY_FEATURES

    split_info = {}
    for s in SPLITS:
        m = split == s
        split_info[s] = {"window": SPLITS[s], "rows": int(m.sum()), "noshows": int(y[m].sum()),
                         "noshow_rate": float(y[m].mean()), "same_day_share": float(same_day[m].mean()),
                         "share_with_history": float((X.loc[m, "prior_appts"] > 0).mean()),
                         "non_same_day_rows": int((m & (same_day == 0)).sum()),
                         "non_same_day_noshow_rate": float(y[m & (same_day == 0)].mean())}
    print("\n[splits]\n" + pd.DataFrame(split_info).T.drop(columns="window").to_string())
    print(f"neighbourhood groups kept: {len(grouper.keep_)} (+OTHER)")

    # ---------------- leakage diagnostics (validation only, never test)
    va = split == "valid"
    single = {}
    for c in features:
        v = X.loc[va, c]
        v = v.cat.codes if hasattr(v, "cat") else v.fillna(-1)
        a = roc_auc_score(y[va], v)
        single[c] = round(float(max(a, 1 - a)), 4)
    print("\n[leakage] single-feature validation AUC (max(a,1-a)); >0.90 would be suspicious:")
    print(json.dumps(dict(sorted(single.items(), key=lambda kv: -kv[1])), indent=1))

    # ---------------- main models
    models = fit_models(X, y, split, features, [Prevalence, LeadOnly, LogReg, LGBM])
    ev = split == eval_split
    preds = {n: m.predict_proba(X.loc[ev, features]) for n, m in models.items()}
    preds_valid = {n: m.predict_proba(X.loc[va, features]) for n, m in models.items()}
    tr = split == "train"
    train_auc = {n: float(roc_auc_score(y[tr], m.predict_proba(X.loc[tr, features])))
                 for n, m in models.items() if n != "prevalence"}
    print(f"\n[lgbm] best_iteration={models['lgbm'].best_iteration_}; train ROC-AUC by model: "
          + json.dumps({k: round(v, 4) for k, v in train_auc.items()}))

    yv, sdv, pidv = y[ev], same_day[ev], pid[ev]
    res_models = {}
    for n, p in preds.items():
        r = headline(yv, p, sdv, pidv, ci=(n != "prevalence"))
        thr_all = float(np.quantile(preds_valid[n][same_day[va] >= 0], 1 - TOP_SHARE))
        thr_nsd = float(np.quantile(preds_valid[n][same_day[va] == 0], 1 - TOP_SHARE))
        if n != "prevalence":
            r["calibration_all_rows"] = calibration(yv, p)
            r["top20_all_rows"] = top_k_capture(yv, p, thr_all)
            r["top20_non_same_day"] = top_k_capture(yv[sdv == 0], p[sdv == 0], thr_nsd)
        res_models[n] = r

    # ---------------- sensitivity 1: no history features
    nohist = fit_models(X, y, split, BASE_FEATURES, [LGBM])["lgbm"]
    p_nh = nohist.predict_proba(X.loc[ev, BASE_FEATURES])
    sens = {"lgbm_no_history": {"best_iteration": nohist.best_iteration_, **headline(yv, p_nh, sdv, pidv)}}

    # ---------------- sensitivity 2: duplicates kept (R4 skipped)
    df_d, log_d = clean(raw, dedupe=False)
    Xd, yd, sd, _ = make_dataset(df_d)
    yd = yd.to_numpy(); sd = sd.to_numpy()
    m_d = fit_models(Xd, yd, sd, features, [LGBM])["lgbm"]
    evd = sd == eval_split
    p_d = m_d.predict_proba(Xd.loc[evd, features])
    sd_d = Xd.loc[evd, "same_day"].to_numpy(); pid_d = df_d.loc[evd, "patient_id"].to_numpy()
    on_dedup = df_d.loc[evd, "appointment_id"].isin(df.loc[ev, "appointment_id"]).to_numpy()
    sens["lgbm_duplicates_kept"] = {
        "best_iteration": m_d.best_iteration_, "rows_after_cleaning": log_d.end_rows,
        "own_eval_set": headline(yd[evd], p_d, sd_d, pid_d),
        "on_deduplicated_eval_rows": headline(yd[evd][on_dedup], p_d[on_dedup], sd_d[on_dedup], pid_d[on_dedup]),
    }

    # ---------------- sensitivity 3: seen vs unseen patients
    train_patients = set(pid[split == "train"])
    seen = np.array([q in train_patients for q in pidv])
    sens["seen_vs_unseen"] = {}
    for label, mask in [("seen_in_train", seen), ("unseen_in_train", ~seen)]:
        sens["seen_vs_unseen"][label] = {
            n: headline(yv[mask], preds[n][mask], sdv[mask], pidv[mask], ci=(n == "lgbm"))
            for n in ["lead_only", "logreg", "lgbm"]}
        sens["seen_vs_unseen"][label]["share_of_eval_rows"] = float(mask.mean())

    # ---------------- save
    pred_df = pd.DataFrame({"appointment_id": df.loc[ev, "appointment_id"].to_numpy(), "patient_id": pidv,
                            "no_show": yv, "same_day": sdv, "seen_in_train": seen.astype(int),
                            **{f"p_{n}": p for n, p in preds.items()}, "p_lgbm_no_history": p_nh})
    pred_df.to_csv(results_dir / f"predictions_{eval_split}.csv", index=False)

    results = {
        "mode": "final" if args.final else "dev", "evaluated_split": eval_split,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__},
        "cleaning": log.as_dict(), "label_disagreeing_groups_remaining": label_disagreeing_groups(df),
        "splits": split_info, "features": features, "excluded_from_features": ["sms_received"],
        "leakage_single_feature_valid_auc": single, "train_roc_auc": train_auc,
        "lgbm_best_iteration": models["lgbm"].best_iteration_,
        "top_share": TOP_SHARE, "models": res_models, "sensitivity": sens,
    }
    out_path.write_text(json.dumps(results, indent=2))
    if args.final:
        with open(results_dir / "test_runs.log", "a") as f:
            f.write(f"{results['run_utc']} final test evaluation, raw sha256 {results['raw_sha256'][:12]}\n")
        models["lgbm"].model_.booster_.save_model(str(ROOT / "models" / "lgbm.txt"),
                                                  num_iteration=models["lgbm"].best_iteration_)
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
