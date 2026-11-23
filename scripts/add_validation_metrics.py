"""Copy the validation-set metrics from the dev run into results/metrics.json.

Does not touch the test split. Before copying it checks that the dev run and
the final run trained the same models:
  * identical cleaning log, split sizes, feature list, LightGBM best iteration,
    and training-set ROC-AUC for every model;
  * the frozen booster (models/lgbm.txt, saved by the final run) re-scores the
    validation rows to the same probabilities as results/predictions_valid.csv.

Usage: python scripts/add_validation_metrics.py data/raw/KaggleV2-May-2016.csv
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from noshow.clean import clean, load_raw  # noqa: E402
from noshow.features import BASE_FEATURES, HISTORY_FEATURES, make_dataset  # noqa: E402


def main(raw_csv):
    final_p, dev_p = ROOT / "results/metrics.json", ROOT / "results/dev_metrics.json"
    final, dev = json.loads(final_p.read_text()), json.loads(dev_p.read_text())
    assert dev["evaluated_split"] == "valid" and final["evaluated_split"] == "test"
    for k in ["raw_sha256", "cleaning", "splits", "features", "lgbm_best_iteration", "train_roc_auc", "versions"]:
        assert dev[k] == final[k], f"dev and final runs differ in '{k}'"
    print("dev and final runs match on: cleaning, splits, features, best_iteration, train AUCs, versions")

    df, _ = clean(load_raw(raw_csv))
    X, y, split, _ = make_dataset(df)
    va = (split == "valid").to_numpy()
    booster = lgb.Booster(model_file=str(ROOT / "models/lgbm.txt"))
    p_frozen = booster.predict(X.loc[va, BASE_FEATURES + HISTORY_FEATURES])
    pv = pd.read_csv(ROOT / "results/predictions_valid.csv")
    assert (pv["appointment_id"].to_numpy() == df.loc[va, "appointment_id"].to_numpy()).all()
    diff = float(np.max(np.abs(p_frozen - pv["p_lgbm"].to_numpy())))
    print(f"frozen booster vs saved validation predictions: max |diff| = {diff:.2e}")
    assert diff < 1e-9

    final["validation"] = {
        "provenance": (f"Copied from dev run {dev['run_utc']} (results/dev_metrics.json): same code, seed, data and "
                       f"trained models as the final run; LightGBM validation predictions re-verified against the "
                       f"frozen booster models/lgbm.txt (max abs diff {diff:.1e})."),
        "models": dev["models"],
        "sensitivity": dev["sensitivity"],
    }
    final_p.write_text(json.dumps(final, indent=2))
    for n, m in dev["models"].items():
        print(f"valid {n:11s} ALL roc={m['all_rows']['roc_auc']:.4f} pr={m['all_rows']['pr_auc']:.4f} | "
              f"NSD roc={m['non_same_day']['roc_auc']:.4f} pr={m['non_same_day']['pr_auc']:.4f}")
    print("updated results/metrics.json -> key 'validation'")


if __name__ == "__main__":
    main(sys.argv[1])
