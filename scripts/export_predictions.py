"""Recreate results/predictions_{valid,test}.csv WITHOUT computing any metric.

Uses the frozen LightGBM (models/lgbm.txt) and refits the deterministic
baselines (prevalence, lead-time-only, logistic regression) on TRAIN only.
Scoring is left to scripts/verify_from_raw.py, which is independent of this code.

Usage: python scripts/export_predictions.py data/raw/KaggleV2-May-2016.csv [out_dir]
"""
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from noshow.clean import clean, load_raw  # noqa: E402
from noshow.features import BASE_FEATURES, HISTORY_FEATURES, make_dataset  # noqa: E402
from noshow.models import LeadOnly, LogReg, Prevalence  # noqa: E402

FEATURES = BASE_FEATURES + HISTORY_FEATURES


def main(raw_csv, out_dir=None):
    out_dir = Path(out_dir) if out_dir else ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df, _ = clean(load_raw(raw_csv))
    X, y, split, _ = make_dataset(df)
    split = split.to_numpy(); tr = split == "train"
    models = {m.name: m.fit(X.loc[tr, FEATURES], y[tr]) for m in [Prevalence(), LeadOnly(), LogReg()]}
    booster = lgb.Booster(model_file=str(ROOT / "models/lgbm.txt"))
    train_patients = set(df.loc[tr, "patient_id"])
    for s in ["valid", "test"]:
        m = split == s
        Xs = X.loc[m, FEATURES]
        out = pd.DataFrame({"appointment_id": df.loc[m, "appointment_id"].to_numpy(),
                            "patient_id": df.loc[m, "patient_id"].to_numpy(),
                            "no_show": y[m].to_numpy(), "same_day": Xs["same_day"].to_numpy(),
                            "seen_in_train": df.loc[m, "patient_id"].isin(train_patients).astype(int).to_numpy()})
        for name in ["prevalence", "lead_only", "logreg"]:
            out[f"p_{name}"] = models[name].predict_proba(Xs)
        out["p_lgbm"] = booster.predict(Xs)
        path = out_dir / f"predictions_{s}.csv"
        out.to_csv(path, index=False)
        print(f"wrote {path} ({len(out):,} rows; no metrics computed)")


if __name__ == "__main__":
    main(*sys.argv[1:])
