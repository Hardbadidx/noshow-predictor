"""Standalone check of the test-set metrics, independent of the pipeline.

Imports nothing from src/noshow. Re-implements cleaning rules R1-R4 from the raw
CSV, selects test rows (AppointmentDay 2016-06-01..2016-06-08) with labels from
the RAW file, joins the saved predictions by AppointmentID, and prints sklearn
ROC-AUC / PR-AUC for all rows and non-same-day rows. Same-day is recomputed here
from the raw dates; only the prediction columns come from the predictions file.

Usage:
  python scripts/verify_from_raw.py data/raw/KaggleV2-May-2016.csv results/predictions_test.csv [results/metrics.json]
"""
import json
import sys

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def clean_raw(path):
    r = pd.read_csv(path, dtype=str, keep_default_na=False)
    n0 = len(r)
    r["AppointmentID"] = r["AppointmentID"].astype("int64")
    r["Age"] = r["Age"].astype(int)
    sched_date = pd.to_datetime(r["ScheduledDay"].str[:10])
    appt_date = pd.to_datetime(r["AppointmentDay"].str[:10])
    r["lead_days"] = (appt_date - sched_date).dt.days
    r["appt_date"] = appt_date
    r = r[r["Age"] >= 0]                      # R1
    r = r[r["Age"] <= 110]                    # R2
    r = r[r["lead_days"] >= 0]                # R3
    dedup_cols = [c for c in ["PatientId", "Gender", "ScheduledDay", "AppointmentDay", "Age", "Neighbourhood",
                              "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap",
                              "SMS_received", "No-show"]]
    r = r.sort_values("AppointmentID").drop_duplicates(subset=dedup_cols, keep="first")  # R4
    print(f"raw rows {n0:,} -> after R1-R4 {len(r):,}; no-shows {(r['No-show'] == 'Yes').sum():,}")
    return r


def main(raw_path, pred_path, metrics_path=None):
    r = clean_raw(raw_path)
    test = r[r["appt_date"].between("2016-06-01", "2016-06-08")].copy()
    test["y"] = (test["No-show"] == "Yes").astype(int)      # 'Yes' = did NOT attend
    test["same_day"] = (test["lead_days"] == 0).astype(int)
    print(f"test rows from raw: {len(test):,}; no-shows {test['y'].sum():,}")

    pred = pd.read_csv(pred_path)
    pcols = [c for c in pred.columns if c.startswith("p_")]
    m = test[["AppointmentID", "y", "same_day"]].merge(
        pred[["appointment_id"] + pcols], left_on="AppointmentID", right_on="appointment_id",
        how="outer", indicator=True)
    print("join:", m["_merge"].value_counts().to_dict())
    assert (m["_merge"] == "both").all(), "test rows and prediction rows do not match one-to-one"

    ref = json.load(open(metrics_path)) if metrics_path else None
    print(f"\n{'model':18s} {'ALL roc':>9s} {'ALL pr':>9s} {'NSD roc':>9s} {'NSD pr':>9s}")
    for c in pcols:
        nsd = m["same_day"] == 0
        vals = [roc_auc_score(m["y"], m[c]), average_precision_score(m["y"], m[c]),
                roc_auc_score(m.loc[nsd, "y"], m.loc[nsd, c]), average_precision_score(m.loc[nsd, "y"], m.loc[nsd, c])]
        line = f"{c[2:]:18s} " + " ".join(f"{v:9.4f}" for v in vals)
        name = c[2:]
        if ref and name in ref["models"]:
            j = ref["models"][name]
            jv = [j["all_rows"]["roc_auc"], j["all_rows"]["pr_auc"], j["non_same_day"]["roc_auc"], j["non_same_day"]["pr_auc"]]
            line += "   max|diff| vs metrics.json = %.1e" % max(abs(a - b) for a, b in zip(vals, jv))
        print(line)


if __name__ == "__main__":
    main(*sys.argv[1:])
