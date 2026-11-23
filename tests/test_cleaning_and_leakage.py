"""Cleaning-count assertions (need the raw CSV) and leakage tests (synthetic + real)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from noshow.clean import clean, label_disagreeing_groups, load_raw, typed
from noshow.features import HISTORY_FEATURES, assign_split, build_features, make_dataset, patient_history

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "KaggleV2-May-2016.csv"
needs_raw = pytest.mark.skipif(not RAW.exists(), reason="raw Kaggle CSV not present")


@pytest.fixture(scope="module")
def raw():
    return load_raw(RAW)


# ---------------------------------------------------------------- cleaning counts
@needs_raw
def test_cleaning_log_counts(raw):
    df, log = clean(raw)
    assert (log.start_rows, log.start_noshows) == (110_527, 22_319)
    per_rule = [(s["rows_dropped"], s["noshows_dropped"]) for s in log.steps]
    assert per_rule == [(1, 0), (5, 3), (5, 5), (617, 213)]
    assert (log.end_rows, log.end_noshows) == (109_899, 22_098)
    assert round(log.end_noshows / log.end_rows * 100, 2) == 20.11
    assert len(df) == 109_899


@needs_raw
def test_cleaned_data_invariants(raw):
    df, _ = clean(raw)
    assert df["age"].between(0, 110).all()
    assert (df["lead_days"] >= 0).all()
    assert df["appointment_id"].is_unique
    assert label_disagreeing_groups(df) == 29
    assert df["no_show"].isin([0, 1]).all()


@needs_raw
def test_label_coding_yes_means_no_show(raw):
    t = typed(raw)
    assert t["no_show"].sum() == (raw["No-show"] == "Yes").sum() == 22_319


@needs_raw
def test_dedupe_keeps_lowest_appointment_id(raw):
    df, _ = clean(raw)
    ids = set(df["appointment_id"])
    assert 5590627 in ids and 5590628 not in ids and 5590629 not in ids


@needs_raw
def test_duplicates_kept_variant(raw):
    df, log = clean(raw, dedupe=False)
    assert len(df) == 110_516 and len(log.steps) == 3


# ---------------------------------------------------------------- leakage: synthetic
def _toy():
    rows = [  # patient, scheduled, appointment, no_show
        ("p1", "2016-05-01 09:00", "2016-05-02", 1),
        ("p1", "2016-05-02 10:00", "2016-05-05", 0),  # booked on day of first appt: outcome NOT yet known
        ("p1", "2016-05-03 10:00", "2016-05-06", 1),  # first appt outcome known
        ("p1", "2016-05-10 10:00", "2016-05-10", 0),  # three prior outcomes known
        ("p2", "2016-05-01 09:00", "2016-05-20", 1),
    ]
    df = pd.DataFrame(rows, columns=["patient_id", "scheduled_ts", "appointment_date", "no_show"])
    df["scheduled_ts"] = pd.to_datetime(df["scheduled_ts"])
    df["scheduled_date"] = df["scheduled_ts"].dt.normalize()
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    return df


def test_history_uses_only_known_outcomes():
    h = patient_history(_toy())
    assert h["prior_appts"].tolist() == [0, 0, 1, 3, 0]
    assert h["prior_noshows"].tolist() == [0, 0, 1, 2, 0]
    assert np.isnan(h["prior_noshow_rate"].iloc[0])


def test_history_ignores_future_label_changes():
    df = _toy()
    base = patient_history(df)
    df.loc[df["appointment_date"] >= "2016-05-06", "no_show"] = 1 - df["no_show"]
    pert = patient_history(df)
    known = df["scheduled_date"] <= pd.Timestamp("2016-05-06")
    pd.testing.assert_frame_equal(base[known], pert[known])


# ---------------------------------------------------------------- leakage: real data
@needs_raw
def test_real_history_invariant_to_test_window_labels(raw):
    """Flip every test-window label: no feature of any row booked before the test window may change."""
    df, _ = clean(raw)
    base = build_features(df)
    flipped = df.copy()
    tw = flipped["appointment_date"] >= pd.Timestamp("2016-06-01")
    flipped.loc[tw, "no_show"] = 1 - flipped.loc[tw, "no_show"]
    pert = build_features(flipped)
    booked_before = df["scheduled_date"] <= pd.Timestamp("2016-06-01")
    pd.testing.assert_frame_equal(base.loc[booked_before, HISTORY_FEATURES], pert.loc[booked_before, HISTORY_FEATURES])


@needs_raw
def test_splits_are_disjoint_and_ordered(raw):
    df, _ = clean(raw)
    s = assign_split(df)
    d = df.groupby(s)["appointment_date"].agg(["min", "max"])
    assert d.loc["train", "max"] < d.loc["valid", "min"] <= d.loc["valid", "max"] < d.loc["test", "min"]
    assert s.notna().all()


@needs_raw
def test_no_forbidden_columns_in_features(raw):
    df, _ = clean(raw)
    X, y, split, _ = make_dataset(df)
    forbidden = {"sms_received", "no_show", "patient_id", "appointment_id", "scheduled_ts", "appointment_date"}
    assert not forbidden & set(X.columns)


@needs_raw
def test_neighbourhood_grouper_fitted_on_train_only(raw):
    df, _ = clean(raw)
    X, y, split, grouper = make_dataset(df)
    train_counts = df.loc[split == "train", "neighbourhood"].value_counts()
    assert set(grouper.keep_) == set(train_counts[train_counts >= 50].index)
