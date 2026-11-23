"""Booking-time features and the time-based split (rules R6-R8).

Prediction moment = when the appointment is booked (scheduled_ts). Every
feature must be knowable at that moment:
  * SMS_received is excluded (sent after booking; R6).
  * Patient history counts only earlier appointments whose appointment date is
    strictly before this booking's date, i.e. whose outcome was already known.
"""
import numpy as np
import pandas as pd

SPLITS = {  # inclusive appointment-date windows (R7)
    "train": ("2016-04-29", "2016-05-19"),
    "valid": ("2016-05-20", "2016-05-31"),
    "test": ("2016-06-01", "2016-06-08"),
}

BASE_FEATURES = [
    "age", "is_female", "scholarship", "hypertension", "diabetes", "alcoholism", "handicap",
    "lead_days", "same_day", "appt_weekday", "sched_weekday", "sched_hour", "neighbourhood_grp",
]
HISTORY_FEATURES = ["prior_appts", "prior_noshows", "prior_noshow_rate"]
CATEGORICAL = ["neighbourhood_grp"]
RARE_MIN_TRAIN_ROWS = 50


def assign_split(df):
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, (lo, hi) in SPLITS.items():
        m = df["appointment_date"].between(pd.Timestamp(lo), pd.Timestamp(hi))
        out[m] = name
    if out.isna().any():
        raise ValueError(f"{int(out.isna().sum())} rows fall outside every split window")
    return out


def patient_history(df):
    """prior_appts / prior_noshows: the patient's appointments with
    appointment_date < this row's scheduled_date (outcomes known at booking)."""
    daily = (df.groupby(["patient_id", "appointment_date"])["no_show"]
               .agg(n="size", ns="sum").reset_index()
               .sort_values(["patient_id", "appointment_date"]))
    daily["prior_appts"] = daily.groupby("patient_id")["n"].cumsum()
    daily["prior_noshows"] = daily.groupby("patient_id")["ns"].cumsum()
    left = df[["patient_id", "scheduled_date"]].reset_index().sort_values("scheduled_date")
    right = daily[["patient_id", "appointment_date", "prior_appts", "prior_noshows"]].sort_values("appointment_date")
    m = pd.merge_asof(left, right, left_on="scheduled_date", right_on="appointment_date",
                      by="patient_id", direction="backward", allow_exact_matches=False)
    m = m.set_index("index").sort_index()
    out = pd.DataFrame(index=df.index)
    out["prior_appts"] = m["prior_appts"].fillna(0).astype(int)
    out["prior_noshows"] = m["prior_noshows"].fillna(0).astype(int)
    out["prior_noshow_rate"] = np.where(out["prior_appts"] > 0,
                                        out["prior_noshows"] / out["prior_appts"].clip(lower=1), np.nan)
    return out


def build_features(df):
    """Row-level features that need no fitting. Neighbourhood grouping is fitted later on train only."""
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"]
    X["is_female"] = (df["gender"] == "F").astype(int)
    for c in ["scholarship", "hypertension", "diabetes", "alcoholism", "handicap"]:
        X[c] = df[c]
    X["lead_days"] = df["lead_days"]
    X["same_day"] = (df["lead_days"] == 0).astype(int)
    X["appt_weekday"] = df["appointment_date"].dt.weekday
    X["sched_weekday"] = df["scheduled_ts"].dt.weekday
    X["sched_hour"] = df["scheduled_ts"].dt.hour
    X = X.join(patient_history(df))
    X["neighbourhood"] = df["neighbourhood"]
    return X


class NeighbourhoodGrouper:
    """R8: neighbourhoods with < RARE_MIN_TRAIN_ROWS training rows (or unseen) -> 'OTHER'."""

    def fit(self, neighbourhood_train):
        vc = neighbourhood_train.value_counts()
        self.keep_ = sorted(vc[vc >= RARE_MIN_TRAIN_ROWS].index)
        self.categories_ = self.keep_ + ["OTHER"]
        return self

    def transform(self, neighbourhood):
        g = neighbourhood.where(neighbourhood.isin(self.keep_), "OTHER")
        return pd.Categorical(g, categories=self.categories_)


def make_dataset(df):
    """Return feature frame, labels, split labels, fitted grouper."""
    X = build_features(df)
    split = assign_split(df)
    grouper = NeighbourhoodGrouper().fit(X.loc[split == "train", "neighbourhood"])
    X["neighbourhood_grp"] = grouper.transform(X["neighbourhood"])
    X = X.drop(columns=["neighbourhood"])
    return X, df["no_show"].astype(int), split, grouper
