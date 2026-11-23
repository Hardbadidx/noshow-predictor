"""Cleaning rules R1-R5 approved at the Stage 1 gate.

Rules are applied in order; the log records rows and no-shows removed by each
rule *at the point it is applied* (so a row failing two rules is counted once,
under the first).

R1  drop Age < 0
R2  drop Age > 110 (the five 115s); ages 101-110 kept
R3  drop rows whose appointment calendar date is before the scheduling date
R4  rows identical on every column except AppointmentID -> keep lowest AppointmentID
R5  PatientId kept as text ID; Handcap kept as 0-4; timestamps read as written
"""
from dataclasses import dataclass, field

import pandas as pd

RAW_COLUMNS = [
    "PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay", "Age",
    "Neighbourhood", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
    "Handcap", "SMS_received", "No-show",
]


@dataclass
class CleaningLog:
    start_rows: int = 0
    start_noshows: int = 0
    steps: list = field(default_factory=list)

    def add(self, rule, before, after):
        self.steps.append({
            "rule": rule,
            "rows_dropped": int(len(before) - len(after)),
            "noshows_dropped": int(before["no_show"].sum() - after["no_show"].sum()),
            "rows_after": int(len(after)),
            "noshows_after": int(after["no_show"].sum()),
        })

    @property
    def end_rows(self):
        return self.steps[-1]["rows_after"] if self.steps else self.start_rows

    @property
    def end_noshows(self):
        return self.steps[-1]["noshows_after"] if self.steps else self.start_noshows

    def as_dict(self):
        return {
            "start_rows": self.start_rows, "start_noshows": self.start_noshows,
            "steps": self.steps, "end_rows": self.end_rows, "end_noshows": self.end_noshows,
            "end_noshow_rate": round(self.end_noshows / self.end_rows, 6),
        }

    def to_text(self):
        lines = [f"{'rule':<52}{'rows -':>9}{'no-shows -':>12}{'rows left':>11}{'no-shows':>10}",
                 f"{'start':<52}{'':>9}{'':>12}{self.start_rows:>11,}{self.start_noshows:>10,}"]
        for s in self.steps:
            lines.append(f"{s['rule']:<52}{s['rows_dropped']:>9,}{s['noshows_dropped']:>12,}"
                         f"{s['rows_after']:>11,}{s['noshows_after']:>10,}")
        lines.append(f"end: {self.end_rows:,} rows, {self.end_noshows:,} no-shows, "
                     f"rate {self.end_noshows / self.end_rows:.4%}")
        return "\n".join(lines)


def load_raw(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"raw file is missing columns: {sorted(missing)}")
    return df


def typed(raw):
    """Parse types without dropping anything. 'Yes' in No-show means did NOT attend."""
    labels = set(raw["No-show"].unique())
    if not labels <= {"Yes", "No"}:
        raise ValueError(f"unexpected No-show values: {labels}")
    df = pd.DataFrame({
        "patient_id": raw["PatientId"].astype(str),
        "appointment_id": raw["AppointmentID"].astype("int64"),
        "gender": raw["Gender"],
        # 'Z' suffix ignored: values are treated as clinic-local wall-clock time (R5)
        "scheduled_ts": pd.to_datetime(raw["ScheduledDay"].str.rstrip("Z")),
        "appointment_date": pd.to_datetime(raw["AppointmentDay"].str.rstrip("Z")).dt.normalize(),
        "age": raw["Age"].astype(int),
        "neighbourhood": raw["Neighbourhood"],
        "scholarship": raw["Scholarship"].astype(int),
        "hypertension": raw["Hipertension"].astype(int),
        "diabetes": raw["Diabetes"].astype(int),
        "alcoholism": raw["Alcoholism"].astype(int),
        "handicap": raw["Handcap"].astype(int),
        "sms_received": raw["SMS_received"].astype(int),
        "no_show": (raw["No-show"] == "Yes").astype(int),
    })
    df["scheduled_date"] = df["scheduled_ts"].dt.normalize()
    df["lead_days"] = (df["appointment_date"] - df["scheduled_date"]).dt.days
    return df


DEDUP_COLUMNS = [  # every raw column except AppointmentID
    "patient_id", "gender", "scheduled_ts", "appointment_date", "age", "neighbourhood",
    "scholarship", "hypertension", "diabetes", "alcoholism", "handicap", "sms_received", "no_show",
]


def clean(raw, dedupe=True):
    """Apply R1-R4 (R4 optional for the duplicates-kept sensitivity run)."""
    df = typed(raw)
    log = CleaningLog(start_rows=len(df), start_noshows=int(df["no_show"].sum()))

    nxt = df[df["age"] >= 0]
    log.add("R1 age < 0", df, nxt); df = nxt

    nxt = df[df["age"] <= 110]
    log.add("R2 age > 110", df, nxt); df = nxt

    nxt = df[df["lead_days"] >= 0]
    log.add("R3 appointment date before scheduling date", df, nxt); df = nxt

    if dedupe:
        nxt = (df.sort_values("appointment_id")
                 .drop_duplicates(subset=DEDUP_COLUMNS, keep="first")
                 .sort_index())
        log.add("R4 duplicate except AppointmentID (keep lowest)", df, nxt); df = nxt

    return df.reset_index(drop=True), log


def label_disagreeing_groups(df):
    """Groups sharing (patient, scheduling timestamp, appointment date) whose labels differ."""
    g = df.groupby(["patient_id", "scheduled_ts", "appointment_date"])["no_show"].nunique()
    return int((g > 1).sum())
