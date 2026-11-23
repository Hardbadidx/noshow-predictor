"""Stage 1: raw-data audit. Reads the untouched Kaggle CSV and prints facts.

Nothing here cleans or changes the data; it only counts. Every number
printed is computed directly from the raw file passed on the command line.

Usage: python scripts/audit_raw.py data/raw/KaggleV2-May-2016.csv
"""
import json
import sys

import pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main(path):
    # Read everything as strings first so nothing is silently coerced.
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}

    section("1. Shape, columns, blanks")
    print(f"rows={len(raw):,}  columns={raw.shape[1]}")
    print("columns:", list(raw.columns))
    blanks = (raw == "").sum()
    print("blank cells per column (0 everywhere = no missing values):")
    print(blanks.to_string())
    out["rows"] = len(raw)
    out["blank_cells_total"] = int(blanks.sum())

    section("2. Label coding: column 'No-show'")
    vc = raw["No-show"].value_counts()
    print(vc.to_string())
    print("Dataset convention: 'Yes' = patient did NOT attend (no-show); 'No' = attended.")
    rate = (raw["No-show"] == "Yes").mean()
    print(f"no-show rate (share of 'Yes') = {rate:.4%}")
    out["label_values"] = vc.to_dict()
    out["noshow_rate_raw"] = round(float(rate), 6)

    section("3. Identifier integrity")
    pid_numeric = pd.to_numeric(raw["PatientId"], errors="coerce")
    print(f"PatientId non-numeric: {pid_numeric.isna().sum()}")
    non_int = raw["PatientId"].str.contains(r"[.eE]", regex=True)
    print(f"PatientId strings with '.' or exponent (float-formatted): {non_int.sum()}")
    print("  examples:", raw.loc[non_int, "PatientId"].head(5).tolist())
    frac = pid_numeric[non_int] % 1 != 0
    print(f"  of those, with a genuine fractional part: {int(frac.sum())}")
    print(f"unique PatientId (as written): {raw['PatientId'].nunique():,}")
    print(f"unique PatientId (as number):  {pid_numeric.nunique():,}")
    print(f"unique AppointmentID: {raw['AppointmentID'].nunique():,} of {len(raw):,} rows")
    out["patients_unique"] = int(pid_numeric.nunique())
    out["appointment_id_dupes"] = int(raw["AppointmentID"].duplicated().sum())

    section("4. Duplicates")
    full = raw.duplicated().sum()
    print(f"exact duplicate rows (all 14 columns): {full}")
    no_id = raw.drop(columns=["AppointmentID"]).duplicated(keep=False)
    print(f"rows identical on everything except AppointmentID (all copies counted): {int(no_id.sum()):,}")
    print(f"  -> extra copies beyond the first: {int(raw.drop(columns=['AppointmentID']).duplicated().sum()):,}")
    key = ["PatientId", "ScheduledDay", "AppointmentDay"]
    k = raw.duplicated(subset=key, keep=False)
    print(f"rows sharing (PatientId, ScheduledDay timestamp, AppointmentDay): {int(k.sum()):,}")
    grp = raw[k].groupby(key)["No-show"].nunique()
    print(f"  groups of such rows: {len(grp):,}; groups whose copies DISAGREE on label: {int((grp > 1).sum())}")
    same_day = raw.duplicated(subset=["PatientId", "AppointmentDay"], keep=False)
    print(f"rows where the patient has >1 appointment on the same AppointmentDay: {int(same_day.sum()):,}")
    out["exact_dupes"] = int(full)
    out["dupes_except_appointment_id_extra"] = int(raw.drop(columns=["AppointmentID"]).duplicated().sum())

    section("5. Ages")
    age = pd.to_numeric(raw["Age"], errors="coerce")
    print(f"non-numeric ages: {int(age.isna().sum())}")
    print(age.describe().to_string())
    print(f"age < 0:   {int((age < 0).sum())}   values: {sorted(age[age < 0].unique().tolist())}")
    print(f"age == 0:  {int((age == 0).sum()):,}  (plausible: infants)")
    print(f"age > 100: {int((age > 100).sum())}   values: {age[age > 100].value_counts().sort_index().to_dict()}")
    print(f"age > 110: {int((age > 110).sum())}")
    print("rows with age > 100 (first 15):")
    print(raw.loc[age > 100, ["PatientId", "Age", "Gender", "AppointmentDay", "No-show"]].head(15).to_string())
    out["age_negative"] = int((age < 0).sum())
    out["age_over_100"] = int((age > 100).sum())
    out["age_zero"] = int((age == 0).sum())

    section("6. Dates")
    sched = pd.to_datetime(raw["ScheduledDay"], utc=True)
    appt = pd.to_datetime(raw["AppointmentDay"], utc=True)
    print(f"ScheduledDay   range: {sched.min()}  ->  {sched.max()}")
    print(f"AppointmentDay range: {appt.min()}  ->  {appt.max()}")
    t = appt.dt.strftime("%H:%M:%S").value_counts()
    print(f"AppointmentDay time-of-day values: {t.to_dict()}  (no appointment time is recorded)")
    lead_ts = (appt - sched).dt.total_seconds() / 86400
    lead_days = (appt.dt.normalize() - sched.dt.normalize()).dt.days
    print(f"appointment timestamp before scheduling timestamp: {int((lead_ts < 0).sum()):,}"
          "  <- mostly SAME-DAY bookings, because AppointmentDay is midnight")
    print(f"appointment CALENDAR DATE before scheduling date (true errors): {int((lead_days < 0).sum())}")
    print(raw.loc[lead_days < 0, ["PatientId", "ScheduledDay", "AppointmentDay", "No-show"]].to_string())
    print(f"same-day bookings (lead = 0 days): {int((lead_days == 0).sum()):,}")
    print("lead_days describe (calendar days):")
    print(lead_days.describe().to_string())
    out["appt_before_sched_timestamp"] = int((lead_ts < 0).sum())
    out["appt_before_sched_date"] = int((lead_days < 0).sum())
    out["same_day_bookings"] = int((lead_days == 0).sum())

    print("\nappointments per AppointmentDay date:")
    per_day = appt.dt.date.value_counts().sort_index()
    print(per_day.to_string())
    print(f"distinct appointment dates: {per_day.size}")
    wd = appt.dt.day_name().value_counts()
    print("weekday of appointment:", wd.to_dict())
    out["appointment_dates"] = int(per_day.size)

    section("7. Categorical / flag columns")
    for c in ["Gender", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "SMS_received"]:
        print(f"{c:13s}", raw[c].value_counts().sort_index().to_dict())
    print("NOTE: 'Handcap' is not binary (0-4). Column names misspell Hypertension/Handicap.")
    nb = raw["Neighbourhood"].value_counts()
    print(f"Neighbourhood: {nb.size} distinct; smallest 6: {nb.tail(6).to_dict()}")

    section("8. Raw no-show rate by key fields (descriptive only, no model)")
    y = (raw["No-show"] == "Yes").astype(int)
    df = pd.DataFrame({"y": y, "sms": raw["SMS_received"], "lead": lead_days})
    print("by SMS_received:")
    print(df.groupby("sms")["y"].agg(["mean", "size"]).to_string())
    bins = pd.cut(df["lead"], [-999, -1, 0, 2, 7, 14, 30, 999],
                  labels=["<0", "0", "1-2", "3-7", "8-14", "15-30", ">30"])
    print("by lead-days bucket:")
    print(df.groupby(bins, observed=True)["y"].agg(["mean", "size"]).to_string())
    print("SMS_received=1 share by lead bucket (is SMS only sent for longer leads?):")
    print(df.assign(s=(df.sms == "1").astype(int)).groupby(bins, observed=True)["s"].mean().to_string())

    section("9. Repeat patients")
    per_pat = pid_numeric.value_counts()
    print(f"patients with 1 appointment: {int((per_pat == 1).sum()):,}; with >=2: {int((per_pat >= 2).sum()):,};"
          f" max per patient: {int(per_pat.max())}")
    print(f"rows belonging to repeat patients: {int(per_pat[per_pat >= 2].sum()):,}")

    with open("reports/raw_audit.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nwrote reports/raw_audit.json")


if __name__ == "__main__":
    main(sys.argv[1])
