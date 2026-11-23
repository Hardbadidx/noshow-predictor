"""Stage 1 follow-up: drill into near-duplicates, odd IDs, and candidate time-split windows."""
import sys
import pandas as pd
raw = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
raw["y"] = (raw["No-show"] == "Yes").astype(int)
sched = pd.to_datetime(raw["ScheduledDay"], utc=True)
appt = pd.to_datetime(raw["AppointmentDay"], utc=True)
raw["appt_date"] = appt.dt.date
raw["lead"] = (appt.dt.normalize() - sched.dt.normalize()).dt.days

print("--- A. Rows identical except AppointmentID ---")
cols = [c for c in raw.columns if c not in ("AppointmentID", "y", "appt_date", "lead")]
dmask = raw.duplicated(subset=cols, keep=False)
g = raw[dmask].groupby(cols, sort=False)
sizes = g.size()
print("group sizes:", sizes.value_counts().sort_index().to_dict())
print(f"no-show rate inside these rows: {raw.loc[dmask,'y'].mean():.4f} (overall {raw.y.mean():.4f})")
print(f"same-day lead among them: {(raw.loc[dmask,'lead']==0).mean():.3f}")
ex = raw[dmask].sort_values(["PatientId", "ScheduledDay"]).head(6)
print(ex[["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay", "Age", "SMS_received", "No-show"]].to_string())

print("\n--- B. Same (patient, scheduled ts, appt day) but differing in some other column ---")
key = ["PatientId", "ScheduledDay", "AppointmentDay"]
k = raw[raw.duplicated(subset=key, keep=False)]
diffcols = {c: int((k.groupby(key)[c].nunique() > 1).sum()) for c in cols if c not in key}
print("groups where column varies:", {c: v for c, v in diffcols.items() if v})

print("\n--- C. Patient with multiple appointments on the same day ---")
sd = raw.duplicated(subset=["PatientId", "AppointmentDay"], keep=False)
print(f"rows={int(sd.sum()):,}  no-show rate={raw.loc[sd,'y'].mean():.4f}  vs others={raw.loc[~sd,'y'].mean():.4f}")
gg = raw[sd].groupby(["PatientId", "AppointmentDay"])["y"].agg(["mean", "size"])
print(f"groups={len(gg):,}; all-no-show={int((gg['mean']==1).sum()):,}; all-show={int((gg['mean']==0).sum()):,}; mixed={int(((gg['mean']>0)&(gg['mean']<1)).sum()):,}")

print("\n--- D. Float-formatted PatientIds ---")
fm = raw["PatientId"].str.contains(r"[eE.]")
print(raw.loc[fm, ["PatientId", "AppointmentID", "Age", "Gender", "Neighbourhood", "No-show"]].to_string())

print("\n--- E. Handcap >1 ---")
print(raw[raw.Handcap.astype(int) > 1].groupby("Handcap")["y"].agg(["mean", "size"]).to_string())

print("\n--- F. Candidate time split on AppointmentDay ---")
for name, lo, hi in [("train", "2016-04-29", "2016-05-19"), ("valid", "2016-05-20", "2016-05-31"), ("test", "2016-06-01", "2016-06-08")]:
    m = (raw.appt_date >= pd.Timestamp(lo).date()) & (raw.appt_date <= pd.Timestamp(hi).date())
    print(f"{name:5s} {lo}..{hi}: rows={int(m.sum()):6,} ({m.mean():.1%})  no-show={raw.loc[m,'y'].mean():.4f}  same-day share={(raw.loc[m,'lead']==0).mean():.3f}  median lead={raw.loc[m,'lead'].median():.0f}")
tr = set(raw.loc[raw.appt_date <= pd.Timestamp('2016-05-19').date(), "PatientId"])
te = raw.loc[raw.appt_date >= pd.Timestamp('2016-06-01').date()]
print(f"test rows whose patient also appears in train: {te.PatientId.isin(tr).mean():.3f}")

print("\n--- G. Scheduling hour (UTC as written) ---")
print(sched.dt.hour.value_counts().sort_index().to_dict())
