"""Descriptive export for Power BI: cleaned appointments (R1-R4), no model outputs.

Patient IDs are dropped. Neighbourhoods with fewer than MIN_NEIGHBOURHOOD_APPTS
appointments are shown as one 'Other' group so no small-sample rate is displayed.
"""
import pandas as pd

MIN_NEIGHBOURHOOD_APPTS = 300
LEAD_BINS = [-1, 0, 2, 7, 14, 30, float("inf")]
LEAD_LABELS = ["Same day", "1-2 days", "3-7 days", "8-14 days", "15-30 days", "31+ days"]
AGE_BINS = [-1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 200]
AGE_LABELS = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
OTHER = f"Other (<{MIN_NEIGHBOURHOOD_APPTS} appts)"


def build_export(df):
    out = pd.DataFrame({
        "appointment_id": df["appointment_id"],
        "appointment_date": df["appointment_date"].dt.date,
        "appointment_week_start": (df["appointment_date"] - pd.to_timedelta(df["appointment_date"].dt.weekday, unit="D")).dt.date,
        "appointment_weekday": df["appointment_date"].dt.day_name(),
        "appointment_weekday_order": df["appointment_date"].dt.weekday + 1,
        "scheduled_date": df["scheduled_date"].dt.date,
        "scheduled_hour": df["scheduled_ts"].dt.hour,
        "lead_days": df["lead_days"],
    })
    lb = pd.cut(df["lead_days"], LEAD_BINS, labels=LEAD_LABELS)
    out["lead_bucket"] = lb.astype(str)
    out["lead_bucket_order"] = lb.cat.codes + 1
    out["same_day"] = (df["lead_days"] == 0).map({True: "Same day", False: "Booked ahead"})
    out["sms_received"] = df["sms_received"].map({1: "SMS sent", 0: "No SMS"})
    ab = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS)
    out["age"] = df["age"]
    out["age_band"] = ab.astype(str)
    out["age_band_order"] = ab.cat.codes + 1
    out["gender"] = df["gender"]
    counts = df["neighbourhood"].map(df["neighbourhood"].value_counts())
    out["neighbourhood"] = df["neighbourhood"]
    out["neighbourhood_appointments"] = counts
    out["neighbourhood_display"] = df["neighbourhood"].where(counts >= MIN_NEIGHBOURHOOD_APPTS, OTHER)
    for c, lab in [("scholarship", "Scholarship (Bolsa Familia)"), ("hypertension", "Hypertension"),
                   ("diabetes", "Diabetes"), ("alcoholism", "Alcoholism")]:
        out[c] = df[c].map({1: "Yes", 0: "No"})
    out["handicap_level"] = df["handicap"]
    out["no_show"] = df["no_show"]  # 1 = did NOT attend (raw 'Yes')
    out["attendance"] = df["no_show"].map({1: "No-show", 0: "Attended"})
    return out


def export_checks(out):
    """Values the Power BI cards/tables must reproduce."""
    g = out.groupby(["lead_bucket_order", "lead_bucket"])
    lead = g.agg(appointments=("no_show", "size"), no_shows=("no_show", "sum"),
                 sms_sent=("sms_received", lambda s: int((s == "SMS sent").sum())))
    lead["no_show_rate"] = (lead["no_shows"] / lead["appointments"]).round(4)
    sms = (out.groupby(["lead_bucket_order", "lead_bucket", "sms_received"])["no_show"]
              .agg(["size", "mean"]).round(4).reset_index())
    nb = out["neighbourhood_display"]
    return {
        "rows": int(len(out)), "no_shows": int(out["no_show"].sum()),
        "no_show_rate": round(float(out["no_show"].mean()), 4),
        "appointment_date_min": str(out["appointment_date"].min()),
        "appointment_date_max": str(out["appointment_date"].max()),
        "min_neighbourhood_appts": MIN_NEIGHBOURHOOD_APPTS,
        "neighbourhoods_total": int(out["neighbourhood"].nunique()),
        "neighbourhoods_shown": int((nb != OTHER).groupby(out["neighbourhood"]).first().sum()),
        "rows_in_other_group": int((nb == OTHER).sum()),
        "by_lead_bucket": lead.reset_index().drop(columns="lead_bucket_order").to_dict(orient="records"),
        "by_lead_bucket_and_sms": sms.drop(columns="lead_bucket_order").rename(
            columns={"size": "appointments", "mean": "no_show_rate"}).to_dict(orient="records"),
    }
