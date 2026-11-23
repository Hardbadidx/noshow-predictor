# Power BI recipe: descriptive no-show dashboard

Descriptive views only. **No model scores** go into this dashboard. The input is the cleaned CSV you generate locally (it is git-ignored and must not be committed). The check values below come from `results/powerbi_export_check.json`, which is committed.

Every rate visual uses a **minimum-count rule**: any cell with fewer than 300 appointments is left blank. Neighbourhoods with fewer than 300 appointments are also grouped as `Other (<300 appts)` in the CSV.

---

## 0. Generate the CSV (once)

```bash
python scripts/export_powerbi.py data/raw/KaggleV2-May-2016.csv
# -> data/processed/noshow_powerbi.csv  (109,899 rows, 26 columns)
```

## 1. Load the data

1. Open **Power BI Desktop** and go to **Home → Get data → Text/CSV**. Select `data/processed/noshow_powerbi.csv`.
2. In the preview dialog, set **File Origin** to `65001: Unicode (UTF-8)` so neighbourhood names with accents (e.g. `SÃO JOSÉ`) display correctly. Set **Delimiter** to `Comma`.
3. Click **Transform Data**. In Power Query, check the column types (click the type icon at the left of each header):
   - `appointment_date`, `appointment_week_start`, `scheduled_date`: **Date**
   - `appointment_id`, `appointment_weekday_order`, `scheduled_hour`, `lead_days`, `lead_bucket_order`, `age`, `age_band_order`, `neighbourhood_appointments`, `handicap_level`, `no_show`: **Whole Number**
   - everything else: **Text**
4. Rename the query to `appointments` (right-click it in the Queries list → **Rename**).
5. Click **Home → Close & Apply**.

## 2. Sort orders (Table view)

Click the **Table view** icon on the left. For each pair below, select the text column, then use **Column tools → Sort by column**:

| Column | Sort by column |
|---|---|
| `lead_bucket` | `lead_bucket_order` |
| `age_band` | `age_band_order` |
| `appointment_weekday` | `appointment_weekday_order` |

## 3. Measures

In the **Data** pane, right-click `appointments` → **New measure**. Create each measure below, then select it and set the format under **Measure tools**. The two rate measures use **Percentage** with 1 decimal place.

```DAX
Appointments = COUNTROWS ( appointments )
```
```DAX
No-shows = SUM ( appointments[no_show] )
```
```DAX
No-show rate = DIVIDE ( [No-shows], [Appointments] )
```
```DAX
No-show rate (min 300) = IF ( [Appointments] >= 300, [No-show rate] )
```

`no_show = 1` means the patient did **not** attend (the raw `No-show = "Yes"`). **Every rate visual below uses `No-show rate (min 300)`**, never `No-show rate`, so small groups (for example the 39 Saturday appointments, `handicap_level` 2–4, or booking hours 20–21) show as blank instead of a noisy rate.

## 4. Caveat banner (put it on every page)

1. On page 1, choose **Insert → Text box** and drag it full width across the top of the page.
2. Paste this text:

   > **Descriptive only.** Public-clinic appointments from one Brazilian city (Vitória, Espírito Santo), 29 Apr – 8 Jun 2016 (six weeks). Rates show associations, not causes. Groups with fewer than 300 appointments are hidden. SMS was not randomly assigned.

3. In the **Format** pane, give the text box a light grey background and 11 pt text.
4. Once all the pages exist, select the text box, press **Ctrl+C**, then on each other page press **Ctrl+V** (it pastes in the same position).

## 5. Page 1: Overview

Rename the page (double-click the tab) to **Overview**.

1. **Three cards.** In **Visualizations**, click *Card*, then drag in `Appointments`. Repeat for `No-shows` and `No-show rate`.
   **Check:** 109,899 · 22,098 · 20.1%.
2. **No-show rate by lead time.** Choose *Clustered column chart*. Set **X-axis** to `lead_bucket` and **Y-axis** to `No-show rate (min 300)`. Drag `Appointments` into **Tooltips**. Then use **⋯ (More options) → Sort axis → lead_bucket** and **Sort ascending**.
   **Check:** Same day 4.6%, 1-2 days 22.7%, 3-7 days 24.9%, 8-14 days 30.5%, 15-30 days 32.5%, 31+ days 32.9%.
3. **Daily trend.** Choose *Line chart*. Set **X-axis** to `appointment_date` (click the dropdown on the field and choose `appointment_date`, not *Date Hierarchy*) and **Y-axis** to `No-show rate (min 300)`. Saturday 14 May (39 appointments) will be blank because of the minimum-count rule.

## 6. Page 2: Lead time & SMS

Rename it **Lead time & SMS**. **Do not** build any visual that compares `SMS sent` vs `No SMS` without `lead_bucket`. SMS was only sent for bookings made 3+ days ahead, so a raw comparison mainly reflects lead time. Within each lead-time bucket, the direction reverses.

1. **Matrix.** Choose *Matrix*. Set **Rows** to `lead_bucket`, **Columns** to `sms_received`, and **Values** to `No-show rate (min 300)` and `Appointments`. In **Format → Cell elements**, turn **Background color** on for the rate.
   **Check:**

   | lead_bucket | No SMS | SMS sent |
   |---|---|---|
   | Same day | 4.6% (38,493) | blank (none sent) |
   | 1-2 days | 22.7% (11,860) | blank (none sent) |
   | 3-7 days | 26.5% (8,572) | 23.8% (11,548) |
   | 8-14 days | 33.8% (4,917) | 28.1% (7,004) |
   | 15-30 days | 36.6% (6,705) | 29.8% (10,519) |
   | 31+ days | 37.4% (3,871) | 30.2% (6,410) |

2. **Clustered column chart.** Set **X-axis** to `lead_bucket`, **Legend** to `sms_received`, and **Y-axis** to `No-show rate (min 300)`.
3. **Text box** under the charts:
   > SMS reminders were sent only for bookings made 3 or more days ahead. Within each lead-time bucket, appointments with an SMS had a lower no-show rate. This is an association: SMS was not randomly assigned, so it is not evidence of an SMS effect.

## 7. Page 3: Patients

Rename it **Patients**.

1. *Clustered column chart*: **X-axis** `age_band` (sorted by `age_band`, ascending), **Y-axis** `No-show rate (min 300)`, **Tooltips** `Appointments`.
2. Four small *Clustered bar charts*, each with **Y-axis** set to one of `scholarship`, `hypertension`, `diabetes`, `alcoholism` and **X-axis** `No-show rate (min 300)`.
3. *Clustered bar chart*: **Y-axis** `gender`, **X-axis** `No-show rate (min 300)`.
4. **Slicer:** `same_day`, shown as a dropdown (**Format → Slicer settings → Style: Dropdown**). Comparing patient groups within "Booked ahead" removes most of the lead-time mix.

## 8. Page 4: Neighbourhoods

Rename it **Neighbourhoods**.

1. *Clustered bar chart*: **Y-axis** `neighbourhood_display`, **X-axis** `No-show rate (min 300)`, **Tooltips** `Appointments`. Then **⋯ → Sort axis → No-show rate (min 300)** and **Sort descending**.
2. In the **Filters** pane, under *Filters on this visual*, set `neighbourhood_display` to **is not** `Other (<300 appts)`. Keep the Other group out of the ranking, since it mixes 16 small areas.
   **Check:** 65 neighbourhoods shown. 2,049 appointments sit in the Other group.
3. **Text box:**
   > Neighbourhood differences can reflect differences in lead time, age and clinic mix, not place itself.

## 9. Page 5: Calendar

1. *Clustered column chart*: **X-axis** `appointment_weekday` (sorted by `appointment_weekday`, ascending), **Y-axis** `No-show rate (min 300)`. Saturday is blank (39 appointments).
2. *Line chart*: **X-axis** `appointment_week_start`, **Y-axis** `No-show rate (min 300)`.
3. *Clustered column chart*: **X-axis** `scheduled_hour`, **Y-axis** `No-show rate (min 300)`. Hours with fewer than 300 bookings are blank.

## 10. Validate, save, share

1. Compare the cards and the Page 2 matrix with `results/powerbi_export_check.json`. Every number should match. If one doesn't, check the column types (step 1.3) and that the rate visuals use the `(min 300)` measure.
2. **File → Save as** `noshow_dashboard.pbix`, stored **outside the repo**. A `.pbix` embeds the data, so don't commit it.
3. **File → Export → Export to PDF**. Also take a PNG screenshot of each page and save them to `reports/powerbi/`. Commit only those images and the PDF.
