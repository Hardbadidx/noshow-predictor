<!-- GENERATED FILE: edit docs/README.template.md, then run `python scripts/build_readme.py`.
     Every number below is read from results/*.json; tests/test_readme_drift.py fails if this file drifts. -->

# Appointment no-show predictor: time-split, leakage-checked, with SHAP

Predicts, **at the moment an appointment is booked**, whether the patient will not attend. The project uses the public Kaggle *Medical Appointment No Shows* data and includes:

- a time-based split;
- three baselines and LightGBM, with no hyperparameter search;
- a test set evaluated once ({{d:test_lines}} logged run in `results/test_runs.log`);
- SHAP explanations;
- a descriptive Power BI dashboard recipe.

> **Bottom line, stated plainly.** LightGBM is **tied with logistic regression**: the paired test difference is {{pair:all_rows/roc_auc_diff_mean|+.3f}} ROC-AUC, 95% CI [{{pair:all_rows/roc_auc_diff_ci95/0|+.3f}}, {{pair:all_rows/roc_auc_diff_ci95/1|+.3f}}].
> Most of the signal is **lead time** (same-day vs booked ahead). Among appointments booked at least a day ahead, test ROC-AUC is only **{{m:models/lgbm/non_same_day/roc_auc|.2f}}**.
> Predictions run {{d:val_gap_pts|.1f}} pts (validation) and {{d:test_gap_pts|.1f}} pts (test) above the observed no-show rate. The no-show rate and the lead-time mix shift over the six weeks, and no recalibration was applied.
> The data comes from **one Brazilian city over six weeks**.

## Headline results (test set: appointments 1–8 June 2016)

ROC-AUC and PR-AUC are shown with equal weight for **all rows** and for **non-same-day rows**. PR-AUC here means **scikit-learn `average_precision_score`** (step-wise average precision), not the trapezoidal area. Brackets are 95% intervals from a patient-clustered bootstrap with 500 resamples.

| Model | All rows ROC-AUC | All rows PR-AUC | Non-same-day ROC-AUC | Non-same-day PR-AUC |
|---|---|---|---|---|
| Rows / no-show rate | {{m:models/lgbm/all_rows/n|,}} / {{m:models/lgbm/all_rows/prevalence|.1%}} | | {{m:models/lgbm/non_same_day/n|,}} / {{m:models/lgbm/non_same_day/prevalence|.1%}} | |
| Prevalence (constant) | {{m:models/prevalence/all_rows/roc_auc|.3f}} | {{m:models/prevalence/all_rows/pr_auc|.3f}} | {{m:models/prevalence/non_same_day/roc_auc|.3f}} | {{m:models/prevalence/non_same_day/pr_auc|.3f}} |
| Lead-time only | {{m:models/lead_only/all_rows/roc_auc|.3f}} [{{m:models/lead_only/all_rows/roc_auc_ci95/0|.3f}}–{{m:models/lead_only/all_rows/roc_auc_ci95/1|.3f}}] | {{m:models/lead_only/all_rows/pr_auc|.3f}} [{{m:models/lead_only/all_rows/pr_auc_ci95/0|.3f}}–{{m:models/lead_only/all_rows/pr_auc_ci95/1|.3f}}] | {{m:models/lead_only/non_same_day/roc_auc|.3f}} [{{m:models/lead_only/non_same_day/roc_auc_ci95/0|.3f}}–{{m:models/lead_only/non_same_day/roc_auc_ci95/1|.3f}}] | {{m:models/lead_only/non_same_day/pr_auc|.3f}} [{{m:models/lead_only/non_same_day/pr_auc_ci95/0|.3f}}–{{m:models/lead_only/non_same_day/pr_auc_ci95/1|.3f}}] |
| Logistic regression | {{m:models/logreg/all_rows/roc_auc|.3f}} [{{m:models/logreg/all_rows/roc_auc_ci95/0|.3f}}–{{m:models/logreg/all_rows/roc_auc_ci95/1|.3f}}] | {{m:models/logreg/all_rows/pr_auc|.3f}} [{{m:models/logreg/all_rows/pr_auc_ci95/0|.3f}}–{{m:models/logreg/all_rows/pr_auc_ci95/1|.3f}}] | {{m:models/logreg/non_same_day/roc_auc|.3f}} [{{m:models/logreg/non_same_day/roc_auc_ci95/0|.3f}}–{{m:models/logreg/non_same_day/roc_auc_ci95/1|.3f}}] | {{m:models/logreg/non_same_day/pr_auc|.3f}} [{{m:models/logreg/non_same_day/pr_auc_ci95/0|.3f}}–{{m:models/logreg/non_same_day/pr_auc_ci95/1|.3f}}] |
| **LightGBM** | **{{m:models/lgbm/all_rows/roc_auc|.3f}}** [{{m:models/lgbm/all_rows/roc_auc_ci95/0|.3f}}–{{m:models/lgbm/all_rows/roc_auc_ci95/1|.3f}}] | **{{m:models/lgbm/all_rows/pr_auc|.3f}}** [{{m:models/lgbm/all_rows/pr_auc_ci95/0|.3f}}–{{m:models/lgbm/all_rows/pr_auc_ci95/1|.3f}}] | **{{m:models/lgbm/non_same_day/roc_auc|.3f}}** [{{m:models/lgbm/non_same_day/roc_auc_ci95/0|.3f}}–{{m:models/lgbm/non_same_day/roc_auc_ci95/1|.3f}}] | **{{m:models/lgbm/non_same_day/pr_auc|.3f}}** [{{m:models/lgbm/non_same_day/pr_auc_ci95/0|.3f}}–{{m:models/lgbm/non_same_day/pr_auc_ci95/1|.3f}}] |

**LightGBM vs logistic regression** (paired, patient-clustered bootstrap on test):

| Rows | ROC-AUC difference | 95% CI | PR-AUC difference | 95% CI |
|---|---|---|---|---|
| All rows | {{pair:all_rows/roc_auc_diff_mean|+.4f}} | [{{pair:all_rows/roc_auc_diff_ci95/0|+.4f}}, {{pair:all_rows/roc_auc_diff_ci95/1|+.4f}}] | {{pair:all_rows/pr_auc_diff_mean|+.4f}} | [{{pair:all_rows/pr_auc_diff_ci95/0|+.4f}}, {{pair:all_rows/pr_auc_diff_ci95/1|+.4f}}] |
| Non-same-day | {{pair:non_same_day/roc_auc_diff_mean|+.4f}} | [{{pair:non_same_day/roc_auc_diff_ci95/0|+.4f}}, {{pair:non_same_day/roc_auc_diff_ci95/1|+.4f}}] | {{pair:non_same_day/pr_auc_diff_mean|+.4f}} | [{{pair:non_same_day/pr_auc_diff_ci95/0|+.4f}}, {{pair:non_same_day/pr_auc_diff_ci95/1|+.4f}}] |

Every interval includes zero: **the two models are tied.**

**Validation set (20–31 May), same models:**

| Model | ROC-AUC (all rows) | PR-AUC (all rows) | ROC-AUC (non-same-day) | PR-AUC (non-same-day) |
|---|---|---|---|---|
| LightGBM | {{m:validation/models/lgbm/all_rows/roc_auc|.3f}} | {{m:validation/models/lgbm/all_rows/pr_auc|.3f}} | {{m:validation/models/lgbm/non_same_day/roc_auc|.3f}} | {{m:validation/models/lgbm/non_same_day/pr_auc|.3f}} |
| Logistic regression | {{m:validation/models/logreg/all_rows/roc_auc|.3f}} | {{m:validation/models/logreg/all_rows/pr_auc|.3f}} | {{m:validation/models/logreg/non_same_day/roc_auc|.3f}} | {{m:validation/models/logreg/non_same_day/pr_auc|.3f}} |

### Calibration

| Measure (LightGBM, test) | Value |
|---|---|
| Brier score | {{m:models/lgbm/calibration_all_rows/brier|.4f}} |
| Calibration slope | {{m:models/lgbm/calibration_all_rows/calibration_slope|.2f}} |
| Calibration intercept | {{m:models/lgbm/calibration_all_rows/calibration_intercept|.2f}} |
| Expected calibration error (deciles) | {{m:models/lgbm/calibration_all_rows/ece_decile|.3f}} |
| Mean prediction vs observed rate | {{m:models/lgbm/calibration_all_rows/mean_pred|.1%}} vs {{m:models/lgbm/calibration_all_rows/observed_rate|.1%}} |

Predictions run {{d:val_gap_pts|.1f}} pts (validation) and {{d:test_gap_pts|.1f}} pts (test) above the observed rate. The no-show rate (train {{m:splits/train/noshow_rate|.1%}} → test {{m:splits/test/noshow_rate|.1%}}) and the lead-time mix shift over the six weeks. **No recalibration was applied.**

![reliability](reports/figures/reliability_valid_vs_test.png)

### Flagging the top 20% by risk

The cutoff is the 80th percentile of **validation** scores (LightGBM: {{m:models/lgbm/top20_all_rows/threshold|.3f}}). It was applied unchanged to test:

| Result on test | Value |
|---|---|
| Share of appointments flagged | **{{m:models/lgbm/top20_all_rows/flagged_share|.1%}}** (not 20%, because test has fewer no-shows) |
| Share of no-shows captured | **{{m:models/lgbm/top20_all_rows/noshows_captured_share|.1%}}** |
| Precision among flagged | {{m:models/lgbm/top20_all_rows/precision_in_flagged|.1%}} |
| Lift over the base rate | {{m:models/lgbm/top20_all_rows/lift_over_prevalence|.2f}}× |

On non-same-day rows, with a threshold chosen on validation non-same-day rows, it flags {{m:models/lgbm/top20_non_same_day/flagged_share|.1%}} and captures {{m:models/lgbm/top20_non_same_day/noshows_captured_share|.1%}} of no-shows.

The highest-risk tenth has {{lift:lgbm/9|.2f}}× the base rate; the lowest tenth has {{lift:lgbm/0|.2f}}×.

![gains](reports/figures/cumulative_gains_test.png)
![pr](reports/figures/precision_recall_test.png)
![lift](reports/figures/lift_by_decile_test.png)

### Sensitivity runs (LightGBM, test)

| Run | ROC-AUC (all rows) | PR-AUC (all rows) | ROC-AUC (non-same-day) | PR-AUC (non-same-day) |
|---|---|---|---|---|
| Main model | {{m:models/lgbm/all_rows/roc_auc|.3f}} | {{m:models/lgbm/all_rows/pr_auc|.3f}} | {{m:models/lgbm/non_same_day/roc_auc|.3f}} | {{m:models/lgbm/non_same_day/pr_auc|.3f}} |
| Without patient-history features | {{m:sensitivity/lgbm_no_history/all_rows/roc_auc|.3f}} | {{m:sensitivity/lgbm_no_history/all_rows/pr_auc|.3f}} | {{m:sensitivity/lgbm_no_history/non_same_day/roc_auc|.3f}} | {{m:sensitivity/lgbm_no_history/non_same_day/pr_auc|.3f}} |
| Duplicates kept (own test set, n={{m:sensitivity/lgbm_duplicates_kept/own_eval_set/all_rows/n|,}}) | {{m:sensitivity/lgbm_duplicates_kept/own_eval_set/all_rows/roc_auc|.3f}} | {{m:sensitivity/lgbm_duplicates_kept/own_eval_set/all_rows/pr_auc|.3f}} | {{m:sensitivity/lgbm_duplicates_kept/own_eval_set/non_same_day/roc_auc|.3f}} | {{m:sensitivity/lgbm_duplicates_kept/own_eval_set/non_same_day/pr_auc|.3f}} |
| Patients seen in training ({{m:sensitivity/seen_vs_unseen/seen_in_train/share_of_eval_rows|.1%}} of test rows) | {{m:sensitivity/seen_vs_unseen/seen_in_train/lgbm/all_rows/roc_auc|.3f}} | {{m:sensitivity/seen_vs_unseen/seen_in_train/lgbm/all_rows/pr_auc|.3f}} | {{m:sensitivity/seen_vs_unseen/seen_in_train/lgbm/non_same_day/roc_auc|.3f}} | {{m:sensitivity/seen_vs_unseen/seen_in_train/lgbm/non_same_day/pr_auc|.3f}} |
| Patients not seen in training ({{d:unseen_share|.1%}}) | {{m:sensitivity/seen_vs_unseen/unseen_in_train/lgbm/all_rows/roc_auc|.3f}} | {{m:sensitivity/seen_vs_unseen/unseen_in_train/lgbm/all_rows/pr_auc|.3f}} | {{m:sensitivity/seen_vs_unseen/unseen_in_train/lgbm/non_same_day/roc_auc|.3f}} | {{m:sensitivity/seen_vs_unseen/unseen_in_train/lgbm/non_same_day/pr_auc|.3f}} |

- **History features add little.** One reason: history is cut off at the start of the data window, so the share of rows with any history rises from {{m:splits/train/share_with_history|.0%}} in train to {{m:splits/test/share_with_history|.0%}} in test.
- **Duplicates don't change the result.**
- **No sign of memorising patients.** On all rows, patients seen in training score *lower* ROC-AUC ({{m:sensitivity/seen_vs_unseen/seen_in_train/lgbm/all_rows/roc_auc|.3f}} vs {{m:sensitivity/seen_vs_unseen/unseen_in_train/lgbm/all_rows/roc_auc|.3f}}). On non-same-day rows the two are close ({{m:sensitivity/seen_vs_unseen/seen_in_train/lgbm/non_same_day/roc_auc|.3f}} vs {{m:sensitivity/seen_vs_unseen/unseen_in_train/lgbm/non_same_day/roc_auc|.3f}}). Their higher PR-AUC follows their higher no-show rate ({{m:sensitivity/seen_vs_unseen/seen_in_train/lgbm/all_rows/prevalence|.1%}} vs {{m:sensitivity/seen_vs_unseen/unseen_in_train/lgbm/all_rows/prevalence|.1%}}).

### Post-hoc subgroup: same-day bookings (hypothesis only)

This subgroup was examined **after** the final run, from saved predictions, and was not part of the pre-declared evaluation.

Same-day bookings are {{m:splits/test/same_day_share|.1%}} of test rows, with a no-show rate of {{post:test/lgbm/prevalence|.1%}}.

| Subgroup | n | LightGBM ROC-AUC | Logistic regression ROC-AUC | Paired difference (LightGBM − LR), 95% CI |
|---|---|---|---|---|
| Test, same-day | {{post:test/lgbm/n|,}} | **{{post:test/lgbm/roc_auc|.2f}}** [{{post:test/lgbm/roc_auc_ci95/0|.2f}}–{{post:test/lgbm/roc_auc_ci95/1|.2f}}] | {{post:test/logreg/roc_auc|.2f}} | ROC-AUC [{{post:test/lgbm_minus_logreg/roc_auc_diff_ci95/0|+.3f}}, {{post:test/lgbm_minus_logreg/roc_auc_diff_ci95/1|+.3f}}]; PR-AUC [{{post:test/lgbm_minus_logreg/pr_auc_diff_ci95/0|+.3f}}, {{post:test/lgbm_minus_logreg/pr_auc_diff_ci95/1|+.3f}}] |
| Validation, same-day | {{post:valid/lgbm/n|,}} | **{{post:valid/lgbm/roc_auc|.2f}}** [{{post:valid/lgbm/roc_auc_ci95/0|.2f}}–{{post:valid/lgbm/roc_auc_ci95/1|.2f}}] | {{post:valid/logreg/roc_auc|.2f}} | ROC-AUC [{{post:valid/lgbm_minus_logreg/roc_auc_diff_ci95/0|+.3f}}, {{post:valid/lgbm_minus_logreg/roc_auc_diff_ci95/1|+.3f}}]; PR-AUC [{{post:valid/lgbm_minus_logreg/pr_auc_diff_ci95/0|+.3f}}, {{post:valid/lgbm_minus_logreg/pr_auc_diff_ci95/1|+.3f}}] |

**Why same-day rows can be ranked at all is a hypothesis, not a finding.** One idea is that some "same-day" rows are walk-ins registered on arrival, with booking hour acting as a proxy. The model's attributions do **not** support booking hour as the driver: within same-day test rows its centred mean |SHAP| is {{shap:lgbm_mean_abs_centred_shap/test_same_day/sched_hour|.3f}}. The larger contributions are:

- age: {{shap:lgbm_mean_abs_centred_shap/test_same_day/age|.3f}}
- neighbourhood: {{shap:lgbm_mean_abs_centred_shap/test_same_day/neighbourhood_grp|.3f}}
- prior no-show rate: {{shap:lgbm_mean_abs_centred_shap/test_same_day/prior_noshow_rate|.3f}}

The walk-in idea is untested. In this subgroup only, LightGBM ranks better than logistic regression on ROC-AUC, though not on PR-AUC. That is a post-hoc result and does not change the overall tie.

## SHAP: what the model uses (not what causes no-shows)

SHAP values describe the **frozen LightGBM's** reliance on each feature. They are associations inside a predictive model, **not causal effects**. Checks run on {{d:rows_explained|,}} validation and test rows:

- SHAP values plus the base value equal the model's log-odds output to within {{shap:checks/additivity_max_abs_err|.1e}};
- TreeExplainer matches LightGBM's own `pred_contrib`;
- the model reproduces the saved predictions.

| Feature | Mean \|SHAP\|, all test rows | Mean \|SHAP − subset mean\|, non-same-day test rows |
|---|---|---|
| lead_days | {{shap:lgbm_mean_abs_shap/test_all/lead_days|.3f}} | {{shap:lgbm_mean_abs_centred_shap/test_non_same_day/lead_days|.3f}} |
| age | {{shap:lgbm_mean_abs_shap/test_all/age|.3f}} | {{shap:lgbm_mean_abs_centred_shap/test_non_same_day/age|.3f}} |
| neighbourhood | {{shap:lgbm_mean_abs_shap/test_all/neighbourhood_grp|.3f}} | {{shap:lgbm_mean_abs_centred_shap/test_non_same_day/neighbourhood_grp|.3f}} |
| booking hour | {{shap:lgbm_mean_abs_shap/test_all/sched_hour|.3f}} | {{shap:lgbm_mean_abs_centred_shap/test_non_same_day/sched_hour|.3f}} |
| prior no-show rate | {{shap:lgbm_mean_abs_shap/test_all/prior_noshow_rate|.3f}} | {{shap:lgbm_mean_abs_centred_shap/test_non_same_day/prior_noshow_rate|.3f}} |
| same_day | {{shap:lgbm_mean_abs_shap/test_all/same_day|.3f}} | {{shap:lgbm_mean_abs_centred_shap/test_non_same_day/same_day|.3f}} |

How to read this table:

- **Lead time dominates.** `same_day` gets exactly 0 because the trees split on `lead_days = 0`, which carries the same information. Read `lead_days` as "lead time including same-day".
- **The non-same-day column is centred.** Those rows carry a constant `lead_days` offset of {{shap:lgbm_mean_shap_offset_by_subset/test_non_same_day/lead_days|+.2f}} log-odds relative to the training baseline, which includes same-day bookings. That offset is removed, so the column shows only how much each feature separates one non-same-day booking from another. There, lead time and age carry about equal weight.

**Cross-check against logistic regression.** The logistic regression's linear contributions rank features differently, feature by feature (Spearman ρ = {{shap:rank_agreement_lgbm_vs_logreg/test_all/spearman_rho|.2f}}). Almost all of that gap comes from the redundant pair: LR puts its weight on `same_day`, while LightGBM puts it on `lead_days`. After merging `lead_days` + `same_day`, and the three history features, the rankings agree (ρ = {{shap:rank_agreement_grouped/test_all/spearman_rho|.2f}}, p = {{shap:rank_agreement_grouped/test_all/p_value|.4f}}).

- Both models put lead time first and age second.
- On all test rows, LightGBM ranks neighbourhood third ({{shap:lgbm_mean_abs_shap_grouped/test_all/neighbourhood_grp|.3f}}), while logistic regression ranks patient history third ({{shap:logreg_mean_abs_linear_contribution_grouped/test_all/history (3 features)|.3f}} vs neighbourhood {{shap:logreg_mean_abs_linear_contribution_grouped/test_all/neighbourhood_grp|.3f}}).
- On non-same-day test rows the top three match.

![shap importance](reports/figures/shap_importance_all_vs_non_same_day.png)
![shap all](reports/figures/shap_beeswarm_test_all.png)
![shap non-same-day](reports/figures/shap_beeswarm_test_non_same_day.png)

## Data

- **Source:** Kaggle, *Medical Appointment No Shows*, file `KaggleV2-May-2016.csv` (Hoppen, 2017). It comes from public health appointments in **Vitória, Espírito Santo, Brazil** (Salazar et al., 2022).
- **SHA-256 of the file used:** `{{m:raw_sha256}}`
- **Not redistributed here.** Download it from <https://www.kaggle.com/joniarroba/noshowappointments> into `data/raw/`, and check the dataset's licence on that page before reuse.
- **Raw file:** {{audit:rows|,}} rows × 14 columns. No-show rate {{audit:noshow_rate_raw|.2%}}, where `No-show = "Yes"` means the patient **did not** attend. Appointments fall on {{audit:appointment_dates}} dates (29 Apr – 8 Jun 2016). There are no appointment times: every timestamp is midnight, so {{audit:appt_before_sched_timestamp|,}} rows *look* booked after their appointment when comparing timestamps; only {{audit:appt_before_sched_date}} do by date.

### Cleaning (applied in order; asserted in tests)

| Rule | Rows dropped | No-shows dropped | Rows left |
|---|---|---|---|
| Start | | | {{m:cleaning/start_rows|,}} |
| R1 age < 0 | {{m:cleaning/steps/0/rows_dropped}} | {{m:cleaning/steps/0/noshows_dropped}} | {{m:cleaning/steps/0/rows_after|,}} |
| R2 age > 110 (five rows aged 115; ages 102 kept) | {{m:cleaning/steps/1/rows_dropped}} | {{m:cleaning/steps/1/noshows_dropped}} | {{m:cleaning/steps/1/rows_after|,}} |
| R3 appointment date before booking date | {{m:cleaning/steps/2/rows_dropped}} | {{m:cleaning/steps/2/noshows_dropped}} | {{m:cleaning/steps/2/rows_after|,}} |
| R4 identical except AppointmentID (keep lowest ID) | {{m:cleaning/steps/3/rows_dropped}} | {{m:cleaning/steps/3/noshows_dropped}} | **{{m:cleaning/end_rows|,}}** ({{m:cleaning/end_noshow_rate|.2%}} no-show) |

{{m:label_disagreeing_groups_remaining}} groups share the same patient, booking timestamp and appointment date but have different labels. They remain in the data.

### Split by appointment date (R7)

| Split | Dates | Rows | No-show rate | Same-day share |
|---|---|---|---|---|
| Train | {{m:splits/train/window/0}} → {{m:splits/train/window/1}} | {{m:splits/train/rows|,}} | {{m:splits/train/noshow_rate|.1%}} | {{m:splits/train/same_day_share|.1%}} |
| Validation | {{m:splits/valid/window/0}} → {{m:splits/valid/window/1}} | {{m:splits/valid/rows|,}} | {{m:splits/valid/noshow_rate|.1%}} | {{m:splits/valid/same_day_share|.1%}} |
| Test | {{m:splits/test/window/0}} → {{m:splits/test/window/1}} | {{m:splits/test/rows|,}} | {{m:splits/test/noshow_rate|.1%}} | {{m:splits/test/same_day_share|.1%}} |

## Method

**Prediction moment: booking time.** Features:

- **Booking details:** age, sex, scholarship, conditions, handicap level (0–4), lead days, a same-day flag, appointment weekday, booking weekday, booking hour.
- **Neighbourhood:** areas with fewer than 50 training rows are grouped, using training data only.
- **Patient history:** prior appointments, prior no-shows and prior no-show rate. These count **only** earlier appointments whose date is strictly before this booking's date, so the outcome was already known.

**`SMS_received` is excluded.** It is sent after booking, never for bookings with less than 3 days' lead time, and inconsistent between copies of the same booking. It stays in the dashboard CSV for descriptive use.

**Models.** All were declared before any results were seen, with no hyperparameter search:

| Model | Definition |
|---|---|
| Prevalence | Constant score |
| Lead-time only | Training no-show rate of the row's lead-time bucket |
| Logistic regression | scikit-learn defaults on standardised and one-hot features. It also gets a `has_history` indicator (with `prior_noshow_rate` set to 0 when there is no history), because linear models can't use missing values. |
| LightGBM | Library defaults, with the tree cap raised to 5,000 and early-stopped on validation log-loss (patience 100; stopped at {{m:lgbm_best_iteration}} trees) |

**Leakage checks (all in `tests/`):**

- History features are unchanged when every test-window label is flipped.
- The splits are disjoint and in date order.
- The ID, label, SMS and timestamp columns are not used as features.
- The neighbourhood grouping is fitted on training data only.
- The strongest single feature, `lead_days`, has a validation AUC of {{m:leakage_single_feature_valid_auc/lead_days|.3f}}.
- LightGBM's training ROC-AUC is {{m:train_roc_auc/lgbm|.3f}} against {{m:models/lgbm/all_rows/roc_auc|.3f}} on test.

**Test set touched once.** `run_pipeline.py --final` refuses to run if `results/metrics.json` exists, and appends every run to `results/test_runs.log` ({{d:test_lines}} entry). All debugging used the validation split.

## Power BI (descriptive only)

`docs/POWERBI_RECIPE.md` gives click-by-click instructions for building the dashboard from the cleaned CSV (`scripts/export_powerbi.py`, generated locally and not committed). It shows no model scores. Every page carries a caveat banner, and rates are hidden for groups with fewer than {{pbi:min_neighbourhood_appts}} appointments.

SMS is only ever shown **within lead-time buckets**. For bookings 15–30 days ahead, the no-show rate is {{pbi:by_lead_bucket_and_sms/7/no_show_rate|.1%}} with an SMS vs {{pbi:by_lead_bucket_and_sms/6/no_show_rate|.1%}} without. That is the reverse of the raw comparison, and it is an association, not an effect.

## Limitations

- **One Brazilian city, six weeks** (29 Apr – 8 Jun 2016), public clinics only. Results may not transfer to other health systems, seasons or years.
- **Observational data.** Neither SHAP nor the dashboard identifies causes, and SMS was not randomised.
- **No recalibration.** Probabilities are biased upward on the later weeks.
- **Weak signal beyond lead time.** Non-same-day ROC-AUC is about {{m:models/lgbm/non_same_day/roc_auc|.2f}}.
- **Patient history is cut off** at the start of the window.
- **The same-day subgroup result is post-hoc.**

## Reproduce

```bash
git clone https://github.com/Hardbadidx/noshow-predictor.git && cd noshow-predictor
python -m venv .venv && source .venv/bin/activate      # Python 3.13.18
pip install -r requirements.txt
# put KaggleV2-May-2016.csv in data/raw/ and confirm its SHA-256 matches the one above
python scripts/audit_raw.py data/raw/KaggleV2-May-2016.csv         # Stage 1 audit
python -m pytest -q                                                 # full suite incl. cleaning counts
python scripts/run_pipeline.py data/raw/KaggleV2-May-2016.csv       # dev run: validation only
# The one test evaluation is already recorded (results/metrics.json); --final refuses to run again.
python scripts/export_predictions.py data/raw/KaggleV2-May-2016.csv   # frozen model, computes no metric
python scripts/verify_from_raw.py data/raw/KaggleV2-May-2016.csv results/predictions_test.csv results/metrics.json
python scripts/export_powerbi.py data/raw/KaggleV2-May-2016.csv       # dashboard CSV (local only)
python scripts/build_readme.py --check                              # README matches results files
```

`results/predictions_*.csv` are not committed because they contain patient IDs.

- `export_predictions.py` rebuilds them from the frozen LightGBM (`models/lgbm.txt`) and the deterministic baselines, without computing any metric.
- `verify_from_raw.py` shares no code with the pipeline: it re-applies the cleaning rules to the raw CSV and scores the predictions itself.

## References

Hoppen, J. (2017). *Medical appointment no shows* [Data set]. Kaggle. https://www.kaggle.com/joniarroba/noshowappointments

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*.

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence, 2*(1), 56–67. https://doi.org/10.1038/s42256-019-0138-9

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Salazar, L. H. A., Leithardt, V. R. Q., Parreira, W. D., da Rocha Fernandes, A. M., Barbosa, J. L. V., & Correia, S. D. (2022). Application of machine learning techniques to predict a patient's no-show in the healthcare sector. *Future Internet, 14*(1), 3. https://doi.org/10.3390/fi14010003
