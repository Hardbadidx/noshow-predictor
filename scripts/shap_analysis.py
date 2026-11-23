"""Stage 3: SHAP explanation of the FROZEN LightGBM (models/lgbm.txt).

No performance metric is computed here (no sklearn.metrics import). The test
rows are only explained; the one comparison against the saved test predictions
is a consistency check that this is the same model.

Checks (assertions):
  * SHAP values + base value == model raw margin (log-odds), every row
  * shap.TreeExplainer agrees with LightGBM's built-in pred_contrib
  * sigmoid(margin) == saved predictions (validation and test)
  * the refitted logistic regression reproduces its saved predictions

Outputs: results/shap_summary.json, reports/figures/shap_*.png

SHAP describes what the model relies on, not what causes no-shows.

Usage: python scripts/shap_analysis.py data/raw/KaggleV2-May-2016.csv
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from noshow.clean import clean, load_raw  # noqa: E402
from noshow.features import BASE_FEATURES, HISTORY_FEATURES, make_dataset  # noqa: E402
from noshow.models import LogReg  # noqa: E402
from noshow.plotstyle import apply_style, INK, INK2  # noqa: E402

FEATURES = BASE_FEATURES + HISTORY_FEATURES
FIG = ROOT / "reports" / "figures"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#a9c9f0", "#2a78d6", "#0d2f5c"])


def mean_abs(sv, mask):
    return pd.Series(np.abs(sv[mask]).mean(axis=0), index=FEATURES)


def beeswarm(expl, mask, title, path, centre=False):
    """centre=True subtracts each feature's mean SHAP over the subset, removing the constant
    offset the subset gets relative to the training baseline (annotated on the plot)."""
    e = expl[mask]
    note = None
    if centre:
        mu = e.values.mean(axis=0)
        e = shap.Explanation(values=e.values - mu, base_values=np.ravel(e.base_values) + mu.sum(),
                             data=e.data, feature_names=e.feature_names)
        big = [(f, v) for f, v in zip(FEATURES, mu) if abs(v) >= 0.05]
        note = "Centred within this subset. Offsets removed (mean SHAP): " + \
               ", ".join(f"{f} {v:+.2f}" for f, v in big) + " log-odds"
    plt.figure()
    shap.plots.beeswarm(e, max_display=len(FEATURES), show=False, color=SEQ,
                        color_bar_label="feature value (low → high)", plot_size=(8.5, 6.8))
    ax = plt.gca()
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_xlabel("SHAP value minus subset mean (log-odds)" if centre
                  else "SHAP value (log-odds contribution to no-show risk)", color=INK2)
    if note:
        plt.figtext(0.01, 0.005, note, fontsize=8.5, color=INK2, ha="left")
    plt.tight_layout(); plt.savefig(path, dpi=160, facecolor="#fcfcfb"); plt.close("all")


def main(raw_csv):
    apply_style(); FIG.mkdir(parents=True, exist_ok=True)
    df, _ = clean(load_raw(raw_csv))
    X, y, split, _ = make_dataset(df)
    split = split.to_numpy()
    sel = np.isin(split, ["valid", "test"])
    Xs = X.loc[sel, FEATURES].reset_index(drop=True)
    sp = split[sel]; sd = Xs["same_day"].to_numpy()
    ids = df.loc[sel, "appointment_id"].to_numpy()

    booster = lgb.Booster(model_file=str(ROOT / "models/lgbm.txt"))
    margin = booster.predict(Xs, raw_score=True)

    # consistency with saved predictions (same model, not a metric)
    cons = {}
    for s in ["valid", "test"]:
        saved = pd.read_csv(ROOT / f"results/predictions_{s}.csv")
        m = sp == s
        assert (saved["appointment_id"].to_numpy() == ids[m]).all()
        cons[s] = float(np.max(np.abs(1 / (1 + np.exp(-margin[m])) - saved["p_lgbm"].to_numpy())))
    print("frozen model vs saved predictions, max |dp|:", cons)
    assert max(cons.values()) < 1e-9

    explainer = shap.TreeExplainer(booster)
    expl = explainer(Xs)
    sv = expl.values
    base = float(np.ravel(expl.base_values)[0])
    add_err = float(np.max(np.abs(sv.sum(axis=1) + np.ravel(expl.base_values) - margin)))
    contrib = booster.predict(Xs, pred_contrib=True)
    lgb_err = float(np.max(np.abs(contrib[:, :-1] - sv)))
    print(f"additivity: max |sum(SHAP)+base - margin| = {add_err:.2e}  (base value {base:.4f} log-odds)")
    print(f"shap.TreeExplainer vs LightGBM pred_contrib: max |diff| = {lgb_err:.2e}")
    assert add_err < 1e-6 and lgb_err < 1e-6

    subsets = {"valid_all": sp == "valid", "valid_non_same_day": (sp == "valid") & (sd == 0),
               "test_all": sp == "test", "test_non_same_day": (sp == "test") & (sd == 0),
               "valid_same_day": (sp == "valid") & (sd == 1), "test_same_day": (sp == "test") & (sd == 1)}
    imp = pd.DataFrame({k: mean_abs(sv, m) for k, m in subsets.items()})
    # centred: removes the constant offset a subset gets relative to the training baseline,
    # leaving only how much a feature separates rows *within* that subset
    imp_c = pd.DataFrame({k: pd.Series(np.abs(sv[m] - sv[m].mean(axis=0)).mean(axis=0), index=FEATURES)
                          for k, m in subsets.items()})
    offsets = pd.DataFrame({k: pd.Series(sv[m].mean(axis=0), index=FEATURES) for k, m in subsets.items()})

    # logistic regression cross-check: refit on TRAIN (deterministic), verify vs saved preds,
    # then linear contributions coef * (x - train mean) in standardised space = linear SHAP
    tr = split == "train"
    lr = LogReg().fit(X.loc[tr, FEATURES], y[tr])
    p_lr = lr.predict_proba(Xs)
    for s in ["valid", "test"]:
        saved = pd.read_csv(ROOT / f"results/predictions_{s}.csv")["p_logreg"].to_numpy()
        d = float(np.max(np.abs(p_lr[sp == s] - saved)))
        print(f"refit logreg vs saved {s} predictions: max |dp| = {d:.2e}"); assert d < 1e-8
    ct = lr.pipe_.named_steps["ct"]; coef = lr.pipe_.named_steps["lr"].coef_.ravel()
    Z = ct.transform(lr._prep(Xs)); Z = Z.toarray() if hasattr(Z, "toarray") else Z
    names = ct.get_feature_names_out()
    Ztr = ct.transform(lr._prep(X.loc[tr, FEATURES])); Ztr = Ztr.toarray() if hasattr(Ztr, "toarray") else Ztr
    lin = (Z - Ztr.mean(axis=0)) * coef

    def orig(n):
        n = n.split("__", 1)[1]
        for f in ["neighbourhood_grp", "appt_weekday", "sched_weekday"]:
            if n.startswith(f + "_"):
                return f
        return "prior_noshow_rate" if n == "has_history" else n  # has_history folded in
    groups = pd.Series([orig(n) for n in names])
    lin_feat = pd.DataFrame({f: lin[:, (groups == f).to_numpy()].sum(axis=1) for f in FEATURES})
    lr_imp = pd.DataFrame({k: lin_feat[m].abs().mean() for k, m in subsets.items()})

    rank_cmp = {}
    for k in subsets:
        rho, pval = spearmanr(imp[k], lr_imp[k])
        rank_cmp[k] = {"spearman_rho": float(rho), "p_value": float(pval),
                       "lgbm_top5": imp[k].sort_values(ascending=False).index[:5].tolist(),
                       "logreg_top5": lr_imp[k].sort_values(ascending=False).index[:5].tolist()}
    std_coef = {n.split("__", 1)[1]: float(c) for n, c in zip(names, coef) if n.startswith("num__")}

    # grouped comparison: redundant columns merged (SHAP is additive, so group = sum of members)
    GROUPS = {"lead_time (lead_days+same_day)": ["lead_days", "same_day"],
              "history (3 features)": HISTORY_FEATURES}
    gmap = {f: g for g, fs in GROUPS.items() for f in fs}
    gcols = [gmap.get(f, f) for f in FEATURES]
    sv_g = pd.DataFrame(sv, columns=FEATURES).T.groupby(gcols, sort=False).sum().T
    lin_g = lin_feat.T.groupby(gcols, sort=False).sum().T
    imp_g = pd.DataFrame({k: sv_g[m].abs().mean() for k, m in subsets.items()})
    lr_imp_g = pd.DataFrame({k: lin_g[m].abs().mean() for k, m in subsets.items()})
    rank_cmp_grouped = {}
    for k in subsets:
        rho, pval = spearmanr(imp_g[k], lr_imp_g[k])
        rank_cmp_grouped[k] = {"spearman_rho": float(rho), "p_value": float(pval), "n_features": len(imp_g),
                               "lgbm_top5": imp_g[k].sort_values(ascending=False).index[:5].tolist(),
                               "logreg_top5": lr_imp_g[k].sort_values(ascending=False).index[:5].tolist()}

    # figures
    beeswarm(expl, subsets["test_all"], "SHAP, test set, all rows (n=%d)" % subsets["test_all"].sum(),
             FIG / "shap_beeswarm_test_all.png")
    beeswarm(expl, subsets["test_non_same_day"],
             "SHAP (centred), test set, non-same-day rows (n=%d)" % subsets["test_non_same_day"].sum(),
             FIG / "shap_beeswarm_test_non_same_day.png", centre=True)
    beeswarm(expl, subsets["test_same_day"],
             "SHAP (centred), test set, same-day rows (n=%d)" % subsets["test_same_day"].sum(),
             FIG / "shap_beeswarm_test_same_day.png", centre=True)
    beeswarm(expl, subsets["valid_all"], "SHAP, validation set, all rows (n=%d)" % subsets["valid_all"].sum(),
             FIG / "shap_beeswarm_valid_all.png")
    beeswarm(expl, subsets["valid_non_same_day"],
             "SHAP (centred), validation set, non-same-day rows (n=%d)" % subsets["valid_non_same_day"].sum(),
             FIG / "shap_beeswarm_valid_non_same_day.png", centre=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    panels = [(axes[0], imp["test_all"], "All test rows: mean |SHAP|"),
              (axes[1], imp_c["test_non_same_day"], "Non-same-day test rows: mean |SHAP - subset mean|")]
    for ax, ser, ttl in panels:
        s = ser.sort_values()
        ax.barh(s.index, s.values, color="#2a78d6", height=0.6)
        for i, v in enumerate(s.values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_title(ttl, loc="left", color=INK, fontsize=10.5); ax.set_xlabel("log-odds")
        ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    fig.suptitle("What the LightGBM model relies on (associations, not causes)", x=0.01, ha="left", color=INK)
    off = offsets.loc["lead_days", "test_non_same_day"]
    fig.text(0.01, 0.01, f"Right panel is centred: in non-same-day rows lead_days carries a constant offset of {off:+.2f} "
             "log-odds versus the training baseline (which includes same-day bookings); it is removed so the bars show "
             "only within-group separation. same_day = 0 because the model splits on lead_days = 0 instead.",
             fontsize=8, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(FIG / "shap_importance_all_vs_non_same_day.png", dpi=160); plt.close(fig)

    out = {
        "model": "models/lgbm.txt (frozen; final-run LightGBM)", "base_value_log_odds": base,
        "checks": {"additivity_max_abs_err": add_err, "treeexplainer_vs_pred_contrib_max_abs_diff": lgb_err,
                   "frozen_vs_saved_predictions_max_abs_dp": cons},
        "rows_explained": {k: int(m.sum()) for k, m in subsets.items()},
        "lgbm_mean_abs_shap": imp.round(5).to_dict(),
        "logreg_mean_abs_linear_contribution": lr_imp.round(5).to_dict(),
        "rank_agreement_lgbm_vs_logreg": rank_cmp,
        "lgbm_mean_abs_centred_shap": imp_c.round(5).to_dict(),
        "lgbm_mean_shap_offset_by_subset": offsets.round(5).to_dict(),
        "lgbm_mean_abs_shap_grouped": imp_g.round(5).to_dict(),
        "logreg_mean_abs_linear_contribution_grouped": lr_imp_g.round(5).to_dict(),
        "rank_agreement_grouped": rank_cmp_grouped,
        "logreg_standardised_coefficients_numeric": std_coef,
        "note": "SHAP shows what the model uses, not causal effects. lead_days and same_day are redundant "
                "(same_day = lead_days==0), so credit is split between them. LR has_history is folded into "
                "prior_noshow_rate.",
    }
    (ROOT / "results/shap_summary.json").write_text(json.dumps(out, indent=2))

    pd.set_option("display.width", 160)
    print("\nLightGBM mean |SHAP| (log-odds):"); print(imp.sort_values("test_all", ascending=False).round(4).to_string())
    print("\nLightGBM mean |SHAP - subset mean| (centred, within-subset spread):")
    print(imp_c.sort_values("test_non_same_day", ascending=False).round(4).to_string())
    print("\nLogReg mean |linear contribution| (log-odds):"); print(lr_imp.sort_values("test_all", ascending=False).round(4).to_string())
    print("\nrank agreement (16 raw features):")
    for k, v in rank_cmp.items():
        print(f"  {k:20s} rho={v['spearman_rho']:.3f} p={v['p_value']:.3f}")
    both = pd.concat({"lgbm_SHAP": imp_g, "logreg_linear": lr_imp_g}, axis=1)
    print("\nGROUPED mean |contribution| (log-odds):")
    print(both.sort_values(("lgbm_SHAP", "test_all"), ascending=False).round(4).to_string())
    print("\nrank agreement (grouped, %d features):" % len(imp_g))
    for k, v in rank_cmp_grouped.items():
        print(f"  {k:20s} rho={v['spearman_rho']:.3f} p={v['p_value']:.4f}  lgbm top5={v['lgbm_top5']}  logreg top5={v['logreg_top5']}")
    print("\nLR standardised coefficients (numeric):", json.dumps({k: round(v, 3) for k, v in std_coef.items()}))
    print("\nwrote results/shap_summary.json and reports/figures/shap_*.png")


if __name__ == "__main__":
    main(sys.argv[1])
