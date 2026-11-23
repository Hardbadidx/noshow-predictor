"""Reliability, precision-recall, cumulative-gains and decile-lift plots.

Reads only saved files: results/predictions_{valid,test}.csv and results/metrics.json.
No model is loaded and nothing is re-scored; thresholds come from metrics.json
(chosen on validation in the final run).

Usage: python scripts/plot_eval.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from noshow.plotstyle import INK, INK2, LABELS, SERIES, apply_style  # noqa: E402

FIG = ROOT / "reports" / "figures"
MODELS = ["lgbm", "logreg", "lead_only"]


def reliability(ax, d, title):
    for m in MODELS:
        b = pd.qcut(d[f"p_{m}"], 10, duplicates="drop")
        t = d.groupby(b, observed=True).agg(pred=(f"p_{m}", "mean"), obs=("no_show", "mean"))
        ax.plot(t["pred"], t["obs"], marker="o", ms=5, color=SERIES[m], label=LABELS[m])
    ax.plot([0, 0.5], [0, 0.5], ls="--", lw=1, color=INK2)
    ax.text(0.44, 0.465, "perfect", color=INK2, fontsize=8, rotation=38)
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.5)
    ax.set_xlabel("mean predicted no-show probability (decile)"); ax.set_ylabel("observed no-show rate")
    ax.set_title(f"{title}\nobserved rate {d.no_show.mean():.1%}, mean LightGBM prediction {d.p_lgbm.mean():.1%}",
                 loc="left", fontsize=10.5)


def gains(d, score):
    """Tie-aware cumulative gains: one point per distinct score threshold."""
    g = d.groupby(score, sort=True)["no_show"].agg(["size", "sum"]).iloc[::-1]
    x = np.r_[0, g["size"].cumsum() / len(d)]; y = np.r_[0, g["sum"].cumsum() / d.no_show.sum()]
    return x, y


def main():
    apply_style(); FIG.mkdir(parents=True, exist_ok=True)
    r = json.loads((ROOT / "results/metrics.json").read_text())
    va = pd.read_csv(ROOT / "results/predictions_valid.csv")
    te = pd.read_csv(ROOT / "results/predictions_test.csv")
    nsd = te[te.same_day == 0]

    # 1. reliability: validation vs test (shows drift-induced miscalibration)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    reliability(axes[0], va, "Validation (20-31 May)")
    reliability(axes[1], te, "Test (1-8 June)")
    axes[1].legend(loc="upper left")
    fig.suptitle("Calibration: predicted vs observed no-show rate by risk decile", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "reliability_valid_vs_test.png", dpi=160); plt.close(fig)

    # 2. precision-recall, all rows and non-same-day
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, d, ttl in [(axes[0], te, "All test rows"), (axes[1], nsd, "Non-same-day test rows")]:
        for m in MODELS:
            p, rc, _ = precision_recall_curve(d.no_show, d[f"p_{m}"])
            p, rc = p[:-1], rc[:-1]  # drop sklearn's synthetic (recall 0, precision 1) end point
            ax.plot(rc, p, color=SERIES[m], marker="o" if m == "lead_only" else None, ms=4, label=f"{LABELS[m]} (PR-AUC {average_precision_score(d.no_show, d[f'p_{m}']):.3f})")
        prev = d.no_show.mean()
        ax.axhline(prev, ls="--", lw=1, color=INK2)
        ax.text(0.01, prev - 0.035, f"prevalence {prev:.3f}", ha="left", fontsize=8, color=INK2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("recall (share of no-shows found)"); ax.set_ylabel("precision")
        ax.set_title(f"{ttl} (n={len(d):,})", loc="left", fontsize=10.5); ax.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Precision-recall, test set", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "precision_recall_test.png", dpi=160); plt.close(fig)

    # 3. cumulative gains with the validation-chosen LightGBM threshold marked
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, d, ttl, key in [(axes[0], te, "All test rows", "top20_all_rows"),
                            (axes[1], nsd, "Non-same-day test rows", "top20_non_same_day")]:
        for m in MODELS:
            x, y = gains(d, f"p_{m}")
            ax.plot(x, y, color=SERIES[m], label=LABELS[m])
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
        ax.text(0.62, 0.56, "random", color=INK2, fontsize=8, rotation=38)
        t = r["models"]["lgbm"][key]
        ax.scatter([t["flagged_share"]], [t["noshows_captured_share"]], s=60, color=SERIES["lgbm"],
                   edgecolor="#fcfcfb", linewidth=2, zorder=5)
        ax.annotate(f"validation-chosen threshold:\nflags {t['flagged_share']:.1%}, captures {t['noshows_captured_share']:.1%}",
                    (t["flagged_share"], t["noshows_captured_share"]), xytext=(0.30, 0.15),
                    textcoords="data", fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("share of appointments flagged (highest risk first)"); ax.set_ylabel("share of no-shows captured")
        ax.set_title(f"{ttl} (n={len(d):,})", loc="left", fontsize=10.5); ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Cumulative gains, test set", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "cumulative_gains_test.png", dpi=160); plt.close(fig)

    # 4. lift by risk decile (observed rate / overall rate), deciles of each model's own test scores
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.38; prev = te.no_show.mean(); rows = {}
    for i, m in enumerate(["lgbm", "logreg"]):
        dec = pd.qcut(te[f"p_{m}"].rank(method="first"), 10, labels=range(1, 11))
        lift = te.groupby(dec, observed=True)["no_show"].mean() / prev
        rows[m] = lift
        ax.bar(np.arange(10) + (i - 0.5) * w, lift.values, width=w - 0.04, color=SERIES[m], label=LABELS[m])
    ax.axhline(1, ls="--", lw=1, color=INK2); ax.text(-0.45, 1.05, "no lift", fontsize=8, color=INK2)
    for i, v in enumerate(rows["lgbm"].values):
        if i in (0, 9):
            ax.text(i - w / 2, v + 0.04, f"{v:.2f}×", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(10), [f"D{k}" for k in range(1, 11)])
    ax.set_xlabel("predicted-risk decile (D10 = highest risk)"); ax.set_ylabel("lift (observed rate / overall rate)")
    ax.set_title(f"Lift by risk decile, all test rows (overall no-show rate {prev:.1%})", loc="left")
    ax.legend(loc="upper left"); ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "lift_by_decile_test.png", dpi=160); plt.close(fig)

    summary = {m: [round(float(v), 3) for v in rows[m].values] for m in rows}
    (ROOT / "results/lift_by_decile_test.json").write_text(json.dumps(summary, indent=1))
    print("lift by decile D1..D10:", json.dumps(summary))
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
