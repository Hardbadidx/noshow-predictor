"""Shared matplotlib style (reference data-viz palette, light mode)."""
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = {"lgbm": "#2a78d6", "logreg": "#eb6834", "lead_only": "#1baf7a"}  # categorical slots 1-3
LABELS = {"lgbm": "LightGBM", "logreg": "Logistic regression", "lead_only": "Lead-time only"}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 12, "lines.linewidth": 2, "legend.frameon": False,
        "text.color": INK,
    })
