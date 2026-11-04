"""Figure 5: what each check said, model by model, out of the nine new cohorts.

Each cell counts the cohorts in which that check raised an alarm (p < 0.05). The first column is
the builder's own check on last year's data, which exists only for the regression models; the
rest are checks on next year's students, which apply to every model.

    python fig5_verdicts.py stable
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

import figstyle as fs

FSET = sys.argv[1] if len(sys.argv) > 1 else "stable"
PROJECT = Path(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"))
SUM = PROJECT / "results" / "summary"
FIG = PROJECT / "figures"
ca = pd.read_csv(SUM / "check_a_all.csv")
ca = ca[ca.feature_set == FSET]
cb = pd.read_csv(SUM / "check_b_all.csv")
cb = cb[cb.feature_set == FSET]
NP = ca.pair.nunique()


def slope_p(y, p):
    """Wald test that the calibration slope is 1: logistic regression of the outcome on logit(p)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    f = sm.GLM(np.asarray(y), sm.add_constant(np.log(p / (1 - p))), family=sm.families.Binomial()).fit()
    b, se = np.asarray(f.params)[1], np.asarray(f.bse)[1]
    return float(2 * norm.sf(abs((b - 1) / se)))


# the "Calibration slope" rows in check A are descriptive (no p-value), so the test is done here
SLOPE = []
for d in sorted((PROJECT / "results" / FSET).glob("*_20*")):
    pr = pd.read_csv(d / "preds.csv.gz")
    for m in fs.MODELS:
        if m in pr:
            SLOPE.append(dict(pair=d.name, model=m, test="slope = 1 (Wald)", p_value=slope_p(pr.y, pr[m])))
ca = pd.concat([ca, pd.DataFrame(SLOPE)], ignore_index=True)

# (column label, source, test name per model)
OWN = {"lr": "HL", "ridge": "Hosmer-Lemeshow (shrinkage-corrected)"}
COLS = [("Own data,\nlast year", "b", None),
        ("Right on\naverage", "a", "Calibration-in-the-large"),
        ("Right\nspread", "a", "slope = 1 (Wald)"),
        ("Hosmer\u2013\nLemeshow", "a", "Hosmer-Lemeshow (external)"),
        ("EDGE", "a", "edge.gof ten groups (G = 10)"),
        ("le\nCessie", "a", "le Cessie (external)"),
        ("Deep-\nGOF-1", "a", "DeepGOF-1 external (combined)"),
        ("Locali-\nzation", "a", "Localize (any part named)")]


def count(m, src, test):
    if src == "b":
        if m not in OWN:
            return None
        p = cb[(cb.model == m) & (cb.test == OWN[m])].p_value
    else:
        p = ca[(ca.model == m) & (ca.test == test)].p_value
    assert len(p) == NP and p.notna().all(), (m, test, len(p))
    return int((p < 0.05).sum())


def colour(k):
    if k is None:
        return "white"
    return "#CDE8D3" if k <= 2 else ("#FBE3B3" if k <= 5 else "#F4B9A8")


fs.use()
fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH_IN, 2.55))
rows = fs.MODELS
for i, m in enumerate(rows):
    for j, (lab, src, test) in enumerate(COLS):
        k = count(m, src, test)
        x = j + (0.35 if j > 0 else 0)          # a gap after the own-data column
        ax.add_patch(plt.Rectangle((x, len(rows) - 1 - i), 0.94, 0.9, facecolor=colour(k),
                                   edgecolor="#BBBBBB" if k is None else "none", lw=0.6,
                                   ls=(0, (2, 2)) if k is None else "-"))
        ax.text(x + 0.47, len(rows) - 1 - i + 0.45, "no such\ncheck" if k is None else f"{k}/{NP}",
                ha="center", va="center", fontsize=7.5 if k is None else 9,
                color=fs.GREY if k is None else "#222222", fontweight="normal")
for i, m in enumerate(rows):
    ax.text(-0.12, len(rows) - 1 - i + 0.45, fs.LABEL[m], ha="right", va="center", fontsize=8.5,
            fontweight="bold" if m in fs.FOUNDATION else "normal")
for j, (lab, *_r) in enumerate(COLS):
    x = j + (0.35 if j > 0 else 0)
    ax.text(x + 0.47, len(rows) + 0.08, lab, ha="center", va="bottom", fontsize=7.5)
ax.text(0.47, len(rows) + 0.95, "Builder's check", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.text(1.35 + (len(COLS) - 1) / 2 - 0.03, len(rows) + 0.95, "Checks on next year's students",
        ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.plot([1.35, len(COLS) + 0.29], [len(rows) + 0.9] * 2, color="#222222", lw=0.6)
ax.plot([0, 0.94], [len(rows) + 0.9] * 2, color="#222222", lw=0.6)
ax.set_xlim(-2.6, len(COLS) + 0.4)
ax.set_ylim(-0.1, len(rows) + 1.4)
ax.axis("off")
# key, with what a cell counts
ax.text(0.94, -0.48, "Each cell: courses with an alarm, out of 9", ha="right", va="center", fontsize=7.5,
        fontweight="bold")
for x0, (lab, c) in zip([1.35, 4.05, 5.15], [("0\u20132 alarms", "#CDE8D3"), ("3\u20135", "#FBE3B3"),
                                              ("6\u20139", "#F4B9A8")]):
    ax.add_patch(plt.Rectangle((x0, -0.62), 0.28, 0.28, facecolor=c, clip_on=False))
    ax.text(x0 + 0.36, -0.48, lab, va="center", fontsize=7.5)

for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig5_verdicts_{FSET}.{ext}", dpi=300)
print("written", FIG / f"fig5_verdicts_{FSET}.pdf")
