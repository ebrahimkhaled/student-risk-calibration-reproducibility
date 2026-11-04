"""Figure 3: ranking ability and trustworthiness, every model, nine new cohorts.

(a) Ranking: AUC, the chance the model scores a student who later failed above one who passed.
(b) Trust: the average gap, in percentage points, between the risk the model promised and the
    share who actually failed (integrated calibration index on the new cohort).
(c) Direction: mean promised risk minus the share who actually failed. Above zero = too alarmist.

Small grey points are the nine cohorts; the large marker is their average with a 95% t interval
across cohorts.

    python fig3_headline.py stable
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

FSET = sys.argv[1] if len(sys.argv) > 1 else "stable"
PROJECT = Path(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"))
SUM = PROJECT / "results" / "summary"
FIG = PROJECT / "figures"
met = pd.read_csv(SUM / "metrics_all.csv")
met = met[met.feature_set == FSET]

fs.use()
fig, axes = plt.subplots(1, 3, figsize=(fs.TEXT_WIDTH_IN, 2.5), sharey=True,
                         gridspec_kw=dict(wspace=0.12))
y = np.arange(len(fs.MODELS))[::-1]
panels = [("auc", 1, "a   Ranking (AUC)", "higher is better"),
          ("emp_ici", 100, "b   Average miss (points)", "lower is better"),
          ("emp_citl_prob", 100, "c   Promised minus actual", "0 = right on average")]
rng = np.random.default_rng(3)
for ax, (col, k, title, sub) in zip(axes, panels):
    for yi, m in zip(y, fs.MODELS):
        v = met[met.model == m][col].to_numpy() * k
        ax.scatter(v, yi + rng.uniform(-0.18, 0.18, len(v)), s=7, color="#BBBBBB", lw=0, zorder=2)
        # average over the cohorts with a t interval across cohorts: the question is how a model
        # does from one course to the next, so the courses are the units
        est = v.mean(); h = 2.306 * v.std(ddof=1) / np.sqrt(len(v)); lo, hi = est - h, est + h
        ax.plot([lo, hi], [yi, yi], color=fs.COLOUR[m], lw=1.6, zorder=3, solid_capstyle="butt")
        ax.plot(est, yi, fs.MARKER[m], color=fs.COLOUR[m], ms=6, mec="white", mew=0.6, zorder=4)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(0, -0.13, sub, transform=ax.transAxes, fontsize=7.5, color=fs.GREY, va="top")
    ax.grid(axis="x", color="#EEEEEE", lw=0.6, zorder=0)
    ax.tick_params(axis="y", length=0)
axes[2].axvline(0, color="#222222", lw=0.9, zorder=2.5)
axes[0].set_yticks(y, [fs.LABEL[m] for m in fs.MODELS])
for t, m in zip(axes[0].get_yticklabels(), fs.MODELS):
    if m in fs.FOUNDATION:
        t.set_fontweight("bold")
axes[0].set_ylim(-0.6, len(fs.MODELS) - 0.4)

for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig3_headline_{FSET}.{ext}", dpi=300)
print("written", FIG / f"fig3_headline_{FSET}.pdf")
