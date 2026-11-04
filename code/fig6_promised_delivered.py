"""Figure 6: when the system says 70%, how many students actually fail or withdraw?

(a) For every model, 100 students it scored between 65% and 75% (pooled over the nine new
    cohorts): one square per student, filled if the student actually failed or withdrew. The line
    under the seventh row is the promise: 70 of 100.
(b) The same question cohort by cohort, with 95% intervals: the promise holds in some courses and
    breaks in others, for every model at once.

    python fig6_promised_delivered.py stable
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from statsmodels.stats.proportion import proportion_confint

import figstyle as fs

FSET = sys.argv[1] if len(sys.argv) > 1 else "stable"
PROJECT = Path(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"))
RES = PROJECT / "results"
FIG = PROJECT / "figures"
FIG.mkdir(exist_ok=True)
LO, HI = 0.65, 0.75

head = pd.read_csv(RES / "summary" / f"headline_{FSET}.csv").set_index(["model", "measure"])
preds = pd.concat([pd.read_csv(d / "preds.csv.gz").assign(pair=d.name)
                   for d in sorted((RES / FSET).glob("*_20*"))], ignore_index=True)

fs.use()
fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, 4.3))
top = fig.add_gridspec(1, 5, left=0.02, right=0.98, top=0.885, bottom=0.585, wspace=0.18)
bot = fig.add_gridspec(1, 1, left=0.085, right=0.98, top=0.40, bottom=0.085)

# ------------------------------------------------------------------ (a) 100 students per model
for i, m in enumerate(fs.MODELS):
    ax = fig.add_subplot(top[0, i])
    delivered = head.loc[(m, "band_delivered"), "estimate"]
    lo, hi = head.loc[(m, "band_delivered"), ["lo", "hi"]]
    k = int(round(100 * delivered))
    for c in range(100):
        r, col = divmod(c, 10)
        ax.add_patch(Rectangle((col + 0.08, 9 - r + 0.08), 0.84, 0.84, lw=0,
                               facecolor=fs.COLOUR[m] if c < k else fs.LIGHT))
    ax.plot([-0.15, 10.15], [3, 3], color="#222222", lw=1.3, solid_capstyle="butt")
    ax.set_xlim(-0.2, 10.2); ax.set_ylim(-0.2, 10.2); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(fs.LABEL[m] if m != "ridge" else "Ridge", fontsize=8.5, pad=3,
                 fontweight="bold" if m in fs.FOUNDATION else "normal")
    ax.text(5, -0.9, f"{k} of 100 failed", ha="center", va="top", fontsize=8)
    ax.text(5, -2.2, f"(95% CI {100 * lo:.0f}\u2013{100 * hi:.0f})", ha="center", va="top", fontsize=7.5, color=fs.GREY)
fig.text(0.02, 0.975, "a", fontweight="bold", fontsize=10, va="top")
fig.text(0.05, 0.975, "100 students the system scored about 70% at risk. The line marks the promise: 70 should fail.",
         fontsize=8.5, va="top")

# ------------------------------------------------------------------ (b) cohort by cohort
ax = fig.add_subplot(bot[0, 0])
pairs = list(fs.COHORT)
width = 0.13
for j, m in enumerate(fs.MODELS):
    xs, ys, los, his = [], [], [], []
    for x, pr in enumerate(pairs):
        g = preds[(preds.pair == pr) & preds[m].between(LO, HI, inclusive="left")]
        if len(g) < 5:
            continue
        k, n = int(g.y.sum()), len(g)
        l, h = proportion_confint(k, n, method="wilson")
        xs.append(x + (j - 2) * width); ys.append(k / n); los.append(k / n - l); his.append(h - k / n)
    ax.errorbar(xs, ys, yerr=[los, his], fmt=fs.MARKER[m], ms=3.8, color=fs.COLOUR[m], mec="white", mew=0.4,
                elinewidth=0.7, capsize=0, label=fs.LABEL[m], zorder=3)
ax.axhline(0.70, color="#222222", lw=1.0, zorder=2)
ax.text(1, 0.705, "promised: 70%", ha="center", va="bottom", fontsize=7.5)
for x in range(len(pairs)):
    if x % 2 == 0:
        ax.axvspan(x - 0.5, x + 0.5, color="#F4F4F4", lw=0, zorder=0)
ax.set_xticks(range(len(pairs)), [fs.COHORT[p] for p in pairs])
ax.set_xlim(-0.5, len(pairs) - 0.5)
ax.set_ylim(0.25, 1.0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
ax.set_ylabel("Actually failed\nor withdrew")
ax.set_xlabel("New cohort (course, start month), 2014")
ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=1.0,
          fontsize=7.5)
fig.text(0.02, 0.475, "b", fontweight="bold", fontsize=10, va="top")

for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig6_promised_delivered_{FSET}.{ext}", dpi=300)
print("written", FIG / f"fig6_promised_delivered_{FSET}.pdf")
