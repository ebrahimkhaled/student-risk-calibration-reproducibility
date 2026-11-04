"""Figure 4: calibration curves on next year's students, every model.

Promised risk (the model's score) against the share who actually failed or withdrew, all nine new
cohorts together. The curve is a logistic regression of the outcome on a natural cubic spline of
the score's logit (4 degrees of freedom); the band is 95% over 200 bootstrap resamples of students
within cohorts. On the diagonal, a score of 70% means 70% fail. The strip at the bottom shows how
the scores are spread. Thin grey lines: each of the nine cohorts on its own.

    python fig4_calibration.py stable
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

import figstyle as fs

FSET = sys.argv[1] if len(sys.argv) > 1 else "stable"
PROJECT = Path(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"))
RES = PROJECT / "results"
FIG = PROJECT / "figures"
B = 200
GRID = np.linspace(0.05, 0.95, 91)

preds = pd.concat([pd.read_csv(d / "preds.csv.gz").assign(pair=d.name)
                   for d in sorted((RES / FSET).glob("*_20*"))], ignore_index=True)
preds = preds.dropna(subset=fs.MODELS).reset_index(drop=True)
logit = lambda p: np.log(p / (1 - p))   # noqa: E731


def curve(y, p):
    z = logit(np.clip(p, 1e-4, 1 - 1e-4))
    lo, hi = np.percentile(z, [1, 99])
    spec = "cr(z, df=4, lower_bound=lb, upper_bound=ub, constraints='center') - 1"
    X = dmatrix(spec, {"z": np.clip(z, lo, hi), "lb": lo, "ub": hi}, return_type="dataframe")
    f = sm.GLM(y, sm.add_constant(X.to_numpy()), family=sm.families.Binomial()).fit()
    zg = np.clip(logit(GRID), lo, hi)
    Xg = dmatrix(X.design_info, {"z": zg, "lb": lo, "ub": hi}, return_type="dataframe")
    out = f.predict(sm.add_constant(Xg.to_numpy(), has_constant="add"))
    out[(GRID < np.quantile(p, 0.01)) | (GRID > np.quantile(p, 0.99))] = np.nan   # no extrapolation
    return out


rng = np.random.default_rng(20261104)
idx = [np.flatnonzero(preds.pair.to_numpy() == k) for k in preds.pair.unique()]
draws = [np.concatenate([rng.choice(v, len(v)) for v in idx]) for _ in range(B)]

fs.use()
fig, axes = plt.subplots(1, 5, figsize=(fs.TEXT_WIDTH_IN, 1.95), sharey=True, gridspec_kw=dict(wspace=0.25))
y = preds.y.to_numpy()
for ax, m in zip(axes, fs.MODELS):
    p = preds[m].to_numpy()
    est = curve(y, p)
    bs = np.array([curve(y[t], p[t]) for t in draws])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    ax.plot([0, 1], [0, 1], color=fs.GREY, lw=0.7, ls=(0, (3, 2)))
    # each course on its own: the pooled curve hides how far single cohorts miss
    for v in idx:
        ax.plot(GRID, curve(y[v], p[v]), color="#9A9A9A", lw=0.45, alpha=0.8, zorder=1)
    ax.fill_between(GRID, lo, hi, color=fs.COLOUR[m], alpha=0.25, lw=0)
    ax.plot(GRID, est, color=fs.COLOUR[m], lw=1.6)
    # where the scores fall
    h, e = np.histogram(p, bins=40, range=(0, 1))
    ax.bar(e[:-1], 0.10 * h / h.max(), width=e[1] - e[0], align="edge", color=fs.COLOUR[m], alpha=0.45, lw=0)
    ax.axvline(0.7, color="#222222", lw=0.5, ymax=0.7, ls=":")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1], ["0", "50", "100"])
    ax.set_yticks([0, 0.5, 1], ["0", "50", "100"])
    ax.set_title(fs.LABEL[m] if m != "ridge" else "Ridge", fontsize=8.5,
                 fontweight="bold" if m in fs.FOUNDATION else "normal")
axes[0].set_ylabel("Actually failed (%)")
axes[2].set_xlabel("Promised risk, the model's score (%)")

for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig4_calibration_{FSET}.{ext}", dpi=300)
print("written", FIG / f"fig4_calibration_{FSET}.pdf")
