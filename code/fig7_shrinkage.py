"""Figure 7: after ridge, the routine check raises false alarms; the corrected check does not.

(a) The model is correct by construction, so every alarm is false. Share of 1,000 simulated
    cohorts flagged at the 5% level, against how strongly ridge shrinks (multiples of the
    strength cross-validation chose on the real cohort).
(b) The model leaves out a real curved effect. With no shrinkage both checks are valid, so that
    rate is the honest chance of catching the problem; at the cross-validated strength the
    corrected check keeps most of it.

    python fig7_shrinkage.py
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

PROJECT = Path(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"))
# keep_default_na=False: otherwise pandas reads the condition name "null" as a missing value
S = pd.read_csv(PROJECT / "results" / "plasmode" / "summary.csv", keep_default_na=False)
FIG = PROJECT / "figures"
ROUTINE, CORRECTED = "#CC79A7", "#000000"
TEST = {"hl": ("HL", "o"), "edge": ("EDGE", "s")}

fs.use()
fig, (a, b) = plt.subplots(1, 2, figsize=(fs.TEXT_WIDTH_IN, 2.75), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))

# ------------------------------------------------------------------ (a) false alarms
nul = S[S.truth == "null"].sort_values("mult")
x = np.arange(len(nul))
for t, (name, mk) in TEST.items():
    for kind, col, ls in (("routine", ROUTINE, "--"), ("corrected", CORRECTED, "-")):
        r, se = nul[f"{t}_{kind}"].to_numpy(), nul[f"{t}_{kind}_se"].to_numpy()
        a.fill_between(x, np.clip(r - 1.96 * se, 0, 1), np.clip(r + 1.96 * se, 0, 1), color=col, alpha=0.12, lw=0)
        a.plot(x, r, ls=ls, marker=mk, color=col, ms=4, lw=1.1, mfc="white" if kind == "routine" else col,
               label=f"{name}, {kind}")
a.axhline(0.05, color=fs.GREY, lw=0.7, ls=":")
a.text(3.1, 0.085, "5%: what a valid check gives", ha="left", va="bottom", fontsize=7.5, color=fs.GREY)
a.set_xticks(x, ["none" if m == 0 else f"\u00d7{m:g}" for m in nul.mult])
a.set_xlabel("Ridge shrinkage (\u00d71 = the strength cross-validation chose)")
a.set_ylabel("False alarms (share of\ncorrect models flagged)")
a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
a.set_ylim(0, 1.03)
a.legend(loc="upper left", fontsize=7.5, handlelength=2.2)
a.set_title("a   The model is correct", loc="left", fontweight="bold")

# ------------------------------------------------------------------ (b) catching a real problem
mis = S[S.truth == "misfit"]
strengths = sorted(mis.strength.unique())
w = 0.13
for i, t in enumerate(TEST):
    name, mk = TEST[t]
    for j, s in enumerate(strengths):
        x0 = mis[(mis.strength == s) & (mis["mult"] == 0)].iloc[0]
        x1 = mis[(mis.strength == s) & (mis["mult"] == 1)].iloc[0]
        base = j + (i - 0.5) * 3.2 * w
        bars = [(x0[f"{t}_corrected"], x0[f"{t}_corrected_se"], fs.LIGHT, fs.LIGHT, "no shrinkage (honest)"),
                (x1[f"{t}_corrected"], x1[f"{t}_corrected_se"], CORRECTED, CORRECTED, "ridge, corrected"),
                (x1[f"{t}_routine"], x1[f"{t}_routine_se"], "white", ROUTINE, "ridge, routine")]
        for k, (r, se, fc, ec, lab) in enumerate(bars):
            b.bar(base + (k - 1) * w, r, w * 0.92, color=fc, edgecolor=ec, lw=0.8,
                  hatch="////" if fc == "white" else None, label=lab if (i == 0 and j == 0) else None)
            b.errorbar(base + (k - 1) * w, r, yerr=1.96 * se, color="#444444", lw=0.6, capsize=0)
for j in range(len(strengths)):
    for i, t in enumerate(TEST):
        b.text(j + (i - 0.5) * 3.2 * w, -0.02, "HL" if t == "hl" else "EDGE", ha="center", va="top",
               fontsize=7.5, color=fs.GREY)
b.set_xticks(range(len(strengths)), ["small", "medium", "large"])
b.tick_params(axis="x", pad=14, length=0)
b.set_xlabel("Size of the missed effect")
b.set_ylabel("Problem caught")
b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
b.set_ylim(0, 1.03)
b.legend(loc="upper left", fontsize=7.5)
b.set_title("b   The model misses a real effect", loc="left", fontweight="bold")

for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig7_shrinkage.{ext}", dpi=300)
print("written", FIG / "fig7_shrinkage.pdf")
