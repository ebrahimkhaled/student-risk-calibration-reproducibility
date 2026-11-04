"""Step 10: the headline numbers, with bootstrap intervals.

For each model, pooled over the nine new cohorts (test students):
  - ranking: AUC, and its difference from logistic regression
  - the 70% question: among students the system scores 65-75%, the promised risk (their mean
    score) against the delivered risk (the share who actually failed or withdrew)
  - the flag: among students scored 70% or more, promised against delivered
  - average calibration: mean score minus observed rate

Intervals: 2,000 bootstrap resamples of students within each cohort (cohort sizes kept), the
same resamples for every model so differences between models are paired.

    python 10_headline.py stable
"""
import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FSET = sys.argv[1] if len(sys.argv) > 1 else "stable"
PROJECT = Path(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"))
RES = PROJECT / "results"
OUT = RES / "summary"
MODELS = ["lr", "ridge", "lgbm", "tabpfn", "tabicl"]
B = 2000
BAND = (0.65, 0.75)
FLAG = 0.70

frames = []
for d in sorted((RES / FSET).glob("*_20*")):
    p = pd.read_csv(d / "preds.csv.gz")
    p["pair"] = d.name
    frames.append(p)
df = pd.concat(frames, ignore_index=True)
df = df.dropna(subset=MODELS)       # a model that failed on a cohort drops that cohort for all
pairs = df.pair.unique()
print(f"{FSET}: {len(pairs)} cohorts, {len(df)} students, {int(df.y.sum())} at risk")


def stats(d):
    out = {}
    y = d.y.to_numpy()
    for m in MODELS:
        s = d[m].to_numpy()
        band = (s >= BAND[0]) & (s < BAND[1])
        flag = s >= FLAG
        # AUC within each cohort, then averaged: the pooled AUC would also reward ranking cohorts
        out[(m, "auc")] = np.mean([roc_auc_score(g.y, g[m]) for _, g in d.groupby("pair")])
        out[(m, "band_n")] = band.sum()
        out[(m, "band_promised")] = s[band].mean()
        out[(m, "band_delivered")] = y[band].mean()
        out[(m, "flag_n")] = flag.sum()
        out[(m, "flag_promised")] = s[flag].mean()
        out[(m, "flag_delivered")] = y[flag].mean()
        out[(m, "mean_gap")] = s.mean() - y.mean()
    for m in MODELS:
        out[(m, "auc_vs_lr")] = out[(m, "auc")] - out[("lr", "auc")]
        out[(m, "band_gap")] = out[(m, "band_promised")] - out[(m, "band_delivered")]
        out[(m, "flag_gap")] = out[(m, "flag_promised")] - out[(m, "flag_delivered")]
    return out


point = stats(df)
rng = np.random.default_rng(20261104)
idx = {k: np.flatnonzero(df.pair.to_numpy() == k) for k in pairs}
boots = []
for b in range(B):
    take = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in idx.values()])
    boots.append(stats(df.iloc[take]))
boots = pd.DataFrame(boots)

rows = []
for (m, k), v in point.items():
    lo, hi = np.nanpercentile(boots[(m, k)], [2.5, 97.5])
    rows.append(dict(feature_set=FSET, model=m, measure=k, estimate=v, lo=lo, hi=hi))
tab = pd.DataFrame(rows)
tab.to_csv(OUT / f"headline_{FSET}.csv", index=False)

pd.set_option("display.width", 200)
show = tab.assign(txt=lambda t: t.apply(
    lambda r: f"{r.estimate:.0f}" if r.measure.endswith("_n") else f"{r.estimate:.3f} [{r.lo:.3f}, {r.hi:.3f}]", axis=1))
print(show.pivot(index="measure", columns="model", values="txt")[MODELS].to_string())
json.dump({"band": BAND, "flag": FLAG, "B": B, "seed": 20261104, "students": len(df), "cohorts": len(pairs)},
          open(OUT / f"headline_{FSET}.json", "w"), indent=1)
