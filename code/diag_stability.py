# -*- coding: utf-8 -*-
"""Pilot diagnostic: does a feature mean the same thing in both years of a module?

For each of the nine year-on-year pairs, how far each feature group moved between
the training cohort and the test cohort, and whether the first assessment changed
its marking (share of full marks). A feature whose meaning changes between years
cannot be used to predict next year, however well it predicted last year.
"""
import os
import json
import numpy as np
import pandas as pd

P = os.path.join(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"), "data")
df = pd.read_csv(f"{P}/features/day28_features.csv.gz")
g = json.load(open(f"{P}/features/feature_groups.json"))
pairs = pd.read_csv(f"{P}/scope/year_on_year_pairs.csv")
for c in g["cont"]:
    if c.startswith("clicks_"):
        df[c] = np.log1p(df[c])

groups = {
    "click types": [c for c in g["cont"] if c.startswith("clicks_")
                    and c not in ("clicks_total", "clicks_last_week")],
    "click volume": ["clicks_total", "clicks_last_week", "active_days"],
    "assess score": ["assess_mean_score"],
    "assess behaviour": ["assess_submitted", "assess_missed", "assess_mean_days_late"],
    "background": ["num_of_prev_attempts", "studied_credits", "date_registration",
                   "gender", "disability"],
}


def smd(a, b):
    s = np.sqrt((a.var() + b.var()) / 2)
    return 0.0 if s == 0 else (b.mean() - a.mean()) / s


rows = []
for _, r in pairs.iterrows():
    tr = df[(df.code_module == r.module) & (df.code_presentation == r.train)]
    te = df[(df.code_module == r.module) & (df.code_presentation == r.test)]
    row = {"pair": f"{r.module} {r.train}>{r.test}"}
    for name, cols in groups.items():
        row[name] = max(abs(smd(tr[c], te[c])) for c in cols)
    for lab, d in (("full_marks_train", tr), ("full_marks_test", te)):
        s = d.loc[d.assess_submitted > 0, "assess_mean_score"]
        row[lab] = (s == 100).mean() if len(s) else np.nan
    rows.append(row)

t = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print("largest |standardized shift| within each feature group (0.2 small, 0.5 medium, 0.8 large):\n")
print(t.round(2).to_string(index=False))
print("\npairs where a group shifts by more than 0.5:")
for name in groups:
    bad = t.loc[t[name] > 0.5, "pair"].tolist()
    print(f"  {name:<17} {len(bad)} of {len(t)}   {', '.join(bad)}")
