# -*- coding: utf-8 -*-
"""Pilot diagnostic: why is the calibration slope so low on BBB 2013J -> 2014J?

Two questions. Which features changed most between the two cohorts (standardized
mean difference)? And does the slope of a plain logistic regression recover when
the suspect feature groups are left out? Diagnostic only - nothing here feeds
the paper's results.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

P = os.path.join(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"), "data/features")
df = pd.read_csv(f"{P}/day28_features.csv.gz")
g = json.load(open(f"{P}/feature_groups.json"))
for c in g["cont"]:
    if c.startswith("clicks_"):
        df[c] = np.log1p(df[c])

tr = df[(df.code_module == "BBB") & (df.code_presentation == "2013J")]
te = df[(df.code_module == "BBB") & (df.code_presentation == "2014J")]

print("=== biggest shifts between the cohorts (standardized mean difference) ===")
num = g["cont"] + g["bin"]
smd = {c: (te[c].mean() - tr[c].mean()) / np.sqrt((tr[c].var() + te[c].var()) / 2 + 1e-12)
       for c in num}
for c, v in sorted(smd.items(), key=lambda kv: -abs(kv[1]))[:8]:
    print(f"  {c:<24} {v:+.2f}")

print("\n=== first-assessment score: how it is spread in each year ===")
for name, d in (("2013J", tr), ("2014J", te)):
    s = d.loc[d.assess_submitted > 0, "assess_mean_score"]
    print(f"  {name}: share scoring 100 = {(s == 100).mean():.1%}   "
          f"quartiles = {s.quantile([.25, .5, .75]).round(0).tolist()}")


def slope_auc(features):
    cont = [c for c in g["cont"] + g["bin"] if c in features]
    cat = [c for c in g["cat"] if c in features]
    pre = ColumnTransformer([("n", StandardScaler(), cont),
                             ("c", OneHotEncoder(handle_unknown="ignore", drop="first"), cat)])
    m = make_pipeline(pre, LogisticRegression(C=1e6, max_iter=5000))
    m.fit(tr[cont + cat], tr.at_risk)
    p = np.clip(m.predict_proba(te[cont + cat])[:, 1], 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    cal = LogisticRegression(C=1e6).fit(lp[:, None], te.at_risk)
    from sklearn.metrics import roc_auc_score
    return cal.coef_[0, 0], roc_auc_score(te.at_risk, p), p.mean() - te.at_risk.mean()


allf = g["cont"] + g["bin"] + g["cat"]
assess = [c for c in allf if c.startswith("assess_")]
clicks = [c for c in allf if c.startswith("clicks_") or c == "active_days"]
print("\n=== plain logistic regression, test-cohort slope / AUC / mean bias ===")
for label, feats in [
        ("all features", allf),
        ("without assessment score", [c for c in allf if c != "assess_mean_score"]),
        ("without any assessment feature", [c for c in allf if c not in assess]),
        ("without click features", [c for c in allf if c not in clicks]),
        ("background only", g["bin"] + g["cat"] + ["num_of_prev_attempts", "studied_credits"])]:
    s, a, b = slope_auc(feats)
    print(f"  {label:<32} slope {s:5.2f}   AUC {a:.3f}   bias {b:+.3f}")
