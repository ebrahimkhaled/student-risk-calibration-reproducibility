"""Step 9: gather every pair's results into three tidy tables for the figures and the paper.

    results/summary/metrics_all.csv    accuracy and calibration, one row per (set, pair, model)
    results/summary/check_a_all.csv    new-cohort checks, one row per (set, pair, model, test)
    results/summary/check_b_all.csv    own-data checks, regression models only

and prints the headline numbers.

    python 09_collect.py
"""
import os
from pathlib import Path

import pandas as pd

PROJECT = Path(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"))
RES = PROJECT / "results"
OUT = RES / "summary"
OUT.mkdir(exist_ok=True)
MODELS = ["lr", "ridge", "lgbm", "tabpfn", "tabicl"]


def gather(name):
    frames = []
    for fs in ["stable", "full"]:
        for d in sorted((RES / fs).glob("*_20*")):
            f = d / name
            if f.exists():
                x = pd.read_csv(f)
                x["pair"] = d.name
                x["feature_set"] = fs
                frames.append(x)
    return pd.concat(frames, ignore_index=True)


met = gather("metrics.csv")
met = met[met.model.isin(MODELS)]
met.to_csv(OUT / "metrics_all.csv", index=False)
ca = gather("check_a.csv")
ca.to_csv(OUT / "check_a_all.csv", index=False)
cb = gather("check_b.csv")
cb.to_csv(OUT / "check_b_all.csv", index=False)

pd.set_option("display.width", 200)
for fs in ["stable", "full"]:
    m = met[met.feature_set == fs]
    print(f"\n=== {fs}: {m.pair.nunique()} pairs, {m[m.model == 'lr'].n.sum()} test students ===")
    print(m.pivot_table(index="model", values=["auc", "emp_slope", "emp_citl_prob", "emp_ici", "brier"],
                        aggfunc="mean").reindex(MODELS).round(3))
    print("\nAUC by pair")
    print(m.pivot(index="pair", columns="model", values="auc")[MODELS].round(3))
    print("\ncalibration slope by pair")
    print(m.pivot(index="pair", columns="model", values="emp_slope")[MODELS].round(2))
    print("\nobserved minus predicted? (citl prob = mean predicted - observed)")
    print(m.pivot(index="pair", columns="model", values="emp_citl_prob")[MODELS].round(3))

print("\n\ncheck A tests:", sorted(ca.test.unique()))
print("check B tests:", sorted(cb.test.unique()))
