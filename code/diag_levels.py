# -*- coding: utf-8 -*-
"""Which category levels are missing from a training cohort but present in its test cohort?

A model fitted on last year cannot estimate anything for a group that had no students last year.
Checks the raw levels, then the proposed groups, across all nine year-on-year pairs. Diagnostic only.
"""
import os
import pandas as pd

P = os.path.join(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"), "data")
df = pd.read_csv(f"{P}/features/day28_features.csv.gz")
pairs = pd.read_csv(f"{P}/scope/year_on_year_pairs.csv")

GROUPS = {
    "age_band": {"0-35": "under 35", "35-55": "35 and over", "55<=": "35 and over"},
    "highest_education": {"No Formal quals": "below A level", "Lower Than A Level": "below A level",
                          "A Level or Equivalent": "A level", "HE Qualification": "higher education",
                          "Post Graduate Qualification": "higher education"},
    "region": {"London Region": "London and South East", "South East Region": "London and South East",
               "South Region": "South, South West and East", "South West Region": "South, South West and East",
               "East Anglian Region": "South, South West and East",
               "West Midlands Region": "Midlands and North", "East Midlands Region": "Midlands and North",
               "Yorkshire Region": "Midlands and North", "North Western Region": "Midlands and North",
               "North Region": "Midlands and North",
               "Scotland": "Scotland, Wales and Ireland", "Wales": "Scotland, Wales and Ireland",
               "Ireland": "Scotland, Wales and Ireland"},
}


def imd_group(v):
    if v == "Unknown":
        return "unknown"
    lo = int(str(v).split("-")[0])
    return "most deprived" if lo < 30 else ("middle" if lo < 70 else "least deprived")


grouped = df.copy()
for c, m in GROUPS.items():
    unmapped = set(grouped[c].unique()) - set(m)
    assert not unmapped, f"{c}: levels without a group {unmapped}"
    grouped[c] = grouped[c].map(m)
grouped["imd_band"] = grouped["imd_band"].map(imd_group)


def report(d, label):
    print(f"=== {label} ===")
    worst = {}
    gaps = 0
    for _, r in pairs.iterrows():
        tr = d[(d.code_module == r.module) & (d.code_presentation == r.train)]
        te = d[(d.code_module == r.module) & (d.code_presentation == r.test)]
        for c in ["age_band", "highest_education", "imd_band", "region"]:
            miss = sorted(set(te[c]) - set(tr[c]))
            if miss:
                gaps += 1
                print(f"  {r.module} {r.train}: {c} has no students in {miss}")
            n_min = tr[c].value_counts().min()
            worst[c] = min(worst.get(c, 10**9), n_min)
    print(f"  pairs x variables with a gap: {gaps}")
    print("  smallest group in any training cohort: " +
          ", ".join(f"{c} {n}" for c, n in worst.items()))
    print()


report(df, "raw levels")
report(grouped, "proposed groups")
