# -*- coding: utf-8 -*-
"""Step 1: which module presentations can the study use?

The design trains on the earlier presentation of a module and tests on the later
one, at day 28. So for every presentation we need: how many students are still
enrolled on day 28 (the ones a model can actually be asked about), and how many
of those end up failing or withdrawing. A module only enters the study if it has
two presentations and both are big enough to fit and to check.
"""
import os
import pandas as pd

RAW = os.path.join(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"), "data/raw")
OUT = os.path.join(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"), "data/scope")
os.makedirs(OUT, exist_ok=True)

CUTOFF = 28          # day of the module at which the prediction is made

# OULAD writes missing values as "?" rather than leaving the cell empty
info = pd.read_csv(os.path.join(RAW, "studentInfo.csv"), na_values="?")
reg = pd.read_csv(os.path.join(RAW, "studentRegistration.csv"), na_values="?")
courses = pd.read_csv(os.path.join(RAW, "courses.csv"))

key = ["code_module", "code_presentation", "id_student"]
df = info.merge(reg, on=key, how="left")

# a student who unregistered on or before the cutoff has already gone: there is
# nothing left to predict for them, so they are not part of the cohort
df["gone_by_cutoff"] = df["date_unregistration"].notna() & (df["date_unregistration"] <= CUTOFF)
df["at_risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

rows = []
for (m, p), g in df.groupby(["code_module", "code_presentation"]):
    kept = g[~g["gone_by_cutoff"]]
    rows.append({
        "module": m, "presentation": p,
        "enrolled": len(g),
        "gone_by_day28": int(g["gone_by_cutoff"].sum()),
        "cohort_at_day28": len(kept),
        "at_risk": int(kept["at_risk"].sum()),
        "at_risk_rate": round(kept["at_risk"].mean(), 3),
        "fail": int((kept["final_result"] == "Fail").sum()),
        "withdrawn": int((kept["final_result"] == "Withdrawn").sum()),
    })
tab = pd.DataFrame(rows).merge(
    courses.rename(columns={"code_module": "module", "code_presentation": "presentation",
                            "module_presentation_length": "length_days"}),
    on=["module", "presentation"])

# presentation codes sort in time order: 2013B < 2013J < 2014B < 2014J
tab = tab.sort_values(["module", "presentation"]).reset_index(drop=True)
tab.to_csv(os.path.join(OUT, "presentations_day28.csv"), index=False)

pd.set_option("display.width", 160)
print(tab.to_string(index=False))

# candidate train -> test pairs: consecutive presentations of the same module
pairs = []
for m, g in tab.groupby("module"):
    g = g.sort_values("presentation").reset_index(drop=True)
    for i in range(len(g) - 1):
        a, b = g.iloc[i], g.iloc[i + 1]
        pairs.append({"module": m, "train": a["presentation"], "test": b["presentation"],
                      "n_train": a["cohort_at_day28"], "n_test": b["cohort_at_day28"],
                      "risk_train": a["at_risk_rate"], "risk_test": b["at_risk_rate"],
                      "events_test": b["at_risk"]})
pairs = pd.DataFrame(pairs)
pairs.to_csv(os.path.join(OUT, "candidate_pairs.csv"), index=False)
print("\ncandidate pairs (consecutive presentations):")
print(pairs.to_string(index=False))
print("\nstudents at day 28 in total:", int(tab["cohort_at_day28"].sum()),
      "| overall at-risk rate:", round(tab["at_risk"].sum() / tab["cohort_at_day28"].sum(), 3))
