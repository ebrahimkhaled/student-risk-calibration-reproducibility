# -*- coding: utf-8 -*-
"""Step 2: what a university knows about each student on day 28.

One row per student per module presentation, for students still enrolled on day
28. Every feature uses only information dated on or before day 28, so a model
can never see the future. The outcome is whether the student went on to fail or
withdraw.

Feature groups
  background   - gender, region, prior education, deprivation band, age band,
                 previous attempts, credits being studied, declared disability
  registration - how many days before the start the student registered
  activity     - online learning activity up to day 28: total clicks, active
                 days, clicks in the most recent week, clicks on the main
                 activity types
  assessment   - assessments due by day 28: how many were submitted, how many
                 missed, average score, average days late

One assumption is stated in the paper: marks for work submitted by day 28 are
taken as known on day 28. Computer-marked quizzes satisfy this; a university
with slower tutor marking would move the prediction date later.
"""
import os
import numpy as np
import pandas as pd

RAW = os.path.join(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"), "data/raw")
OUT = os.path.join(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"), "data/features")
os.makedirs(OUT, exist_ok=True)

CUTOFF = 28
RECENT_FROM = 22          # "the most recent week" = days 22 to 28
KEY = ["code_module", "code_presentation", "id_student"]
MOD = ["code_module", "code_presentation"]

# the activity types that carry most of the clicks; the rest are pooled as "other"
MAIN_TYPES = ["homepage", "oucontent", "forumng", "subpage", "resource", "url", "quiz"]


def read(name, **kw):
    return pd.read_csv(os.path.join(RAW, name), na_values="?", **kw)


# ---------------------------------------------------------------- who is in the cohort
info = read("studentInfo.csv")
reg = read("studentRegistration.csv")
df = info.merge(reg, on=KEY, how="left")
gone = df["date_unregistration"].notna() & (df["date_unregistration"] <= CUTOFF)
df = df.loc[~gone].copy()
df["at_risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

# ---------------------------------------------------------------- online activity
vle = read("vle.csv", usecols=["id_site", "activity_type"])
site_type = dict(zip(vle["id_site"], vle["activity_type"]))

parts = []
for chunk in read("studentVle.csv", chunksize=2_000_000,
                  dtype={"id_student": "int32", "id_site": "int32",
                         "date": "int16", "sum_click": "int32"}):
    # pre-start activity (negative days) is known by day 28, so it is kept
    parts.append(chunk.loc[chunk["date"] <= CUTOFF])
clicks = pd.concat(parts, ignore_index=True)
del parts
clicks["type"] = clicks["id_site"].map(site_type)
clicks.loc[~clicks["type"].isin(MAIN_TYPES), "type"] = "other"

act = clicks.groupby(KEY).agg(clicks_total=("sum_click", "sum"),
                              active_days=("date", "nunique")).reset_index()
recent = (clicks.loc[clicks["date"] >= RECENT_FROM].groupby(KEY)["sum_click"].sum()
          .rename("clicks_last_week").reset_index())
by_type = (clicks.pivot_table(index=KEY, columns="type", values="sum_click",
                              aggfunc="sum", fill_value=0)
           .add_prefix("clicks_").reset_index())
by_type.columns.name = None

for t in (act, recent, by_type):
    df = df.merge(t, on=KEY, how="left")
click_cols = [c for c in df.columns if c.startswith("clicks_")] + ["active_days"]
df[click_cols] = df[click_cols].fillna(0)     # no record = no activity

# ---------------------------------------------------------------- assessments due by day 28
ass = read("assessments.csv")
ass = ass.loc[(ass["assessment_type"] != "Exam") & ass["date"].notna() & (ass["date"] <= CUTOFF)]
sa = read("studentAssessment.csv")
sa = sa.merge(ass[["id_assessment", "code_module", "code_presentation", "date"]],
              on="id_assessment", how="inner")
sa = sa.loc[sa["date_submitted"] <= CUTOFF].copy()
sa["days_late"] = (sa["date_submitted"] - sa["date"]).clip(lower=0)

due = ass.groupby(MOD).size().rename("assess_due").reset_index()
sub = sa.groupby(KEY).agg(assess_submitted=("id_assessment", "nunique"),
                          assess_mean_score=("score", "mean"),
                          assess_mean_days_late=("days_late", "mean")).reset_index()
df = df.merge(due, on=MOD, how="left").merge(sub, on=KEY, how="left")
df["assess_due"] = df["assess_due"].fillna(0).astype(int)
df["assess_submitted"] = df["assess_submitted"].fillna(0).astype(int)
df["assess_missed"] = (df["assess_due"] - df["assess_submitted"]).clip(lower=0)
# a student who submitted nothing has no score: 0 is the honest value for "nothing handed in",
# and the missed count already tells the model that it is absence rather than failure
df["assess_mean_score"] = df["assess_mean_score"].fillna(0)
df["assess_mean_days_late"] = df["assess_mean_days_late"].fillna(0)

# ---------------------------------------------------------------- background
# Grouped so that every level holds students in every training cohort (deviation D-07): a model
# fitted on last year has no estimate for a group that had no students last year. The raw levels
# left BBB 2013B with no student aged 55+ and GGG 2013J with none from Ireland or Wales.
AGE = {"0-35": "under 35", "35-55": "35 and over", "55<=": "35 and over"}
EDUCATION = {"No Formal quals": "below A level", "Lower Than A Level": "below A level",
             "A Level or Equivalent": "A level", "HE Qualification": "higher education",
             "Post Graduate Qualification": "higher education"}


def deprivation(v):
    # an unknown band (4% of students, 3 in GGG 2013J) joins the middle band
    if pd.isna(v) or v == "Unknown":
        return "middle"
    lo = int(str(v).split("-")[0])
    return "most deprived" if lo < 30 else ("middle" if lo < 70 else "least deprived")


df["imd_band_raw"] = df["imd_band"].fillna("Unknown")        # kept for the fairness check
df["age_band_raw"] = df["age_band"]
df["highest_education_raw"] = df["highest_education"]
for col, m in (("age_band", AGE), ("highest_education", EDUCATION)):
    unmapped = set(df[col].dropna().unique()) - set(m)
    if unmapped:
        raise SystemExit(f"{col}: levels without a group: {unmapped}")
    df[col] = df[col].map(m)
df["imd_band"] = df["imd_band"].map(deprivation)
df["disability"] = (df["disability"] == "Y").astype(int)
df["gender"] = (df["gender"] == "M").astype(int)          # 1 = male
df["date_registration"] = df["date_registration"].fillna(df["date_registration"].median())

# Region is not used (D-07): it is a weak predictor (at-risk rates differ by at most 6.6 points
# across region groups), unstable between years (GGG 2013J had 1 student from Scotland, Wales or
# Ireland), and predicting an individual's risk from where they live is questionable in itself.
# "Assessments due" is constant within a cohort, so "missed" = due - submitted adds nothing a model
# can use and makes the design exactly collinear; both are dropped and "submitted" is kept.
FEATURES = {
    "cont": ["num_of_prev_attempts", "studied_credits", "date_registration",
             "clicks_total", "active_days", "clicks_last_week"]
            + [f"clicks_{t}" for t in MAIN_TYPES + ["other"]]
            + ["assess_submitted", "assess_mean_score", "assess_mean_days_late"],
    "bin": ["gender", "disability"],
    "cat": ["highest_education", "imd_band", "age_band"],
}
cols = MOD + ["id_student"] + FEATURES["cont"] + FEATURES["bin"] + FEATURES["cat"] + \
       ["region", "imd_band_raw", "age_band_raw", "highest_education_raw", "final_result", "at_risk"]
missing = [c for c in cols if c not in df.columns]
if missing:
    raise SystemExit(f"columns not built: {missing}")
df = df[cols].sort_values(KEY).reset_index(drop=True)

# ---------------------------------------------------------------- leakage and sanity checks
assert df[FEATURES["cont"] + FEATURES["bin"]].notna().all().all(), "missing values left"
assert df.duplicated(KEY).sum() == 0, "duplicate student rows"
assert (df["clicks_total"] >= df["clicks_last_week"]).all(), "recent clicks exceed total"

out = os.path.join(OUT, "day28_features.csv.gz")
df.to_csv(out, index=False, compression="gzip")
pd.Series(FEATURES).to_json(os.path.join(OUT, "feature_groups.json"))

print(f"rows: {len(df):,}   presentations: {df.groupby(MOD).ngroups}   "
      f"features: {sum(len(v) for v in FEATURES.values())}")
print(f"at-risk rate: {df['at_risk'].mean():.3f}")
print(f"students with no online activity by day 28: {(df['clicks_total'] == 0).mean():.1%}")
print(f"written: {out}")
print()
print("how the at-risk rate moves with activity (quartiles of clicks by day 28):")
q = pd.qcut(df["clicks_total"].rank(method="first"), 4, labels=["lowest", "2nd", "3rd", "highest"])
print(df.groupby(q, observed=True)["at_risk"].agg(["mean", "size"]).round(3).to_string())
