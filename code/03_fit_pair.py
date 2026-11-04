# -*- coding: utf-8 -*-
"""Step 3: fit every model on one module's earlier cohort, predict its later cohort.

Uses the model engine built for the tabular-foundation-model study (#11) unchanged:
the same fitters, the same R arms, the same checks on what comes back. Only the
data are new.

    python 03_fit_pair.py BBB 2013J 2014J            # one pair
    python 03_fit_pair.py BBB 2013J 2014J --device cpu

Writes, under results/<module>_<train>_<test>/:
    preds.csv.gz     one row per test student: outcome, and each model's predicted risk
    metrics.csv      ranking and calibration measures for each model
    run.json         versions, timings, seeds, data fingerprint
"""
import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get("ICT_ROOT", r"D:/Research/paper_ict_student_risk"))
ENGINE = Path(os.environ.get("TFM_REBUILD_ROOT", r"D:/Research/paper_tfm_calibration/rebuild")) / "code"
sys.path.insert(0, str(ENGINE))

import models   # noqa: E402  (the #11 engine)
import metrics  # noqa: E402

ARMS = ["lr", "ridge", "lgbm", "tabpfn", "tabicl"]
CLICK_COLS_PREFIX = "clicks_"

# Two feature sets (deviation D-01, adopted after the pilot):
#   stable - only information whose meaning does not depend on how a particular year's course
#            was designed: background, registration timing, overall online engagement. This is
#            the main analysis.
#   full   - adds the mix of click types and the assessment features. In the pilot the course
#            website was reorganised between years in 7 of 9 modules, and in BBB the first
#            assignment changed from graded to near pass/fail (0% -> 58% full marks), so these
#            features describe a different course each year. Kept as a secondary analysis.
UNSTABLE = {"clicks_homepage", "clicks_oucontent", "clicks_forumng", "clicks_subpage",
            "clicks_resource", "clicks_url", "clicks_quiz", "clicks_other",
            "assess_due", "assess_submitted", "assess_missed", "assess_mean_score",
            "assess_mean_days_late"}


def seed_for(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def load_features(feature_set: str = "stable"):
    df = pd.read_csv(PROJECT / "data/features/day28_features.csv.gz")
    groups = json.loads((PROJECT / "data/features/feature_groups.json").read_text())
    cont, binary, cat = groups["cont"], groups["bin"], groups["cat"]
    if feature_set == "stable":
        cont = [c for c in cont if c not in UNSTABLE]
    elif feature_set != "full":
        raise ValueError(f"unknown feature set {feature_set!r}")

    # click counts are heavily skewed; every model sees them on the same log(1 + x) scale
    for c in cont:
        if c.startswith(CLICK_COLS_PREFIX):
            df[c] = np.log1p(df[c])

    # category codes fixed from the whole table, so a level means the same in every cohort
    levels = {c: sorted(df[c].astype(str).unique()) for c in cat}
    for c in cat:
        df[c] = df[c].astype(str).map({v: i for i, v in enumerate(levels[c])}).astype(int)
    return df, cont, binary, cat, levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("train")
    ap.add_argument("test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--features", default="stable", choices=["stable", "full"])
    a = ap.parse_args()
    arms = a.arms.split(",")

    models.configure_environment(allow_download=False, threads=a.threads)
    df, cont, binary, cat, levels = load_features(a.features)
    columns = cont + binary + cat

    tr = df[(df.code_module == a.module) & (df.code_presentation == a.train)]
    te = df[(df.code_module == a.module) & (df.code_presentation == a.test)]
    if tr.empty or te.empty:
        raise SystemExit(f"no rows for {a.module} {a.train} -> {a.test}")

    tag = f"{a.module}_{a.train}_{a.test}"
    out = PROJECT / "results" / a.features / tag
    out.mkdir(parents=True, exist_ok=True)

    ctx = models.FitContext(
        X_dev=tr[columns].to_numpy(float), y_dev=tr["at_risk"].to_numpy(float),
        X_pred=te[columns].to_numpy(float), p_star_pred=np.zeros(len(te)),
        columns=columns, cont=cont, bin=binary, cat=cat,
        cat_levels={c: len(levels[c]) for c in cat},
        seeds={arm: seed_for("ict2026", tag, arm) for arm in arms + ["lgbm_raw"]},
        device=a.device, threads=a.threads)

    t0 = time.perf_counter()
    res = models.run_arms(arms, ctx)
    # TabPFN scores each new student against the training cohort only, never against other new
    # students, so predicting the new cohort in parts gives the same numbers with less GPU memory.
    # Used only when the whole cohort does not fit (full set, FFF 2013J, on a 6 GB card).
    if "tabpfn" in res and "out of memory" in str(res["tabpfn"].info.get("error", "")):
        parts = np.array_split(np.arange(len(te)), 2)
        rs = [models.fit_tabpfn(dataclasses.replace(ctx, X_pred=ctx.X_pred[i], p_star_pred=ctx.p_star_pred[i]))[0]
              for i in parts]
        if all(r.status == "ok" for r in rs):
            info0 = dict(rs[0].info, seconds=sum(r.info["seconds"] for r in rs), pred_parts=len(parts))
            res["tabpfn"] = models.ArmResult("tabpfn", np.concatenate([r.p for r in rs]), "ok", info0)
            print("  tabpfn     out of memory on the whole cohort; predicted in 2 parts")
    wall = time.perf_counter() - t0

    y = te["at_risk"].to_numpy(int)
    preds = pd.DataFrame({"id_student": te["id_student"].to_numpy(), "y": y})
    rows, info = [], {}
    for name, r in res.items():
        info[name] = {"status": r.status, **{k: v for k, v in r.info.items()
                                            if isinstance(v, (int, float, str, bool, type(None)))}}
        if r.status != "ok":
            print(f"  {name:<10} FAILED  {r.info.get('error', '')[:150]}")
            continue
        preds[name] = r.p
        m = metrics.empirical_metrics(y, r.p)
        m["model"] = name
        m["seconds"] = r.info.get("seconds", r.info.get("r_call_seconds"))
        rows.append(m)

    preds.to_csv(out / "preds.csv.gz", index=False, compression="gzip")
    met = pd.DataFrame(rows)
    met.to_csv(out / "metrics.csv", index=False)

    fp = hashlib.sha256(np.ascontiguousarray(ctx.X_dev).tobytes()
                        + np.ascontiguousarray(ctx.X_pred).tobytes()).hexdigest()[:16]
    (out / "run.json").write_text(json.dumps({
        "pair": tag, "feature_set": a.features, "n_train": len(tr), "n_test": len(te),
        "events_train": int(tr["at_risk"].sum()), "events_test": int(y.sum()),
        "rate_train": float(tr["at_risk"].mean()), "rate_test": float(y.mean()),
        "columns": columns, "seeds": ctx.seeds, "device": a.device,
        "wall_seconds": wall, "data_fingerprint": fp,
        "environment": models.environment_info(), "arms": info}, indent=2, default=str))

    show = ["model", "auc", "emp_citl_prob", "emp_slope", "emp_ici", "ece10", "brier", "seconds"]
    show = [c for c in show if c in met.columns]
    print(f"\n{tag}: train {len(tr)} (risk {tr['at_risk'].mean():.3f}) -> "
          f"test {len(te)} (risk {y.mean():.3f})   wall {wall:.0f}s")
    print(met[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
