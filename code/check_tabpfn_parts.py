"""Does TabPFN give the same predictions when the new cohort is predicted in two parts?

Run once on a pair that fits in memory whole (full set, EEE 2013J -> 2014J) and compare with the
saved whole-cohort predictions.

    python check_tabpfn_parts.py
"""
import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("fit_pair", HERE / "03_fit_pair.py")
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)
models = fp.models

module, train, test, fset = "EEE", "2013J", "2014J", "full"
models.configure_environment(allow_download=False, threads=4)
df, cont, binary, cat, levels = fp.load_features(fset)
columns = cont + binary + cat
tr = df[(df.code_module == module) & (df.code_presentation == train)]
te = df[(df.code_module == module) & (df.code_presentation == test)]
tag = f"{module}_{train}_{test}"
ctx = models.FitContext(
    X_dev=tr[columns].to_numpy(float), y_dev=tr["at_risk"].to_numpy(float),
    X_pred=te[columns].to_numpy(float), p_star_pred=np.zeros(len(te)),
    columns=columns, cont=cont, bin=binary, cat=cat, cat_levels={c: len(levels[c]) for c in cat},
    seeds={arm: fp.seed_for("ict2026", tag, arm) for arm in fp.ARMS + ["lgbm_raw"]},
    device="cuda", threads=4)
parts = np.array_split(np.arange(len(te)), 2)
p = np.concatenate([models.fit_tabpfn(dataclasses.replace(ctx, X_pred=ctx.X_pred[i], p_star_pred=ctx.p_star_pred[i]))[0].p
                    for i in parts])
saved = pd.read_csv(fp.PROJECT / "results" / fset / tag / "preds.csv.gz")
assert (saved.id_student.to_numpy() == te.id_student.to_numpy()).all()
d = np.abs(p - saved.tabpfn.to_numpy())
print(f"{tag} ({fset}): {len(te)} students; max |parts - whole| = {d.max():.2e}, mean {d.mean():.2e}")
sys.exit(0 if d.max() < 1e-4 else 1)
