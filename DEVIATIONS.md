# Deviations from the plan

Every change to `ICT 2026 Paper Plan.md` made after looking at data, with its reason.
The paper's methods section states each one.

## D-01 — Two feature sets; the main analysis uses only "stable" features (pilot, 3 Nov 2026)

**Plan:** background, registration, online activity (total and by type), assessments due by day 28.

**Change:** the main analysis uses only features whose meaning does not depend on how a
particular year's course was designed — background (gender, region, prior education,
deprivation band, age band, previous attempts, credits, disability), registration timing, and
overall online engagement (total clicks, clicks in the last week, active days). Twelve
features. The full set is kept as a secondary analysis.

**Why:** the first pilot run (BBB, 2013J → 2014J, all features) gave a calibration slope of
0.26 for logistic regression. The cause was the course, not the models:

- In BBB the first assignment changed from graded to near pass/fail: 0.1% of students scored
  100 in 2013J, 57.7% in 2014J (quartiles 65/74/80 → 1/100/100). Dropping that one feature
  moved the slope from 0.26 to 0.77 and the AUC from 0.647 to 0.707.
- Across the nine pairs, the mix of click types shifted by more than half a standard deviation
  in 7 of 9 — the course websites were reorganised between years. Total engagement shifted that
  much in 1 of 9, background in 0 of 9.

The rule is conceptual — a feature whose definition is tied to one year's course design cannot
describe next year's — but it was adopted after the pilot, so it is declared here.

**Side effect:** the stable set contains no assessment marks, so the main analysis no longer
needs the assumption that marks for work submitted by day 28 are known on day 28. That
assumption now applies only to the secondary analysis.

## D-02 — Same-term, year-on-year pairs (pilot, 3 Nov 2026)

**Plan:** train on the earlier presentation of a module, test on the later one, every module
with two presentations.

**Change:** pairs are the same term one year apart — 2013B → 2014B and 2013J → 2014J. Nine
pairs, 10,809 test students, 4,507 of them at risk.

**Why:** "J" presentations start in October and "B" in February, so consecutive pairs mix a
four-month gap with an eight-month one. Same-term pairs are what "last year's model, this
year's students" actually means. Module CCC ran in one year only, so it has no pair.

## D-03 — localize.external is given student groups (pilot, 3 Nov 2026)

**Plan:** the external battery as used in #11.

**Change:** `localize.external` receives small integer-coded student groups — age band, sex,
disability, prior education (three levels), deprivation (three bands plus "unknown" as its own
level), and engagement quartile within the cohort — rather than the model inputs. This follows
#11's rule (pre-registration 10.3). The other covariate checks (le Cessie, DeepGOF-1) keep the
numeric model inputs.

**Why:** given the model inputs, `localize.external` failed on every model with "all interior
knots match left boundary knot". The cause is columns with many values but one dominant one —
previous attempts (89% zeros) and credits (75% on a single value) — which collapse its spline
knots. Coded groups with two to four levels all work.

## D-04 — Click counts on a log scale for every model

**Not in the plan.** Click counts are heavily skewed, so every model receives
`log(1 + clicks)`. A standard choice, applied identically to all arms, so that the regression
models are not handicapped by a careless specification. Tree models and the foundation models
are essentially unaffected by a monotone transformation.

## D-06 — Shrinkage plasmode: a no-shrinkage control, common outcomes, a third misfit strength

**Plan:** Figure 7 shows the routine check misleading after ridge, demonstrated rather than asserted.

**Design (fixed before the full run):** real BBB 2013J covariates; truth = the unpenalized fit to
the real outcomes, so the model is correct and every rejection is a false alarm; ridge at 0, 0.5,
1, 2, 4 and 8 times the cross-validated strength; 1,000 replicates per level; B = 199 reference
draws. Each replicate uses the same simulated cohort at every level (common random numbers), so
differences across levels come from shrinkage alone.

**Power control.** A misfit truth adds a curved effect of online activity the fitted model omits.
Each strength runs at no shrinkage, where both checks are valid and their rate is the honest power,
and at the cross-validated strength, where the corrected check should keep it. Without the
no-shrinkage arm, a corrected check that rarely rejects could mean lost power or an undetectable
misfit, and the first smoke test could not tell them apart.

**Change after the smoke test:** strengths 0.25 and 0.5 were fixed in advance. With no shrinkage
they were caught in at most 1 of 4 smoke replicates, so a third strength, **1.0**, was added before
the full run. This affects only the power control; the null conditions are unchanged.

## D-07 — Background categories grouped; region and two assessment columns dropped (3 Nov 2026)

**Plan:** background variables at their original OULAD levels.

**Change**, applied identically to every model:
- age band → two groups (under 35; 35 and over)
- prior education → three (below A level; A level; higher education)
- deprivation band → three (most deprived 0–30%; middle 30–70%; least deprived 70–100%), with an
  unknown band merged into the middle
- **region not used**
- full set only: "assessments due" and "assessments missed" dropped; "submitted" kept

**Why.** The first main run failed Check B in two pairs because a model fitted on last year had no
estimate for a group with no students last year: BBB 2013B had no student aged 55+, GGG 2013J none
from Ireland or Wales. Logistic regression cannot predict for such a group; the tree and foundation
models silently push those students through, so the comparison was not even equal. After grouping,
no pair has an unseen level and the smallest group in any training cohort holds 31 students.

Region is dropped rather than grouped because, grouped, GGG 2013J still had a single student from
Scotland, Wales or Ireland; it is weak (at-risk rates differ by at most 6.6 points across region
groups); and predicting an individual's risk from where they live is questionable in a paper about
trustworthy AI. The unknown deprivation band (4% of students; 3 in GGG 2013J) is merged into the
middle band for the models. The fairness check (localize) keeps the ORIGINAL levels, with unknown
deprivation as its own group.

"Assessments due" is constant within a cohort, so "missed" equals due minus submitted: it adds
nothing a model can use and makes the design exactly collinear, which made the correction inside
`shrink.gof` fail on a singular matrix.

**Effect:** the stable set has 13 model columns instead of 36. Every result was re-run from
scratch on the new features; nothing from the earlier run is used.

## D-08 — Check B leaves out columns that are constant in the training cohort (4 Nov 2026)

In a module's first year (AAA, EEE and GGG 2013J) no student has a previous attempt, so that
column is all zeros. Logistic regression gives it no coefficient, so the fitted model and its
predictions are unchanged, but several in-sample tests (EDGE, the DEF tests, Copas, McCullagh,
the information-matrix test) stopped on the singular matrix. Check B now runs on the same fit with
training-constant columns removed; the script asserts the fitted values are identical. Check A and
every model's predictions are untouched.

## D-09 — TabPFN predicts one cohort in two parts (4 Nov 2026)

On the full feature set, FFF 2013J → 2014J (1,982 training and 1,859 new students), TabPFN ran
out of memory on the 6 GB GPU twice. TabPFN scores each new student against the training cohort
only, so predicting the new cohort in two halves is the same calculation. Verified on a pair that
fits whole (full set, EEE 2013J → 2014J): two-part and whole-cohort predictions agree to 3e-8
(`code/check_tabpfn_parts.py`). `03_fit_pair.py` falls back to two parts only after an
out-of-memory failure, and records it in run.json. No other pair is affected.

## D-10 — Figure 5's "right spread" column uses a slope test computed from the predictions

The external battery reports the calibration slope without a p-value, so the slope-equals-1 test
(Wald test from a logistic regression of the outcome on the logit of the score) is computed in
`fig5_verdicts.py` from the saved predictions. Presentation only; no model or check is changed.

## D-05 — Check B in the paper reports the recommended core (presentation only)

`run.all.gof()` returns 28 in-sample tests. Some of them — the deviance test, the grouped F-test,
Xie's test — are known to be unreliable with continuous covariates (the benchmark paper,
JICDS 2(2), 97–128). The paper reports the core that study recommends; every test is kept in the
result files.
