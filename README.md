# Reproduction materials: When the System Says 70%, Do 70% of Students Really Fail?

Code, derived data and results for the paper *When the System Says 70%, Do 70% of Students Really Fail?
Checking AI Foundation Models for Student-Risk Predictions* by Ebrahim Khaled Ebrahim (Department
of Statistics, Faculty of Business, Alexandria University, Egypt; ORCID 0009-0006-7839-8778),
16th International Conference on ICT in Our Lives, Alexandria University, 19–21 December 2026.

The study trains five models — logistic regression, ridge regression, LightGBM and the tabular
foundation models TabPFN and TabICL — on one year's students of an Open University course and
scores the next year's students, at day 28 of the course. It asks whether the models rank students
correctly (AUC) and whether their percentages can be taken at face value (calibration), using the
goodness-of-fit checks of the R package
[ebrahim.gof](https://CRAN.R-project.org/package=ebrahim.gof). A controlled experiment on real
student information shows that the routine check raises false alarms after ridge penalization and
that the corrected check does not.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | Every step, numbered in the order it runs (below), and the figure scripts. |
| `data/` | The derived day-28 features (`features/`) and the course pairs (`scope/`). No raw data; see `data/README.md` for how to get OULAD. |
| `results/` | Per course pair and feature set: every model's predictions for every new student, accuracy and calibration measures, and the results of every check. `summary/` holds the tables behind the paper; `plasmode/` the controlled experiment, one row per simulated cohort. |
| `figures/` | Figures 1–7 as PDF and PNG; Figures 1 and 2 also as TikZ source. |
| `DEVIATIONS.md` | Every design decision made after looking at the data, with its reason (D-01 to D-10). |

## Steps

| Step | Script | What it does |
|---|---|---|
| 1 | `code/01_scope_cohorts.py` | Finds the course presentations and the nine same-term, year-on-year pairs. |
| 2 | `code/02_build_features.py` | Builds the day-28 features for every student. |
| 3 | `code/03_fit_pair.py` | Fits the five models on one year and scores the next (GPU for the foundation models). |
| 4 | `code/04_check_a.R` | Checks on next year's students, for every model. |
| 5 | `code/05_check_b.R` | The builder's routine checks on the training data, for logistic and ridge regression. |
| 6 | `code/06_run_all.ps1` | Runs steps 3–5 for every pair; resumable. |
| 7 | `code/07_plasmode_shrinkage.R` | The controlled experiment on ridge penalization (Figure 7); writes every simulated cohort to disk and resumes. |
| 8 | `code/08_overnight.ps1` | Runs the main analysis, the experiment and the secondary analysis in that order. |
| 9 | `code/09_collect.py` | Gathers the per-pair results into `results/summary/`. |
| 10 | `code/10_headline.py` | The headline numbers with bootstrap intervals. |
| Figures | `code/fig3_*.py` … `code/fig7_*.py`, `figures/fig1_*.tex`, `figures/fig2_*.tex` | Figures 3–7 (matplotlib, shared style in `code/figstyle.py`); Figures 1–2 (XeLaTeX). |

`code/check_tabpfn_parts.py` verifies that predicting a cohort in two parts gives TabPFN's
whole-cohort predictions (deviation D-09). `code/check_snippet.R` runs the code shown in Section 5
of the paper. The `diag_*` scripts are the pilot diagnostics behind deviations D-01, D-03 and D-07.

## How to reproduce

The study ran on Windows 11 with Python 3.11, R 4.4 and one 6 GB NVIDIA GPU.

1. **The model engine.** Models are fitted and checked with the code of an earlier study, used
   unchanged: [tfm-calibration-reproducibility](https://github.com/ebrahimkhaled/tfm-calibration-reproducibility)
   (doi:[10.5281/zenodo.23737378](https://doi.org/10.5281/zenodo.23737378)). Clone it and follow its
   setup, which builds the Python environment, the R library and ebrahim.gof 2.10.0, and downloads
   the TabPFN weights after you accept their licence.
2. **Paths.** Set `ICT_ROOT` to this folder and `TFM_REBUILD_ROOT` to the engine's folder.
3. **Data.** Download OULAD into `data/raw/` (see `data/README.md`), then run steps 1 and 2.
   The derived features in `data/features/` let you skip this.
4. **Run.** `powershell -File code/08_overnight.ps1` (about two hours on the machine above), then
   steps 9 and 10 and the figure scripts.

To redraw the figures and tables without refitting anything, run steps 9 and 10 and the figure
scripts on the results already in `results/`.

## Model weights

No model weights are included. TabPFN weights are distributed by Prior Labs under a licence that
does not allow redistribution; the engine's setup downloads them after you accept it.

## Licence

Code: MIT. Derived data and results: CC BY 4.0, as derived from the Open University Learning
Analytics Dataset (Kuzilek, Hlosta and Zdrahal, 2017, *Scientific Data* 4, 170171,
doi:10.1038/sdata.2017.171), which is under CC BY 4.0.

## Citation

See `CITATION.cff`.
