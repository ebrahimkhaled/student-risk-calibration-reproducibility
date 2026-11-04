# Data

## Raw data (not included)

The Open University Learning Analytics Dataset (OULAD; Kuzilek, Hlosta and Zdrahal, 2017,
*Scientific Data* 4, 170171, doi:10.1038/sdata.2017.171), CC BY 4.0. The study used the copy on the
UCI Machine Learning Repository:

    https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip
    MD5 ea9f1293b549c3a93b70b8d349b442e2

Unzip it so that the seven CSV files (`studentInfo.csv`, `studentVle.csv`, …) sit in `data/raw/`.
Missing values in the files are written as `?`.

## Included

| File | What it holds |
|---|---|
| `features/day28_features.csv.gz` | One row per student and course presentation (27,538 rows): the day-28 features used by the models, the original background levels used by the fairness check (`*_raw`), the final result and the outcome `at_risk` (fail or withdrawn). Students who withdrew before day 28 are excluded. |
| `features/feature_groups.json` | Which columns are continuous, binary and categorical. |
| `scope/presentations_day28.csv` | Every course presentation with its size and at-risk rate. |
| `scope/candidate_pairs.csv`, `scope/year_on_year_pairs.csv` | The course pairs; the nine same-term, year-on-year pairs are the ones analysed. |

These derived files are under CC BY 4.0, as is OULAD.
