## Pilot diagnostic: which kind of covariate makes localize.external fail with
## "all interior knots match left boundary knot"? Tries the test cohort's columns one at a
## time against one model's predictions. Diagnostic only.
Sys.setenv(TFM_REBUILD_ROOT = Sys.getenv("TFM_REBUILD_ROOT", "D:/Research/paper_tfm_calibration/rebuild"))
source(file.path(Sys.getenv("TFM_REBUILD_ROOT"), "R", "gof_layer.R"))
use_project_lib()

P <- Sys.getenv("ICT_ROOT", "D:/Research/paper_ict_student_risk")
pred <- read.csv(gzfile(file.path(P, "results/stable/BBB_2013J_2014J/preds.csv.gz")))
feat <- read.csv(gzfile(file.path(P, "data/features/day28_features.csv.gz")))
te <- feat[feat$code_module == "BBB" & feat$code_presentation == "2014J", ]
te <- te[match(pred$id_student, te$id_student), ]
stopifnot(identical(te$id_student, pred$id_student))

## candidate group variables, as integer codes
edu <- c("No Formal quals" = 0, "Lower Than A Level" = 0, "A Level or Equivalent" = 1,
         "HE Qualification" = 2, "Post Graduate Qualification" = 2)
imd <- suppressWarnings(as.integer(sub("-.*", "", te$imd_band)))   # "0-10%" -> 0, "Unknown" -> NA
cand <- data.frame(
  age_band   = as.integer(factor(te$age_band)) - 1L,
  male       = te$gender,
  disability = te$disability,
  education  = unname(edu[te$highest_education]),
  deprived   = ifelse(is.na(imd), 1L, ifelse(imd < 30, 0L, ifelse(imd < 70, 1L, 2L))),
  engagement = as.integer(cut(rank(te$clicks_total, ties.method = "first"),
                              4, labels = FALSE)) - 1L,
  prev_attempts = te$num_of_prev_attempts,
  credits    = te$studied_credits,
  registration = te$date_registration)

y <- pred$y; p <- pred$tabpfn
show <- function(label, X) {
  r <- tryCatch(localize.external(y, p, as.matrix(X), M = 199, seed = 1, plot = FALSE),
                error = function(e) e)
  if (inherits(r, "error")) cat(sprintf("  %-28s FAILS: %s\n", label, conditionMessage(r)))
  else cat(sprintf("  %-28s ok    named = %s\n", label,
                   if (length(r$named)) paste(r$named, collapse = "+") else "none"))
}
cat("distinct values and largest share per candidate:\n")
for (v in names(cand))
  cat(sprintf("  %-14s %4d distinct, top value holds %.0f%%\n", v,
              length(unique(cand[[v]])), 100 * max(table(cand[[v]])) / nrow(cand)))
cat("\nlocalize.external with one candidate plus engagement (COV needs two columns):\n")
for (v in setdiff(names(cand), "engagement")) show(paste(v, "+ engagement"), cand[, c(v, "engagement")])
cat("\nthe proposed group set:\n")
show("age+male+education+deprived+engagement+disability",
     cand[, c("age_band", "male", "education", "deprived", "engagement", "disability")])
