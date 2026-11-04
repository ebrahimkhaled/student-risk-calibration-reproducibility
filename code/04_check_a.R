## Step 4: Check A - the new-cohort check, for every model.
##
## Runs the external battery of the tabular-foundation-model study (#11) unchanged on each
## model's predictions for the test cohort: the calibration tests of run.all.external,
## EDGE in external mode, localize.external (which group of students a model is wrong
## about) and DeepGOF-1 external. The layer refuses to run unless ebrahim.gof is the
## pinned build, so the numbers are produced by exactly the code #11 used.
##
##   Rscript 04_check_a.R stable BBB_2013J_2014J
##
## Reads  results/<set>/<pair>/preds.csv.gz  and the test cohort's covariates;
## writes results/<set>/<pair>/check_a.csv

args <- commandArgs(TRUE)
if (length(args) < 2) stop("usage: Rscript 04_check_a.R <feature set> <pair>")
fset <- args[1]; pair <- args[2]

PROJECT <- Sys.getenv("ICT_ROOT", "D:/Research/paper_ict_student_risk")
ENGINE  <- Sys.getenv("TFM_REBUILD_ROOT", "D:/Research/paper_tfm_calibration/rebuild")
Sys.setenv(TFM_REBUILD_ROOT = ENGINE)
source(file.path(ENGINE, "R", "gof_layer.R"))
use_project_lib()                     # stops unless ebrahim.gof is build 2.10.0@19480aa

dir  <- file.path(PROJECT, "results", fset, pair)
pred <- read.csv(gzfile(file.path(dir, "preds.csv.gz")))
run  <- jsonlite::fromJSON(file.path(dir, "run.json"))

## the covariates the test students had, in the same order as the predictions
feat <- read.csv(gzfile(file.path(PROJECT, "data", "features", "day28_features.csv.gz")))
bits <- strsplit(pair, "_")[[1]]
te <- feat[feat$code_module == bits[1] & feat$code_presentation == bits[3], ]
te <- te[match(pred$id_student, te$id_student), ]
stopifnot(identical(te$id_student, pred$id_student), all(te$at_risk == pred$y))

## the battery's covariate checks (le Cessie, localize, DeepGOF) use the numeric predictors
## the models saw; categorical background is left to the calibration tests
num <- setdiff(run$columns, c("highest_education", "imd_band", "age_band"))
X <- as.matrix(te[, num])
for (j in grep("^clicks_", colnames(X))) X[, j] <- log1p(X[, j])

## localize.external gets student GROUPS rather than the model inputs (the rule of #11, 10.3):
## small integer codes it can split on. Columns with many values but one dominant one (previous
## attempts, 89% zeros; credits, 75% on one value) make its spline knots collapse, so they are not
## used here. An unknown deprivation band keeps its own level rather than being imputed.
edu <- c("No Formal quals" = 0, "Lower Than A Level" = 0, "A Level or Equivalent" = 1,
         "HE Qualification" = 2, "Post Graduate Qualification" = 2)
## the fairness groups use the ORIGINAL levels (kept as *_raw), not the model's grouped ones:
## unknown deprivation keeps its own level here even though the models merge it into "middle"
imd <- suppressWarnings(as.integer(sub("-.*", "", te$imd_band_raw)))   # "0-10%" -> 0; "Unknown" -> NA
XL <- cbind(
  age_band   = as.integer(factor(te$age_band_raw, levels = c("0-35", "35-55", "55<="))) - 1L,
  male       = te$gender,
  disability = te$disability,
  education  = unname(edu[te$highest_education_raw]),
  deprived   = ifelse(is.na(imd), 3L, ifelse(imd < 30, 0L, ifelse(imd < 70, 1L, 2L))),
  engagement = as.integer(cut(rank(te$clicks_total, ties.method = "first"), 4, labels = FALSE)) - 1L)
stopifnot(!anyNA(XL))
XL <- XL[, apply(XL, 2, function(v) length(unique(v)) > 1), drop = FALSE]   # no constant column

localize_block <- function(y, p, seed_base, M_loc = 999L) {
  r <- timed(localize.external(y, clamp_p(p), XL, M = M_loc,
                               seed = task_seed(seed_base, "localize"), plot = FALSE))
  if (inherits(r$val, "gof_fail") || inherits(r$val, "error"))
    return(one_row("Localize (any part named)", "Localize", note = paste("Failed:", r$val), secs = r$secs))
  L <- r$val; g <- names(L$single)
  named <- if (length(L$named)) paste(L$named, collapse = "+") else "none"
  rbind(one_row("Localize (any part named)", "Localize", length(L$named), NA, min(L$adjusted),
                sprintf("named=%s; M=%d", named, L$draws), r$secs),
        one_row(paste0("Localize ", g, " (named)"), "Localize", NA, NA, L$adjusted,
                ifelse(g %in% L$named, "named", "not named"), NA),
        one_row(paste0("Localize ", g, " (alone)"), "Localize", NA, NA, L$single, "single-group p", NA))
}

models <- setdiff(names(pred), c("id_student", "y"))
out <- list()
for (m in models) {
  t0 <- Sys.time()
  sb <- paste("ict2026", fset, pair, m)
  r <- rbind(gof_battery(pred$y, pred[[m]], X = X, seed_base = sb, do_localize = FALSE),
             localize_block(pred$y, pred[[m]], sb))
  r$model <- m
  out[[m]] <- r
  cat(sprintf("  %-9s %d rows  %.0fs\n", m, nrow(r), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
res <- do.call(rbind, out)
res$pair <- pair; res$feature_set <- fset
write.csv(res, file.path(dir, "check_a.csv"), row.names = FALSE)

## every test side by side, leaving out the per-group localize rows
pick <- res[!grepl("^Localize .*\\((named|alone)\\)$", res$test) & !is.na(res$p_value), ]
w <- reshape(pick[, c("test", "model", "p_value")], idvar = "test", timevar = "model",
             direction = "wide")
names(w) <- sub("^p_value\\.", "", names(w))
cat("\np-values (rows: test, columns: model)\n")
print(format(w, digits = 2), row.names = FALSE)

loc <- res[res$test == "Localize (any part named)", c("model", "note")]
cat("\nwhat kind of miscalibration localize names, per model:\n")
for (i in seq_len(nrow(loc))) cat(sprintf("  %-9s %s\n", loc$model[i], loc$note[i]))
