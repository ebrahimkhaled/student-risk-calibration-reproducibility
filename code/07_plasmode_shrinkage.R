## Step 7: does the routine check mislead after ridge, and does shrink.gof fix it?
##
## A plasmode study on real student data, so the answer is known by construction.
##
##   covariates  the real day-28 covariates of one training cohort (BBB 2013J, stable set)
##   truth       a logistic model whose coefficients are the unpenalized fit to the real outcomes,
##               so the model being checked is CORRECT and every alarm is a false alarm
##   outcomes    simulated from that truth, a fresh draw per replicate
##   fit         ridge at fixed strengths: multiples of the strength cross-validation chose on the
##               real cohort (0 = no shrinkage, where the routine check is valid)
##   checks      shrink.gof(..., uncorrected = TRUE): the routine Hosmer-Lemeshow and EDGE a
##               practitioner would compute on the ridge fit, and the corrected tests
##
## A valid check raises a false alarm at its nominal rate, 5%. One extra condition, "misfit", draws
## the outcomes from a truth with a curved effect of online activity that the fitted model leaves
## out, to confirm the corrected check still catches a real problem.
##
## Every replicate is written to disk as it finishes, one file per (condition, worker), and a rerun
## skips what is done. Outcomes are drawn from a per-replicate seed, never from the loop's stream,
## and each draw's fingerprint is stored.
##
##   Rscript 07_plasmode_shrinkage.R            # full run
##   Rscript 07_plasmode_shrinkage.R --smoke    # 4 replicates per condition, 2 workers

args  <- commandArgs(TRUE)
SMOKE <- "--smoke" %in% args
R_REP <- if (SMOKE) 4L else 1000L
R_MIS <- if (SMOKE) 4L else 500L
B     <- 199L                       # reference draws per test: p-values on a 1/200 grid
NWORK <- if (SMOKE) 2L else 20L
MULT  <- c(0, 0.5, 1, 2, 4, 8)      # ridge strength as a multiple of the cross-validated choice

PROJECT <- Sys.getenv("ICT_ROOT", "D:/Research/paper_ict_student_risk")
ENGINE  <- Sys.getenv("TFM_REBUILD_ROOT", "D:/Research/paper_tfm_calibration/rebuild")
OUT <- file.path(PROJECT, "results", if (SMOKE) "plasmode_smoke" else "plasmode")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(TFM_REBUILD_ROOT = ENGINE)
source(file.path(ENGINE, "R", "gof_layer.R"))
use_project_lib()
suppressPackageStartupMessages({ library(glmnet); library(parallel) })

## ------------------------------------------------ the real covariates
pair <- "BBB_2013J_2014J"
run  <- jsonlite::fromJSON(file.path(PROJECT, "results", "stable", pair, "run.json"))
feat <- read.csv(gzfile(file.path(PROJECT, "data", "features", "day28_features.csv.gz")))
for (v in grep("^clicks_", names(feat), value = TRUE)) feat[[v]] <- log1p(feat[[v]])
for (v in c("highest_education", "imd_band", "age_band"))
  feat[[v]] <- factor(as.character(feat[[v]]), levels = sort(unique(as.character(feat[[v]]))))
dev <- feat[feat$code_module == "BBB" & feat$code_presentation == "2013J", c(run$columns, "at_risk")]
f <- as.formula(paste("at_risk ~", paste(run$columns, collapse = " + ")))
x <- model.matrix(f, dev)[, -1, drop = FALSE]
n <- nrow(x)
## standardized exactly as glmnet standardizes (population sd), the scale shrink.gof penalizes on
sdp <- apply(x, 2, function(v) sqrt(mean((v - mean(v))^2)))
x <- x[, sdp > 0, drop = FALSE]; sdp <- sdp[sdp > 0]
Xs <- sweep(sweep(x, 2, colMeans(x)), 2, sdp, "/")

## the truth: the unpenalized fit to the real outcomes, on the standardized scale
fit0 <- glm(dev$at_risk ~ Xs, family = binomial)
stopifnot(fit0$converged, !anyNA(coef(fit0)))
eta_null <- as.numeric(cbind(1, Xs) %*% coef(fit0))

## misfit truth: add a curved effect of total online activity that the fitted model omits, at two
## strengths fixed in advance (a quarter and a half of the linear effect's scale, on the squared
## z-score). Neither is tuned to any result.
##
## Each misfit strength is run twice: with NO shrinkage (x0), where the routine and corrected checks
## are both valid, so their rejection rate is the honest power against this misfit; and at the
## cross-validated strength (x1), where the corrected check should keep that power. Without the x0
## control, a corrected check that rarely rejects could mean lost power or an undetectable misfit;
## the smoke test (4 replicates) could not tell the two apart.
z <- as.numeric(scale(dev$clicks_total))
b_clicks <- abs(coef(fit0)[["Xsclicks_total"]])
## 1.0 added after the smoke test (deviation D-06): with no shrinkage, 0.25 and 0.5 were caught in
## at most 1 of 4 replicates by checks that are valid there, so the power control had almost nothing
## to detect. The null conditions - the false-alarm rates the figure is about - are unchanged.
MISFIT <- c(0.25, 0.5, 1.0)
eta_misfit <- lapply(setNames(MISFIT, sprintf("%g", MISFIT)), function(s) eta_null + s * b_clicks * (z^2 - 1))

## the strength cross-validation chose on the real cohort, read from Check B's record of that fit
lam_file <- file.path(PROJECT, "results", "stable", pair, "ridge_lambda.json")
if (!file.exists(lam_file)) stop("run Check B on ", pair, " first: ", lam_file, " is missing")
lam_cv <- jsonlite::fromJSON(lam_file)$lambda_min_glmnet       # glmnet scale
stopifnot(jsonlite::fromJSON(lam_file)$columns == ncol(Xs))    # the same design as here
conds <- rbind(
  data.frame(cond = sprintf("null_x%g", MULT), truth = "null", strength = 0, mult = MULT, reps = R_REP),
  do.call(rbind, lapply(MISFIT, function(s)
    data.frame(cond = sprintf("misfit%g_x%g", s, c(0, 1)), truth = "misfit", strength = s,
               mult = c(0, 1), reps = R_MIS))))

seed_of <- function(cond, r) task_seed("ict2026", "plasmode", cond, r)

one_rep <- function(i) {
  cd <- conds[i$k, ]; r <- i$r
  eta <- if (cd$truth == "null") eta_null else eta_misfit[[sprintf("%g", cd$strength)]]
  ## the outcomes depend on (truth, strength, replicate) only, so every shrinkage level of a
  ## replicate checks the SAME simulated cohort (common random numbers); the reference draws
  ## inside shrink.gof get a seed of their own per condition
  reset_rng(seed_of(paste(cd$truth, cd$strength), r))
  y <- rbinom(n, 1, plogis(eta))
  lam <- n * lam_cv * cd$mult
  t0 <- proc.time()[["elapsed"]]
  s <- tryCatch(shrink.gof(Xs, y, lambda = lam, G = 10, basis = c("edge", "decile"), B = B,
                           seed = seed_of(cd$cond, r), uncorrected = TRUE),
                error = function(e) e)
  secs <- proc.time()[["elapsed"]] - t0
  base <- data.frame(cond = cd$cond, truth = cd$truth, strength = cd$strength, mult = cd$mult,
                     rep = r, n = n,
                     lambda_theory = lam, events = sum(y), y_hash = data_fingerprint(y), secs = secs)
  if (inherits(s, "error"))
    return(cbind(base, hl_routine = NA, hl_corrected = NA, edge_routine = NA, edge_corrected = NA,
                 error = conditionMessage(s)))
  cbind(base, hl_routine = s$SC.HL$p.uncorrected, hl_corrected = s$SC.HL$p.value,
        edge_routine = s$SC.EDGE$p.uncorrected, edge_corrected = s$SC.EDGE$p.value, error = "")
}

## ------------------------------------------------ resume: what is already on disk
done_keys <- character()
for (fp in list.files(OUT, pattern = "^part_.*\\.csv$", full.names = TRUE)) {
  d <- tryCatch(read.csv(fp), error = function(e) NULL)
  if (!is.null(d) && nrow(d)) done_keys <- c(done_keys, paste(d$cond, d$rep))
}
todo <- list()
for (k in seq_len(nrow(conds))) for (r in seq_len(conds$reps[k]))
  if (!(paste(conds$cond[k], r) %in% done_keys)) todo[[length(todo) + 1L]] <- list(k = k, r = r)
cat(sprintf("n = %d students, %d columns; truth events rate %.3f\n", n, ncol(Xs), mean(plogis(eta_null))))
cat(sprintf("%d replicates to run (%d already on disk), %d workers\n",
            length(todo), length(done_keys), NWORK))
if (!length(todo)) quit(save = "no")

## ------------------------------------------------ run: each worker appends its own file
cl <- makeCluster(NWORK)
on.exit(stopCluster(cl), add = TRUE)
clusterExport(cl, c("conds", "eta_null", "eta_misfit", "Xs", "n", "lam_cv", "B", "OUT", "ENGINE",
                    "seed_of", "one_rep"), envir = environment())
invisible(clusterEvalQ(cl, {
  Sys.setenv(TFM_REBUILD_ROOT = ENGINE)
  source(file.path(ENGINE, "R", "gof_layer.R")); use_project_lib(); NULL }))
chunks <- split(todo, rep_len(seq_len(NWORK), length(todo)))
t0 <- Sys.time()
res_par <- parLapply(cl, seq_along(chunks), function(w, chunks) {
  fp <- file.path(OUT, sprintf("part_w%02d_%d.csv", w, Sys.getpid()))
  for (it in chunks[[w]]) {
    row <- one_rep(it)
    write.table(row, fp, sep = ",", row.names = FALSE, col.names = !file.exists(fp), append = TRUE)
  }
  NULL
}, chunks = chunks)
cat(sprintf("done in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))

## ------------------------------------------------ summary with Monte Carlo standard errors
all <- do.call(rbind, lapply(list.files(OUT, pattern = "^part_.*\\.csv$", full.names = TRUE), read.csv))
all <- all[!duplicated(paste(all$cond, all$rep)), ]
all$error[is.na(all$error)] <- ""                               # an empty cell reads back as NA
stopifnot(!anyDuplicated(paste(all$cond, all$y_hash)))         # every draw within a condition distinct
## common random numbers: the same replicate must carry the same cohort at every shrinkage level
crn <- tapply(all$y_hash, paste(all$truth, all$strength, all$rep), function(h) length(unique(h)))
stopifnot(all(crn == 1))
rate <- function(p) { p <- p[!is.na(p)]; k <- mean(p <= 0.05); c(rate = k, se = sqrt(k * (1 - k) / length(p)), R = length(p)) }
summ <- do.call(rbind, lapply(split(all, all$cond), function(d) {
  data.frame(cond = d$cond[1], truth = d$truth[1], strength = d$strength[1], mult = d$mult[1],
             hl_routine = rate(d$hl_routine)[["rate"]], hl_routine_se = rate(d$hl_routine)[["se"]],
             hl_corrected = rate(d$hl_corrected)[["rate"]], hl_corrected_se = rate(d$hl_corrected)[["se"]],
             edge_routine = rate(d$edge_routine)[["rate"]], edge_routine_se = rate(d$edge_routine)[["se"]],
             edge_corrected = rate(d$edge_corrected)[["rate"]], edge_corrected_se = rate(d$edge_corrected)[["se"]],
             R = nrow(d), failed = sum(d$error != ""), mean_secs = mean(d$secs))
}))
summ <- summ[order(summ$truth, summ$strength, summ$mult), ]
write.csv(summ, file.path(OUT, "summary.csv"), row.names = FALSE)
cat("\nrejection rate at the 5% level (Monte Carlo SE in brackets)\n")
for (i in seq_len(nrow(summ))) with(summ[i, ], cat(sprintf(
  "  %-10s  HL routine %.3f (%.3f)  corrected %.3f (%.3f)   EDGE routine %.3f (%.3f)  corrected %.3f (%.3f)   R=%d fail=%d\n",
  cond, hl_routine, hl_routine_se, hl_corrected, hl_corrected_se,
  edge_routine, edge_routine_se, edge_corrected, edge_corrected_se, R, failed)))
