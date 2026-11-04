## Step 5: Check B - the routine check an analyst runs on their own fitted model.
##
## Only the regression models have one. On the TRAINING cohort:
##   logistic regression - the in-sample battery, run.all.gof()
##   ridge               - the Hosmer-Lemeshow statistic a practitioner would compute on the
##                         ridge fit (uncorrected), beside shrink.gof's corrected test
##
## The ridge fit is rebuilt with the engine's own recipe (same folds, same lambda path,
## lambda.min) and must reproduce Check A's ridge predictions on the test cohort, so both checks
## are about the same model. The foundation models and LightGBM have no Check B: they are not
## parametric models, so the new-cohort check is the only one available for them.
##
##   Rscript 05_check_b.R stable BBB_2013J_2014J

args <- commandArgs(TRUE)
if (length(args) < 2) stop("usage: Rscript 05_check_b.R <feature set> <pair>")
fset <- args[1]; pair <- args[2]

PROJECT <- Sys.getenv("ICT_ROOT", "D:/Research/paper_ict_student_risk")
ENGINE  <- Sys.getenv("TFM_REBUILD_ROOT", "D:/Research/paper_tfm_calibration/rebuild")
Sys.setenv(TFM_REBUILD_ROOT = ENGINE)
source(file.path(ENGINE, "R", "gof_layer.R"))
use_project_lib()
suppressPackageStartupMessages(library(glmnet))

dir  <- file.path(PROJECT, "results", fset, pair)
run  <- jsonlite::fromJSON(file.path(dir, "run.json"))
pred <- read.csv(gzfile(file.path(dir, "preds.csv.gz")))
bits <- strsplit(pair, "_")[[1]]

## ------------------------------------------------ the data exactly as the engine saw them
feat <- read.csv(gzfile(file.path(PROJECT, "data", "features", "day28_features.csv.gz")))
cat_cols <- c("highest_education", "imd_band", "age_band")
for (v in grep("^clicks_", names(feat), value = TRUE)) feat[[v]] <- log1p(feat[[v]])
for (v in cat_cols) feat[[v]] <- factor(as.character(feat[[v]]), levels = sort(unique(as.character(feat[[v]]))))
cols <- run$columns
dev <- feat[feat$code_module == bits[1] & feat$code_presentation == bits[2], c(cols, "at_risk")]
new <- feat[feat$code_module == bits[1] & feat$code_presentation == bits[3], ]
new <- new[match(pred$id_student, new$id_student), c(cols, "at_risk")]
names(dev)[names(dev) == "at_risk"] <- "y"
stopifnot(nrow(dev) == run$n_train, sum(dev$y) == run$events_train, all(new$at_risk == pred$y))
f_main <- as.formula(paste("y ~", paste(cols, collapse = " + ")))
n <- nrow(dev)

out <- list()

## ------------------------------------------------ logistic regression: in-sample battery
fit <- glm(f_main, family = binomial, data = dev)
stopifnot(fit$converged)
## the engine's LR predictions must be this fit's predictions
p_new <- predict(fit, newdata = new, type = "response")
cat(sprintf("LR reproduces Check A's predictions: max |diff| = %.1e\n", max(abs(p_new - pred$lr))))
stopifnot(max(abs(p_new - pred$lr)) < 1e-6)

## In a module's first year nobody has a previous attempt, so that column is constant and glm
## gives it no coefficient (NA). The fit is the same without it, but several tests stop on the
## singular matrix, so the battery gets the fit with training-constant columns removed (D-08).
const <- cols[sapply(dev[cols], function(v) length(unique(v)) == 1L)]
if (length(const)) {
  cat("constant in the training cohort, left out of the battery:", const, "\n")
  fit <- glm(as.formula(paste("y ~", paste(setdiff(cols, const), collapse = " + "))),
             family = binomial, data = dev)
  stopifnot(fit$converged, max(abs(fitted(fit) - fitted(glm(f_main, family = binomial, data = dev)))) < 1e-8)
}
Xnum <- model.matrix(formula(fit), dev)[, -1, drop = FALSE]
t0 <- Sys.time()
b <- suppressWarnings(run.all.gof(fit, X = Xnum, include_slow = FALSE, install = "no"))
b <- as.data.frame(b)
cat(sprintf("run.all.gof on the LR fit: %d tests, %.0fs\n", nrow(b),
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))
out$lr <- data.frame(model = "lr", test = b$Test, family = b$Family, statistic = b$Statistic,
                     df = b$df, p_value = b$p_value, note = b$Note, stringsAsFactors = FALSE)

## ------------------------------------------------ ridge: rebuild the engine's fit exactly
x  <- model.matrix(f_main, dev)[, -1, drop = FALSE]
xn <- model.matrix(as.formula(paste("~", paste(cols, collapse = " + "))), new)[, -1, drop = FALSE]
seed <- run$seeds$ridge
set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
fold <- integer(n)
for (cl in unique(dev$y)) { w <- which(dev$y == cl); fold[w] <- sample(rep(seq_len(10L), length.out = length(w))) }
lmax <- max(glmnet(x, dev$y, family = "binomial", alpha = 0, nlambda = 5)$lambda)
lam <- exp(seq(log(lmax), log(lmax * 1e-6), length.out = 200L))
glmnet.control(fdev = 0)
cv <- cv.glmnet(x, dev$y, family = "binomial", alpha = 0, foldid = fold, lambda = lam,
                type.measure = "deviance")
glmnet.control(factory = TRUE)
p_new <- as.numeric(predict(cv, newx = xn, s = "lambda.min", type = "response"))
cat(sprintf("ridge reproduces Check A's predictions: max |diff| = %.1e\n", max(abs(p_new - pred$ridge))))
stopifnot(max(abs(p_new - pred$ridge)) < 1e-6)
lam_min <- cv$lambda.min

## shrink.gof fits its own ridge in base R, unstandardized. glmnet standardized x internally
## (population sd, 1/n), so the same model is X standardized that way with lambda = n * lambda_min.
mu <- colMeans(x)
sdp <- apply(x, 2, function(v) sqrt(mean((v - mean(v))^2)))
keep <- sdp > 0
Xs <- sweep(sweep(x[, keep, drop = FALSE], 2, mu[keep]), 2, sdp[keep], "/")
## the same model, two ways: fitted probabilities from glmnet on Xs must match the CV fit
chk <- glmnet(Xs, dev$y, family = "binomial", alpha = 0, lambda = lam, standardize = FALSE)
p_chk <- as.numeric(predict(chk, newx = Xs, s = lam_min, type = "response", exact = FALSE))
p_cv  <- as.numeric(predict(cv, newx = x, s = "lambda.min", type = "response"))
cat(sprintf("standardized refit matches the CV fit in-sample: max |diff| = %.1e\n", max(abs(p_chk - p_cv))))

t0 <- Sys.time()
sg <- shrink.gof(Xs, dev$y, lambda = n * lam_min, G = 10, basis = c("edge", "decile"), B = 999,
                 seed = task_seed(paste("ict2026", fset, pair, "ridge"), "shrink"), uncorrected = TRUE)
cat(sprintf("shrink.gof: %.0fs\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
print(sg)
saveRDS(sg, file.path(dir, "check_b_shrink.rds"))
## two rows per basis: the uncorrected statistic a practitioner would compute, and the corrected one
for (b in intersect(c("SC.HL", "SC.EDGE"), names(sg))) {
  base <- if (b == "SC.HL") "Hosmer-Lemeshow" else "EDGE"
  out[[paste0("ridge_", b)]] <- data.frame(
    model = "ridge", test = paste(base, c("(routine, uncorrected)", "(shrinkage-corrected)")),
    family = "Shrinkage", statistic = c(NA, sg[[b]]$statistic), df = NA,
    p_value = c(sg[[b]]$p.uncorrected, sg[[b]]$p.value),
    note = sprintf("lambda=%.4g (theory scale); G=%d; B=%d", sg$lambda, sg$G, sg$B),
    stringsAsFactors = FALSE)
}

res <- do.call(rbind, out)
res$pair <- pair; res$feature_set <- fset
write.csv(res, file.path(dir, "check_b.csv"), row.names = FALSE)
cat("\nLR in-sample battery (p-values):\n")
print(format(res[, c("test", "family", "p_value")], digits = 2), row.names = FALSE)
cat(sprintf("\nridge: lambda.min = %.4g (theory scale %.4g)\n", lam_min, n * lam_min))
## recorded for the shrinkage plasmode (07), which must use THIS fit's strength rather than a typed value
jsonlite::write_json(list(pair = pair, feature_set = fset, n = n, lambda_min_glmnet = lam_min,
                          lambda_theory = n * lam_min, columns = ncol(Xs)),
                     file.path(dir, "ridge_lambda.json"), auto_unbox = TRUE, digits = NA)
