## Does the code shown in Section 5 of the paper run as printed, on one real cohort?
##   Rscript check_snippet.R
ENGINE <- Sys.getenv("TFM_REBUILD_ROOT", "D:/Research/paper_tfm_calibration/rebuild")
Sys.setenv(TFM_REBUILD_ROOT = ENGINE)
source(file.path(ENGINE, "R", "gof_layer.R")); use_project_lib()   # the pinned 2.10.0
library(ebrahim.gof)
cat("ebrahim.gof", as.character(packageVersion("ebrahim.gof")), "\n")
p_all <- read.csv(gzfile(file.path(Sys.getenv("ICT_ROOT", "D:/Research/paper_ict_student_risk"),
                           "results/stable/BBB_2013J_2014J/preds.csv.gz")))
y <- p_all$y
p <- p_all$tabpfn
groups <- data.frame(activity = cut(rank(p), 4, labels = FALSE))   # any coded student groups
print(run.all.external(y, p))
print(localize.external(y, p, groups))
