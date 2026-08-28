#!/usr/bin/env Rscript

# Conventional local numerical Synth fit for the CRAN tidysynth Proposition 99
# README design. Source and frozen-six-decimal results are kept separate.
.libPaths(c(file.path(".work", "Rlib"), .libPaths()))
suppressPackageStartupMessages(library(Synth))

arguments <- commandArgs(trailingOnly = TRUE)
source_path <- if (length(arguments) == 1) {
  arguments[[1]]
} else {
  file.path(".work", "tidysynth", "data", "smoking.rda")
}
load(source_path)
if (!exists("smoking")) {
  stop("source path does not define smoking")
}

treated <- "California"
states <- c(treated, sort(setdiff(unique(smoking$state), treated)))
smoking$id <- match(smoking$state, states)
prepared <- dataprep(
  foo = as.data.frame(smoking),
  predictors = c("lnincome", "retprice", "age15to24"),
  predictors.op = "mean",
  dependent = "cigsale",
  unit.variable = "id",
  time.variable = "year",
  special.predictors = list(
    list("beer", 1984:1988, "mean"),
    list("cigsale", 1975, "mean"),
    list("cigsale", 1980, "mean"),
    list("cigsale", 1988, "mean")
  ),
  treatment.identifier = 1,
  controls.identifier = 2:length(states),
  time.predictors.prior = 1980:1988,
  time.optimize.ssr = 1970:1988,
  unit.names.variable = "state",
  time.plot = 1970:2000
)
source_fit <- synth(data.prep.obj = prepared, verbose = FALSE)

frozen <- read.csv(
  file.path("instances", "empirical", "prop99_full.csv"),
  check.names = FALSE
)
predictor_names <- names(frozen)[2:8]
validation_names <- paste0("cigsale_", 1970:1988)
frozen_prepared <- prepared
frozen_prepared$X1 <- matrix(
  as.numeric(frozen[1, predictor_names]),
  ncol = 1,
  dimnames = list(predictor_names, treated)
)
frozen_prepared$X0 <- t(as.matrix(frozen[-1, predictor_names]))
rownames(frozen_prepared$X0) <- predictor_names
colnames(frozen_prepared$X0) <- as.character(2:length(states))
frozen_prepared$Z1 <- matrix(
  as.numeric(frozen[1, validation_names]),
  ncol = 1,
  dimnames = list(1970:1988, treated)
)
frozen_prepared$Z0 <- t(as.matrix(frozen[-1, validation_names]))
rownames(frozen_prepared$Z0) <- as.character(1970:1988)
colnames(frozen_prepared$Z0) <- as.character(2:length(states))
frozen_fit <- synth(data.prep.obj = frozen_prepared, verbose = FALSE)

effect_for <- function(fit, values) {
  as.numeric(values$mean_cigsale_1989_2000[1]) -
    sum(as.numeric(fit$solution.w) * as.numeric(values$mean_cigsale_1989_2000[-1]))
}
source_post <- vapply(
  states,
  function(state) mean(
    smoking$cigsale[smoking$state == state & smoking$year %in% 1989:2000]
  ),
  numeric(1)
)
source_effect <- source_post[1] - sum(as.numeric(source_fit$solution.w) * source_post[-1])
frozen_effect <- effect_for(frozen_fit, frozen)

cat("status=conventional local numerical Synth output; not a global certificate\n")
cat("R=", as.character(getRversion()), "\n", sep = "")
for (package in c("Synth", "kernlab", "optimx", "rgenoud")) {
  cat(package, "=", as.character(packageVersion(package)), "\n", sep = "")
}
cat("source_loss_v=", sprintf("%.17g", as.numeric(source_fit$loss.v)), "\n", sep = "")
cat("source_effect_mean_1989_2000=", sprintf("%.17g", source_effect), "\n", sep = "")
cat("frozen_loss_v=", sprintf("%.17g", as.numeric(frozen_fit$loss.v)), "\n", sep = "")
cat("frozen_effect_mean_1989_2000=", sprintf("%.17g", frozen_effect), "\n", sep = "")
for (index in seq_along(frozen_fit$solution.v)) {
  cat(
    "v,", predictor_names[index], ",",
    sprintf("%.17g", frozen_fit$solution.v[index]), "\n",
    sep = ""
  )
}
for (index in seq_along(frozen_fit$solution.w)) {
  if (frozen_fit$solution.w[index] > 1e-8) {
    cat(
      "w,", frozen$state[index + 1], ",",
      sprintf("%.17g", frozen_fit$solution.w[index]), "\n",
      sep = ""
    )
  }
}
