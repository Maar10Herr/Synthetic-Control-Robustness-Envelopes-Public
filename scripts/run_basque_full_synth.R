#!/usr/bin/env Rscript

# Reproduce the conventional local numerical Synth fit for comparison only.
.libPaths(c(file.path(".work", "Rlib"), .libPaths()))
suppressPackageStartupMessages(library(Synth))
data(basque, package = "Synth")

prepared <- dataprep(
  foo = basque,
  predictors = c(
    "school.illit", "school.prim", "school.med", "school.high",
    "school.post.high", "invest"
  ),
  predictors.op = "mean",
  dependent = "gdpcap",
  unit.variable = "regionno",
  time.variable = "year",
  special.predictors = list(
    list("gdpcap", 1960:1969, "mean"),
    list("sec.agriculture", seq(1961, 1969, 2), "mean"),
    list("sec.energy", seq(1961, 1969, 2), "mean"),
    list("sec.industry", seq(1961, 1969, 2), "mean"),
    list("sec.construction", seq(1961, 1969, 2), "mean"),
    list("sec.services.venta", seq(1961, 1969, 2), "mean"),
    list("sec.services.nonventa", seq(1961, 1969, 2), "mean"),
    list("popdens", 1969, "mean")
  ),
  treatment.identifier = 17,
  controls.identifier = c(2:16, 18),
  time.predictors.prior = 1964:1969,
  time.optimize.ssr = 1960:1969,
  unit.names.variable = "regionname",
  time.plot = 1955:1997
)
prepared$X1["school.high", ] <-
  prepared$X1["school.high", ] + prepared$X1["school.post.high", ]
prepared$X1 <- as.matrix(
  prepared$X1[-which(rownames(prepared$X1) == "school.post.high"), ]
)
prepared$X0["school.high", ] <-
  prepared$X0["school.high", ] + prepared$X0["school.post.high", ]
prepared$X0 <- prepared$X0[
  -which(rownames(prepared$X0) == "school.post.high"),
]

fit <- synth(data.prep.obj = prepared, verbose = FALSE)
post_years <- 1970:1975
treated_post <- mean(
  basque$gdpcap[basque$regionno == 17 & basque$year %in% post_years]
)
donor_ids <- as.integer(rownames(fit$solution.w))
donor_post <- vapply(
  donor_ids,
  function(unit) mean(
    basque$gdpcap[basque$regionno == unit & basque$year %in% post_years]
  ),
  numeric(1)
)
effect <- treated_post - sum(as.numeric(fit$solution.w) * donor_post)

# Evaluate the same numerical donor weights on the frozen six-decimal fixture
# used by the exact certificate.  The official Synth fit above uses the source
# package's full floating-point values, so these are deliberately separate
# apples-to-apples comparison quantities rather than replacements for fit$loss.v.
frozen <- read.csv(
  file.path("instances", "empirical", "basque_k2.csv"),
  check.names = FALSE
)
frozen_treated <- frozen[frozen$regionno == 17, , drop = FALSE]
frozen_donors <- frozen[
  match(donor_ids, frozen$regionno),
  ,
  drop = FALSE
]
frozen_pre_names <- paste0("gdpcap_", 1960:1969)
frozen_pre_gap <- as.numeric(frozen_treated[, frozen_pre_names]) -
  t(as.matrix(frozen_donors[, frozen_pre_names, drop = FALSE])) %*%
    as.numeric(fit$solution.w)
frozen_loss <- mean(as.numeric(frozen_pre_gap)^2)
frozen_effect <- as.numeric(frozen_treated$mean_gdpcap_1970_1975) -
  sum(
    as.numeric(fit$solution.w) *
      as.numeric(frozen_donors$mean_gdpcap_1970_1975)
  )

cat("status=conventional local numerical Synth output; not a global certificate\n")
cat("R=", as.character(getRversion()), "\n", sep = "")
for (package in c("Synth", "kernlab", "optimx", "rgenoud")) {
  cat(package, "=", as.character(packageVersion(package)), "\n", sep = "")
}
cat("loss_v=", sprintf("%.17g", as.numeric(fit$loss.v)), "\n", sep = "")
cat("effect_1970_1975=", sprintf("%.17g", effect), "\n", sep = "")
cat("frozen_six_decimal_loss=", sprintf("%.17g", frozen_loss), "\n", sep = "")
cat(
  "frozen_six_decimal_effect_1970_1975=",
  sprintf("%.17g", frozen_effect), "\n",
  sep = ""
)
for (index in seq_along(fit$solution.v)) {
  cat(
    "v,", names(fit$solution.v)[index], ",",
    sprintf("%.17g", fit$solution.v[index]), "\n",
    sep = ""
  )
}
for (index in seq_along(fit$solution.w)) {
  if (fit$solution.w[index] > 1e-8) {
    cat(
      "w,", rownames(fit$solution.w)[index], ",",
      sprintf("%.17g", fit$solution.w[index]), "\n",
      sep = ""
    )
  }
}
