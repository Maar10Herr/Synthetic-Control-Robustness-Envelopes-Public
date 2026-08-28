#!/usr/bin/env Rscript

# Replay the decimal transformation recorded in synth_demo_k2.json.
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1) {
  stop("usage: extract_synth_demo_k2.R /path/to/synth.data.RData")
}

load(arguments[[1]])
if (!exists("synth.data")) {
  stop("input does not define synth.data")
}

treated <- 7
controls <- c(29, 2, 13, 17, 32, 38)
units <- c(treated, controls)

unit_value <- function(unit, variable, years, operation = mean) {
  values <- synth.data[
    synth.data$unit.num == unit & synth.data$year %in% years,
    variable
  ]
  operation(values, na.rm = TRUE)
}

for (variable in c("X1", "X2")) {
  values <- vapply(
    units,
    unit_value,
    numeric(1),
    variable = variable,
    years = 1984:1989
  )
  cat(variable, paste(units, sprintf("%.9f", values), sep = ":", collapse = ","), "\n")
}

for (year in 1984:1991) {
  values <- vapply(
    units,
    unit_value,
    numeric(1),
    variable = "Y",
    years = year,
    operation = function(value, na.rm) value[[1]]
  )
  cat("Y", year, paste(units, sprintf("%.1f", values), sep = ":", collapse = ","), "\n")
}

