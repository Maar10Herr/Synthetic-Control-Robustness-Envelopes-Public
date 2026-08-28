#!/usr/bin/env Rscript

# Replay the official 13 transformed Synth predictors. Validation and outcome
# columns are already frozen in basque_k2.csv from the same source object.
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1) {
  stop("usage: extract_basque_full.R /path/to/basque.RData")
}
load(arguments[[1]])
if (!exists("basque")) {
  stop("input does not define basque")
}

units <- c(17, 2:16, 18)
regular_years <- 1964:1969
sector_years <- seq(1961, 1969, 2)
predictor_names <- c(
  "mean_school_illit_1964_1969",
  "mean_school_prim_1964_1969",
  "mean_school_med_1964_1969",
  "mean_school_high_combined_1964_1969",
  "mean_invest_1964_1969",
  "mean_gdpcap_1960_1969",
  "mean_sec_agriculture_odd_1961_1969",
  "mean_sec_energy_odd_1961_1969",
  "mean_sec_industry_odd_1961_1969",
  "mean_sec_construction_odd_1961_1969",
  "mean_sec_services_venta_odd_1961_1969",
  "mean_sec_services_nonventa_odd_1961_1969",
  "popdens_1969"
)
header <- c("regionno", "regionname", predictor_names)
cat(paste(header, collapse = ","), "\n", sep = "")
for (unit in units) {
  data <- basque[basque$regionno == unit, ]
  regular_mean <- function(variable) {
    mean(data[[variable]][data$year %in% regular_years], na.rm = TRUE)
  }
  sector_mean <- function(variable) {
    mean(data[[variable]][data$year %in% sector_years], na.rm = TRUE)
  }
  values <- c(
    regular_mean("school.illit"),
    regular_mean("school.prim"),
    regular_mean("school.med"),
    regular_mean("school.high") + regular_mean("school.post.high"),
    regular_mean("invest"),
    mean(data$gdpcap[data$year %in% 1960:1969], na.rm = TRUE),
    sector_mean("sec.agriculture"),
    sector_mean("sec.energy"),
    sector_mean("sec.industry"),
    sector_mean("sec.construction"),
    sector_mean("sec.services.venta"),
    sector_mean("sec.services.nonventa"),
    data$popdens[data$year == 1969]
  )
  cat(
    paste(
      c(unit, paste0('"', unique(data$regionname), '"'), sprintf("%.6f", values)),
      collapse = ","
    ),
    "\n",
    sep = ""
  )
}
