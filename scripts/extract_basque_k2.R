#!/usr/bin/env Rscript

# Replay the six-decimal transformation recorded in basque_k2.csv.
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1) {
  stop("usage: extract_basque_k2.R /path/to/basque.RData")
}
load(arguments[[1]])
if (!exists("basque")) {
  stop("input does not define basque")
}

units <- c(17, 2:16, 18)
header <- c(
  "regionno", "regionname", "mean_gdpcap_1960_1969",
  "mean_invest_1964_1969", paste0("gdpcap_", 1960:1969),
  "mean_gdpcap_1970_1975"
)
cat(paste(header, collapse = ","), "\n", sep = "")
for (unit in units) {
  data <- basque[basque$regionno == unit, ]
  values <- c(
    mean(data$gdpcap[data$year %in% 1960:1969], na.rm = TRUE),
    mean(data$invest[data$year %in% 1964:1969], na.rm = TRUE),
    vapply(1960:1969, function(year) data$gdpcap[data$year == year], numeric(1)),
    mean(data$gdpcap[data$year %in% 1970:1975], na.rm = TRUE)
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
