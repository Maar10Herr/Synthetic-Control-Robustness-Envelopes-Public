#!/usr/bin/env Rscript

# Replay the six-decimal Proposition 99 transformation declared by the CRAN
# tidysynth 0.2.1 README. The output fixes California first and sorts all
# control states alphabetically.
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1) {
  stop("usage: extract_prop99_full.R /path/to/smoking.rda")
}
load(arguments[[1]])
if (!exists("smoking")) {
  stop("input does not define smoking")
}

treated <- "California"
states <- c(treated, sort(setdiff(unique(smoking$state), treated)))
predictor_names <- c(
  "mean_lnincome_1980_1988",
  "mean_retprice_1980_1988",
  "mean_age15to24_1980_1988",
  "mean_beer_1984_1988",
  "cigsale_1975",
  "cigsale_1980",
  "cigsale_1988"
)
validation_names <- paste0("cigsale_", 1970:1988)
header <- c("state", predictor_names, validation_names, "mean_cigsale_1989_2000")
cat(paste(header, collapse = ","), "\n", sep = "")

for (state in states) {
  data <- smoking[smoking$state == state, ]
  window_mean <- function(variable, years) {
    values <- data[[variable]][data$year %in% years]
    if (anyNA(values) || length(values) != length(years)) {
      stop(paste("incomplete", variable, "window for", state))
    }
    mean(values)
  }
  at_year <- function(variable, year) {
    value <- data[[variable]][data$year == year]
    if (length(value) != 1 || is.na(value)) {
      stop(paste("missing", variable, year, "for", state))
    }
    value
  }
  values <- c(
    window_mean("lnincome", 1980:1988),
    window_mean("retprice", 1980:1988),
    window_mean("age15to24", 1980:1988),
    window_mean("beer", 1984:1988),
    at_year("cigsale", 1975),
    at_year("cigsale", 1980),
    at_year("cigsale", 1988),
    vapply(1970:1988, function(year) at_year("cigsale", year), numeric(1)),
    window_mean("cigsale", 1989:2000)
  )
  escaped_state <- gsub('"', '""', state, fixed = TRUE)
  cat(
    paste(c(paste0('"', escaped_state, '"'), sprintf("%.6f", values)), collapse = ","),
    "\n",
    sep = ""
  )
}
