"""Exact loader for the frozen full Proposition 99 design."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from .scm_oracle import Matrix, Rat, Vector


PREDICTOR_NAMES = (
    "mean_lnincome_1980_1988",
    "mean_retprice_1980_1988",
    "mean_age15to24_1980_1988",
    "mean_beer_1984_1988",
    "cigsale_1975",
    "cigsale_1980",
    "cigsale_1988",
)
VALIDATION_NAMES = tuple(f"cigsale_{year}" for year in range(1970, 1989))
POST_NAME = "mean_cigsale_1989_2000"
EXPECTED_HEADER = ("state",) + PREDICTOR_NAMES + VALIDATION_NAMES + (POST_NAME,)


@dataclass(frozen=True)
class Prop99Design:
    treated_label: str
    donor_labels: Tuple[str, ...]
    predictor_names: Tuple[str, ...]
    predictor_target: Vector
    predictor_donors: Matrix
    validation_names: Tuple[str, ...]
    validation_target: Vector
    validation_donors: Matrix
    treated_post: Rat
    donor_post: Vector


def load_prop99_full(path: Path) -> Prop99Design:
    """Load the positional CSV schema without collapsing duplicate headers.

    Three cigarette-sale predictors also occur in the validation sequence, so
    name-keyed CSV readers can silently collapse columns.  Requiring the exact
    header and reading by position makes that ambiguity impossible.
    """

    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or tuple(rows[0]) != EXPECTED_HEADER:
        raise ValueError("unexpected Proposition 99 fixture header")
    data = rows[1:]
    if len(data) != 39 or data[0][0] != "California":
        raise ValueError("fixture must contain California and 38 controls")
    if any(len(row) != len(EXPECTED_HEADER) for row in data):
        raise ValueError("fixture row width does not match its header")
    donor_labels = tuple(row[0] for row in data[1:])
    if any(not label for label in donor_labels) or len(set(donor_labels)) != len(
        donor_labels
    ):
        raise ValueError("control states must have unique nonempty labels")
    if donor_labels != tuple(sorted(donor_labels)):
        raise ValueError("control states must be sorted")

    predictor_start = 1
    validation_start = predictor_start + len(PREDICTOR_NAMES)
    post_index = validation_start + len(VALIDATION_NAMES)
    return Prop99Design(
        data[0][0],
        tuple(row[0] for row in data[1:]),
        PREDICTOR_NAMES,
        tuple(Rat(value) for value in data[0][predictor_start:validation_start]),
        tuple(
            tuple(Rat(row[column]) for row in data[1:])
            for column in range(predictor_start, validation_start)
        ),
        VALIDATION_NAMES,
        tuple(Rat(value) for value in data[0][validation_start:post_index]),
        tuple(
            tuple(Rat(row[column]) for row in data[1:])
            for column in range(validation_start, post_index)
        ),
        Rat(data[0][post_index]),
        tuple(Rat(row[post_index]) for row in data[1:]),
    )
