"""Transparent exact oracle for tiny rational SCM instances.

The oracle enumerates nonempty supports and checks exact KKT conditions using
``fractions.Fraction``. Its bilevel routine is exact over a caller-supplied
finite grid of predictor weights. Separate LP/QP support routines exactly
accept or reject active supports proposed by other solvers. This module alone
does not optimize over the continuous predictor-weight simplex.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple


Rat = Fraction
Vector = Tuple[Rat, ...]
Matrix = Tuple[Vector, ...]


def _f(value: object) -> Rat:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        raise TypeError("use integers, strings, or Fraction; binary floats are not exact")
    return Fraction(value)  # type: ignore[arg-type]


def as_vector(values: Iterable[object]) -> Vector:
    return tuple(_f(value) for value in values)


def as_matrix(rows: Iterable[Iterable[object]]) -> Matrix:
    matrix = tuple(as_vector(row) for row in rows)
    if matrix and any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("ragged matrix")
    return matrix


def _dot(left: Sequence[Rat], right: Sequence[Rat]) -> Rat:
    return sum((a * b for a, b in zip(left, right)), Rat(0))


def _rank(rows: Sequence[Sequence[Rat]]) -> int:
    if not rows:
        return 0
    work = [list(row) for row in rows]
    ncols = len(work[0])
    pivot_row = 0
    for col in range(ncols):
        pivot = next((r for r in range(pivot_row, len(work)) if work[r][col]), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][col]
        work[pivot_row] = [value / scale for value in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or not work[row][col]:
                continue
            scale = work[row][col]
            work[row] = [a - scale * b for a, b in zip(work[row], work[pivot_row])]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def _solve_square(matrix: Sequence[Sequence[Rat]], rhs: Sequence[Rat]) -> Optional[Vector]:
    n = len(matrix)
    if n == 0 or len(rhs) != n or any(len(row) != n for row in matrix):
        raise ValueError("expected nonempty square system")
    work = [list(row) + [rhs_value] for row, rhs_value in zip(matrix, rhs)]
    for col in range(n):
        pivot = next((row for row in range(col, n) if work[row][col]), None)
        if pivot is None:
            return None
        work[col], work[pivot] = work[pivot], work[col]
        scale = work[col][col]
        work[col] = [value / scale for value in work[col]]
        for row in range(n):
            if row == col or not work[row][col]:
                continue
            scale = work[row][col]
            work[row] = [a - scale * b for a, b in zip(work[row], work[col])]
    return tuple(work[row][-1] for row in range(n))


def _solve_linear_system(
    matrix: Sequence[Sequence[Rat]],
    rhs: Sequence[Rat],
    variable_count: int,
) -> Optional[Vector]:
    """Return one rational solution of a rectangular system, with free values zero."""

    if len(matrix) != len(rhs) or any(len(row) != variable_count for row in matrix):
        raise ValueError("linear-system dimensions do not match")
    if not matrix:
        return tuple(Rat(0) for _ in range(variable_count))
    work = [list(row) + [value] for row, value in zip(matrix, rhs)]
    pivot_row = 0
    pivot_columns: List[int] = []
    for column in range(variable_count):
        pivot = next(
            (row for row in range(pivot_row, len(work)) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [value / scale for value in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or not work[row][column]:
                continue
            scale = work[row][column]
            work[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(work[row], work[pivot_row])
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(work):
            break
    if any(
        all(value == 0 for value in row[:variable_count]) and row[-1] != 0
        for row in work
    ):
        return None
    solution = [Rat(0) for _ in range(variable_count)]
    for row, column in enumerate(pivot_columns):
        solution[column] = work[row][-1]
    return tuple(solution)


def _solve_linear_system_near(
    matrix: Sequence[Sequence[Rat]],
    rhs: Sequence[Rat],
    variable_count: int,
    proposal: Sequence[Rat],
) -> Optional[Vector]:
    """Solve exact equalities while retaining proposed values for free variables."""

    if len(matrix) != len(rhs) or len(proposal) != variable_count:
        raise ValueError("linear system dimensions do not match")
    if not matrix:
        return tuple(proposal)
    if any(len(row) != variable_count for row in matrix):
        raise ValueError("ragged linear system")
    work = [list(row) + [value] for row, value in zip(matrix, rhs)]
    pivot_row = 0
    pivot_columns: List[int] = []
    for column in range(variable_count):
        pivot = next(
            (row for row in range(pivot_row, len(work)) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [value / scale for value in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or not work[row][column]:
                continue
            scale = work[row][column]
            work[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(work[row], work[pivot_row])
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(work):
            break
    if any(
        all(value == 0 for value in row[:variable_count]) and row[-1] != 0
        for row in work
    ):
        return None
    solution = list(proposal)
    for row, column in reversed(tuple(enumerate(pivot_columns))):
        solution[column] = work[row][-1] - sum(
            (
                work[row][other] * solution[other]
                for other in range(variable_count)
                if other != column
            ),
            Rat(0),
        )
    return tuple(solution)


def _independent_equalities(rows: Matrix, rhs: Vector) -> Tuple[Matrix, Vector]:
    if len(rows) != len(rhs):
        raise ValueError("equality row and right-hand-side counts do not match")
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("ragged equality matrix")
    kept_rows: List[Vector] = []
    kept_rhs: List[Rat] = []
    rank = 0
    for row, value in zip(rows, rhs):
        candidate_rank = _rank(kept_rows + [row])
        if candidate_rank > rank:
            kept_rows.append(row)
            kept_rhs.append(value)
            rank = candidate_rank
            continue
        augmented = [tuple(r) + (b,) for r, b in zip(kept_rows + [row], kept_rhs + [value])]
        if _rank(augmented) > rank:
            raise ValueError("inconsistent equality constraints")
    return tuple(kept_rows), tuple(kept_rhs)


def _least_squares_data(target: Vector, donors: Matrix, weights: Vector) -> Tuple[Matrix, Vector]:
    if len(target) != len(donors) or len(weights) != len(donors):
        raise ValueError("row dimensions do not match")
    if not donors:
        raise ValueError("at least one row is required")
    if any(weight < 0 for weight in weights):
        raise ValueError("least-squares weights must be nonnegative")
    donor_count = len(donors[0])
    if any(len(row) != donor_count for row in donors):
        raise ValueError("ragged donor matrix")
    h = tuple(
        tuple(
            sum((weights[k] * donors[k][i] * donors[k][j] for k in range(len(donors))), Rat(0))
            for j in range(donor_count)
        )
        for i in range(donor_count)
    )
    g = tuple(
        sum((weights[k] * donors[k][i] * target[k] for k in range(len(donors))), Rat(0))
        for i in range(donor_count)
    )
    return h, g


def polytope_vertices(equality_rows: Matrix, equality_rhs: Vector) -> Tuple[Vector, ...]:
    """Enumerate vertices of ``{w >= 0 : equality_rows @ w = rhs}`` exactly."""

    if not equality_rows:
        raise ValueError("at least one equality is required")
    if len(equality_rows) != len(equality_rhs):
        raise ValueError("equality row and right-hand-side counts do not match")
    donor_count = len(equality_rows[0])
    if any(len(row) != donor_count for row in equality_rows):
        raise ValueError("ragged equality matrix")
    vertices = set()
    support_bound = _rank(equality_rows)
    for support_size in range(1, min(donor_count, support_bound) + 1):
        for support in combinations(range(donor_count), support_size):
            restricted = tuple(tuple(row[j] for j in support) for row in equality_rows)
            if _rank(restricted) < support_size:
                continue
            gram = tuple(
                tuple(sum((restricted[r][i] * restricted[r][j] for r in range(len(restricted))), Rat(0))
                      for j in range(support_size))
                for i in range(support_size)
            )
            projected_rhs = tuple(
                sum((restricted[r][i] * equality_rhs[r] for r in range(len(restricted))), Rat(0))
                for i in range(support_size)
            )
            active = _solve_square(gram, projected_rhs)
            if active is None or any(value <= 0 for value in active):
                continue
            w = [Rat(0) for _ in range(donor_count)]
            for donor, value in zip(support, active):
                w[donor] = value
            w_tuple = tuple(w)
            if all(_dot(row, w_tuple) == value for row, value in zip(equality_rows, equality_rhs)):
                vertices.add(w_tuple)
    if not vertices:
        raise ValueError("equality polytope is empty")
    return tuple(sorted(vertices))


def squared_loss(target: Vector, donors: Matrix, weights: Vector, w: Vector) -> Rat:
    residuals = tuple(target[k] - _dot(donors[k], w) for k in range(len(target)))
    return sum((weights[k] * residuals[k] * residuals[k] for k in range(len(target))), Rat(0))


@dataclass(frozen=True)
class QPSolutionSet:
    value: Rat
    vertices: Tuple[Vector, ...]


@dataclass(frozen=True)
class QPKKTCertificate:
    value: Rat
    optimizer: Vector
    support: Tuple[int, ...]
    equality_rows: Matrix
    equality_rhs: Vector
    multipliers: Vector
    reduced_gradients: Vector


@dataclass(frozen=True)
class LPKKTCertificate:
    value: Rat
    optimizer: Vector
    support: Tuple[int, ...]
    equality_rows: Matrix
    equality_rhs: Vector
    multipliers: Vector
    reduced_costs: Vector


def certify_lp_support(
    cost: Vector,
    equality_rows: Matrix,
    equality_rhs: Vector,
    support: Sequence[int],
    multiplier_binding_positions: Optional[Sequence[int]] = None,
    multiplier_near_proposal: Optional[Vector] = None,
) -> LPKKTCertificate:
    """Construct and exactly verify an LP KKT certificate on one basic support."""

    variable_count = len(cost)
    normalized_support = tuple(sorted(set(support)))
    if not normalized_support or any(
        index < 0 or index >= variable_count for index in normalized_support
    ):
        raise ValueError("support indices are empty or out of range")
    eq_rows, eq_rhs = _independent_equalities(equality_rows, equality_rhs)
    restricted_rows = tuple(
        tuple(row[index] for index in normalized_support) for row in eq_rows
    )
    if _rank(restricted_rows) != len(normalized_support):
        raise ValueError("proposed LP support is not basic")
    active = _solve_linear_system(restricted_rows, eq_rhs, len(normalized_support))
    if active is None or any(value <= 0 for value in active):
        raise ValueError("proposed LP support is infeasible or not strictly positive")
    optimizer = [Rat(0) for _ in range(variable_count)]
    for index, value in zip(normalized_support, active):
        optimizer[index] = value
    optimizer_tuple = tuple(optimizer)
    if any(_dot(row, optimizer_tuple) != value for row, value in zip(eq_rows, eq_rhs)):
        raise ValueError("LP support solution violates equality constraints")

    multiplier_count = len(eq_rows)
    active_rows = tuple(
        tuple(row[index] for row in eq_rows) for index in normalized_support
    )
    active_rhs = tuple(-cost[index] for index in normalized_support)
    inactive = tuple(
        index for index in range(variable_count) if index not in normalized_support
    )
    inactive_rows = tuple(tuple(row[index] for row in eq_rows) for index in inactive)
    inactive_rhs = tuple(-cost[index] for index in inactive)
    multipliers = None
    if multiplier_near_proposal is not None:
        if len(multiplier_near_proposal) != multiplier_count:
            raise ValueError("multiplier proposal has wrong dimension")
        binding_positions = tuple(
            sorted(set(multiplier_binding_positions or ()))
        )
        if any(
            position < 0 or position >= len(inactive)
            for position in binding_positions
        ):
            raise ValueError("multiplier binding position is out of range")
        rows = active_rows + tuple(
            inactive_rows[position] for position in binding_positions
        )
        values = active_rhs + tuple(
            inactive_rhs[position] for position in binding_positions
        )
        candidate = _solve_linear_system_near(
            rows,
            values,
            multiplier_count,
            multiplier_near_proposal,
        )
        if candidate is not None and all(
            _dot(row, candidate) >= value
            for row, value in zip(inactive_rows, inactive_rhs)
        ):
            multipliers = candidate
    elif multiplier_binding_positions is not None:
        binding_positions = tuple(sorted(set(multiplier_binding_positions)))
        if any(
            position < 0 or position >= len(inactive)
            for position in binding_positions
        ):
            raise ValueError("multiplier binding position is out of range")
        rows = active_rows + tuple(
            inactive_rows[position] for position in binding_positions
        )
        values = active_rhs + tuple(
            inactive_rhs[position] for position in binding_positions
        )
        candidate = _solve_linear_system(rows, values, multiplier_count)
        if candidate is not None and all(
            _dot(row, candidate) >= value
            for row, value in zip(inactive_rows, inactive_rhs)
        ):
            multipliers = candidate
    else:
        free_multiplier_count = multiplier_count - _rank(active_rows)
        for binding_count in range(
            min(free_multiplier_count, len(inactive)) + 1
        ):
            for binding_positions in combinations(
                range(len(inactive)), binding_count
            ):
                rows = active_rows + tuple(
                    inactive_rows[position] for position in binding_positions
                )
                values = active_rhs + tuple(
                    inactive_rhs[position] for position in binding_positions
                )
                candidate = _solve_linear_system(
                    rows,
                    values,
                    multiplier_count,
                )
                if candidate is None:
                    continue
                if all(
                    _dot(row, candidate) >= value
                    for row, value in zip(inactive_rows, inactive_rhs)
                ):
                    multipliers = candidate
                    break
            if multipliers is not None:
                break
    if multipliers is None:
        raise ValueError("inactive LP dual feasibility failed")
    reduced_costs = tuple(
        cost[index]
        + sum(
            (row[index] * multiplier for row, multiplier in zip(eq_rows, multipliers)),
            Rat(0),
        )
        for index in range(variable_count)
    )
    if any(reduced_costs[index] != 0 for index in normalized_support):
        raise AssertionError("active LP stationarity failed")
    if any(reduced_costs[index] < 0 for index in inactive):
        raise ValueError("inactive LP dual feasibility failed")
    return LPKKTCertificate(
        _dot(cost, optimizer_tuple),
        optimizer_tuple,
        normalized_support,
        eq_rows,
        eq_rhs,
        multipliers,
        reduced_costs,
    )


def certify_qp_support(
    target: Vector,
    donors: Matrix,
    row_weights: Vector,
    equality_rows: Matrix,
    equality_rhs: Vector,
    support: Sequence[int],
    multiplier_binding_positions: Optional[Sequence[int]] = None,
    multiplier_near_proposal: Optional[Vector] = None,
) -> QPKKTCertificate:
    """Construct and exactly verify a convex-QP KKT certificate on one support."""

    h, g = _least_squares_data(target, donors, row_weights)
    donor_count = len(h)
    normalized_support = tuple(sorted(set(support)))
    if not normalized_support or any(
        index < 0 or index >= donor_count for index in normalized_support
    ):
        raise ValueError("support indices are empty or out of range")
    eq_rows, eq_rhs = _independent_equalities(equality_rows, equality_rhs)
    kept_restricted_rows: List[Vector] = []
    kept_rhs: List[Rat] = []
    restricted_rank = 0
    for full_row, value in zip(eq_rows, eq_rhs):
        restricted_row = tuple(full_row[index] for index in normalized_support)
        candidate_rank = _rank(kept_restricted_rows + [restricted_row])
        if candidate_rank > restricted_rank:
            kept_restricted_rows.append(restricted_row)
            kept_rhs.append(value)
            restricted_rank = candidate_rank
            continue
        augmented = [
            tuple(row) + (rhs_value,)
            for row, rhs_value in zip(
                kept_restricted_rows + [restricted_row], kept_rhs + [value]
            )
        ]
        if _rank(augmented) > restricted_rank:
            raise ValueError("proposed support makes equality constraints inconsistent")
    restricted_rows = tuple(kept_restricted_rows)
    restricted_rhs = tuple(kept_rhs)
    support_size = len(normalized_support)
    equality_count = len(restricted_rows)
    system_size = support_size + equality_count
    kkt = [[Rat(0) for _ in range(system_size)] for _ in range(system_size)]
    rhs = [Rat(0) for _ in range(system_size)]
    for row_position, donor_i in enumerate(normalized_support):
        rhs[row_position] = g[donor_i]
        for column_position, donor_j in enumerate(normalized_support):
            kkt[row_position][column_position] = h[donor_i][donor_j]
        for equality_position, equality_row in enumerate(restricted_rows):
            kkt[row_position][support_size + equality_position] = equality_row[row_position]
    for equality_position, equality_row in enumerate(restricted_rows):
        rhs[support_size + equality_position] = restricted_rhs[equality_position]
        for column_position in range(support_size):
            kkt[support_size + equality_position][column_position] = equality_row[column_position]

    solution = _solve_square(kkt, rhs)
    if solution is None:
        raise ValueError("support KKT system is singular")
    active = solution[:support_size]
    if any(value <= 0 for value in active):
        raise ValueError("proposed support is not strictly positive")

    optimizer = [Rat(0) for _ in range(donor_count)]
    for index, value in zip(normalized_support, active):
        optimizer[index] = value
    optimizer_tuple = tuple(optimizer)
    if any(_dot(row, optimizer_tuple) != value for row, value in zip(eq_rows, eq_rhs)):
        raise ValueError("support solution violates equality constraints")

    multiplier_count = len(eq_rows)
    active_multiplier_rows = tuple(
        tuple(row[index] for row in eq_rows) for index in normalized_support
    )
    active_multiplier_rhs = tuple(
        g[index]
        - sum(
            (h[index][j] * optimizer_tuple[j] for j in range(donor_count)),
            Rat(0),
        )
        for index in normalized_support
    )
    inactive = tuple(index for index in range(donor_count) if index not in normalized_support)
    inactive_multiplier_rows = tuple(
        tuple(row[index] for row in eq_rows) for index in inactive
    )
    inactive_multiplier_rhs = tuple(
        g[index]
        - sum(
            (h[index][j] * optimizer_tuple[j] for j in range(donor_count)),
            Rat(0),
        )
        for index in inactive
    )
    multipliers = None
    if multiplier_near_proposal is not None:
        if len(multiplier_near_proposal) != multiplier_count:
            raise ValueError("multiplier proposal has wrong dimension")
        binding_positions = tuple(
            sorted(set(multiplier_binding_positions or ()))
        )
        if any(
            position < 0 or position >= len(inactive)
            for position in binding_positions
        ):
            raise ValueError("multiplier binding position is out of range")
        rows = active_multiplier_rows + tuple(
            inactive_multiplier_rows[position] for position in binding_positions
        )
        values = active_multiplier_rhs + tuple(
            inactive_multiplier_rhs[position] for position in binding_positions
        )
        candidate = _solve_linear_system_near(
            rows,
            values,
            multiplier_count,
            multiplier_near_proposal,
        )
        if candidate is not None and all(
            _dot(row, candidate) >= value
            for row, value in zip(inactive_multiplier_rows, inactive_multiplier_rhs)
        ):
            multipliers = candidate
    elif multiplier_binding_positions is not None:
        binding_positions = tuple(sorted(set(multiplier_binding_positions)))
        if any(
            position < 0 or position >= len(inactive)
            for position in binding_positions
        ):
            raise ValueError("multiplier binding position is out of range")
        rows = active_multiplier_rows + tuple(
            inactive_multiplier_rows[position] for position in binding_positions
        )
        values = active_multiplier_rhs + tuple(
            inactive_multiplier_rhs[position] for position in binding_positions
        )
        candidate = _solve_linear_system(rows, values, multiplier_count)
        if candidate is not None and all(
            _dot(row, candidate) >= value
            for row, value in zip(inactive_multiplier_rows, inactive_multiplier_rhs)
        ):
            multipliers = candidate
    else:
        free_multiplier_count = multiplier_count - _rank(active_multiplier_rows)
        for binding_count in range(
            min(free_multiplier_count, len(inactive)) + 1
        ):
            for binding_positions in combinations(
                range(len(inactive)), binding_count
            ):
                rows = active_multiplier_rows + tuple(
                    inactive_multiplier_rows[position]
                    for position in binding_positions
                )
                values = active_multiplier_rhs + tuple(
                    inactive_multiplier_rhs[position]
                    for position in binding_positions
                )
                candidate = _solve_linear_system(rows, values, multiplier_count)
                if candidate is None:
                    continue
                if all(
                    _dot(row, candidate) >= value
                    for row, value in zip(
                        inactive_multiplier_rows,
                        inactive_multiplier_rhs,
                    )
                ):
                    multipliers = candidate
                    break
            if multipliers is not None:
                break
    if multipliers is None:
        raise ValueError("inactive KKT dual feasibility failed")

    reduced_gradients = tuple(
        sum(
            (h[index][j] * optimizer_tuple[j] for j in range(donor_count)),
            Rat(0),
        )
        - g[index]
        + sum(
            (row[index] * multiplier for row, multiplier in zip(eq_rows, multipliers)),
            Rat(0),
        )
        for index in range(donor_count)
    )
    if any(reduced_gradients[index] != 0 for index in normalized_support):
        raise AssertionError("active KKT stationarity failed")
    if any(
        reduced_gradients[index] < 0
        for index in range(donor_count)
        if index not in normalized_support
    ):
        raise ValueError("inactive KKT dual feasibility failed")

    return QPKKTCertificate(
        squared_loss(target, donors, row_weights, optimizer_tuple),
        optimizer_tuple,
        normalized_support,
        eq_rows,
        eq_rhs,
        multipliers,
        reduced_gradients,
    )


def equality_constrained_qp_vertices(
    target: Vector,
    donors: Matrix,
    row_weights: Vector,
    equality_rows: Matrix,
    equality_rhs: Vector,
) -> QPSolutionSet:
    """Enumerate vertices of the minimizer set of a tiny convex QP.

    First enumerate every support and solve its equality-constrained stationary
    problem. The least loss among all feasible faces is global. Weighted fitted
    values are unique across global minimizers, so adjoining those values to the
    original equalities turns the full minimizer set into a polytope whose
    vertices can be enumerated independently.
    """

    h, g = _least_squares_data(target, donors, row_weights)
    donor_count = len(h)
    eq_rows, eq_rhs = _independent_equalities(equality_rows, equality_rhs)
    candidates = set()
    positive_fit_rows = tuple(
        row for row, weight in zip(donors, row_weights) if weight > 0
    )
    support_bound = _rank(eq_rows + positive_fit_rows)

    for support_size in range(1, min(donor_count, support_bound) + 1):
        for support in combinations(range(donor_count), support_size):
            restricted_all = tuple(tuple(row[j] for j in support) for row in eq_rows)
            try:
                restricted_rows, restricted_rhs = _independent_equalities(restricted_all, eq_rhs)
            except ValueError:
                continue
            eq_count = len(restricted_rows)
            size = support_size + eq_count
            kkt = [[Rat(0) for _ in range(size)] for _ in range(size)]
            rhs = [Rat(0) for _ in range(size)]
            for row_pos, donor_i in enumerate(support):
                rhs[row_pos] = g[donor_i]
                for col_pos, donor_j in enumerate(support):
                    kkt[row_pos][col_pos] = h[donor_i][donor_j]
                for eq_pos, eq_row in enumerate(restricted_rows):
                    kkt[row_pos][support_size + eq_pos] = eq_row[row_pos]
            for eq_pos, eq_row in enumerate(restricted_rows):
                rhs[support_size + eq_pos] = restricted_rhs[eq_pos]
                for col_pos in range(support_size):
                    kkt[support_size + eq_pos][col_pos] = eq_row[col_pos]

            solution = _solve_square(kkt, rhs)
            if solution is None:
                continue
            active = solution[:support_size]
            if any(value <= 0 for value in active):
                continue
            w = [Rat(0) for _ in range(donor_count)]
            for donor, value in zip(support, active):
                w[donor] = value
            w_tuple = tuple(w)
            if any(_dot(row, w_tuple) != value for row, value in zip(eq_rows, eq_rhs)):
                continue
            candidates.add(w_tuple)

    if not candidates:
        raise ValueError("no feasible stationary face found")
    values = {w: squared_loss(target, donors, row_weights, w) for w in candidates}
    optimum = min(values.values())
    representative = next(w for w, value in values.items() if value == optimum)

    optimizer_rows: List[Vector] = list(eq_rows)
    optimizer_rhs: List[Rat] = list(eq_rhs)
    for row, weight in zip(donors, row_weights):
        if weight > 0:
            optimizer_rows.append(row)
            optimizer_rhs.append(_dot(row, representative))
    final_rows, final_rhs = _independent_equalities(tuple(optimizer_rows), tuple(optimizer_rhs))
    vertices = polytope_vertices(final_rows, final_rhs)
    if any(squared_loss(target, donors, row_weights, w) != optimum for w in vertices):
        raise AssertionError("reconstructed minimizer polytope changed objective value")
    return QPSolutionSet(value=optimum, vertices=vertices)


def simplex_qp_vertices(target: Vector, donors: Matrix, row_weights: Vector) -> QPSolutionSet:
    donor_count = len(donors[0])
    return equality_constrained_qp_vertices(
        target,
        donors,
        row_weights,
        (tuple(Rat(1) for _ in range(donor_count)),),
        (Rat(1),),
    )


@dataclass(frozen=True)
class BilevelCandidate:
    predictor_weights: Vector
    donor_weights: Vector
    outer_loss: Rat
    effect: Rat


@dataclass(frozen=True)
class GridEnvelope:
    outer_optimum: Rat
    lower_effect: Rat
    upper_effect: Rat
    optimizers: Tuple[BilevelCandidate, ...]


def exact_grid_envelope(
    predictor_target: Vector,
    predictor_donors: Matrix,
    validation_target: Vector,
    validation_donors: Matrix,
    predictor_weight_grid: Sequence[Vector],
    treated_post: Rat,
    donor_post: Vector,
) -> GridEnvelope:
    """Exact bilevel envelope over a finite, explicitly supplied V grid."""

    donor_count = len(predictor_donors[0])
    all_candidates: List[BilevelCandidate] = []
    validation_weights = tuple(Rat(1, len(validation_target)) for _ in validation_target)

    for v in predictor_weight_grid:
        if any(value < 0 for value in v) or sum(v, Rat(0)) != 1:
            raise ValueError("predictor weights must lie on the simplex")
        inner = simplex_qp_vertices(predictor_target, predictor_donors, v)
        representative = inner.vertices[0]
        equality_rows: List[Vector] = [tuple(Rat(1) for _ in range(donor_count))]
        equality_rhs: List[Rat] = [Rat(1)]
        for row_index, value in enumerate(v):
            if value > 0:
                equality_rows.append(predictor_donors[row_index])
                equality_rhs.append(_dot(predictor_donors[row_index], representative))
        outer = equality_constrained_qp_vertices(
            validation_target,
            validation_donors,
            validation_weights,
            tuple(equality_rows),
            tuple(equality_rhs),
        )
        for w in outer.vertices:
            effect = treated_post - _dot(donor_post, w)
            all_candidates.append(BilevelCandidate(v, w, outer.value, effect))

    outer_optimum = min(candidate.outer_loss for candidate in all_candidates)
    optimizers = tuple(candidate for candidate in all_candidates if candidate.outer_loss == outer_optimum)
    effects = tuple(candidate.effect for candidate in optimizers)
    return GridEnvelope(outer_optimum, min(effects), max(effects), optimizers)
