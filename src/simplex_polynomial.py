"""Exact homogeneous-polynomial certificates on a rational simplex.

A homogeneous polynomial in barycentric coordinates is a linear combination
of the degree-matched Bernstein basis monomials.  Consequently, nonnegative
coefficients prove nonnegativity on the whole simplex, while strictly negative
coefficients prove strict negativity.  Rational longest-edge subdivision turns
those elementary bounds into independently replayable semialgebraic branch
certificates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
from typing import Dict, Optional, Sequence, Tuple

from .scm_oracle import Rat


Exponent = Tuple[int, ...]
Polynomial = Dict[Exponent, Rat]
SimplexVertex = Tuple[Rat, ...]
SimplexCell = Tuple[SimplexVertex, ...]


@lru_cache(maxsize=None)
def _degree_exponents(variable_count: int, degree: int) -> Tuple[Exponent, ...]:
    """Enumerate every barycentric monomial exponent of one total degree."""

    if variable_count < 1 or degree < 0:
        raise ValueError("invalid homogeneous polynomial dimensions")
    if variable_count == 1:
        return ((degree,),)
    return tuple(
        (first,) + remainder
        for first in range(degree + 1)
        for remainder in _degree_exponents(variable_count - 1, degree - first)
    )


def _all_degree_coefficients_strictly_negative(
    polynomial: Polynomial,
    variable_count: int,
) -> bool:
    """Check a strict negative Bernstein upper bound, including zero terms."""

    if not polynomial:
        return False
    degree = polynomial_degree(polynomial)
    return all(
        polynomial.get(exponent, Rat(0)) < 0
        for exponent in _degree_exponents(variable_count, degree)
    )


def polynomial_constant(value: Rat, variable_count: int) -> Polynomial:
    if variable_count < 1:
        raise ValueError("a polynomial needs at least one variable")
    return {(0,) * variable_count: value} if value else {}


def polynomial_linear(coefficients: Sequence[Rat]) -> Polynomial:
    variable_count = len(coefficients)
    if variable_count < 1:
        raise ValueError("a polynomial needs at least one variable")
    result: Polynomial = {}
    for index, coefficient in enumerate(coefficients):
        if coefficient:
            exponent = [0] * variable_count
            exponent[index] = 1
            result[tuple(exponent)] = coefficient
    return result


def polynomial_add(*polynomials: Polynomial) -> Polynomial:
    if not polynomials:
        return {}
    result = defaultdict(Rat)
    for polynomial in polynomials:
        for exponent, coefficient in polynomial.items():
            result[exponent] += coefficient
    return {
        exponent: coefficient
        for exponent, coefficient in result.items()
        if coefficient
    }


def polynomial_scale(polynomial: Polynomial, scalar: Rat) -> Polynomial:
    return {
        exponent: scalar * coefficient
        for exponent, coefficient in polynomial.items()
        if scalar * coefficient
    }


def polynomial_multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    if not left or not right:
        return {}
    result = defaultdict(Rat)
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            if len(left_exponent) != len(right_exponent):
                raise ValueError("polynomial variable counts do not match")
            exponent = tuple(
                left_power + right_power
                for left_power, right_power in zip(
                    left_exponent,
                    right_exponent,
                )
            )
            result[exponent] += left_coefficient * right_coefficient
    return {
        exponent: coefficient
        for exponent, coefficient in result.items()
        if coefficient
    }


def polynomial_power(polynomial: Polynomial, power: int) -> Polynomial:
    if power < 0:
        raise ValueError("polynomial power must be nonnegative")
    if not polynomial:
        if power == 0:
            raise ValueError("zero polynomial does not reveal its variable count")
        return {}
    variable_count = len(next(iter(polynomial)))
    result = polynomial_constant(Rat(1), variable_count)
    base = polynomial
    remaining = power
    while remaining:
        if remaining & 1:
            result = polynomial_multiply(result, base)
        remaining //= 2
        if remaining:
            base = polynomial_multiply(base, base)
    return result


def polynomial_degree(polynomial: Polynomial) -> int:
    if not polynomial:
        return -1
    return max(sum(exponent) for exponent in polynomial)


def polynomial_homogenize(
    polynomial: Polynomial,
    target_degree: Optional[int] = None,
) -> Polynomial:
    """Homogenize with ``sum(z)``, which equals one on the simplex."""

    if not polynomial:
        return {}
    variable_count = len(next(iter(polynomial)))
    degree = polynomial_degree(polynomial)
    if target_degree is None:
        target_degree = degree
    if target_degree < degree:
        raise ValueError("target degree is below the polynomial degree")
    coordinate_sum = polynomial_linear(tuple(Rat(1) for _ in range(variable_count)))
    result: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        term = {exponent: coefficient}
        missing = target_degree - sum(exponent)
        if missing:
            term = polynomial_multiply(
                term,
                polynomial_power(coordinate_sum, missing),
            )
        result = polynomial_add(result, term)
    return result


def polynomial_determinant(
    matrix: Sequence[Sequence[Polynomial]],
    variable_count: Optional[int] = None,
) -> Polynomial:
    size = len(matrix)
    if size < 1 or any(len(row) != size for row in matrix):
        raise ValueError("polynomial determinant needs a nonempty square matrix")
    first = matrix[0][0]
    inferred_variable_count = (
        len(next(iter(first)))
        if first
        else next(
            (
                len(next(iter(entry)))
                for row in matrix
                for entry in row
                if entry
            ),
            None,
        )
    )
    if variable_count is None:
        variable_count = inferred_variable_count
    elif inferred_variable_count is not None and inferred_variable_count != variable_count:
        raise ValueError("polynomial determinant variable count does not match")
    if variable_count is None or variable_count < 1:
        raise ValueError("zero matrix does not reveal its variable count")
    result: Polynomial = {}
    for permutation in permutations(range(size)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(size)
            for right in range(left + 1, size)
        )
        term = polynomial_constant(Rat(1), variable_count)
        for row, column in enumerate(permutation):
            term = polynomial_multiply(term, matrix[row][column])
        result = polynomial_add(
            result,
            polynomial_scale(term, Rat(-1) if inversions % 2 else Rat(1)),
        )
    return result


def standard_simplex(variable_count: int) -> SimplexCell:
    if variable_count < 1:
        raise ValueError("a simplex needs at least one vertex")
    return tuple(
        tuple(
            Rat(1) if row == column else Rat(0)
            for column in range(variable_count)
        )
        for row in range(variable_count)
    )


def split_simplex_longest_edge(cell: SimplexCell) -> Tuple[SimplexCell, SimplexCell]:
    variable_count = len(cell)
    if variable_count < 2 or any(len(vertex) != variable_count for vertex in cell):
        raise ValueError("cell must contain a full barycentric simplex")
    edge = max(
        combinations(range(variable_count), 2),
        key=lambda pair: sum(
            (cell[pair[0]][coordinate] - cell[pair[1]][coordinate]) ** 2
            for coordinate in range(variable_count)
        ),
    )
    midpoint = tuple(
        (left + right) / 2
        for left, right in zip(cell[edge[0]], cell[edge[1]])
    )
    left = list(cell)
    right = list(cell)
    left[edge[0]] = midpoint
    right[edge[1]] = midpoint
    return tuple(left), tuple(right)


def polynomial_substitute_simplex(
    polynomial: Polynomial,
    cell: SimplexCell,
) -> Polynomial:
    """Express a homogeneous global polynomial in a cell's coordinates."""

    if not polynomial:
        return {}
    variable_count = len(next(iter(polynomial)))
    if len(cell) != variable_count or any(
        len(vertex) != variable_count for vertex in cell
    ):
        raise ValueError("simplex cell dimensions do not match the polynomial")
    degree = polynomial_degree(polynomial)
    powers = []
    for coordinate in range(variable_count):
        base = polynomial_linear(
            tuple(vertex[coordinate] for vertex in cell)
        )
        coordinate_powers = [polynomial_constant(Rat(1), variable_count)]
        for _ in range(degree):
            coordinate_powers.append(
                polynomial_multiply(coordinate_powers[-1], base)
            )
        powers.append(coordinate_powers)
    result: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        term = polynomial_constant(coefficient, variable_count)
        for coordinate, power in enumerate(exponent):
            if power:
                term = polynomial_multiply(term, powers[coordinate][power])
        result = polynomial_add(result, term)
    return result


@dataclass(frozen=True)
class SimplexCertificateLeaf:
    path: Tuple[int, ...]
    disposition: str
    constraint_index: Optional[int] = None


@dataclass(frozen=True)
class SimplexCoefficientCertificate:
    variable_count: int
    leaves: Tuple[SimplexCertificateLeaf, ...]
    complete: bool
    unresolved_paths: Tuple[Tuple[int, ...], ...]

    @property
    def max_depth(self) -> int:
        paths = tuple(leaf.path for leaf in self.leaves) + self.unresolved_paths
        return max((len(path) for path in paths), default=0)


def _cell_from_path(variable_count: int, path: Sequence[int]) -> SimplexCell:
    cell = standard_simplex(variable_count)
    for direction in path:
        if direction not in (0, 1):
            raise ValueError("simplex certificate path must be binary")
        cell = split_simplex_longest_edge(cell)[direction]
    return cell


def certify_semialgebraic_nonnegative(
    objective: Polynomial,
    constraints: Sequence[Polynomial] = (),
    *,
    max_depth: int = 18,
    max_nodes: int = 200_000,
) -> SimplexCoefficientCertificate:
    """Prove ``objective >= 0`` wherever every constraint is nonnegative."""

    if not objective or max_depth < 0 or max_nodes < 1:
        raise ValueError("invalid objective or certificate limits")
    variable_count = len(next(iter(objective)))
    objective = polynomial_homogenize(objective)
    constraints = tuple(polynomial_homogenize(item) for item in constraints)
    stack = [((), standard_simplex(variable_count), 0)]
    leaves = []
    unresolved = []
    visited = 0
    while stack:
        path, cell, depth = stack.pop()
        visited += 1
        local_objective = polynomial_substitute_simplex(objective, cell)
        if not local_objective or min(local_objective.values()) >= 0:
            leaves.append(SimplexCertificateLeaf(path, "objective-nonnegative"))
            continue
        pruned = False
        for constraint_index, constraint in enumerate(constraints):
            local = polynomial_substitute_simplex(constraint, cell)
            if _all_degree_coefficients_strictly_negative(local, variable_count):
                leaves.append(
                    SimplexCertificateLeaf(
                        path,
                        "constraint-negative",
                        constraint_index,
                    )
                )
                pruned = True
                break
        if pruned:
            continue
        if depth >= max_depth or visited + len(stack) >= max_nodes:
            unresolved.append(path)
            continue
        left, right = split_simplex_longest_edge(cell)
        stack.append((path + (1,), right, depth + 1))
        stack.append((path + (0,), left, depth + 1))
    return SimplexCoefficientCertificate(
        variable_count,
        tuple(leaves),
        not unresolved,
        tuple(unresolved),
    )


def _verify_complete_paths(paths: Sequence[Tuple[int, ...]]) -> bool:
    path_set = set(paths)
    if not path_set or len(path_set) != len(paths):
        return False
    if any(direction not in (0, 1) for path in path_set for direction in path):
        return False

    internal_prefixes = {
        path[:length]
        for path in path_set
        for length in range(len(path))
    }
    if path_set & internal_prefixes:
        return False
    nodes = path_set | internal_prefixes
    return all(
        prefix + (0,) in nodes and prefix + (1,) in nodes
        for prefix in internal_prefixes
    )


def verify_semialgebraic_certificate(
    objective: Polynomial,
    constraints: Sequence[Polynomial],
    certificate: SimplexCoefficientCertificate,
) -> None:
    if not certificate.complete or certificate.unresolved_paths:
        raise ValueError("certificate is incomplete")
    if not _verify_complete_paths(tuple(leaf.path for leaf in certificate.leaves)):
        raise ValueError("certificate leaves do not partition the simplex")
    objective = polynomial_homogenize(objective)
    constraints = tuple(polynomial_homogenize(item) for item in constraints)
    if len(next(iter(objective))) != certificate.variable_count:
        raise ValueError("certificate variable count does not match")
    for leaf in certificate.leaves:
        cell = _cell_from_path(certificate.variable_count, leaf.path)
        if leaf.disposition == "objective-nonnegative":
            local = polynomial_substitute_simplex(objective, cell)
            if local and min(local.values()) < 0:
                raise ValueError("objective leaf has a negative coefficient")
        elif leaf.disposition == "constraint-negative":
            index = leaf.constraint_index
            if index is None or index < 0 or index >= len(constraints):
                raise ValueError("constraint leaf has an invalid index")
            local = polynomial_substitute_simplex(constraints[index], cell)
            if not _all_degree_coefficients_strictly_negative(
                local, certificate.variable_count
            ):
                raise ValueError("constraint leaf is not strictly infeasible")
        else:
            raise ValueError("unknown simplex certificate disposition")
