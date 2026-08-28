"""Numerical support proposal followed by exact rational KKT acceptance.

SciPy is an optional discovery dependency. Numerical output is never accepted
as an optimum: a proposed active support must pass ``certify_qp_support`` in
exact rational arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .scm_oracle import (
    Matrix,
    LPKKTCertificate,
    QPKKTCertificate,
    Rat,
    Vector,
    _independent_equalities,
    certify_lp_support,
    certify_qp_support,
)


@dataclass(frozen=True)
class ScalableQPCertificate:
    exact: QPKKTCertificate
    numerical_objective: float
    numerical_iterations: int
    support_threshold: float


@dataclass(frozen=True)
class ScalableLPCertificate:
    exact: LPKKTCertificate
    numerical_objective: float


@dataclass(frozen=True)
class StrictFeasiblePointCertificate:
    point: Vector
    margin: Rat
    exact: LPKKTCertificate


def _rational_vector_proposals(values):
    """Return increasingly accurate rationalizations of a floating proposal."""

    for denominator_limit in (10**3, 10**6, 10**9, 10**12):
        yield tuple(
            Rat(str(float(value))).limit_denominator(denominator_limit)
            for value in values
        )


def propose_and_certify_lp(
    cost: Vector,
    equality_rows: Matrix,
    equality_rhs: Vector,
) -> ScalableLPCertificate:
    """Use HiGHS to propose a basic LP support, then certify it exactly."""

    try:
        import numpy as np
        from scipy.optimize import linprog
    except ImportError as error:
        raise RuntimeError(
            "scalable support discovery requires optional numpy and scipy"
        ) from error
    independent_rows, independent_rhs = _independent_equalities(
        equality_rows, equality_rhs
    )
    cost_array = np.asarray([float(value) for value in cost])
    equality_array = np.asarray(
        [[float(value) for value in row] for row in independent_rows]
    )
    rhs_array = np.asarray([float(value) for value in independent_rhs])
    numerical = linprog(
        cost_array,
        A_eq=equality_array,
        b_eq=rhs_array,
        bounds=(0, None),
        method="highs",
    )
    if not numerical.success:
        raise ValueError(f"numerical LP support search failed: {numerical.message}")

    variable_count = len(cost)

    def certify_support_with_dual_proposal(support):
        inactive = tuple(
            index for index in range(variable_count) if index not in support
        )
        support_array = np.asarray(support, dtype=int)
        active_rows = equality_array[:, support_array].T
        active_rhs = -cost_array[support_array]
        if inactive:
            inactive_array = np.asarray(inactive, dtype=int)
            inactive_rows = equality_array[:, inactive_array].T
            inactive_rhs = -cost_array[inactive_array]
            multiplier_proposal = linprog(
                np.zeros(len(independent_rows)),
                A_ub=-inactive_rows,
                b_ub=-inactive_rhs,
                A_eq=active_rows,
                b_eq=active_rhs,
                bounds=(None, None),
                method="highs",
            )
            if not multiplier_proposal.success:
                raise ValueError("numerical LP multiplier proposal failed")
            multiplier_values = multiplier_proposal.x
            slacks = inactive_rows @ multiplier_proposal.x - inactive_rhs
            ordering = tuple(int(index) for index in np.argsort(np.abs(slacks)))
            free_count = max(
                0,
                len(independent_rows) - int(np.linalg.matrix_rank(active_rows)),
            )
            binding_proposals = [()]
            for threshold in (1e-6, 1e-8, 1e-10, 1e-12):
                binding_proposals.append(
                    tuple(
                        index
                        for index, slack in enumerate(slacks)
                        if abs(slack) <= threshold
                    )
                )
            for count in range(
                max(0, free_count - 1),
                min(len(inactive), free_count + 2) + 1,
            ):
                binding_proposals.append(tuple(sorted(ordering[:count])))
        else:
            multiplier_values, *_ = np.linalg.lstsq(
                active_rows,
                active_rhs,
                rcond=None,
            )
            binding_proposals = [()]

        attempted_bindings = set()
        for binding_positions in binding_proposals:
            normalized = tuple(sorted(set(binding_positions)))
            if normalized in attempted_bindings:
                continue
            attempted_bindings.add(normalized)
            for rational_proposal in _rational_vector_proposals(multiplier_values):
                try:
                    return certify_lp_support(
                        cost,
                        independent_rows,
                        independent_rhs,
                        support,
                        multiplier_binding_positions=normalized,
                        multiplier_near_proposal=rational_proposal,
                    )
                except ValueError:
                    continue
            try:
                return certify_lp_support(
                    cost,
                    independent_rows,
                    independent_rhs,
                    support,
                    multiplier_binding_positions=normalized,
                )
            except ValueError:
                continue
        raise ValueError("no proposed LP multiplier support passed exact checks")

    attempted = set()
    for threshold in (1e-6, 1e-8, 1e-10, 1e-12):
        support = tuple(index for index, value in enumerate(numerical.x) if value > threshold)
        if not support or support in attempted:
            continue
        attempted.add(support)
        try:
            exact = certify_support_with_dual_proposal(support)
        except ValueError:
            continue
        return ScalableLPCertificate(exact, float(numerical.fun))

    fixed_zero_variables = set()
    for row, value in zip(independent_rows, independent_rhs):
        nonzero = tuple(index for index, coefficient in enumerate(row) if coefficient)
        if value == 0 and len(nonzero) == 1:
            fixed_zero_variables.add(nonzero[0])
    if fixed_zero_variables:
        kept_variables = tuple(
            index
            for index in range(variable_count)
            if index not in fixed_zero_variables
        )
        retained = tuple(
            (row, value)
            for row, value in zip(independent_rows, independent_rhs)
            if any(row[index] for index in kept_variables)
        )
        reduced = propose_and_certify_lp(
            tuple(cost[index] for index in kept_variables),
            tuple(
                tuple(row[index] for index in kept_variables)
                for row, _ in retained
            ),
            tuple(value for _, value in retained),
        )
        lifted_support = tuple(
            kept_variables[index] for index in reduced.exact.support
        )
        exact = certify_support_with_dual_proposal(lifted_support)
        return ScalableLPCertificate(exact, reduced.numerical_objective)
    raise ValueError("no numerically proposed LP support passed exact KKT verification")


def propose_and_certify_strict_feasible_point(
    equality_rows: Matrix,
    equality_rhs: Vector,
    required_positive_indices: Tuple[int, ...],
) -> StrictFeasiblePointCertificate:
    """Find an exact feasible point strictly positive on declared coordinates."""

    if not equality_rows or len(equality_rows) != len(equality_rhs):
        raise ValueError("equality dimensions do not match")
    variable_count = len(equality_rows[0])
    if any(len(row) != variable_count for row in equality_rows):
        raise ValueError("equality matrix is ragged")
    required = tuple(sorted(set(required_positive_indices)))
    if not required or any(
        index < 0 or index >= variable_count for index in required
    ):
        raise ValueError("required positive coordinates are empty or out of range")
    extended_variable_count = variable_count + 1 + len(required)
    rows = [
        tuple(row) + tuple(Rat(0) for _ in range(1 + len(required)))
        for row in equality_rows
    ]
    rhs = list(equality_rhs)
    for position, index in enumerate(required):
        row = [Rat(0) for _ in range(extended_variable_count)]
        row[index] = Rat(1)
        row[variable_count] = Rat(-1)
        row[variable_count + 1 + position] = Rat(-1)
        rows.append(tuple(row))
        rhs.append(Rat(0))
    cost = tuple(Rat(0) for _ in range(variable_count)) + (Rat(-1),) + tuple(
        Rat(0) for _ in required
    )
    exact = propose_and_certify_lp(cost, tuple(rows), tuple(rhs)).exact
    margin = exact.optimizer[variable_count]
    point = exact.optimizer[:variable_count]
    if margin <= 0 or any(point[index] < margin for index in required):
        raise ValueError("polytope has no certified strictly positive point")
    return StrictFeasiblePointCertificate(point, margin, exact)


def propose_and_certify_qp(
    target: Vector,
    donors: Matrix,
    row_weights: Vector,
    equality_rows: Matrix,
    equality_rhs: Vector,
) -> ScalableQPCertificate:
    """Use SciPy only to propose a support, then certify it exactly."""

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
    except ImportError as error:
        raise RuntimeError(
            "scalable support discovery requires optional numpy and scipy"
        ) from error

    independent_rows, independent_rhs = _independent_equalities(
        equality_rows, equality_rhs
    )
    donor_count = len(donors[0])
    donor_array = np.asarray([[float(value) for value in row] for row in donors])
    target_array = np.asarray([float(value) for value in target])
    weight_array = np.asarray([float(value) for value in row_weights])
    equality_array = np.asarray(
        [[float(value) for value in row] for row in independent_rows]
    )
    rhs_array = np.asarray([float(value) for value in independent_rhs])

    feasibility = linprog(
        np.zeros(donor_count),
        A_eq=equality_array,
        b_eq=rhs_array,
        bounds=(0, None),
        method="highs",
    )
    if not feasibility.success:
        raise ValueError(f"numerical phase I failed: {feasibility.message}")

    def objective(values):
        residual = donor_array @ values - target_array
        return float(np.dot(weight_array * residual, residual))

    def gradient(values):
        residual = donor_array @ values - target_array
        return 2 * donor_array.T @ (weight_array * residual)

    hessian = 2 * donor_array.T @ (weight_array[:, None] * donor_array)

    numerical = minimize(
        objective,
        feasibility.x,
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(np.zeros(donor_count), np.full(donor_count, np.inf)),
        constraints=(LinearConstraint(equality_array, rhs_array, rhs_array),),
        options={"ftol": 1e-13, "maxiter": 2000, "disp": False},
    )
    if not numerical.success:
        numerical = minimize(
            objective,
            feasibility.x,
            jac=gradient,
            hess=lambda _values: hessian,
            method="trust-constr",
            bounds=Bounds(np.zeros(donor_count), np.full(donor_count, np.inf)),
            constraints=(LinearConstraint(equality_array, rhs_array, rhs_array),),
            options={
                "gtol": 1e-11,
                "xtol": 1e-13,
                "barrier_tol": 1e-13,
                "maxiter": 2000,
                "verbose": 0,
            },
        )
    if not numerical.success:
        raise ValueError(f"numerical support search failed: {numerical.message}")

    half_hessian = donor_array.T @ (weight_array[:, None] * donor_array)
    linear_term = donor_array.T @ (weight_array * target_array)

    def certify_support_with_dual_proposal(support):
        inactive = tuple(
            index for index in range(donor_count) if index not in support
        )
        support_array = np.asarray(support, dtype=int)
        support_size = len(support)
        restricted_equalities = equality_array[:, support_array]
        kkt = np.block(
            [
                [
                    half_hessian[np.ix_(support_array, support_array)],
                    restricted_equalities.T,
                ],
                [
                    restricted_equalities,
                    np.zeros((len(independent_rows), len(independent_rows))),
                ],
            ]
        )
        kkt_rhs = np.concatenate((linear_term[support_array], rhs_array))
        proposal, *_ = np.linalg.lstsq(kkt, kkt_rhs, rcond=None)
        optimizer = np.zeros(donor_count)
        optimizer[support_array] = proposal[:support_size]

        active_multiplier_rows = equality_array[:, support_array].T
        active_multiplier_rhs = (
            linear_term[support_array]
            - half_hessian[support_array] @ optimizer
        )
        if inactive:
            inactive_array = np.asarray(inactive, dtype=int)
            inactive_multiplier_rows = equality_array[:, inactive_array].T
            inactive_multiplier_rhs = (
                linear_term[inactive_array]
                - half_hessian[inactive_array] @ optimizer
            )
            multiplier_proposal = linprog(
                np.zeros(len(independent_rows)),
                A_ub=-inactive_multiplier_rows,
                b_ub=-inactive_multiplier_rhs,
                A_eq=active_multiplier_rows,
                b_eq=active_multiplier_rhs,
                bounds=(None, None),
                method="highs",
            )
            if not multiplier_proposal.success:
                raise ValueError("numerical QP multiplier proposal failed")
            multiplier_values = multiplier_proposal.x
            slacks = (
                inactive_multiplier_rows @ multiplier_proposal.x
                - inactive_multiplier_rhs
            )
            ordering = tuple(int(index) for index in np.argsort(np.abs(slacks)))
            free_count = max(
                0,
                len(independent_rows)
                - int(np.linalg.matrix_rank(active_multiplier_rows)),
            )
            binding_proposals = [()]
            for threshold in (1e-6, 1e-8, 1e-10, 1e-12):
                binding_proposals.append(
                    tuple(
                        index
                        for index, slack in enumerate(slacks)
                        if abs(slack) <= threshold
                    )
                )
            for count in range(
                max(0, free_count - 1),
                min(len(inactive), free_count + 2) + 1,
            ):
                binding_proposals.append(tuple(sorted(ordering[:count])))
        else:
            multiplier_values, *_ = np.linalg.lstsq(
                active_multiplier_rows,
                active_multiplier_rhs,
                rcond=None,
            )
            binding_proposals = [()]

        attempted_bindings = set()
        for binding_positions in binding_proposals:
            normalized = tuple(sorted(set(binding_positions)))
            if normalized in attempted_bindings:
                continue
            attempted_bindings.add(normalized)
            for rational_proposal in _rational_vector_proposals(multiplier_values):
                try:
                    return certify_qp_support(
                        target,
                        donors,
                        row_weights,
                        independent_rows,
                        independent_rhs,
                        support,
                        multiplier_binding_positions=normalized,
                        multiplier_near_proposal=rational_proposal,
                    )
                except ValueError:
                    continue
            try:
                return certify_qp_support(
                    target,
                    donors,
                    row_weights,
                    independent_rows,
                    independent_rhs,
                    support,
                    multiplier_binding_positions=normalized,
                )
            except ValueError:
                continue
        raise ValueError("no proposed QP multiplier support passed exact checks")

    attempted = set()
    thresholds = (1e-5, 1e-7, 1e-9, 1e-11, 1e-13)
    for threshold in thresholds:
        support = tuple(
            index for index, value in enumerate(numerical.x) if value > threshold
        )
        if support in attempted or not support:
            continue
        attempted.add(support)
        try:
            exact = certify_support_with_dual_proposal(support)
        except ValueError:
            continue
        return ScalableQPCertificate(
            exact,
            float(numerical.fun),
            int(getattr(numerical, "nit", 0)),
            threshold,
        )

    positive_fit_rows = donor_array[weight_array > 0]
    fitted_values = positive_fit_rows @ numerical.x
    optimizer_equalities = np.vstack((equality_array, positive_fit_rows))
    optimizer_rhs = np.concatenate((rhs_array, fitted_values))
    proposal_objectives = (
        np.arange(1, donor_count + 1, dtype=float),
        -np.arange(1, donor_count + 1, dtype=float),
        np.asarray(
            [((index * 37) % 101) - 50 for index in range(donor_count)],
            dtype=float,
        ),
    )
    for proposal_objective in proposal_objectives:
        vertex = linprog(
            proposal_objective,
            A_eq=optimizer_equalities,
            b_eq=optimizer_rhs,
            bounds=(0, None),
            method="highs",
        )
        if not vertex.success:
            continue
        for threshold in thresholds:
            support = tuple(
                index for index, value in enumerate(vertex.x) if value > threshold
            )
            if support in attempted or not support:
                continue
            attempted.add(support)
            try:
                exact = certify_support_with_dual_proposal(support)
            except ValueError:
                continue
            return ScalableQPCertificate(
                exact,
                float(numerical.fun),
                int(getattr(numerical, "nit", 0)),
                threshold,
            )

    ordering = tuple(sorted(range(donor_count), key=lambda index: -numerical.x[index]))
    equality_rank = len(independent_rows)
    fitted_rank_bound = min(donor_count, equality_rank + len(target))
    for support_size in range(max(1, equality_rank), fitted_rank_bound + 1):
        support = tuple(sorted(ordering[:support_size]))
        if support in attempted:
            continue
        attempted.add(support)
        try:
            exact = certify_support_with_dual_proposal(support)
        except ValueError:
            continue
        return ScalableQPCertificate(
            exact,
            float(numerical.fun),
            int(getattr(numerical, "nit", 0)),
            -float(support_size),
        )

    fixed_zero_variables = set()
    for row, value in zip(independent_rows, independent_rhs):
        nonzero = tuple(index for index, coefficient in enumerate(row) if coefficient)
        if value == 0 and len(nonzero) == 1:
            fixed_zero_variables.add(nonzero[0])
    if fixed_zero_variables:
        kept_variables = tuple(
            index
            for index in range(donor_count)
            if index not in fixed_zero_variables
        )
        reduced_rows = tuple(
            tuple(row[index] for index in kept_variables)
            for row in independent_rows
            if any(row[index] for index in kept_variables)
        )
        reduced_rhs = tuple(
            value
            for row, value in zip(independent_rows, independent_rhs)
            if any(row[index] for index in kept_variables)
        )
        reduced = propose_and_certify_qp(
            target,
            tuple(
                tuple(row[index] for index in kept_variables)
                for row in donors
            ),
            row_weights,
            reduced_rows,
            reduced_rhs,
        )
        lifted_support = tuple(
            kept_variables[index] for index in reduced.exact.support
        )
        exact = certify_support_with_dual_proposal(lifted_support)
        return ScalableQPCertificate(
            exact,
            reduced.numerical_objective,
            reduced.numerical_iterations,
            reduced.support_threshold,
        )

    raise ValueError("no numerically proposed support passed exact KKT verification")
