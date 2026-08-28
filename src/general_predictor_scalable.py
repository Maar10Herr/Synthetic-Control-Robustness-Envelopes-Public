"""Optional numerical discovery for exact arbitrary-K graph certificates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import List, Optional, Sequence, Tuple

from .general_predictor import (
    GeneralGraphBranch,
    GeneralPredictorGraph,
    SignVector,
    _coordinate_branch,
    _donor_points,
    _faces,
    _facets,
    normal_orthant_system,
    normal_witness_from_system_solution,
)
from .scalable_qp import propose_and_certify_lp
from .scm_oracle import (
    Matrix,
    QPKKTCertificate,
    Rat,
    Vector,
    _dot,
    _rank,
    _solve_linear_system,
    certify_qp_support,
)
from .two_predictor import CertifiedInterval, CertifiedNearEffectEnvelope, GraphPolytope
from .two_predictor_scalable import (
    ScalableBranchResult,
    _augmented_validation,
    _bound_feasible,
    _branch_effect_range,
    _minimum_with_effect_constraint,
    _solve_branch,
)


@dataclass(frozen=True)
class InnerAttainabilityCertificate:
    predictor_weights: Vector
    inner: QPKKTCertificate


@dataclass(frozen=True)
class NormalOrthantDecision:
    witness: Optional[Vector]
    farkas_witness: Optional[Vector]
    system_solution: Optional[Vector] = None


@dataclass(frozen=True)
class NonnegativeSystemDecision:
    solution: Optional[Vector]
    farkas_witness: Optional[Vector]


@dataclass(frozen=True)
class GraphDecisionRecord:
    stage: str
    face: frozenset[int]
    sign: SignVector
    solution: Optional[Vector]
    farkas_witness: Optional[Vector]


@dataclass(frozen=True)
class UniqueNonnegativeSolutionCertificate:
    support: Tuple[int, ...]
    exposing_multipliers: Vector
    exposure_margins: Vector


@dataclass(frozen=True)
class CoordinateRefinementDecision:
    secondary_predictor_weights: Optional[Vector]
    system_solution: Optional[Vector]
    farkas_witness: Optional[Vector]


@dataclass(frozen=True)
class ScalableGeneralPredictorReduction:
    graph: GeneralPredictorGraph
    branch_results: Tuple[ScalableBranchResult, ...]
    outer_optimum: Rat
    exact_optimum_effects: Tuple[Rat, Rat]
    attainable_effects: Tuple[Rat, Rat]
    sign_breakdown_epsilon: Optional[Rat]
    attained_zero_epsilon: Optional[Rat]
    validation_target: Vector
    validation_donors: Matrix
    treated_post: Rat
    donor_post: Vector

    @property
    def branches(self) -> Tuple[GraphPolytope, ...]:
        return tuple(result.branch for result in self.branch_results)

    def minimum_loss_with_effect_bound(
        self,
        cutoff: Rat,
        relation: str,
    ) -> Optional[Rat]:
        if relation not in ("at_most", "at_least", "equal"):
            raise ValueError("unknown effect relation")
        return _minimum_with_effect_constraint(self, cutoff, relation)

    def near_effect_envelope(
        self,
        epsilon: Rat,
        digits: int = 6,
    ) -> CertifiedNearEffectEnvelope:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        if digits < 1:
            raise ValueError("digits must be positive")
        if epsilon == 0:
            lower, upper = self.exact_optimum_effects
            return CertifiedNearEffectEnvelope(
                CertifiedInterval(lower, lower),
                CertifiedInterval(upper, upper),
                0,
            )

        threshold = self.outer_optimum + epsilon
        width = Rat(1, 10**digits)
        steps = 0
        lower, upper = self.attainable_effects[0], self.exact_optimum_effects[0]
        if _bound_feasible(self, lower, "at_most", threshold):
            lower_box = CertifiedInterval(lower, lower)
        else:
            while upper - lower > width:
                midpoint = (lower + upper) / 2
                if _bound_feasible(self, midpoint, "at_most", threshold):
                    upper = midpoint
                else:
                    lower = midpoint
                steps += 1
            lower_box = CertifiedInterval(lower, upper)

        lower, upper = self.exact_optimum_effects[1], self.attainable_effects[1]
        if _bound_feasible(self, upper, "at_least", threshold):
            upper_box = CertifiedInterval(upper, upper)
        else:
            while upper - lower > width:
                midpoint = (lower + upper) / 2
                if _bound_feasible(self, midpoint, "at_least", threshold):
                    lower = midpoint
                else:
                    upper = midpoint
                steps += 1
            upper_box = CertifiedInterval(lower, upper)
        return CertifiedNearEffectEnvelope(lower_box, upper_box, steps)


def propose_and_certify_normal_orthant_witness(
    normals: Sequence[Vector],
    sign: SignVector,
) -> Vector:
    """Use HiGHS to propose a cone support, then accept it with exact LP KKT."""

    rows, rhs = normal_orthant_system(normals, sign)
    cost = tuple(Rat(0) for _ in range(len(rows[0])))
    certificate = propose_and_certify_lp(cost, rows, rhs).exact
    return normal_witness_from_system_solution(
        normals,
        sign,
        certificate.optimizer,
    )


def decide_nonnegative_system(
    rows: Matrix,
    rhs: Vector,
) -> NonnegativeSystemDecision:
    """Exactly decide feasibility of ``rows @ x = rhs, x >= 0``."""

    if not rows or len(rows) != len(rhs):
        raise ValueError("nonnegative system dimensions do not match")
    primal_variable_count = len(rows[0])
    if primal_variable_count < 1 or any(
        len(row) != primal_variable_count for row in rows
    ):
        raise ValueError("nonnegative system matrix is empty or ragged")
    try:
        certificate = propose_and_certify_lp(
            tuple(Rat(0) for _ in range(primal_variable_count)),
            rows,
            rhs,
        ).exact
        return NonnegativeSystemDecision(certificate.optimizer, None)
    except ValueError:
        pass

    equality_count = len(rows)
    dual_variable_count = 2 * equality_count + primal_variable_count
    farkas_rows: List[Vector] = [
        rhs
        + tuple(-value for value in rhs)
        + tuple(Rat(0) for _ in range(primal_variable_count))
    ]
    farkas_rhs: List[Rat] = [Rat(-1)]
    for column in range(primal_variable_count):
        coefficients = tuple(row[column] for row in rows)
        farkas_rows.append(
            coefficients
            + tuple(-value for value in coefficients)
            + tuple(
                Rat(-1) if index == column else Rat(0)
                for index in range(primal_variable_count)
            )
        )
        farkas_rhs.append(Rat(0))
    certificate = propose_and_certify_lp(
        tuple(Rat(0) for _ in range(dual_variable_count)),
        tuple(farkas_rows),
        tuple(farkas_rhs),
    ).exact
    positive = certificate.optimizer[:equality_count]
    negative = certificate.optimizer[equality_count : 2 * equality_count]
    farkas = tuple(left - right for left, right in zip(positive, negative))
    if _dot(rhs, farkas) != -1:
        raise AssertionError("Farkas witness has wrong normalization")
    if any(
        sum(
            (row[column] * farkas[index] for index, row in enumerate(rows)),
            Rat(0),
        )
        < 0
        for column in range(primal_variable_count)
    ):
        raise AssertionError("Farkas witness violates dual feasibility")
    return NonnegativeSystemDecision(None, farkas)


def decide_normal_orthant_intersection(
    normals: Sequence[Vector],
    sign: SignVector,
) -> NormalOrthantDecision:
    """Exactly decide cone/orthant compatibility using primal or Farkas support."""

    rows, rhs = normal_orthant_system(normals, sign)
    decision = decide_nonnegative_system(rows, rhs)
    if decision.solution is not None:
        return NormalOrthantDecision(
            normal_witness_from_system_solution(
                normals,
                sign,
                decision.solution,
            ),
            None,
            decision.solution,
        )
    return NormalOrthantDecision(None, decision.farkas_witness, None)


def certify_unique_nonnegative_solution(
    rows: Matrix,
    rhs: Vector,
    point: Vector,
) -> UniqueNonnegativeSolutionCertificate:
    """Certify that ``point`` uniquely solves ``rows @ x = rhs, x >= 0``."""

    if not rows or len(rows) != len(rhs):
        raise ValueError("nonnegative system dimensions do not match")
    variable_count = len(point)
    if any(len(row) != variable_count for row in rows):
        raise ValueError("nonnegative system matrix is ragged")
    if any(value < 0 for value in point) or any(
        _dot(row, point) != value for row, value in zip(rows, rhs)
    ):
        raise ValueError("proposed point is not feasible")
    support = tuple(index for index, value in enumerate(point) if value > 0)
    inactive = tuple(
        index for index in range(variable_count) if index not in support
    )
    restricted = tuple(tuple(row[index] for index in support) for row in rows)
    if _rank(restricted) != len(support):
        raise ValueError("active columns do not determine the proposed point")
    if not inactive:
        return UniqueNonnegativeSolutionCertificate(
            support,
            tuple(Rat(0) for _ in rows),
            (),
        )

    equality_count = len(rows)
    certificate_variable_count = 2 * equality_count + len(inactive)

    def exposure_row(donor: int, slack_position: Optional[int] = None) -> Vector:
        column = tuple(row[donor] for row in rows)
        return (
            column
            + tuple(-value for value in column)
            + tuple(
                Rat(-1) if position == slack_position else Rat(0)
                for position in range(len(inactive))
            )
        )

    exposure_rows = tuple(exposure_row(donor) for donor in support) + tuple(
        exposure_row(donor, position)
        for position, donor in enumerate(inactive)
    )
    exposure_rhs = tuple(Rat(0) for _ in support) + tuple(
        Rat(1) for _ in inactive
    )
    if any(len(row) != certificate_variable_count for row in exposure_rows):
        raise AssertionError("exposure system has inconsistent width")
    decision = decide_nonnegative_system(exposure_rows, exposure_rhs)
    if decision.solution is None:
        raise ValueError("no strict exposing multiplier was certified")
    positive = decision.solution[:equality_count]
    negative = decision.solution[equality_count : 2 * equality_count]
    multipliers = tuple(left - right for left, right in zip(positive, negative))
    margins = tuple(
        sum(
            (rows[row][donor] * multipliers[row] for row in range(equality_count)),
            Rat(0),
        )
        for donor in range(variable_count)
    )
    if any(margins[donor] != 0 for donor in support) or any(
        margins[donor] < 1 for donor in inactive
    ):
        raise AssertionError("strict exposure verification failed")
    return UniqueNonnegativeSolutionCertificate(support, multipliers, margins)


def coordinate_refinement_system(
    predictor_target: Vector,
    predictor_donors: Matrix,
    donor_weights: Vector,
    coordinate: int,
) -> Tuple[Matrix, Vector, Tuple[int, ...]]:
    """Construct the nonnegative KKT system for coordinate refinement."""

    dimension = len(predictor_target)
    donor_count = len(donor_weights)
    if coordinate < 0 or coordinate >= dimension:
        raise ValueError("coordinate is out of range")
    if len(predictor_donors) != dimension or any(
        len(row) != donor_count for row in predictor_donors
    ):
        raise ValueError("predictor dimensions do not match donor weights")
    if any(value < 0 for value in donor_weights) or sum(
        donor_weights, Rat(0)
    ) != 1:
        raise ValueError("donor weights must lie on the simplex")
    if _dot(predictor_donors[coordinate], donor_weights) != predictor_target[
        coordinate
    ]:
        raise ValueError("donor weights do not match the declared coordinate")

    support = tuple(
        index for index, value in enumerate(donor_weights) if value > 0
    )
    inactive = tuple(
        index for index in range(donor_count) if index not in support
    )
    secondary = tuple(index for index in range(dimension) if index != coordinate)
    residual = tuple(
        predictor_target[index] - _dot(predictor_donors[index], donor_weights)
        for index in range(dimension)
    )
    equality_rows = (
        tuple(Rat(1) for _ in range(donor_count)),
        predictor_donors[coordinate],
    )
    equality_count = len(equality_rows)
    secondary_count = len(secondary)
    system_variable_count = (
        secondary_count + 2 * equality_count + len(inactive)
    )
    system_rows: List[Vector] = [
        tuple(Rat(1) for _ in secondary)
        + tuple(
            Rat(0)
            for _ in range(system_variable_count - secondary_count)
        )
    ]
    system_rhs: List[Rat] = [Rat(1)]

    def stationarity_row(
        donor: int,
        slack_position: Optional[int] = None,
    ) -> Vector:
        multiplier_column = tuple(row[donor] for row in equality_rows)
        return (
            tuple(
                -predictor_donors[index][donor] * residual[index]
                for index in secondary
            )
            + multiplier_column
            + tuple(-value for value in multiplier_column)
            + tuple(
                Rat(-1) if position == slack_position else Rat(0)
                for position in range(len(inactive))
            )
        )

    for donor in support:
        system_rows.append(stationarity_row(donor))
        system_rhs.append(Rat(0))
    for position, donor in enumerate(inactive):
        system_rows.append(stationarity_row(donor, position))
        system_rhs.append(Rat(0))
    return tuple(system_rows), tuple(system_rhs), secondary


def decide_coordinate_refinement(
    predictor_target: Vector,
    predictor_donors: Matrix,
    donor_weights: Vector,
    coordinate: int,
) -> CoordinateRefinementDecision:
    """Decide whether a coordinate-match point passes first-order refinement.

    The refinement minimizes a nonnegative weighted combination of the other
    predictor losses over the exact coordinate-match polytope. Failure is an
    exact necessary-condition certificate against reachability from strictly
    positive predictor weights converging to the coordinate vertex.
    """

    rows, rhs, secondary = coordinate_refinement_system(
        predictor_target,
        predictor_donors,
        donor_weights,
        coordinate,
    )
    decision = decide_nonnegative_system(rows, rhs)
    if decision.solution is None:
        return CoordinateRefinementDecision(
            None,
            None,
            decision.farkas_witness,
        )
    secondary_weights = [Rat(0) for _ in predictor_target]
    for index, value in zip(
        secondary,
        decision.solution[: len(secondary)],
    ):
        secondary_weights[index] = value
    return CoordinateRefinementDecision(
        tuple(secondary_weights),
        decision.solution,
        None,
    )


def _try_inner_certificate(
    predictor_target: Vector,
    predictor_donors: Matrix,
    donor_weights: Vector,
    predictor_weights: Vector,
    donor_support: Sequence[int],
) -> InnerAttainabilityCertificate | None:
    try:
        inner = certify_qp_support(
            predictor_target,
            predictor_donors,
            predictor_weights,
            (tuple(Rat(1) for _ in donor_weights),),
            (Rat(1),),
            donor_support,
        )
    except ValueError:
        return None
    if inner.optimizer != donor_weights:
        return None
    return InnerAttainabilityCertificate(predictor_weights, inner)


def certify_inner_attainability_on_predictor_support(
    predictor_target: Vector,
    predictor_donors: Matrix,
    donor_weights: Vector,
    predictor_support: Sequence[int],
) -> InnerAttainabilityCertificate:
    """Reconstruct V on a declared support and exactly verify inner KKT."""

    dimension = len(predictor_target)
    donor_count = len(donor_weights)
    normalized_predictor_support = tuple(sorted(set(predictor_support)))
    if not normalized_predictor_support or any(
        index < 0 or index >= dimension for index in normalized_predictor_support
    ):
        raise ValueError("predictor support is empty or out of range")
    if len(predictor_donors) != dimension or any(
        len(row) != donor_count for row in predictor_donors
    ):
        raise ValueError("predictor dimensions do not match donor weights")
    if any(value < 0 for value in donor_weights) or sum(donor_weights, Rat(0)) != 1:
        raise ValueError("donor weights must lie on the simplex")
    donor_support = tuple(
        index for index, value in enumerate(donor_weights) if value > 0
    )
    if not donor_support:
        raise ValueError("donor support is empty")
    fitted = tuple(_dot(row, donor_weights) for row in predictor_donors)
    residual = tuple(
        target - value for target, value in zip(predictor_target, fitted)
    )
    points = tuple(
        tuple(row[index] for row in predictor_donors)
        for index in range(donor_count)
    )
    reference = donor_support[0]
    active_rows = tuple(
        tuple(
            residual[k] * (points[donor][k] - points[reference][k])
            for k in normalized_predictor_support
        )
        for donor in donor_support[1:]
    )
    solution = _solve_linear_system(
        (tuple(Rat(1) for _ in normalized_predictor_support),) + active_rows,
        (Rat(1),) + tuple(Rat(0) for _ in active_rows),
        len(normalized_predictor_support),
    )
    if solution is None or any(value <= 0 for value in solution):
        raise ValueError("declared predictor support has no positive stationary solution")
    predictor_weights = [Rat(0) for _ in range(dimension)]
    for index, value in zip(normalized_predictor_support, solution):
        predictor_weights[index] = value
    certificate = _try_inner_certificate(
        predictor_target,
        predictor_donors,
        donor_weights,
        tuple(predictor_weights),
        donor_support,
    )
    if certificate is None:
        raise ValueError("declared predictor support fails exact inner KKT checks")
    return certificate


def propose_and_certify_inner_attainability(
    predictor_target: Vector,
    predictor_donors: Matrix,
    donor_weights: Vector,
) -> InnerAttainabilityCertificate:
    """Certify that fixed donor weights solve the inner problem for some V."""

    dimension = len(predictor_target)
    donor_count = len(donor_weights)
    if len(predictor_donors) != dimension or any(
        len(row) != donor_count for row in predictor_donors
    ):
        raise ValueError("predictor dimensions do not match donor weights")
    if any(value < 0 for value in donor_weights) or sum(donor_weights, Rat(0)) != 1:
        raise ValueError("donor weights must lie on the simplex")
    donor_support = tuple(
        index for index, value in enumerate(donor_weights) if value > 0
    )
    if not donor_support:
        raise ValueError("donor support is empty")
    fitted = tuple(_dot(row, donor_weights) for row in predictor_donors)
    residual = tuple(
        target - value for target, value in zip(predictor_target, fitted)
    )

    for coordinate, value in enumerate(residual):
        if value != 0:
            continue
        predictor_weights = tuple(
            Rat(1) if index == coordinate else Rat(0)
            for index in range(dimension)
        )
        certificate = _try_inner_certificate(
            predictor_target,
            predictor_donors,
            donor_weights,
            predictor_weights,
            donor_support,
        )
        if certificate is not None:
            return certificate

    points = tuple(
        tuple(row[index] for row in predictor_donors)
        for index in range(donor_count)
    )
    reference = donor_support[0]
    active_rows = tuple(
        tuple(
            residual[k] * (points[donor][k] - points[reference][k])
            for k in range(dimension)
        )
        for donor in donor_support[1:]
    )
    maximum_sparse_size = min(dimension, len(donor_support))
    for support_size in range(1, maximum_sparse_size + 1):
        for predictor_support in combinations(range(dimension), support_size):
            try:
                return certify_inner_attainability_on_predictor_support(
                    predictor_target,
                    predictor_donors,
                    donor_weights,
                    predictor_support,
                )
            except ValueError:
                continue

    inactive = tuple(
        donor for donor in range(donor_count) if donor not in donor_support
    )
    variable_count = dimension + len(inactive)
    rows = [
        tuple(Rat(1) for _ in range(dimension))
        + tuple(Rat(0) for _ in inactive)
    ]
    rhs = [Rat(1)]
    for row in active_rows:
        rows.append(row + tuple(Rat(0) for _ in inactive))
        rhs.append(Rat(0))
    for position, donor in enumerate(inactive):
        rows.append(
            tuple(
                residual[k] * (points[donor][k] - points[reference][k])
                for k in range(dimension)
            )
            + tuple(
                Rat(1) if index == position else Rat(0)
                for index in range(len(inactive))
            )
        )
        rhs.append(Rat(0))
    feasibility = propose_and_certify_lp(
        tuple(Rat(0) for _ in range(variable_count)),
        tuple(rows),
        tuple(rhs),
    ).exact
    certificate = _try_inner_certificate(
        predictor_target,
        predictor_donors,
        donor_weights,
        feasibility.optimizer[:dimension],
        donor_support,
    )
    if certificate is None:
        raise ValueError("no proposed predictor weights passed exact inner KKT checks")
    return certificate


def normal_branch_from_certificate(
    face: frozenset[int],
    sign: SignVector,
    witness: Vector,
    predictor_target: Vector,
    predictor_donors: Matrix,
) -> GeneralGraphBranch:
    """Build a normal branch whose saved witnesses are verified separately."""

    donor_count = len(predictor_donors[0])
    dimension = len(predictor_target)
    auxiliary_zeros = tuple(Rat(0) for _ in range(dimension))
    rows: List[Vector] = [
        tuple(Rat(1) for _ in range(donor_count)) + auxiliary_zeros
    ]
    rhs: List[Rat] = [Rat(1)]
    for donor in range(donor_count):
        if donor not in face:
            rows.append(
                tuple(
                    Rat(1) if index == donor else Rat(0)
                    for index in range(donor_count)
                )
                + auxiliary_zeros
            )
            rhs.append(Rat(0))
    for coordinate in range(dimension):
        rows.append(
            tuple(
                Rat(sign[coordinate]) * value
                for value in predictor_donors[coordinate]
            )
            + tuple(
                Rat(1) if index == coordinate else Rat(0)
                for index in range(dimension)
            )
        )
        rhs.append(Rat(sign[coordinate]) * predictor_target[coordinate])
    polytope = GraphPolytope(
        "normal-face-%s-sign-%s"
        % (
            "-".join(map(str, sorted(face))),
            "".join("p" if value > 0 else "m" for value in sign),
        ),
        donor_count,
        tuple(rows),
        tuple(rhs),
    )
    return GeneralGraphBranch(
        polytope,
        "normal-face",
        face,
        residual_sign=sign,
        normal_witness=witness,
    )


def _normal_branch_scalable(
    face: frozenset[int],
    sign: SignVector,
    witness: Vector,
    predictor_target: Vector,
    predictor_donors: Matrix,
) -> Tuple[Optional[GeneralGraphBranch], NonnegativeSystemDecision]:
    branch = normal_branch_from_certificate(
        face,
        sign,
        witness,
        predictor_target,
        predictor_donors,
    )
    decision = decide_nonnegative_system(
        branch.polytope.equality_rows,
        branch.polytope.equality_rhs,
    )
    return (branch if decision.solution is not None else None), decision


def construct_general_predictor_graph_scalable(
    predictor_target: Vector,
    predictor_donors: Matrix,
    decision_log: Optional[List[GraphDecisionRecord]] = None,
) -> GeneralPredictorGraph:
    """Construct the exact arbitrary-K graph with exact primal/Farkas decisions."""

    dimension = len(predictor_target)
    if dimension < 1 or len(predictor_donors) != dimension:
        raise ValueError("predictor dimensions do not match")
    donor_count = len(predictor_donors[0])
    if donor_count < dimension + 1 or any(
        len(row) != donor_count for row in predictor_donors
    ):
        raise ValueError("full-dimensional hull requires at least K+1 donor columns")
    points = _donor_points(predictor_donors)
    differences = tuple(
        tuple(point[k] - points[0][k] for k in range(dimension))
        for point in points[1:]
    )
    if _rank(differences) != dimension:
        raise ValueError("predictor-donor convex hull must be full-dimensional")

    facet_normals = _facets(points)
    facet_sets = tuple(
        sorted(facet_normals, key=lambda face: tuple(sorted(face)))
    )
    faces = _faces(donor_count, facet_sets)
    proper_faces = tuple(face for face in faces if len(face) < donor_count)
    containing_normals = {
        face: tuple(
            normal
            for facet, normal in facet_normals.items()
            if face <= facet
        )
        for face in proper_faces
    }

    branches: List[GeneralGraphBranch] = []
    for coordinate in range(dimension):
        branch = _coordinate_branch(
            coordinate,
            predictor_target,
            predictor_donors,
        )
        if branch is not None:
            branches.append(branch)

    for sign in product((-1, 1), repeat=dimension):
        maximal: List[Tuple[frozenset[int], Vector]] = []
        for face in sorted(proper_faces, key=len, reverse=True):
            if any(face < kept_face for kept_face, _ in maximal):
                continue
            decision = decide_normal_orthant_intersection(
                containing_normals[face],
                sign,
            )
            if decision_log is not None:
                decision_log.append(
                    GraphDecisionRecord(
                        "normal-orthant",
                        face,
                        sign,
                        decision.system_solution,
                        decision.farkas_witness,
                    )
                )
            if decision.witness is not None:
                maximal.append((face, decision.witness))
        for face, witness in maximal:
            branch, decision = _normal_branch_scalable(
                face,
                sign,
                witness,
                predictor_target,
                predictor_donors,
            )
            if decision_log is not None:
                decision_log.append(
                    GraphDecisionRecord(
                        "normal-branch",
                        face,
                        sign,
                        decision.solution,
                        decision.farkas_witness,
                    )
                )
            if branch is not None:
                branches.append(branch)

    return GeneralPredictorGraph(
        dimension,
        donor_count,
        facet_sets,
        faces,
        tuple(branches),
    )


def reduce_general_predictor_problem_scalable(
    predictor_target: Vector,
    predictor_donors: Matrix,
    validation_target: Vector,
    validation_donors: Matrix,
    treated_post: Rat,
    donor_post: Vector,
    graph: Optional[GeneralPredictorGraph] = None,
) -> ScalableGeneralPredictorReduction:
    """Globally reduce an arbitrary-K full-dimensional rational SCM instance."""

    donor_count = len(donor_post)
    if not predictor_target or len(predictor_donors) != len(predictor_target):
        raise ValueError("predictor dimensions do not match")
    if any(len(row) != donor_count for row in predictor_donors):
        raise ValueError("predictor donor dimensions do not match")
    if (
        not validation_target
        or len(validation_target) != len(validation_donors)
        or any(len(row) != donor_count for row in validation_donors)
    ):
        raise ValueError("validation dimensions do not match")

    if graph is None:
        graph = construct_general_predictor_graph_scalable(
            predictor_target,
            predictor_donors,
        )
    elif (
        graph.predictor_dimension != len(predictor_target)
        or graph.donor_count != donor_count
    ):
        raise ValueError("provided graph dimensions do not match problem")
    branch_results = tuple(
        ScalableBranchResult(
            branch.polytope,
            _solve_branch(
                branch.polytope,
                validation_target,
                validation_donors,
            ),
            _branch_effect_range(
                branch.polytope,
                treated_post,
                donor_post,
            ),
        )
        for branch in graph.branches
    )
    if not branch_results:
        raise AssertionError("complete lower graph unexpectedly has no branches")
    outer_optimum = min(
        result.certificate.exact.value for result in branch_results
    )
    global_results = tuple(
        result
        for result in branch_results
        if result.certificate.exact.value == outer_optimum
    )
    exact_effects: List[Rat] = []
    for result in global_results:
        certificate = result.certificate.exact
        augmented_validation = _augmented_validation(
            result.branch,
            validation_donors,
        )
        optimizer_branch = GraphPolytope(
            f"{result.branch.name}-outer-optimizer",
            result.branch.donor_count,
            result.branch.equality_rows + augmented_validation,
            result.branch.equality_rhs
            + tuple(
                _dot(row, certificate.optimizer)
                for row in augmented_validation
            ),
        )
        lower, upper = _branch_effect_range(
            optimizer_branch,
            treated_post,
            donor_post,
        )
        exact_effects.extend((lower, upper))

    attainable = (
        min(result.effect_range[0] for result in branch_results),
        max(result.effect_range[1] for result in branch_results),
    )
    provisional = ScalableGeneralPredictorReduction(
        graph,
        branch_results,
        outer_optimum,
        (min(exact_effects), max(exact_effects)),
        attainable,
        None,
        None,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
    )
    nonpositive = _minimum_with_effect_constraint(
        provisional,
        Rat(0),
        "at_most",
    )
    nonnegative = _minimum_with_effect_constraint(
        provisional,
        Rat(0),
        "at_least",
    )
    zero = _minimum_with_effect_constraint(provisional, Rat(0), "equal")
    span = (
        max(nonpositive, nonnegative) - outer_optimum
        if nonpositive is not None and nonnegative is not None
        else None
    )
    attained_zero = zero - outer_optimum if zero is not None else None
    return ScalableGeneralPredictorReduction(
        graph,
        branch_results,
        outer_optimum,
        provisional.exact_optimum_effects,
        attainable,
        span,
        attained_zero,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
    )
