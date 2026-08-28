"""Exact finite-polytope lower-graph construction for arbitrary fixed K.

The implementation is deliberately exhaustive. It is intended as a structural
oracle for small full-dimensional rational predictor hulls, not as a scalable
high-dimensional face enumerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .scm_oracle import (
    Matrix,
    Rat,
    Vector,
    _dot,
    _rank,
    _solve_square,
    polytope_vertices,
)
from .two_predictor import GraphPolytope


Point = Tuple[Rat, ...]
SignVector = Tuple[int, ...]


@dataclass(frozen=True)
class GeneralGraphBranch:
    polytope: GraphPolytope
    kind: str
    face: FrozenSet[int]
    coordinate: Optional[int] = None
    residual_sign: Optional[SignVector] = None
    normal_witness: Optional[Vector] = None


@dataclass(frozen=True)
class GeneralPredictorGraph:
    predictor_dimension: int
    donor_count: int
    facets: Tuple[FrozenSet[int], ...]
    faces: Tuple[FrozenSet[int], ...]
    branches: Tuple[GeneralGraphBranch, ...]


def _donor_points(predictor_donors: Matrix) -> Tuple[Point, ...]:
    return tuple(
        tuple(row[index] for row in predictor_donors)
        for index in range(len(predictor_donors[0]))
    )


def _facet_normal(points: Sequence[Point], indices: Sequence[int]) -> Optional[Vector]:
    dimension = len(points[0])
    base = points[indices[0]]
    differences = tuple(
        tuple(points[index][coordinate] - base[coordinate] for coordinate in range(dimension))
        for index in indices[1:]
    )
    if _rank(differences) != dimension - 1:
        return None
    for fixed_coordinate in range(dimension):
        fixed_row = tuple(
            Rat(1) if coordinate == fixed_coordinate else Rat(0)
            for coordinate in range(dimension)
        )
        normal = _solve_square(
            differences + (fixed_row,),
            tuple(Rat(0) for _ in differences) + (Rat(1),),
        )
        if normal is None:
            continue
        offsets = tuple(
            _dot(normal, tuple(point[k] - base[k] for k in range(dimension)))
            for point in points
        )
        if all(value <= 0 for value in offsets) and any(value < 0 for value in offsets):
            return normal
        if all(value >= 0 for value in offsets) and any(value > 0 for value in offsets):
            return tuple(-value for value in normal)
        return None
    raise AssertionError("rank-(K-1) differences have no nonzero null vector")


def _facets(points: Sequence[Point]) -> Dict[FrozenSet[int], Vector]:
    dimension = len(points[0])
    facets: Dict[FrozenSet[int], Vector] = {}
    for indices in combinations(range(len(points)), dimension):
        normal = _facet_normal(points, indices)
        if normal is None:
            continue
        base = points[indices[0]]
        face = frozenset(
            index
            for index, point in enumerate(points)
            if _dot(
                normal,
                tuple(point[k] - base[k] for k in range(dimension)),
            )
            == 0
        )
        facets.setdefault(face, normal)
    if not facets:
        raise ValueError("no facets found; predictor hull is not full-dimensional")
    return facets


def _faces(donor_count: int, facets: Sequence[FrozenSet[int]]) -> Tuple[FrozenSet[int], ...]:
    full_face = frozenset(range(donor_count))
    faces = {full_face, *facets}
    queue = list(facets)
    while queue:
        face = queue.pop()
        for facet in facets:
            intersection = face & facet
            if intersection and intersection not in faces:
                faces.add(intersection)
                queue.append(intersection)
    return tuple(sorted(faces, key=lambda face: (len(face), tuple(sorted(face)))))


def _normal_orthant_witness(
    normals: Sequence[Vector],
    sign: SignVector,
) -> Optional[Vector]:
    rows, rhs = normal_orthant_system(normals, sign)
    try:
        variables = polytope_vertices(rows, rhs)[0]
    except ValueError:
        return None
    return normal_witness_from_system_solution(normals, sign, variables)


def normal_orthant_system(
    normals: Sequence[Vector],
    sign: SignVector,
) -> Tuple[Matrix, Vector]:
    """Encode a nonzero normal-cone/closed-orthant intersection as equalities."""

    generator_count = len(normals)
    dimension = len(sign)
    if not normals:
        raise ValueError("at least one normal-cone generator is required")
    if any(len(normal) != dimension for normal in normals):
        raise ValueError("normal and sign dimensions do not match")
    signed_generators = tuple(
        tuple(Rat(sign[k]) * normal[k] for normal in normals)
        for k in range(dimension)
    )
    variable_count = generator_count + dimension
    rows: List[Vector] = []
    rhs: List[Rat] = []
    rows.append(
        tuple(
            sum((signed_generators[k][index] for k in range(dimension)), Rat(0))
            for index in range(generator_count)
        )
        + tuple(Rat(0) for _ in range(dimension))
    )
    rhs.append(Rat(1))
    for coordinate in range(dimension):
        rows.append(
            signed_generators[coordinate]
            + tuple(
                Rat(-1) if index == coordinate else Rat(0)
                for index in range(dimension)
            )
        )
        rhs.append(Rat(0))
    return tuple(rows), tuple(rhs)


def normal_witness_from_system_solution(
    normals: Sequence[Vector],
    sign: SignVector,
    variables: Vector,
) -> Vector:
    """Reconstruct and verify q from a normal-orthant system solution."""

    generator_count = len(normals)
    dimension = len(sign)
    rows, rhs = normal_orthant_system(normals, sign)
    if len(variables) != generator_count + dimension or any(
        value < 0 for value in variables
    ):
        raise ValueError("normal-orthant system solution has invalid dimensions or sign")
    if any(_dot(row, variables) != value for row, value in zip(rows, rhs)):
        raise ValueError("normal-orthant system equalities are violated")
    coefficients = variables[:generator_count]
    witness = tuple(
        sum(
            (coefficient * normal[k] for coefficient, normal in zip(coefficients, normals)),
            Rat(0),
        )
        for k in range(dimension)
    )
    if not any(witness):
        raise AssertionError("normalized normal-cone witness is zero")
    if any(Rat(sign[k]) * witness[k] < 0 for k in range(dimension)):
        raise AssertionError("normal-cone witness left requested orthant")
    return witness


def _coordinate_branch(
    coordinate: int,
    predictor_target: Vector,
    predictor_donors: Matrix,
) -> Optional[GeneralGraphBranch]:
    donor_count = len(predictor_donors[0])
    rows = (
        tuple(Rat(1) for _ in range(donor_count)),
        predictor_donors[coordinate],
    )
    rhs = (Rat(1), predictor_target[coordinate])
    try:
        polytope_vertices(rows, rhs)
    except ValueError:
        return None
    return GeneralGraphBranch(
        GraphPolytope(
            f"coordinate-match-{coordinate}",
            donor_count,
            rows,
            rhs,
        ),
        "coordinate-match",
        frozenset(range(donor_count)),
        coordinate=coordinate,
    )


def _normal_branch(
    face: FrozenSet[int],
    sign: SignVector,
    witness: Vector,
    predictor_target: Vector,
    predictor_donors: Matrix,
) -> Optional[GeneralGraphBranch]:
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
            tuple(Rat(sign[coordinate]) * value for value in predictor_donors[coordinate])
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
    try:
        polytope_vertices(polytope.equality_rows, polytope.equality_rhs)
    except ValueError:
        return None
    return GeneralGraphBranch(
        polytope,
        "normal-face",
        face,
        residual_sign=sign,
        normal_witness=witness,
    )


def construct_general_predictor_graph(
    predictor_target: Vector,
    predictor_donors: Matrix,
) -> GeneralPredictorGraph:
    """Construct the complete continuous-V lower graph for small full-dimensional data."""

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
    facet_sets = tuple(sorted(facet_normals, key=lambda face: tuple(sorted(face))))
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
        compatible: List[Tuple[FrozenSet[int], Vector]] = []
        for face in proper_faces:
            witness = _normal_orthant_witness(containing_normals[face], sign)
            if witness is not None:
                compatible.append((face, witness))
        maximal: List[Tuple[FrozenSet[int], Vector]] = []
        for face, witness in sorted(compatible, key=lambda item: -len(item[0])):
            if any(face < kept_face for kept_face, _ in maximal):
                continue
            maximal.append((face, witness))
        for face, witness in maximal:
            branch = _normal_branch(
                face,
                sign,
                witness,
                predictor_target,
                predictor_donors,
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


def branch_contains_weights(
    branch: GeneralGraphBranch,
    predictor_target: Vector,
    predictor_donors: Matrix,
    donor_weights: Vector,
) -> bool:
    if len(donor_weights) != branch.polytope.donor_count or any(
        value < 0 for value in donor_weights
    ):
        return False
    if branch.kind == "coordinate-match":
        variables = donor_weights
    else:
        assert branch.residual_sign is not None
        fitted = tuple(_dot(row, donor_weights) for row in predictor_donors)
        slacks = tuple(
            Rat(sign) * (target - value)
            for sign, target, value in zip(
                branch.residual_sign,
                predictor_target,
                fitted,
            )
        )
        if any(value < 0 for value in slacks):
            return False
        variables = donor_weights + slacks
    return all(
        _dot(row, variables) == value
        for row, value in zip(
            branch.polytope.equality_rows,
            branch.polytope.equality_rhs,
        )
    )


def witness_predictor_weights(
    branch: GeneralGraphBranch,
    predictor_target: Vector,
    predictor_donors: Matrix,
    donor_weights: Vector,
) -> Vector:
    if not branch_contains_weights(
        branch,
        predictor_target,
        predictor_donors,
        donor_weights,
    ):
        raise ValueError("donor weights do not belong to branch")
    dimension = len(predictor_target)
    if branch.kind == "coordinate-match":
        assert branch.coordinate is not None
        return tuple(
            Rat(1) if index == branch.coordinate else Rat(0)
            for index in range(dimension)
        )
    fitted = tuple(_dot(row, donor_weights) for row in predictor_donors)
    residual = tuple(
        target - value for target, value in zip(predictor_target, fitted)
    )
    zero_coordinate = next(
        (index for index, value in enumerate(residual) if value == 0),
        None,
    )
    if zero_coordinate is not None:
        return tuple(
            Rat(1) if index == zero_coordinate else Rat(0)
            for index in range(dimension)
        )
    assert branch.normal_witness is not None
    unnormalized = tuple(
        normal / value
        for normal, value in zip(branch.normal_witness, residual)
    )
    if any(value < 0 for value in unnormalized):
        raise AssertionError("normal and residual signs are incompatible")
    scale = sum(unnormalized, Rat(0))
    if scale <= 0:
        raise AssertionError("normal witness induced zero predictor weight")
    return tuple(value / scale for value in unnormalized)
