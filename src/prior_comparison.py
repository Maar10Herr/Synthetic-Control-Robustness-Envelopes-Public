"""Certified convex feasible-set comparators from closest prior SCM work.

Ferguson--Ross's constrained-error metric is a relative Euclidean-loss
sublevel over simplex donor weights. Koo--Guo's empirical uncertainty class is
a simplex intersected with an infinity-norm sample-moment constraint. This
module keeps those two one-level objects distinct from the ADH bilevel graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from .scalable_qp import ScalableLPCertificate, propose_and_certify_lp
from .scm_oracle import Matrix, Rat, Vector, _dot
from .simplex_relaxation import ScalableSimplexRelaxation
from .two_predictor import CertifiedNearEffectEnvelope


@dataclass(frozen=True)
class FergusonRossEnvelope:
    """Equation-16 constrained-error envelope at a rational radius ``B``."""

    relative_error_radius: Rat
    squared_loss_gap: Rat
    effect_envelope: CertifiedNearEffectEnvelope


@dataclass(frozen=True)
class MomentRadiusCertificate:
    """Minimum Koo--Guo sample-moment radius, optionally at a fixed effect."""

    radius: Rat
    donor_weights: Vector
    certificate: ScalableLPCertificate


@dataclass(frozen=True)
class KooGuoMomentEnvelope:
    """Exact sensitivity interval over a fixed rational moment radius."""

    radius: Rat
    lower_effect: Rat
    upper_effect: Rat
    lower_weights: Vector
    upper_weights: Vector
    lower_certificate: ScalableLPCertificate
    upper_certificate: ScalableLPCertificate

    @property
    def zero_robust_effect(self) -> Rat:
        """Projection of zero onto the sensitivity interval (their Theorem 2)."""

        if self.lower_effect > 0:
            return self.lower_effect
        if self.upper_effect < 0:
            return self.upper_effect
        return Rat(0)


@dataclass(frozen=True)
class KooGuoAuthorRadius:
    """Author-default-style effective radius, frozen to a decimal rational."""

    nominal_lambda: Rat
    effective_radius: Rat
    nu_used: Rat
    feasibility_attempts: int
    residual_sample_variance: Rat
    maximum_donor_mean_square: Rat


def ferguson_ross_constrained_error_envelope(
    one_level: ScalableSimplexRelaxation,
    relative_error_radius: Rat,
    digits: int = 7,
) -> FergusonRossEnvelope:
    """Map Ferguson--Ross's norm radius to the squared-loss parameterization.

    Their equation (16) admits ``w`` when its Euclidean pre-treatment error is
    at most ``(1+B)`` times the minimum. Thus the equivalent additive squared-
    loss gap is ``((1+B)^2-1) L_star``. The ratio is undefined at zero minimum
    loss, which is rejected explicitly here.
    """

    if relative_error_radius < 0:
        raise ValueError("relative error radius must be nonnegative")
    if one_level.outer_optimum <= 0:
        raise ValueError(
            "Ferguson--Ross relative constrained error is undefined at zero loss"
        )
    squared_loss_gap = (
        (Rat(1) + relative_error_radius) ** 2 - Rat(1)
    ) * one_level.outer_optimum
    return FergusonRossEnvelope(
        relative_error_radius,
        squared_loss_gap,
        one_level.near_effect_envelope(squared_loss_gap, digits=digits),
    )


def sample_moments(
    pre_target: Vector,
    pre_donors: Matrix,
) -> Tuple[Matrix, Vector]:
    """Return exact ``Sigma_hat`` and ``gamma_hat`` from Koo--Guo equation 14."""

    period_count = len(pre_target)
    if (
        period_count < 1
        or len(pre_donors) != period_count
        or not pre_donors
        or any(len(row) != len(pre_donors[0]) for row in pre_donors)
    ):
        raise ValueError("pre-treatment dimensions do not match")
    donor_count = len(pre_donors[0])
    sigma = tuple(
        tuple(
            sum(
                (row[first] * row[second] for row in pre_donors),
                Rat(0),
            )
            / period_count
            for second in range(donor_count)
        )
        for first in range(donor_count)
    )
    gamma = tuple(
        sum(
            (row[donor] * outcome for row, outcome in zip(pre_donors, pre_target)),
            Rat(0),
        )
        / period_count
        for donor in range(donor_count)
    )
    return sigma, gamma


def koo_guo_author_default_radius(
    one_level: ScalableSimplexRelaxation,
    pre_target: Vector,
    pre_donors: Matrix,
    nominal_lambda: Rat,
    minimum_feasible_radius: Rat,
    initial_nu: Rat = Rat(1, 100),
    eta: Rat = Rat(1, 100),
    multiplier: Rat = Rat(5, 4),
) -> KooGuoAuthorRadius:
    """Replay the point-estimation radius loop in the v3 authors' ``helpers.R``.

    The code uses double-precision square roots and logarithms, starts ``nu`` at
    0.01, and multiplies it by 1.25 until the moment class is feasible. We freeze
    each resulting double with 17 significant digits and then compare it with
    the exact rational feasibility floor. Endpoint optimization downstream is
    therefore exact conditional on this explicitly serialized implementation
    radius; no claim is made that the transcendental radius itself is rational.
    """

    if nominal_lambda < 0 or minimum_feasible_radius < 0:
        raise ValueError("lambda and the feasibility floor must be nonnegative")
    period_count = len(pre_target)
    if (
        period_count < 2
        or len(pre_donors) != period_count
        or any(len(row) != len(one_level.optimum.exact.optimizer) for row in pre_donors)
    ):
        raise ValueError("pre-treatment dimensions do not match")
    weights = one_level.optimum.exact.optimizer
    residuals = tuple(
        target - _dot(row, weights)
        for target, row in zip(pre_target, pre_donors)
    )
    residual_mean = sum(residuals, Rat(0)) / period_count
    residual_sample_variance = sum(
        ((value - residual_mean) ** 2 for value in residuals),
        Rat(0),
    ) / (period_count - 1)
    donor_count = len(weights)
    maximum_donor_mean_square = max(
        sum((row[donor] ** 2 for row in pre_donors), Rat(0)) / period_count
        for donor in range(donor_count)
    )
    residual_sd = math.sqrt(float(residual_sample_variance))
    maximum_donor_rms = math.sqrt(float(maximum_donor_mean_square))
    factor = math.sqrt(math.log(max(period_count, donor_count))) / math.sqrt(
        period_count
    )
    nu = float(initial_nu)
    lambda_float = float(nominal_lambda)
    for attempt in range(1, 1001):
        radius_float = lambda_float + factor * (
            nu * residual_sd * maximum_donor_rms
            + float(eta) * lambda_float
        )
        radius = Rat(format(radius_float, ".17g"))
        if radius >= minimum_feasible_radius:
            return KooGuoAuthorRadius(
                nominal_lambda,
                radius,
                Rat(format(nu, ".17g")),
                attempt,
                residual_sample_variance,
                maximum_donor_mean_square,
            )
        nu *= float(multiplier)
    raise ValueError("author-style radius loop did not reach feasibility")


def _minimum_radius_standard_form(
    sigma: Matrix,
    gamma: Vector,
    donor_post: Optional[Vector] = None,
    treated_post: Optional[Rat] = None,
    effect_cutoff: Rat = Rat(0),
) -> Tuple[Vector, Matrix, Vector, int]:
    donor_count = len(gamma)
    if (
        donor_count < 1
        or len(sigma) != donor_count
        or any(len(row) != donor_count for row in sigma)
    ):
        raise ValueError("moment dimensions do not match")
    radius_index = donor_count
    variable_count = 3 * donor_count + 1
    rows = [
        tuple(Rat(1) if index < donor_count else Rat(0) for index in range(variable_count))
    ]
    rhs = [Rat(1)]
    for moment, sigma_row in enumerate(sigma):
        upper = [Rat(0) for _ in range(variable_count)]
        lower = [Rat(0) for _ in range(variable_count)]
        upper[:donor_count] = sigma_row
        lower[:donor_count] = tuple(-value for value in sigma_row)
        upper[radius_index] = Rat(-1)
        lower[radius_index] = Rat(-1)
        upper[donor_count + 1 + moment] = Rat(1)
        lower[2 * donor_count + 1 + moment] = Rat(1)
        rows.extend((tuple(upper), tuple(lower)))
        rhs.extend((gamma[moment], -gamma[moment]))
    if (donor_post is not None or treated_post is not None):
        if (
            donor_post is None
            or treated_post is None
            or len(donor_post) != donor_count
        ):
            raise ValueError("post-treatment dimensions do not match")
        rows.append(
            tuple(donor_post)
            + tuple(Rat(0) for _ in range(variable_count - donor_count))
        )
        rhs.append(treated_post - effect_cutoff)
    cost = tuple(
        Rat(1) if index == radius_index else Rat(0)
        for index in range(variable_count)
    )
    return cost, tuple(rows), tuple(rhs), radius_index


def minimum_koo_guo_moment_radius(
    pre_target: Vector,
    pre_donors: Matrix,
    donor_post: Optional[Vector] = None,
    treated_post: Optional[Rat] = None,
    effect_cutoff: Rat = Rat(0),
) -> MomentRadiusCertificate:
    """Certify the least moment radius, optionally imposing an exact effect."""

    sigma, gamma = sample_moments(pre_target, pre_donors)
    cost, rows, rhs, radius_index = _minimum_radius_standard_form(
        sigma,
        gamma,
        donor_post,
        treated_post,
        effect_cutoff,
    )
    certificate = propose_and_certify_lp(cost, rows, rhs)
    donor_count = len(gamma)
    return MomentRadiusCertificate(
        certificate.exact.optimizer[radius_index],
        certificate.exact.optimizer[:donor_count],
        certificate,
    )


def _fixed_radius_standard_form(
    sigma: Matrix,
    gamma: Vector,
    radius: Rat,
) -> Tuple[Matrix, Vector]:
    if radius < 0:
        raise ValueError("moment radius must be nonnegative")
    donor_count = len(gamma)
    variable_count = 3 * donor_count
    rows = [
        tuple(Rat(1) if index < donor_count else Rat(0) for index in range(variable_count))
    ]
    rhs = [Rat(1)]
    for moment, sigma_row in enumerate(sigma):
        upper = [Rat(0) for _ in range(variable_count)]
        lower = [Rat(0) for _ in range(variable_count)]
        upper[:donor_count] = sigma_row
        lower[:donor_count] = tuple(-value for value in sigma_row)
        upper[donor_count + moment] = Rat(1)
        lower[2 * donor_count + moment] = Rat(1)
        rows.extend((tuple(upper), tuple(lower)))
        rhs.extend((gamma[moment] + radius, -gamma[moment] + radius))
    return tuple(rows), tuple(rhs)


def koo_guo_moment_envelope(
    pre_target: Vector,
    pre_donors: Matrix,
    treated_post: Rat,
    donor_post: Vector,
    radius: Rat,
) -> KooGuoMomentEnvelope:
    """Certify equation-13 endpoints over the empirical moment class at ``radius``."""

    sigma, gamma = sample_moments(pre_target, pre_donors)
    donor_count = len(gamma)
    if len(donor_post) != donor_count:
        raise ValueError("post-treatment dimensions do not match")
    rows, rhs = _fixed_radius_standard_form(sigma, gamma, radius)
    lower_cost = tuple(-value for value in donor_post) + tuple(
        Rat(0) for _ in range(2 * donor_count)
    )
    upper_cost = tuple(donor_post) + tuple(
        Rat(0) for _ in range(2 * donor_count)
    )
    lower = propose_and_certify_lp(lower_cost, rows, rhs)
    upper = propose_and_certify_lp(upper_cost, rows, rhs)
    lower_weights = lower.exact.optimizer[:donor_count]
    upper_weights = upper.exact.optimizer[:donor_count]
    lower_effect = treated_post - _dot(donor_post, lower_weights)
    upper_effect = treated_post - _dot(donor_post, upper_weights)
    if lower_effect > upper_effect:
        raise AssertionError("moment-envelope endpoint ordering failed")
    return KooGuoMomentEnvelope(
        radius,
        lower_effect,
        upper_effect,
        lower_weights,
        upper_weights,
        lower,
        upper,
    )
