"""Contact-spot conductance: Cooper-Mikic-Yovanovich and Mikic elastic.

Both correlations give a conductance per unit of APPARENT area that already
contains the constriction resistance of the discrete contact spots. They say
nothing about what happens across the atomically-thin interface inside a
spot; that is the Kapitza term, and combining the two correctly is the job of
the network module.

Plastic (Cooper-Mikic-Yovanovich):
    h_c = 1.25 * k_s * (m/sigma) * (P/H_c)^0.95

Elastic (Mikic):
    h_c = 1.55 * k_s * (m/sigma) * (sqrt(2) * P / (m * E'))^0.94

with combined surface parameters
    sigma = sqrt(sigma_a^2 + sigma_b^2)
    m     = sqrt(m_a^2 + m_b^2)
and the harmonic mean conductivity
    k_s   = 2 k_a k_b / (k_a + k_b)
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Confidence, Quantity, make_derived
from .regime import Regime

# Fitted range of the plastic CMY correlation. Outside it the correlation is
# extrapolation, not interpolation.
CMY_PRESSURE_MIN = 1e-5
CMY_PRESSURE_MAX = 2e-2

# Correlation constants. Part of the published fits, not tunable parameters.
_CMY_PREFACTOR = 1.25
_CMY_EXPONENT = 0.95
_MIKIC_PREFACTOR = 1.55
_MIKIC_EXPONENT = 0.94


class AreaModel(enum.Enum):
    PLASTIC = "plastic"
    ELASTIC = "elastic"


@dataclass
class ContactResult:
    """One branch of the contact calculation."""

    branch: str                 # "plastic" | "elastic"
    h_c: Quantity               # W/(m^2 K), per apparent area
    xi: Quantity                # real/apparent contact area ratio, -
    sigma_eff: Quantity
    m_eff: Quantity
    k_s: Quantity
    dimensionless_pressure: Quantity   # P/H_c or sqrt(2)P/(mE')


def combined_roughness(sigma_a: Quantity, sigma_b: Quantity) -> Quantity:
    return make_derived(
        math.hypot(sigma_a.si, sigma_b.si),
        unit="um",
        inputs=[sigma_a, sigma_b],
        name="sigma_eff",
    )


def combined_slope(m_a: Quantity, m_b: Quantity) -> Quantity:
    return make_derived(
        math.hypot(m_a.si, m_b.si), unit="-", inputs=[m_a, m_b], name="m_eff"
    )


def harmonic_conductivity(k_a: Quantity, k_b: Quantity) -> Quantity:
    """k_s = 2 k_a k_b / (k_a + k_b)."""
    k_s = 2.0 * k_a.si * k_b.si / (k_a.si + k_b.si)
    return make_derived(k_s, unit="W/mK", inputs=[k_a, k_b], name="k_s")


def _check_area_fraction(xi: float, flags: FlagSet, branch: str) -> float:
    """Clamp xi to a physical range and flag saturation."""
    if xi >= 1.0:
        flags.add(
            Code.CONTACT_SATURATED,
            Severity.CRITICAL,
            f"The {branch} model predicts a real contact area fraction of "
            f"{xi:.3g}, i.e. full contact. The correlation has no meaning here.",
            remedy=(
                "The applied pressure is at or above the level that would "
                "flatten the surfaces entirely. Check the pressure and hardness "
                "inputs; if they are right, the joint is outside the regime any "
                "roughness-based contact model describes."
            ),
            xi=xi,
        )
        return 1.0
    return xi


def contact_conductance_plastic(
    k_s: Quantity,
    sigma_eff: Quantity,
    m_eff: Quantity,
    p_over_hc: Quantity,
    flags: FlagSet,
) -> ContactResult:
    """Plastic CMY branch."""
    x = p_over_hc.si
    cap = None

    if x < CMY_PRESSURE_MIN or x > CMY_PRESSURE_MAX:
        cap = Confidence.LOW
        side = "below" if x < CMY_PRESSURE_MIN else "above"
        flags.add(
            Code.CMY_PRESSURE_RANGE,
            Severity.WARNING,
            f"P/H_c = {x:.3g} is {side} the fitted range of the CMY "
            f"correlation ({CMY_PRESSURE_MIN:g} to {CMY_PRESSURE_MAX:g}).",
            remedy=(
                "The correlation is being extrapolated. At very low P/H_c the "
                "contact is a sparse set of spots where the continuum "
                "constriction picture weakens; at high P/H_c the asperities "
                "interact and the independent-spot assumption fails."
            ),
            p_over_hc=x,
        )

    h = _CMY_PREFACTOR * k_s.si * (m_eff.si / sigma_eff.si) * (x ** _CMY_EXPONENT)
    xi_raw = x                       # plastic: A_r/A_a = P/H_c
    xi_val = _check_area_fraction(xi_raw, flags, "plastic")

    inputs = [k_s, sigma_eff, m_eff, p_over_hc]
    return ContactResult(
        branch="plastic",
        h_c=make_derived(h, unit="W/m2K", inputs=inputs, confidence=cap, name="h_c"),
        xi=make_derived(xi_val, unit="-", inputs=inputs, confidence=cap, name="xi"),
        sigma_eff=sigma_eff,
        m_eff=m_eff,
        k_s=k_s,
        dimensionless_pressure=p_over_hc,
    )


def contact_conductance_elastic(
    k_s: Quantity,
    sigma_eff: Quantity,
    m_eff: Quantity,
    pressure: Quantity,
    E_eff: Quantity,
    flags: FlagSet,
) -> ContactResult:
    """Mikic elastic branch.

    The dimensionless group sqrt(2)P/(m E') plays the role that P/H_c plays in
    the plastic branch, and the real area fraction is taken as that same group.
    That identification is model-dependent -- it is the natural elastic
    analogue rather than an independently derived result -- so the area
    fraction is flagged accordingly, because it is what scales the Kapitza
    resistance downstream.
    """
    group = math.sqrt(2.0) * pressure.si / (m_eff.si * E_eff.si)
    h = _MIKIC_PREFACTOR * k_s.si * (m_eff.si / sigma_eff.si) * (group ** _MIKIC_EXPONENT)
    xi_val = _check_area_fraction(group, flags, "elastic")

    flags.add(
        Code.AREA_FRACTION_MODEL,
        Severity.CAUTION,
        f"Elastic real-area fraction taken as the Mikic group "
        f"sqrt(2)P/(m E') = {group:.3g}.",
        remedy=(
            "This is the natural elastic analogue of P/H_c rather than an "
            "independently derived area. Since this number divides the Kapitza "
            "resistance when the two are combined, an error here scales the "
            "atomistic contribution directly."
        ),
        xi=group,
    )

    inputs = [k_s, sigma_eff, m_eff, pressure, E_eff]
    group_q = make_derived(group, unit="-", inputs=inputs, name="elastic_group")
    return ContactResult(
        branch="elastic",
        h_c=make_derived(h, unit="W/m2K", inputs=inputs, name="h_c"),
        xi=make_derived(xi_val, unit="-", inputs=inputs, confidence=Confidence.LOW, name="xi"),
        sigma_eff=sigma_eff,
        m_eff=m_eff,
        k_s=k_s,
        dimensionless_pressure=group_q,
    )


def mean_plane_separation(
    sigma_eff: Quantity, p_over_hc: Quantity, flags: FlagSet
) -> Quantity:
    """Mean separation of the two mean planes, Y, for the gap model.

        Y = 1.53 * sigma * (P/H_c)^(-0.097)

    This is the Yovanovich correlation. The exponent is small, so Y stays
    within a factor of about two of 1.5*sigma across the whole useful pressure
    range -- which is a useful sanity check on the output.
    """
    x = max(p_over_hc.si, 1e-12)
    y = 1.53 * sigma_eff.si * (x ** -0.097)
    return make_derived(
        y, unit="um", inputs=[sigma_eff, p_over_hc], name="Y",
        note="Yovanovich mean-plane-separation correlation",
    )


def regime_branches(regime: Regime) -> list[AreaModel]:
    if regime is Regime.PLASTIC:
        return [AreaModel.PLASTIC]
    if regime is Regime.ELASTIC:
        return [AreaModel.ELASTIC]
    return [AreaModel.PLASTIC, AreaModel.ELASTIC]
