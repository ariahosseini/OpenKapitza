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

# Fitted range of Yovanovich's 1982 recorrelation of the CMY plastic model,
# as published alongside the 1.25 prefactor and 0.95 exponent used below.
# Outside it the correlation is extrapolation, not interpolation.
CMY_PRESSURE_MIN = 1e-6
CMY_PRESSURE_MAX = 2.3e-2

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
    model: str = "cmy"                 # "cmy" | "truncated_gaussian" | "mikic"


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
    z_trunc: Quantity | None = None,
) -> ContactResult:
    """Plastic CMY branch; truncated Gaussian (Eq. 10) when z_trunc is given."""
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

    xi_raw = x                       # plastic: A_r/A_a = P/H_c (a force balance:
    xi_val = _check_area_fraction(xi_raw, flags, "plastic")   # holds for TG too)

    inputs = [k_s, sigma_eff, m_eff, p_over_hc]
    if z_trunc is not None:
        from .truncated import check_truncation, tg_quantity
        check_truncation(z_trunc.si, x, flags)
        h_q = tg_quantity(k_s, sigma_eff, m_eff, p_over_hc, z_trunc, cap)
        model = "truncated_gaussian"
    else:
        h = _CMY_PREFACTOR * k_s.si * (m_eff.si / sigma_eff.si) * (x ** _CMY_EXPONENT)
        h_q = make_derived(h, unit="W/m2K", inputs=inputs, confidence=cap, name="h_c")
        model = "cmy"
    return ContactResult(
        branch="plastic",
        model=model,
        h_c=h_q,
        xi=make_derived(xi_val, unit="-", inputs=inputs, confidence=cap, name="xi"),
        sigma_eff=sigma_eff,
        m_eff=m_eff,
        k_s=k_s,
        dimensionless_pressure=p_over_hc,
    )


def contact_conductance_unloading(
    k_s: Quantity,
    sigma_eff: Quantity,
    m_eff: Quantity,
    pressure: Quantity,
    max_pressure: Quantity,
    H_max: Quantity,
    flags: FlagSet,
    z_trunc: Quantity | None = None,
):
    """Plastic branch on unloading from max_pressure (contact.unloading).

    Returns (ContactResult, UnloadingResult). Reloading up to max_pressure
    follows the same curve, so this covers relaxed and re-tightened joints too.
    """
    from .unloading import unloading_conductance
    z = z_trunc.si if z_trunc is not None else None
    r = unloading_conductance(k_s.si, sigma_eff.si, m_eff.si, pressure.si, max_pressure.si,
                              H_max.si, z)
    inputs = [k_s, sigma_eff, m_eff, pressure, max_pressure, H_max] + ([z_trunc] if z_trunc else [])
    cap = None
    if z is not None and pressure.si < 4e5:
        flags.add(Code.UNLOADING, Severity.CAUTION,
                  f"Truncated-Gaussian unloading at {pressure.si / 1e3:.0f} kPa: Milanez et al. "
                  "found it accurate above ~300-400 kPa; below, the fully Gaussian unloading "
                  "model predicted their data better.",
                  remedy="Compare with z_trunc unset (fully Gaussian unloading); the truth for "
                         "their surfaces lay nearer that one.")
    if r.at_edge:
        cap = Confidence.LOW
        flags.add(Code.UNLOADING, Severity.WARNING,
                  "Unloaded beyond what the truncated-Gaussian model can describe (the "
                  "truncated tail has fully opened).",
                  remedy="Use the fully Gaussian unloading model (z_trunc unset) at this depth "
                         "of unloading.")
    xi = _check_area_fraction(r.area_fraction, flags, "plastic (unloading)")
    p_over_h = make_derived(pressure.si / H_max.si, unit="-", inputs=[pressure, H_max],
                            name="P_over_Hmax")
    res = ContactResult(
        branch="plastic",
        model="unloading" + ("_truncated" if z is not None else ""),
        h_c=make_derived(r.h_c, unit="W/m2K", inputs=inputs, confidence=cap, name="h_c",
                         note="Mikic unloading (Milanez et al. 2004, Eqs. 16-19)"),
        xi=make_derived(xi, unit="-", inputs=inputs, confidence=cap, name="xi"),
        sigma_eff=sigma_eff, m_eff=m_eff, k_s=k_s, dimensionless_pressure=p_over_h,
    )
    return res, r


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
        model="mikic",
    )


# Light-load truncation (open issue P0-7, knowledge base 8.1). The Gaussian
# model lets the tallest asperities make the first contacts; real surfaces
# lack that tail (fitted truncation 3.5-4.2 sigma: Milanez, Yovanovich &
# Mantelli, AIAA 2003), so at light load the real separation is smaller and
# the conductance higher than CMY predicts. Thresholds are in Y/sigma from the
# Yovanovich separation correlation, set from this package's own benchmark
# (207 Ni 200 / SS 304 vacuum points, measured/predicted geometric mean):
#     Y/sigma < 3.2: 0.90-0.93    3.2-3.4: 1.05    > 3.4: 1.23
# Milanez, Culham & Yovanovich (AIAA 2002-0787) report 40-70 % under-prediction
# at their lightest loads (16 kPa, bead-blasted SS 304).
LIGHT_LOAD_CAUTION_Y = 3.2
LIGHT_LOAD_WARNING_Y = 3.5


def light_load_check(p_over_hc: Quantity, flags: FlagSet) -> float:
    """Flag CMY as a likely UNDER-estimate at light load; returns Y/sigma."""
    y_over_sigma = 1.53 * max(p_over_hc.si, 1e-12) ** -0.097
    if y_over_sigma <= LIGHT_LOAD_CAUTION_Y:
        return y_over_sigma
    severe = y_over_sigma > LIGHT_LOAD_WARNING_Y
    flags.add(
        Code.CMY_LIGHT_LOAD,
        Severity.WARNING if severe else Severity.CAUTION,
        f"Light load: Y/sigma = {y_over_sigma:.2f} (P/H_c = {p_over_hc.si:.2g}), so the "
        f"first contacts come from the Gaussian tail, which real surfaces lack. CMY "
        f"under-predicts here: measured/predicted 1.05 at Y/sigma 3.2-3.4 and 1.23 "
        f"above 3.4 in this package's benchmark; 40-70 % at the lightest loads "
        f"in Milanez, Culham & Yovanovich (2002).",
        remedy=("Treat h_c as a LOWER estimate, or pass z_trunc to use the truncated-"
                "Gaussian model (Milanez, Yovanovich & Mantelli 2004: RMS error "
                "21-48 % -> 5-18 %). z_trunc is fitted to light-load thermal data "
                "(profilometer R_p/sigma is not reliable); z_trunc=\"reference\" uses "
                "their bead-blasted SS 304 / Ni 200 values. On this package's benchmark "
                "it cuts the >3.4 bias from 1.23 to 1.14. A measured h_c is better still. "
                "Heat-sink mounting pressures (0.07-0.17 MPa) usually land here."),
        y_over_sigma=y_over_sigma,
    )
    return y_over_sigma


def mean_plane_separation(
    sigma_eff: Quantity, p_over_hc: Quantity, flags: FlagSet,
    z_trunc: Quantity | None = None,
) -> Quantity:
    """Mean separation of the two mean planes, Y, for the gap model.

        Y = 1.53 * sigma * (P/H_c)^(-0.097)

    This is the Yovanovich correlation. The exponent is small, so Y stays
    within a factor of about two of 1.5*sigma across the whole useful pressure
    range -- which is a useful sanity check on the output.

    With z_trunc, Y is scaled by the exact truncated/Gaussian separation ratio
    (Milanez et al. Eq. 9 over Eq. 4), so z_trunc -> infinity returns the
    correlation unchanged. Carrying the TG separation into the GAP model is
    this package's extension: the paper is vacuum-only. It is the consistent
    choice -- the same truncation that raises h_c closes the gap -- and it
    raises h_g.
    """
    x = max(p_over_hc.si, 1e-12)
    y = 1.53 * sigma_eff.si * (x ** -0.097)
    inputs = [sigma_eff, p_over_hc]
    note = "Yovanovich mean-plane-separation correlation"
    if z_trunc is not None and x < 0.5:
        from .truncated import tg_separation
        y *= tg_separation(x, z_trunc.si) / tg_separation(x, None)
        inputs.append(z_trunc)
        note += ", scaled by the truncated-Gaussian separation ratio (Eq. 9 / Eq. 4)"
    return make_derived(y, unit="um", inputs=inputs, name="Y", note=note)


def regime_branches(regime: Regime) -> list[AreaModel]:
    if regime is Regime.PLASTIC:
        return [AreaModel.PLASTIC]
    if regime is Regime.ELASTIC:
        return [AreaModel.ELASTIC]
    return [AreaModel.PLASTIC, AreaModel.ELASTIC]
