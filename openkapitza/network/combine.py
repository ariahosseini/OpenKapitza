"""Assembling the resistance network.

THE CENTRAL CORRECTION IN THIS PACKAGE LIVES HERE.

h_c from CMY and h_K from MD are not conductances of the same thing:

    h_c  is per unit APPARENT area, and already contains the spreading
         (constriction) resistance of the discrete contact spots.
    h_K  is per unit REAL contact area -- the conductance of an atomically
         flat patch where the two solids actually touch.

Adding 1/h_c + 1/h_K treats both as apparent-area quantities and understates
the Kapitza contribution by the real contact area fraction xi = A_r/A_a. For
plastic contact xi ~ P/H_c, which is typically 1e-3 to 1e-2 -- so the naive
sum is wrong by two to three orders of magnitude in that term.

The correct network, all on an apparent-area basis:

    R_solid = 1/h_c + 1/(xi * h_K)          series: spots, then the interface
    h_joint = 1/R_solid + h_g               parallel: solid path and gas path

The (1-xi) area weighting on the gas path is omitted: with xi <= 1e-2 it is a
sub-percent correction, far below the uncertainty on the accommodation
coefficients that set h_g.

A second consequence of the same scaling: a run of MD at the apparent
pressure P would be modelling the wrong state. Inside a plastically deforming
contact spot the local normal stress is of order H_c -- GPa, not the MPa the
user typed in. The pressure to apply in NEMD is reported here as
`spot_pressure` for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contact.cmy import ContactResult
from ..gap.rarefaction import GapResult
from ..md.dmm import KapitzaEstimate
from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Confidence, Quantity, make_derived

# If sweeping h_K over its plausible band moves the joint conductance by less
# than this, running MD cannot change the answer meaningfully.
NEGLIGIBLE_SENSITIVITY = 0.05
# If the Kapitza term carries more than this share of the total resistance,
# the placeholder is load-bearing and MD is the critical path.
DOMINANT_FRACTION = 0.50


@dataclass
class Contribution:
    label: str
    resistance: float       # m^2 K/W, apparent-area basis
    fraction: float         # share of total series+parallel resistance
    note: str = ""


@dataclass
class NetworkResult:
    h_joint: Quantity           # W/(m^2 K), total joint conductance
    R_total: Quantity           # m^2 K/W
    h_solid: Quantity           # solid path only
    R_constriction: Quantity    # 1/h_c
    R_kapitza: Quantity         # 1/(xi * h_K), apparent-area basis
    R_kapitza_raw: Quantity     # 1/h_K, real-area basis, for reference
    xi: Quantity
    h_gap: Quantity
    spot_pressure: Quantity     # pressure to apply in NEMD, Pa
    branch: str
    contributions: list[Contribution] = field(default_factory=list)
    sensitivity: dict = field(default_factory=dict)
    R_macro: Quantity | None = None   # out-of-flatness constriction, apparent-area basis


def combine(
    contact: ContactResult,
    kapitza: KapitzaEstimate,
    gap: GapResult | None,
    H_c: Quantity,
    flags: FlagSet,
    *,
    gap_conductance_override: Quantity | None = None,
    R_macro: Quantity | None = None,
) -> NetworkResult:
    """Combine the contact, interfacial and gas paths into a joint conductance.

    R_macro (apparent-area basis, contact.macro) adds the out-of-flatness
    constriction in series with the asperity and Kapitza terms.
    """
    h_c = contact.h_c.si
    xi = contact.xi.si
    h_K = kapitza.h_K.si

    if xi <= 0:
        raise ValueError("real contact area fraction must be positive")

    R_constriction = 1.0 / h_c
    R_kapitza = 1.0 / (xi * h_K)            # scaled to apparent area
    R_kapitza_raw = 1.0 / h_K               # real-area basis, reference only
    R_mac = R_macro.si if R_macro is not None else 0.0
    R_solid = R_constriction + R_kapitza + R_mac
    h_solid = 1.0 / R_solid

    if gap_conductance_override is not None:
        h_g_q = gap_conductance_override
    elif gap is not None:
        h_g_q = gap.h_g
    else:
        raise ValueError("pass a GapResult or an explicit gap conductance")
    h_g = h_g_q.si

    h_joint = h_solid + h_g
    R_total = 1.0 / h_joint

    # ---- contribution breakdown -------------------------------------------
    # Series resistances are reported as their share of the solid path, then
    # the solid and gas paths as their share of the parallel combination.
    contributions = [
        Contribution(
            "constriction (contact spots)",
            R_constriction,
            R_constriction / R_solid,
            "CMY/Mikic, per apparent area",
        ),
        Contribution(
            "Kapitza (interface, area-scaled)",
            R_kapitza,
            R_kapitza / R_solid,
            f"1/(xi*h_K) with xi = {xi:.3e}",
        ),
    ]

    if R_macro is not None:
        contributions.append(Contribution(
            "macro constriction (flatness)", R_mac, R_mac / R_solid,
            "Bahrami 2004, crowned flat"))
    inputs = [contact.h_c, contact.xi, kapitza.h_K, h_g_q] + ([R_macro] if R_macro is not None else [])

    # ---- is MD worth running? ---------------------------------------------
    sens = _sensitivity(h_c, xi, h_K, h_g, kapitza, R_mac)
    if sens["relative_span"] < NEGLIGIBLE_SENSITIVITY:
        flags.add(
            Code.KAPITZA_NEGLIGIBLE,
            Severity.INFO,
            f"Sweeping h_K across its plausible band changes the joint "
            f"conductance by only {sens['relative_span'] * 100:.1f} %. The "
            f"interface term is not what limits this joint.",
            remedy=(
                "An NEMD campaign would not meaningfully change this answer. "
                "Effort is better spent on the roughness, slope and hardness "
                "inputs, which dominate here. Run MD only if h_K is of "
                "interest in its own right."
            ),
            relative_span=sens["relative_span"],
        )
    elif R_kapitza / R_solid > DOMINANT_FRACTION:
        flags.add(
            Code.KAPITZA_DOMINANT,
            Severity.WARNING,
            f"The area-scaled Kapitza term carries "
            f"{R_kapitza / R_solid * 100:.0f} % of the solid-path resistance, "
            f"and it currently rests on a placeholder estimate.",
            remedy=(
                "This joint is interface-limited, so the MD calculation is the "
                "critical path and the quoted total should not be trusted "
                "until it is done."
            ),
            kapitza_fraction=R_kapitza / R_solid,
        )

    contributions.append(
        Contribution("solid path (series total)", R_solid, R_total / R_solid,
                     "parallel share of total conductance")
    )
    if h_g > 0:
        contributions.append(
            Contribution("gas gap (parallel)", 1.0 / h_g, R_total * h_g,
                         "share of total conductance")
        )

    return NetworkResult(
        h_joint=make_derived(h_joint, unit="W/m2K", inputs=inputs, name="h_joint"),
        R_total=make_derived(R_total, unit="m2K/W", inputs=inputs, name="R_total"),
        h_solid=make_derived(h_solid, unit="W/m2K", inputs=inputs[:3], name="h_solid"),
        R_constriction=make_derived(
            R_constriction, unit="m2K/W", inputs=[contact.h_c], name="R_constriction"
        ),
        R_kapitza=make_derived(
            R_kapitza, unit="m2K/W", inputs=[contact.xi, kapitza.h_K],
            name="R_kapitza", note="apparent-area basis",
        ),
        R_kapitza_raw=make_derived(
            R_kapitza_raw, unit="m2K/W", inputs=[kapitza.h_K],
            name="R_kapitza_raw", note="real-contact-area basis",
        ),
        xi=contact.xi,
        h_gap=h_g_q,
        spot_pressure=make_derived(
            H_c.si, unit="GPa", inputs=[H_c], confidence=Confidence.LOW,
            name="spot_pressure",
            note="local stress inside a plastic contact spot; the pressure to "
                 "apply in NEMD, NOT the apparent contact pressure",
        ),
        branch=contact.branch,
        contributions=contributions,
        sensitivity=sens,
        R_macro=R_macro,
    )


def _sensitivity(
    h_c: float, xi: float, h_K: float, h_g: float, kapitza: KapitzaEstimate,
    R_mac: float = 0.0,
) -> dict:
    """How much could the joint conductance move if h_K is wrong?

    Sweeps h_K over its stated uncertainty band (and to the perfect-interface
    limit) and reports the resulting spread in h_joint. This is what decides
    whether an MD campaign is worth its node-hours.
    """
    unc = kapitza.h_K.uncertainty
    factor = 1.0 + (unc / h_K if unc else 1.5)

    def joint(hk: float) -> float:
        return 1.0 / (1.0 / h_c + 1.0 / (xi * hk) + R_mac) + h_g

    low = joint(h_K / factor)
    high = joint(h_K * factor)
    perfect = 1.0 / (1.0 / h_c + R_mac) + h_g      # h_K -> infinity
    nominal = joint(h_K)

    return {
        "h_joint_nominal": nominal,
        "h_joint_low": low,
        "h_joint_high": high,
        "h_joint_perfect_interface": perfect,
        "factor_swept": factor,
        "relative_span": (high - low) / nominal if nominal > 0 else 0.0,
        "max_possible_gain": (perfect - nominal) / nominal if nominal > 0 else 0.0,
    }
