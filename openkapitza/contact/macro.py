"""Out-of-flatness (macro-contact) resistance and the clamping check.

CMY assumes conforming flats: the whole apparent area is in contact and only
the asperities limit heat flow. Real flats are crowned or warped. If the
out-of-flatness is large compared with the roughness, the surfaces touch only
over a central macro-contact of radius a_L < b_L, and heat must also squeeze
into that patch -- a macro constriction in series with the asperity one.

Source: M. Bahrami, "Modeling of Thermal Joint Resistance for Sphere-Flat
Contacts in a Vacuum", PhD thesis, University of Waterloo, 2004. Equation
numbers are the thesis's.

    rho  = b_L^2 / (2 delta)                       (4.2, after Clausing & Chao)
    a_H  = (3 F rho / 4 E')^(1/3),  tau = rho / a_H,  alpha = sigma rho / a_H^2   (3.27)
    a_L  = 1.80 a_H sqrt(alpha + 0.31 tau^0.056) / tau^0.028                     (3.30)
    F_c  = 4 E' / (3 rho) * max(0, b_L^2 - 2.25 sigma rho)^1.5   (a_L = b_L above) (4.23)
    R_L  = (1 - a_L/b_L)^1.5 / (2 k_s a_L)                                         (4.6)
    R_j  = R_s (micro, independent of curvature) + R_L                            (4.30)

delta is the COMBINED maximum out-of-flatness (1/rho = 1/rho_1 + 1/rho_2, so
the two surfaces' deltas add); b_L the joint radius (sqrt(A/pi) for a
non-circular footprint); F = P * pi b_L^2. Fitted for 0.005 <= alpha <= 100,
50 <= tau <= 80 000. The thesis calls a joint flat when B = a_L/b_L >= 0.8
(R_L* = k_s b_L R_L <= 0.05), roughly 3 <= delta/sigma <= 30 from very rough
to very smooth surfaces.

Checks (tests/test_macro.py): reproduces the thesis's alpha, tau, B and R_L
columns for 205 non-conforming points (Tables A.3-A.38) within print rounding;
this package's CMY micro term plus R_L predicts the measured R at geometric
mean 1.03, RMS 15.3 % (thesis: 14.8 % for its approximate model); CMY alone,
ignoring flatness, predicts R at 0.14x measured on average, down to 0.008x
-- i.e. it OVER-predicts h by ~7x typically and up to ~125x.

Assumptions carried over: spherical crown, centred load, vacuum (the gas in
the open macro gap is not modelled), elastic bulk. A bolted joint is none of
those: pressure concentrates around the fasteners (Yeh et al. 2001: h rises
as P_mean^2.8-3.8 and 8 bolts beat 4 at the same mean pressure), so for
`clamping="bolted"` the result is an indication, flagged as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Confidence, Quantity, make_derived

ALPHA_RANGE = (0.005, 100.0)
TAU_RANGE = (50.0, 80_000.0)
FLAT_B = 0.8                    # thesis criterion for a "flat" (conforming) joint
CLAMPING = ("uniform", "bolted", "unknown")


@dataclass
class MacroResult:
    rho: float                  # equivalent radius of curvature, m
    a_H: float                  # Hertz radius, m
    alpha: float
    tau: float
    F: float                    # total load, N
    F_c: float                  # load above which the whole face touches, N
    a_L: float                  # macro-contact radius, m
    B: float                    # a_L / b_L
    R_L: float                  # macro constriction, K/W (whole joint)
    R_L_area: Quantity          # R_L * pi b_L^2, m^2 K/W (apparent-area basis)
    conforming: bool            # B >= FLAT_B


def equivalent_radius(b_L: float, delta: float) -> float:
    """Eq. 4.2: rho = b_L^2 / (2 delta)."""
    return b_L * b_L / (2.0 * delta)


def macro_contact(F: float, E_eff: float, sigma: float, delta: float, b_L: float,
                  k_s: float) -> tuple[float, ...]:
    """(rho, a_H, alpha, tau, F_c, a_L, B, R_L), SI throughout."""
    rho = equivalent_radius(b_L, delta)
    a_H = (0.75 * F * rho / E_eff) ** (1.0 / 3.0)
    tau = rho / a_H
    alpha = sigma * rho / (a_H * a_H)
    F_c = 4.0 * E_eff / (3.0 * rho) * max(0.0, b_L * b_L - 2.25 * sigma * rho) ** 1.5
    if F >= F_c:
        a_L = b_L
    else:
        a_L = min(b_L, 1.80 * a_H * math.sqrt(alpha + 0.31 * tau ** 0.056) / tau ** 0.028)
    B = a_L / b_L
    R_L = (1.0 - B) ** 1.5 / (2.0 * k_s * a_L)
    return rho, a_H, alpha, tau, F_c, a_L, B, R_L


def macro_resistance(pressure: Quantity, E_eff: Quantity, sigma_eff: Quantity,
                     flatness: Quantity, joint_radius: Quantity, k_s: Quantity,
                     flags: FlagSet) -> MacroResult:
    """Macro constriction for a crowned joint, with range and conformity flags."""
    b_L = joint_radius.si
    F = pressure.si * math.pi * b_L * b_L
    rho, a_H, alpha, tau, F_c, a_L, B, R_L = macro_contact(
        F, E_eff.si, sigma_eff.si, flatness.si, b_L, k_s.si)
    out_of_range = not (ALPHA_RANGE[0] <= alpha <= ALPHA_RANGE[1]
                        and TAU_RANGE[0] <= tau <= TAU_RANGE[1]) and F < F_c
    if out_of_range:
        flags.add(Code.MACRO_RANGE, Severity.CAUTION,
                  f"Macro-contact correlation used outside its fitted range "
                  f"(alpha = {alpha:.3g}, tau = {tau:.3g}; fitted 0.005-100, 50-80 000).",
                  remedy="tau < 50 means the crown is so tight that the half-space "
                         "assumption weakens; alpha > 100 means roughness dominates the "
                         "crown entirely (then R_L is small anyway).",
                  alpha=alpha, tau=tau)
    cap = Confidence.LOW if out_of_range else None
    area = math.pi * b_L * b_L
    R_area = make_derived(R_L * area, unit="m2K/W",
                          inputs=[pressure, E_eff, sigma_eff, flatness, joint_radius, k_s],
                          confidence=cap, name="R_macro",
                          note="Bahrami 2004 Eqs. 3.30, 4.6, 4.23 (crowned flat, vacuum)")
    return MacroResult(rho, a_H, alpha, tau, F, F_c, a_L, B, R_L, R_area, B >= FLAT_B)


def macro_flags(result: MacroResult | None, R_rest: float, clamping: str | None,
                flatness_given: bool, gas: bool, flags: FlagSet) -> None:
    """Explain the macro term, or its absence, in the joint's own terms.

    R_rest is the rest of the solid path on the apparent-area basis (m^2K/W).
    """
    clamping = clamping or "unknown"
    if clamping not in CLAMPING:
        raise ValueError(f"clamping must be one of {CLAMPING}, got {clamping!r}")

    if clamping == "bolted":
        flags.add(Code.CLAMP_BOLTED, Severity.WARNING,
                  "Bolted/screwed joint: the apparent pressure is an average over a face "
                  "whose real contact concentrates around the fasteners.",
                  remedy=("CMY at the mean pressure over-predicted measured bolted Al 6061 "
                          "joints 5-35x (Yeh et al. 2001: h ~ P_mean^2.8-3.8, 8 bolts beat 4 "
                          "at the same mean pressure). "
                          + ("The flatness correction below assumes a centred load on a "
                             "crowned face, so treat it as an indication. "
                             if flatness_given else
                             "Give the measured out-of-flatness and joint size to get at "
                             "least the crown correction. ")
                          + "A measured h for the actual assembly is the reliable number."))

    if result is None:
        if flatness_given:
            return
        sev = Severity.CAUTION if clamping == "bolted" else Severity.INFO
        flags.add(Code.FLATNESS_ASSUMED, sev,
                  "No out-of-flatness given: the joint is treated as conforming flats "
                  "(whole face in contact, only asperities limit heat flow).",
                  remedy="That holds only if the combined out-of-flatness is within "
                         "~3-30x the RMS roughness (Bahrami 2004). On 205 measured crowned "
                         "joints, ignoring flatness over-predicted h ~7x typically and up to "
                         "~125x. Machined plates, heat-sink bases and lids are usually crowned "
                         "or warped by microns to tens of microns: give flatness_um and the "
                         "joint size if you have them.")
        return

    share = result.R_L_area.si / (result.R_L_area.si + R_rest)
    if result.conforming:
        flags.add(Code.MACRO_CONSTRICTION, Severity.INFO,
                  f"Joint is effectively flat: macro-contact covers {result.B:.0%} of the face "
                  f"(>= 80 %, Bahrami's criterion); macro resistance is {share:.0%} of the "
                  f"solid path.",
                  B=result.B, share=share)
    else:
        sev = Severity.WARNING if share > 0.5 else Severity.CAUTION
        flags.add(Code.MACRO_CONSTRICTION, sev,
                  f"Out-of-flatness limits the contact to the central {result.B:.0%} of the "
                  f"joint radius; the macro constriction is {share:.0%} of the solid-path "
                  f"resistance.",
                  remedy=("Raising the load, lapping the faces flatter, or a TIM that fills "
                          "the macro gap all help more than smoother asperities would."
                          + (" In gas, the open macro gap's own gas conduction is not "
                             "modelled: h_g (computed for the asperity gap) is over-estimated "
                             "over that area." if gas else "")),
                  B=result.B, share=share)
