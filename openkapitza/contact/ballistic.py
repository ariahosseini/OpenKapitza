"""Ballistic contact spots (open issue P0-11; KB 23).

CMY treats every contact spot as a diffusive constriction: each half-space
contributes 1/(4 k a), with a the spot radius. When a is comparable to the
mean free path of the heat carriers (phonons in a semiconductor, electrons in
a metal), heat crosses the spot partly ballistically, and the constriction
resistance is higher (Prasher, Nano Lett. 5, 2155, 2005).

This module estimates how much higher, side by side, and reports it as a flag
with a corrected band. It does not change the primary answer.

Estimate (ours, not Prasher's formula). The resistance of a ballistic
(Sharvin-type) orifice for a gray carrier gas, 4/(C v pi a^2) per side pair,
is added to the diffusive one. With the kinetic mean free path
lambda = 3k/(Cv), this gives, per side,

    R_side / R_side,diffusive = 1 + (8 / 3 pi) * lambda / a .

A spectrum of mean free paths is treated as groups in parallel. A fraction
f_i of k with path lambda_i gives

    F = 1 / sum_i [ f_i / (1 + (8/3pi) lambda_i / a) ] .

Two bounds are reported:
- **low:** one group at the gray kinetic path. For phonons this is computed
  from k, the Debye heat capacity and the mean sound speed; for metals it is
  the tabulated electron path.
- **high:** the material's long-path fraction (`mfp_long_fraction` at
  `mfp_long_nm`, e.g. Si: 40 % of k from MFP > 1 um, Regner et al. 2013),
  with the rest at the gray path.

The truth needs a spectral calculation; the band brackets it for the
materials whose long-path fraction is known.

Mean spot radius, from the CMY/Yovanovich model at the same P/H_c:

    lambda_s = sqrt(2) * erfcinv(2 P/H_c),
    a = (sigma/m) * sqrt(8/pi) * exp(lambda_s^2 / 2) * erfc(lambda_s / sqrt 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..schema.flags import Code, FlagSet, Severity

KB = 1.380649e-23
C_BALLISTIC = 8.0 / (3.0 * math.pi)
REPORT_ABOVE = 0.10          # CAUTION when the high bound exceeds +10 %
NOTE_ABOVE = 0.02            # INFO between +2 % and +10 %


def mean_spot_radius(sigma_m: float, m: float, p_over_hc: float) -> float:
    """Mean contact-spot radius (m) of the CMY/Yovanovich model."""
    from .hardness import _erfcinv
    x = min(max(p_over_hc, 1e-12), 0.49)
    lam = math.sqrt(2.0) * _erfcinv(2.0 * x)
    return (sigma_m / m) * math.sqrt(8.0 / math.pi) * math.exp(lam * lam / 2.0) \
        * math.erfc(lam / math.sqrt(2.0))


def debye_heat_capacity(n: float, theta: float, T: float) -> float:
    """Volumetric Debye heat capacity, J/(m^3 K), for number density n (1/m^3)."""
    x_d = theta / T
    steps = 400
    h = x_d / steps
    s = 0.0
    for i in range(steps + 1):
        x = i * h
        f = 0.0 if x == 0 else x ** 4 * math.exp(x) / (math.exp(x) - 1.0) ** 2
        s += f * (0.5 if i in (0, steps) else 1.0)
    s *= h
    return 9.0 * n * KB * (T / theta) ** 3 * s


@dataclass
class Carrier:
    kind: str                     # "phonon" | "electron"
    mfp_m: float                  # gray (kinetic) path
    long_fraction: float = 0.0    # share of k carried at mfp_long_m
    mfp_long_m: float = 0.0
    source: str = ""


def carrier_of(mat, T: float) -> Carrier | None:
    """The heat carriers' mean free path for a material, or None if unknown."""
    extra = getattr(mat, "ballistic", None) or {}
    if extra.get("carrier") == "electron" and extra.get("mfp_nm"):
        return Carrier("electron", extra["mfp_nm"] * 1e-9, source=extra.get("source", ""))
    k = mat.k.si
    if not all(q is not None for q in (mat.v_long, mat.v_trans, mat.theta_debye,
                                       mat.number_density)):
        return None
    v = (mat.v_long.si + 2.0 * mat.v_trans.si) / 3.0
    C = debye_heat_capacity(mat.number_density.si, mat.theta_debye.si, T)
    lam = 3.0 * k / (C * v)
    return Carrier("phonon", lam, extra.get("mfp_long_fraction", 0.0) or 0.0,
                   (extra.get("mfp_long_nm") or 0.0) * 1e-9,
                   source=extra.get("source", "gray kinetic path 3k/(Cv), Debye C"))


def side_factor(c: Carrier, a: float, *, high: bool) -> float:
    """R/R_diffusive for one side."""
    if high and c.long_fraction > 0:
        groups = [(1.0 - c.long_fraction, c.mfp_m), (c.long_fraction, c.mfp_long_m)]
    else:
        groups = [(1.0, c.mfp_m)]
    return 1.0 / sum(f / (1.0 + C_BALLISTIC * lam / a) for f, lam in groups)


def ballistic_check(mat_a, mat_b, sigma_m: float, m: float, p_over_hc: float,
                    T: float, h_c: float, flags: FlagSet) -> dict | None:
    """Estimate the ballistic penalty on the contact-spot conductance and flag it.

    Returns {a_m, F_low, F_high, h_c_low, h_c_high, sides}, or None when a
    side's carrier path is unknown.
    """
    a = mean_spot_radius(sigma_m, m, p_over_hc)
    ca, cb = carrier_of(mat_a, T), carrier_of(mat_b, T)
    if ca is None or cb is None:
        return None
    wa, wb = 1.0 / mat_a.k.si, 1.0 / mat_b.k.si   # each side's share of 1/(2 k_s a)
    out = {"a_m": a, "sides": {}}
    for tag, high in (("low", False), ("high", True)):
        fa, fb = side_factor(ca, a, high=high), side_factor(cb, a, high=high)
        out[f"F_{tag}"] = (fa * wa + fb * wb) / (wa + wb)
        out["sides"][tag] = (fa, fb)
    out["h_c_low"] = h_c / out["F_high"]
    out["h_c_high"] = h_c / out["F_low"]
    pen_hi = out["F_high"] - 1.0
    kn = {mat_a.key: ca.mfp_m / a, mat_b.key: cb.mfp_m / a}
    detail = (f"Mean contact-spot radius a = {a * 1e9:.0f} nm; carrier paths "
              f"{mat_a.key}: {ca.kind} {ca.mfp_m * 1e9:.0f} nm"
              + (f" (+{ca.long_fraction:.0%} of k at {ca.mfp_long_m * 1e9:.0f} nm)"
                 if ca.long_fraction else "")
              + f", {mat_b.key}: {cb.kind} {cb.mfp_m * 1e9:.0f} nm"
              + (f" (+{cb.long_fraction:.0%} of k at {cb.mfp_long_m * 1e9:.0f} nm)"
                 if cb.long_fraction else "") + ".")
    if pen_hi > REPORT_ABOVE:
        flags.add(
            Code.CONTACT_BALLISTIC, Severity.CAUTION,
            f"Contact spots are small enough for partly ballistic transport. {detail} "
            f"The diffusive CMY spot resistance is then low by "
            f"{(out['F_low'] - 1) * 100:.0f}-{pen_hi * 100:.0f} %, so h_c is "
            f"{out['h_c_low']:.3g}-{out['h_c_high']:.3g} W/m^2K rather than {h_c:.3g}.",
            remedy=("An estimate (gray Sharvin + diffusive; KB 23), not a spectral "
                    "calculation. It matters for polished semiconductor contacts. A "
                    "rougher finish (larger sigma/m) or a measured joint conductance "
                    "removes the question."),
            spot_radius_nm=a * 1e9, F_low=out["F_low"], F_high=out["F_high"],
            knudsen=kn)
    elif pen_hi > NOTE_ABOVE:
        flags.add(Code.CONTACT_BALLISTIC, Severity.INFO,
                  f"Ballistic correction to the spot resistance is small "
                  f"(+{(out['F_low'] - 1) * 100:.1f} to +{pen_hi * 100:.1f} %). {detail}",
                  spot_radius_nm=a * 1e9, F_low=out["F_low"], F_high=out["F_high"])
    return out
