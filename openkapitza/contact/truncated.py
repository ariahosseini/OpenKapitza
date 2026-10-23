"""Truncated-Gaussian (TG) contact conductance for light loads (open issue P0-7).

CMY assumes the asperity heights are Gaussian all the way out. Real surfaces
are Gaussian only to about 3.5-4.5 sigma; the tallest asperities that would
make the first contacts at light load do not exist. The surfaces therefore sit
closer than CMY says, and the conductance is higher.

Source: F. H. Milanez, M. M. Yovanovich & M. B. H. Mantelli, "Thermal Contact
Conductance at Low Contact Pressures", J. Thermophys. Heat Transfer 18(1),
2004 (AIAA 36th Thermophysics Conf., 2003). Equation numbers below are theirs.

    Eq. 10   h_c = 1.25 k_s (m/sigma) (P/H_c)^0.95 (1 + 1/f)^0.9289 sqrt(1 - 1/(1+f))
    Eq. 11   f   = (P/H_c) sqrt(2 pi) z_t exp(z_t^2 / 2)
    Eq. 14   P/H_c = [ (P/H_c)_G^-p + (P/H_c)_TG^-p ]^(-1/p),   p = 3.9 + 52 exp(10 c2)
    Eq. 12   (P/H_c)_G  = [ P / (c1 (1.62 sigma/m)^c2) ]^(1/(1 + 0.071 c2))      (Song-Yovanovich)
    Eq. 15   (P/H_c)_TG = { P / (c1 (2.178 sigma/m)^c2) * erfc(z_t/sqrt2)^(0.4289 c2) }^(1/(1 + 0.5 c2))

sigma/m in um. z_t -> infinity recovers CMY and Song-Yovanovich exactly.

Checks against the paper (tests/test_truncated.py):
  * Eq. 14 reproduces every H_c range in their Table 1 (six tests, both ends)
    within 2.2 %, using the combined sigma, m printed on their Figs 3-8.
  * Their Fig. 2 statement (S3, 15.8 kPa, z_t 4.1 -> 3.3 gives "approximately
    8 to 20 W/m^2K") comes out 8.4 -> 22.6.
  * They state Eq. 10 is within 4 % of the exact model (Eqs. 1, 2, 5, 8, 9) for
    1e-6 < P/H_c < 1e-2, 3 < z_t < 4.5. Evaluated with exact erfc here, it is
    within 4 % for P/H_c >= 1e-5 but reaches -7 % at P/H_c = 1e-6, z_t = 3.5
    (the correlation is LOW there). Their z_t values were fitted with Eq. 10, so
    Eq. 10 stays the default; `tg_conductance_exact` is the cross-check.
  * "For f > 40 the difference ... is less than 1 %": it is 1.1 % at f = 40.

z_t cannot be measured by profilometry (R_p/sigma scatters 2.5-4.5 at fixed
sigma/m, their Fig. 9); it is fitted to light-load thermal data. Their fitted
values for bead-blasted SS 304 and Ni 200 are in REFERENCE_TRUNCATION.

z_t IS TIED TO THE HARDNESS FIT IT WAS FITTED WITH. Their SS 304 values used
c1 = 10.67 GPa, c2 = -0.37 (Ni 200: 4.33, -0.079). On their S1 data (Bahrami
thesis Table A.63) that pairing gives RMS 20.8 % -> 15.5 % (paper: 25.3 ->
16.1); with c1/c2 = 6.27/-0.23 plain CMY is already 14.7 % and TG with the
same z_t over-corrects to 22.4 % (tests/test_benchmarks_ext.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Confidence, Provenance, Quantity, make_derived

_SQRT2 = math.sqrt(2.0)

# Milanez, Yovanovich & Mantelli, Table 1: bead-blasted surface against a lapped
# one, vacuum, first loading. (sigma/m [um], z_trunc), ordered by sigma/m.
REFERENCE_TRUNCATION: dict[str, tuple[tuple[float, float], ...]] = {
    "SS304": ((17.7, 4.1), (26.2, 3.8), (40.0, 3.5)),
    "Ni200": ((23.0, 4.2), (27.0, 4.2), (48.0, 3.9)),
}
# Their Fig. 2 fits z_t from a family of curves spaced 0.2 apart: the fitted
# value is resolved to about that, so it is the 1-sigma we attach.
REFERENCE_TRUNCATION_UNC = 0.2
# Range of z_t over which their correlation was built and checked.
Z_TRUNC_MIN, Z_TRUNC_MAX = 3.0, 4.5
# Above this f the TG correction is ~1 % (paper: "< 1 %"; it is 1.1 % at 40).
F_NEGLIGIBLE = 40.0


def truncation_function(p_over_hc: float, z_trunc: float) -> float:
    """Eq. 11."""
    return p_over_hc * math.sqrt(2.0 * math.pi) * z_trunc * math.exp(0.5 * z_trunc * z_trunc)


def tg_factor(p_over_hc: float, z_trunc: float) -> float:
    """Eq. 10 divided by Eq. 7: the multiplier on CMY (>= 1)."""
    f = truncation_function(p_over_hc, z_trunc)
    return (1.0 + 1.0 / f) ** 0.9289 * math.sqrt(1.0 - 1.0 / (1.0 + f))


def tg_conductance(k_s: float, sigma: float, m: float, p_over_hc: float, z_trunc: float) -> float:
    """Eq. 10, SI in and out."""
    return 1.25 * k_s * (m / sigma) * p_over_hc ** 0.95 * tg_factor(p_over_hc, z_trunc)


def _erfcinv(y: float) -> float:
    from .hardness import _erfcinv as inv
    return inv(y)


def tg_separation(p_over_hc: float, z_trunc: float | None) -> float:
    """Eq. 9: mean-plane separation lambda = Y/sigma (Eq. 4 when z_t is None)."""
    tail = math.erfc(z_trunc / _SQRT2) if z_trunc is not None else 0.0
    return _SQRT2 * _erfcinv(2.0 * p_over_hc + tail)


def tg_conductance_exact(k_s: float, sigma: float, m: float, p_over_hc: float,
                         z_trunc: float | None) -> float:
    """Eqs. 1, 2, 5, 8, 9 without correlations (fully Gaussian when z_t is None)."""
    lam = tg_separation(p_over_hc, z_trunc)
    tail = math.erfc(z_trunc / _SQRT2) if z_trunc is not None else 0.0
    e = math.erfc(lam / _SQRT2)
    n = (m / sigma) ** 2 / 16.0 * math.exp(-lam * lam) / e
    a = (math.sqrt(1.0 - tail / e) * 2.0 * _SQRT2 / math.sqrt(math.pi) * (sigma / m)
         * math.exp(0.5 * lam * lam) * e)
    return 2.0 * k_s * n * a / (1.0 - math.sqrt(p_over_hc)) ** 1.5


def p_over_hc_gaussian(P: float, sigma_over_m_um: float, c1: float, c2: float) -> float:
    """Eq. 12 (Song-Yovanovich)."""
    return (P / (c1 * (1.62 * sigma_over_m_um) ** c2)) ** (1.0 / (1.0 + 0.071 * c2))


def p_over_hc_tg_branch(P: float, sigma_over_m_um: float, c1: float, c2: float,
                        z_trunc: float) -> float:
    """Eq. 15."""
    tail = math.erfc(z_trunc / _SQRT2) ** (0.4289 * c2)
    return (P / (c1 * (2.178 * sigma_over_m_um) ** c2) * tail) ** (1.0 / (1.0 + 0.5 * c2))


def p_over_hc_truncated(P: float, sigma_over_m_um: float, c1: float, c2: float,
                        z_trunc: float) -> float:
    """Eq. 14: the smaller of the two branches, blended with index p."""
    p = 3.9 + 52.0 * math.exp(10.0 * c2)
    g = p_over_hc_gaussian(P, sigma_over_m_um, c1, c2)
    t = p_over_hc_tg_branch(P, sigma_over_m_um, c1, c2, z_trunc)
    return (g ** -p + t ** -p) ** (-1.0 / p)


@dataclass
class TruncationLookup:
    z_trunc: Quantity
    material: str
    interpolated: bool


def reference_truncation(material_key: str, sigma_over_m_um: float,
                         flags: FlagSet) -> TruncationLookup | None:
    """z_t for bead-blasted SS 304 / Ni 200 from Milanez et al. Table 1.

    Linear in sigma/m inside the tested range (as the paper suggests); held at
    the end value outside it, with a CAUTION. Returns None for other materials.
    Always flags (INFO) that the value was fitted with Milanez's own c1/c2.
    """
    key = _match(material_key)
    if key is None:
        return None
    pts = REFERENCE_TRUNCATION[key]
    xs = [p[0] for p in pts]
    inside = xs[0] <= sigma_over_m_um <= xs[-1]
    if sigma_over_m_um <= xs[0]:
        z = pts[0][1]
    elif sigma_over_m_um >= xs[-1]:
        z = pts[-1][1]
    else:
        for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
            if x0 <= sigma_over_m_um <= x1:
                z = z0 + (z1 - z0) * (sigma_over_m_um - x0) / (x1 - x0)
                break
    if not inside:
        flags.add(Code.TG_TRUNCATION_SOURCE, Severity.CAUTION,
                  f"sigma/m = {sigma_over_m_um:.1f} um is outside the tested range for "
                  f"bead-blasted {key} ({xs[0]}-{xs[-1]} um); z_trunc held at {z}.",
                  remedy="z_trunc falls with sigma/m in their data; outside the range the "
                         "trend is unknown. Fit z_trunc to a light-load measurement.",
                  sigma_over_m_um=sigma_over_m_um)
    c1c2 = {"SS304": "10.67 GPa / -0.37", "Ni200": "4.33 GPa / -0.079"}[key]
    flags.add(Code.TG_TRUNCATION_SOURCE, Severity.INFO,
              f"z_trunc = {z:.2f} from Milanez, Yovanovich & Mantelli (bead-blasted {key}), "
              f"fitted with their hardness coefficients c1/c2 = {c1c2}.",
              remedy="z_trunc and c1/c2 were fitted together. With a different hardness fit "
                     "the reference z_trunc can over-correct (their S1 data: RMS 14.7 % "
                     "plain CMY -> 22.4 % TG with c1/c2 = 6.27/-0.23). Prefer z_trunc fitted "
                     "to your own light-load data with your own c1/c2.",
              z_trunc=z)
    q = Quantity(si=z, unit="-", provenance=Provenance.LITERATURE,
                 uncertainty=REFERENCE_TRUNCATION_UNC, name="z_trunc",
                 citation=f"Milanez, Yovanovich & Mantelli 2004, Table 1 (bead-blasted {key})")
    return TruncationLookup(q, key, inside)


def _match(material_key: str) -> str | None:
    k = material_key.replace(" ", "").replace("_", "").upper()
    for ref in REFERENCE_TRUNCATION:
        if k.startswith(ref.upper()):
            return ref
    return None


def check_truncation(z_trunc: float, p_over_hc: float, flags: FlagSet,
                     *, surface_note: str = "") -> float:
    """Flag a z_t outside the correlation's range; return f (Eq. 11)."""
    if not Z_TRUNC_MIN <= z_trunc <= Z_TRUNC_MAX:
        flags.add(Code.TG_TRUNCATION_RANGE, Severity.WARNING,
                  f"z_trunc = {z_trunc:g} is outside 3-4.5, where the truncated-Gaussian "
                  f"correlation was built and checked.",
                  remedy="Below 3 the surface is not Gaussian enough for any CMY-family "
                         "model; above 4.5 truncation no longer matters (use plain CMY).",
                  z_trunc=z_trunc)
    return truncation_function(p_over_hc, z_trunc)


def tg_quantity(k_s: Quantity, sigma: Quantity, m: Quantity, p_over_hc: Quantity,
                z_trunc: Quantity, cap: Confidence | None) -> Quantity:
    h = tg_conductance(k_s.si, sigma.si, m.si, p_over_hc.si, z_trunc.si)
    return make_derived(h, unit="W/m2K", inputs=[k_s, sigma, m, p_over_hc, z_trunc],
                        confidence=cap, name="h_c",
                        note="truncated Gaussian, Milanez, Yovanovich & Mantelli Eq. 10")
