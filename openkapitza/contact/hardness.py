"""Contact microhardness.

The single most common mistake in applying CMY is to substitute a handbook
Vickers number for H_c. Contact microhardness is not a material constant: it
depends on the indentation load, and therefore on the contact pressure and
the surface topography that set the asperity load distribution. Using a bulk
Vickers value systematically overestimates H_c, which understates P/H_c, and
h_c goes as (P/H_c)^0.95 -- so the error passes through almost linearly.

The Song-Yovanovich correlation handles this by expressing the dimensionless
pressure directly in terms of the Vickers coefficients c1 and c2 fitted to
load-dependent indentation data:

    P/H_c = [ P / ( c1 * (1.62 * sigma/m [um])^c2 ) ] ^ ( 1 / (1 + 0.071*c2) )

Note this is explicit, not implicit, despite H_c appearing on both sides of
the physical problem -- the algebra has already been done.

When c1 and c2 are unavailable the code falls back to H_c = Hv of the softer
material and flags it. That fallback is a real approximation, not a formality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..schema.flags import Code, FlagSet, Severity
from ..schema.inputs import MaterialSpec
from ..schema.quantities import Confidence, Provenance, Quantity, make_derived

# The 1.62 and 0.071 are part of the published correlation, not free
# parameters. They are kept here rather than in YAML because changing them
# would mean using a different correlation, not a different material.
_SY_SIGMA_COEFF = 1.62
_SY_EXPONENT_COEFF = 0.071


@dataclass
class HardnessResult:
    H_c: Quantity               # contact microhardness, Pa
    P_over_Hc: Quantity         # dimensionless contact pressure, -
    softer: MaterialSpec
    method: str                 # "song_yovanovich" | "vickers_fallback"


def softer_material(a: MaterialSpec, b: MaterialSpec) -> MaterialSpec:
    """The softer solid governs plastic deformation of the asperities."""
    return a if a.Hv.si <= b.Hv.si else b


def contact_hardness(
    material_a: MaterialSpec,
    material_b: MaterialSpec,
    pressure: Quantity,
    sigma_eff: Quantity,
    m_eff: Quantity,
    flags: FlagSet,
) -> HardnessResult:
    """Compute H_c and P/H_c for the joint.

    Parameters
    ----------
    sigma_eff, m_eff
        The *combined* roughness and slope of the pair, as used by CMY.
    """
    soft = softer_material(material_a, material_b)
    P = pressure.si

    if soft.has_sy_coefficients:
        c1 = soft.c1.si
        c2 = soft.c2.si
        sigma_over_m_um = (sigma_eff.si / m_eff.si) * 1e6
        denom = c1 * (_SY_SIGMA_COEFF * sigma_over_m_um) ** c2
        exponent = 1.0 / (1.0 + _SY_EXPONENT_COEFF * c2)
        p_over_hc = (P / denom) ** exponent
        H_c_si = P / p_over_hc
        method = "song_yovanovich"
        inputs = [pressure, soft.c1, soft.c2, sigma_eff, m_eff]
        cap = None
    else:
        H_c_si = soft.Hv.si
        p_over_hc = P / H_c_si
        method = "vickers_fallback"
        inputs = [pressure, soft.Hv]
        cap = Confidence.LOW
        flags.add(
            Code.HARDNESS_FALLBACK,
            Severity.WARNING,
            f"No Song-Yovanovich coefficients for '{soft.key}'; using the bulk "
            f"Vickers hardness H_c = {soft.Hv.to('GPa'):.3g} GPa directly.",
            remedy=(
                "This ignores the indentation size effect. The Vickers "
                "exponent c2 is negative, so contact microhardness RISES as "
                "the microcontacts shrink: at the light loads of a real joint "
                "H_c exceeds the bulk number. For SS 304 at 2 MPa the fitted "
                "value is 3.41 GPa against a bulk 2.0 GPa. The fallback "
                "therefore UNDERestimates H_c and OVERestimates P/H_c and "
                "h_c. Fit c1 and c2 to load-dependent indentation data on "
                "your own surfaces, or use a published fit for the material."
            ),
            H_c_GPa=H_c_si / 1e9,
        )

    H_c = make_derived(
        H_c_si, unit="GPa", inputs=inputs, confidence=cap,
        name="H_c", note=f"method={method}, softer material={soft.key}",
    )
    ratio = make_derived(
        p_over_hc, unit="-", inputs=inputs, confidence=cap, name="P_over_Hc",
    )
    return HardnessResult(H_c=H_c, P_over_Hc=ratio, softer=soft, method=method)


def _erfcinv(y: float, tol: float = 1e-14) -> float:
    """Inverse complementary error function, by bisection.

    Kept dependency-free on purpose: this is the only place the package needs
    it, and adding scipy for one scalar inverse would be a poor trade.
    """
    if not 0.0 < y < 2.0:
        raise ValueError(f"erfcinv argument must lie in (0, 2), got {y}")
    lo, hi = -6.0, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid) > y:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def contact_hardness_implicit(
    pressure_Pa: float,
    sigma_eff_m: float,
    m_eff: float,
    c1_Pa: float,
    c2: float,
    *,
    d0_um: float = 1.0,
    max_iter: int = 300,
) -> float:
    """Contact microhardness from the implicit Song-Yovanovich system.

    This is the definition the explicit correlation approximates:

        lambda = sqrt(2) * erfcinv(2 P / H_c)
        a      = (sigma/m) * sqrt(8/pi) * exp(lambda^2/2) * erfc(lambda/sqrt2)
        d_V    = sqrt(2 pi) * a
        H_c    = c1 * (d_V / d0)^c2

    Solved by damped fixed-point iteration. Provided mainly so the explicit
    form can be checked against it: a published transcription of the explicit
    equation disagreed with this package's implementation over where the
    constant 1.62 sits, and solving the system settled it -- the implementation
    tracks the implicit solution to about 1 %, the alternative was 45 % low.
    """
    sigma_over_m_um = (sigma_eff_m / m_eff) * 1e6
    H_c = c1_Pa
    for _ in range(max_iter):
        ratio = min(pressure_Pa / H_c, 0.999999)
        lam = math.sqrt(2.0) * _erfcinv(2.0 * ratio)
        a = (
            sigma_over_m_um
            * math.sqrt(8.0 / math.pi)
            * math.exp(lam * lam / 2.0)
            * math.erfc(lam / math.sqrt(2.0))
        )
        d_V = math.sqrt(2.0 * math.pi) * a
        updated = c1_Pa * (d_V / d0_um) ** c2
        if abs(updated - H_c) < 1e-6 * H_c:
            return updated
        H_c = 0.5 * H_c + 0.5 * updated
    return H_c


def effective_modulus(
    material_a: MaterialSpec, material_b: MaterialSpec
) -> Quantity:
    """Effective (reduced) elastic modulus E'.

        1/E' = (1 - nu_a^2)/E_a + (1 - nu_b^2)/E_b
    """
    inv = (
        (1.0 - material_a.nu.si ** 2) / material_a.E.si
        + (1.0 - material_b.nu.si ** 2) / material_b.E.si
    )
    return make_derived(
        1.0 / inv,
        unit="GPa",
        inputs=[material_a.E, material_a.nu, material_b.E, material_b.nu],
        name="E_eff",
    )
