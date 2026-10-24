"""Unloading (hysteresis): a joint relaxed from a higher earlier load.

During first loading the asperities flatten plastically. Unloaded to a lower
pressure P < P_max, they recover only elastically, so the real contact area
and the conductance stay ABOVE the first-loading value at the same P
(Mikic 1971). Reloading up to P_max retraces the unloading curve. For a
package this is the joint whose clamp has relaxed, or one assembled at a
high load and run at a lower one.

Source: Milanez, Yovanovich & Mantelli, JTHT 18(1) 2004, Eqs. 16-19 (Mikic's
model with the truncated-Gaussian geometry; z_trunc -> infinity is Mikic's
fully Gaussian model). Variables are dimensionless: lambda = Y/sigma at
P_max, u = lambda_max + X^2 the height variable, Z the unloading parameter
solved from the pressure.

    P/H_max = N(Z) / [1 + t / (I(Z) - t)]                         (16)
    A_r/A_a = I(Z) - t                                             (17)
    n*      = n*_max(lambda_max + Z^2)        [Eq. 18, integrated] (18)
    a       = sqrt(A_r / (A_a n pi))                               (19)

    I(Z) = (2/sqrt(2 pi)) Int_Z^inf u exp(-u^2/2) X^3 (1 - Z/X)   dX
    N(Z) = (2/sqrt(2 pi)) Int_Z^inf u exp(-u^2/2) X^3 (1 - Z/X)^1.5 dX
    t    = erfc(z_trunc/sqrt2) / 2         (0 for the fully Gaussian model)
    n*   = n (sigma/m)^2 = exp(-u^2) / (16 erfc(u/sqrt2))   (Eq. 2 at u)

and h_c = 2 k_s n a / (1 - sqrt(A_r/A_a))^1.5 (Eq. 1). H_max, lambda_max are
the first-loading values at P_max (Eqs. 14 and 9).

Checks (tests/test_unloading.py): I(0) = erfc(lambda_max/sqrt2)/2 exactly, so
Z = 0 returns the first-loading point (A_r/A_a = P_max/H_max); Eqs. 19 + 2
reproduce Eq. 8's spot radius and Eq. 1 then gives Eq. 6.

EQ. 18 AS PRINTED. The printed integrand's second factor reads
{lambda_max - ...}. Deriving Eq. 18 as the spots lost between heights
lambda_max and lambda_max + Z^2 (n = n_max - Int (-dn/du) 2X dX) gives
{lambda_max + X^2 - ...}, which integrates to n*(lambda_max + Z^2) and goes to
zero when every contact opens. The printed form leaves ~60 % of the spots in
contact at full separation; the two agree within ~1 % for Z <= 0.2. The
derived form is used.

LIMIT OF THE TRUNCATED VERSION. A_r/A_a = I(Z) - t reaches zero at a finite Z
(the truncated tail cannot unload further); deep unloading of a truncated
surface is outside the model. The paper found the truncated unloading model
accurate above ~300-400 kPa in its tests and the fully Gaussian one closer
below; both are flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_SQRT2 = math.sqrt(2.0)
_C = 4.0 * math.pi / (2.0 * math.pi) ** 1.5           # = 2/sqrt(2 pi)
_XG, _WG = np.polynomial.legendre.leggauss(400)


def _quad(f, a: float, b: float) -> float:
    x = 0.5 * (b - a) * _XG + 0.5 * (b + a)
    return 0.5 * (b - a) * float(np.sum(_WG * f(x)))


def _upper(lam: float, Z: float) -> float:
    # exp(-u^2/2) is < 1e-40 once u > 13.6; integrate X far enough past that.
    return Z + math.sqrt(max(14.0 - lam, 0.0) + 1.0) + 1.0


def area_integral(lam: float, Z: float) -> float:
    """I(Z) of Eq. 17."""
    def g(X):
        u = lam + X * X
        return u * np.exp(-0.5 * u * u) * X ** 3 * (1.0 - Z / X)
    return _C * _quad(g, Z, _upper(lam, Z)) if Z > 0 else 0.5 * math.erfc(lam / _SQRT2)


def force_integral(lam: float, Z: float) -> float:
    """N(Z) of Eq. 16."""
    if Z <= 0:
        return 0.5 * math.erfc(lam / _SQRT2)
    def g(X):
        u = lam + X * X
        return u * np.exp(-0.5 * u * u) * X ** 3 * np.clip(1.0 - Z / X, 0.0, None) ** 1.5
    return _C * _quad(g, Z, _upper(lam, Z))


def spot_density(lam: float, Z: float) -> float:
    """n* = n (sigma/m)^2 during unloading: Eq. 2 at u = lambda_max + Z^2."""
    u = lam + Z * Z
    return math.exp(-u * u) / (16.0 * math.erfc(u / _SQRT2))


def pressure_ratio(lam: float, Z: float, t: float) -> float:
    """P/H_max at unloading parameter Z (Eq. 16); NaN beyond the model's edge."""
    I = area_integral(lam, Z)
    if I - t <= 0.0:
        return float("nan")
    return force_integral(lam, Z) / (1.0 + t / (I - t))


def _edge(lam: float, t: float) -> float:
    """Largest Z with A_r/A_a > 0 (infinity-like for the Gaussian model)."""
    if t <= 0.0:
        return 8.0
    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if area_integral(lam, mid) - t > 0.0:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class UnloadingResult:
    h_c: float          # W/m^2K
    Z: float
    area_fraction: float
    spot_density: float  # n*, dimensionless
    lam_max: float
    separation_ratio: float   # (lambda_max + Z^2) / lambda_max, for the gap
    at_edge: bool             # truncated model ran out of tail (deep unloading)


def unloading_conductance(k_s: float, sigma: float, m: float, P: float, P_max: float,
                          H_max: float, z_trunc: float | None = None) -> UnloadingResult:
    """Contact conductance on unloading from P_max to P (SI in and out)."""
    from .truncated import tg_separation
    if not 0.0 < P <= P_max:
        raise ValueError("unloading needs 0 < P <= P_max")
    t = 0.5 * math.erfc(z_trunc / _SQRT2) if z_trunc is not None else 0.0
    lam = tg_separation(P_max / H_max, z_trunc)
    target = P / H_max
    top = _edge(lam, t)
    lo, hi = 0.0, top
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        r = pressure_ratio(lam, mid, t)
        if r == r and r > target:          # r == r: not NaN
            lo = mid
        else:
            hi = mid
    Z = 0.5 * (lo + hi)
    Ar = area_integral(lam, Z) - t
    at_edge = t > 0.0 and Z > 0.999 * top
    if Ar <= 0.0:
        Ar = max(Ar, 1e-300)
        at_edge = True
    n = spot_density(lam, Z)
    h = 2.0 * k_s * (m / sigma) * math.sqrt(n * Ar / math.pi) / (1.0 - math.sqrt(Ar)) ** 1.5
    return UnloadingResult(h, Z, Ar, n, lam, (lam + Z * Z) / lam, at_edge)
