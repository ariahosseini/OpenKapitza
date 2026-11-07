"""Quantum bracket for a classical-MD interface conductance (S-11, KB 31).

Classical MD gives every vibrational mode k_B of heat capacity. Below the
Debye temperature the high-frequency modes are partly frozen out, so a
classical h_K counts heat carriers that, in reality, carry less. The
project's rule (KB "Quantum correction"): the classical value stays the
primary result; a correction is offered only as a labelled bracket.

The bracket here is the heat-capacity weighting used in Landauer-type
estimates: if the interface transmits phonons up to a cutoff frequency with
no strong spectral preference, each mode's contribution scales with its
quantum heat capacity relative to k_B, so

    h_q / h_cl  =  C_Debye(T; theta_c) / (3 N k_B)  =  3/x_c^3 * int_0^x_c x^4 e^x/(e^x-1)^2 dx,
    x_c = theta_c / T,

the Debye heat-capacity fraction at the cutoff theta_c. Which cutoff is
right depends on what crosses:
- **softer side's theta** (elastic transmission stops at the softer side's
  highest frequencies): the milder correction, the upper end of the band;
- **stiffer side's theta** (inelastic exchange lets higher modes take part,
  as classical MD allows freely): the stronger correction, the lower end.

So the band is h_cl * [f(theta_max), f(theta_min)]. It is an estimate of the
size of the effect, not a corrected answer: the classical-MD transmission
spectrum is itself not the quantum one, and temperature-rescaling
corrections are known to be unreliable for transport (KB).
"""

from __future__ import annotations

import math


def debye_fraction(T: float, theta: float, steps: int = 400) -> float:
    """C_V(Debye)/(3 N k_B) at temperature T for Debye temperature theta."""
    if T <= 0 or theta <= 0:
        raise ValueError("T and theta must be positive")
    xc = theta / T
    h = xc / steps
    s = 0.0
    for i in range(steps + 1):
        x = i * h
        f = x * x if x == 0 else x ** 4 * math.exp(x) / math.expm1(x) ** 2
        s += f * (0.5 if i in (0, steps) else 1.0)
    return 3.0 * s * h / xc ** 3


def quantum_band(T: float, theta_a: float, theta_b: float) -> tuple[float, float]:
    """(low, high) factors on a classical h_K."""
    lo, hi = sorted((theta_a, theta_b))
    return debye_fraction(T, hi), debye_fraction(T, lo)
