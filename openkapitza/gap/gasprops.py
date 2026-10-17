"""Temperature- and pressure-dependent gas properties.

Viscosity and conductivity follow Sutherland's law:

    phi(T) = phi_ref * (T_ref + S)/(T + S) * (T/T_ref)^1.5

The molecular mean free path is derived from viscosity via hard-sphere
kinetic theory rather than from a tabulated collision diameter:

    Lambda = (mu / P) * sqrt(pi * R_specific * T / 2)

Viscosity data is better established than collision diameters, and this
relation is exact within the hard-sphere model, so the derived path length
inherits only the viscosity's error rather than adding a second one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..schema.inputs import GasSpec
from ..schema.quantities import Quantity, make_derived
from ..schema.units import R_GAS


@dataclass
class GasState:
    """Gas properties evaluated at a particular temperature and pressure."""

    k_g: Quantity           # thermal conductivity, W/(m*K)
    mu: Quantity            # dynamic viscosity, Pa*s
    mfp: Quantity           # molecular mean free path, m
    beta: Quantity          # 2*gamma/((gamma+1)*Pr), -
    temperature: Quantity
    pressure: Quantity


def _sutherland(
    phi_ref: float, T_ref: float, S: float, T: float
) -> float:
    return phi_ref * ((T_ref + S) / (T + S)) * (T / T_ref) ** 1.5


def gas_state(
    gas: GasSpec, temperature: Quantity, pressure: Quantity | None = None
) -> GasState:
    """Evaluate the gas properties needed by the gap model."""
    T = temperature.si
    p = (pressure or gas.pressure)
    if p is None:
        raise ValueError(
            "gas pressure is required to evaluate the mean free path; "
            "pass it on the GasSpec or to gas_state()"
        )
    P = p.si

    k_si = _sutherland(gas.k_ref.si, gas.k_T_ref.si, gas.k_S.si, T)
    mu_si = _sutherland(gas.mu_ref.si, gas.mu_T_ref.si, gas.mu_S.si, T)

    R_specific = R_GAS / gas.M.si
    mfp_si = (mu_si / P) * math.sqrt(math.pi * R_specific * T / 2.0)

    beta_si = 2.0 * gas.gamma.si / ((gas.gamma.si + 1.0) * gas.Pr.si)

    return GasState(
        k_g=make_derived(
            k_si, unit="W/mK",
            inputs=[gas.k_ref, gas.k_T_ref, gas.k_S, temperature], name="k_g",
        ),
        mu=make_derived(
            mu_si, unit="Pa.s",
            inputs=[gas.mu_ref, gas.mu_T_ref, gas.mu_S, temperature], name="mu",
        ),
        mfp=make_derived(
            mfp_si, unit="nm",
            inputs=[gas.mu_ref, gas.M, temperature, p], name="mfp",
            note="hard-sphere kinetic theory from viscosity",
        ),
        beta=make_derived(
            beta_si, unit="-", inputs=[gas.gamma, gas.Pr], name="beta",
        ),
        temperature=temperature,
        pressure=p,
    )
