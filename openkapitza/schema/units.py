"""Unit handling for OpenKapitza.

CONVENTION, enforced throughout the package:

    Every number stored or passed between physics functions is in base SI.

Conversion happens only at the IO boundary -- when parsing user input, and
when formatting a report. No physics function ever accepts a unit string.
This is deliberate: unit confusion is the single most common source of
silent, order-of-magnitude errors in thermal contact codes, and the only
robust defence is to make the internal representation unambiguous.

LAMMPS `metal` units (Angstrom, eV, ps, bar) are handled by a separate
converter in the md package, applied only when writing input decks.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Multiplicative factors: value_in_SI = value_in_unit * FACTOR[unit]
# Units with an additive offset are handled separately below.
# ---------------------------------------------------------------------------

_FACTORS: dict[str, float] = {
    # dimensionless
    "": 1.0,
    "-": 1.0,
    "dimensionless": 1.0,
    # length -> m
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "micron": 1e-6,
    "nm": 1e-9,
    "A": 1e-10,
    "angstrom": 1e-10,
    # pressure / stress / modulus -> Pa
    "Pa": 1.0,
    "kPa": 1e3,
    "MPa": 1e6,
    "GPa": 1e9,
    "bar": 1e5,
    "mbar": 1e2,
    "atm": 101325.0,
    "torr": 133.322368421,
    "psi": 6894.757293168,
    # thermal conductivity -> W/(m*K)
    "W/mK": 1.0,
    "W/m/K": 1.0,
    "W/(m*K)": 1.0,
    # heat flux -> W/m^2
    "W/m2": 1.0,
    "W/m^2": 1.0,
    # conductance -> W/(m^2*K)
    "W/m2K": 1.0,
    "W/m^2K": 1.0,
    "MW/m2K": 1e6,
    "MW/m^2K": 1e6,
    "GW/m2K": 1e9,
    # resistance -> m^2*K/W
    "m2K/W": 1.0,
    "m^2K/W": 1.0,
    # temperature -> K (multiplicative part)
    "K": 1.0,
    # density -> kg/m^3
    "kg/m3": 1.0,
    "kg/m^3": 1.0,
    "g/cm3": 1e3,
    "g/cm^3": 1e3,
    # molar mass -> kg/mol
    "kg/mol": 1.0,
    "g/mol": 1e-3,
    # velocity -> m/s
    "m/s": 1.0,
    "km/s": 1e3,
    # viscosity -> Pa*s
    "Pa.s": 1.0,
    "Pa*s": 1.0,
    # heat capacity -> J/(m^3*K)
    "J/m3K": 1.0,
    "J/m^3K": 1.0,
}

_OFFSETS: dict[str, float] = {
    "C": 273.15,
    "degC": 273.15,
}

# Physical constants (SI, CODATA-style values)
K_B = 1.380649e-23          # J/K, exact
N_A = 6.02214076e23         # 1/mol, exact
R_GAS = K_B * N_A           # J/(mol*K)
HBAR = 1.054571817e-34      # J*s


class UnitError(ValueError):
    """Raised when a unit string is not recognised."""


def to_si(value: float, unit: str) -> float:
    """Convert `value` expressed in `unit` into base SI."""
    unit = (unit or "").strip()
    if unit in _OFFSETS:
        return float(value) + _OFFSETS[unit]
    try:
        return float(value) * _FACTORS[unit]
    except KeyError:
        raise UnitError(
            f"unknown unit {unit!r}. Known units: {', '.join(sorted(k for k in _FACTORS if k))}"
        ) from None


def from_si(value_si: float, unit: str) -> float:
    """Convert a base-SI `value_si` into `unit`, for display."""
    unit = (unit or "").strip()
    if unit in _OFFSETS:
        return float(value_si) - _OFFSETS[unit]
    try:
        return float(value_si) / _FACTORS[unit]
    except KeyError:
        raise UnitError(f"unknown unit {unit!r}") from None


def is_known(unit: str) -> bool:
    unit = (unit or "").strip()
    return unit in _FACTORS or unit in _OFFSETS
