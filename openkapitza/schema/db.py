"""Loaders for the bundled material and gas databases.

No physical constant is written inline anywhere else in the package. Every
tabulated number lives in data/*.yaml with a `verified` flag and a note, and
reaches the calculation through one of these loaders, which attaches the
right provenance and raises a flag when the entry is unverified.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from .flags import Code, FlagSet, Severity
from .inputs import GasSpec, MaterialSpec
from .quantities import Confidence, Provenance, Quantity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict:
    with open(DATA_DIR / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def available_materials() -> list[str]:
    return [k for k in _load_yaml("materials.yaml") if not k.startswith("_")]


def available_gases() -> list[str]:
    data = _load_yaml("gases.yaml")
    return [k for k in data if not k.startswith("_")
            and k not in ("accommodation_defaults", "accommodation_bands")]


def _q(
    value: float | None,
    unit: str,
    name: str,
    *,
    verified: bool,
    citation: str,
    confidence: Confidence | None = None,
) -> Quantity | None:
    if value is None:
        return None
    return Quantity(
        si=float(value),
        unit=unit,
        provenance=Provenance.LITERATURE,
        confidence=confidence or (Confidence.MEDIUM if verified else Confidence.LOW),
        citation=citation,
        name=name,
    )


def load_material(key: str, flags: FlagSet | None = None) -> MaterialSpec:
    """Load a material by key, flagging it if the entry is unverified."""
    data = _load_yaml("materials.yaml")
    if key not in data:
        raise KeyError(
            f"unknown material {key!r}; available: {', '.join(available_materials())}"
        )
    e = data[key]
    sources = e.get("sources") or {}
    entry = f"openkapitza/data/materials.yaml::{key}"
    core = ("k", "rho", "E", "nu", "Hv")
    verified = all(f in sources for f in core)
    unverified = [f for f in (*core, "v_long", "v_trans", "theta_debye")
                  if e.get(f) is not None and f not in sources]
    if flags is not None and unverified:
        flags.add(
            Code.MATERIAL_UNVERIFIED,
            Severity.CAUTION,
            f"Material '{key}': {', '.join(unverified)} not yet checked against a "
            f"primary source (the other fields cite one in materials.yaml).",
            remedy=(
                "Supply measured values for this sample where it matters, above all "
                "hardness, which varies by a factor of two or more with temper and "
                "surface preparation."
            ),
        )

    def field(name, unit, *, confidence=None, value=None):
        v = e.get(name) if value is None else value
        src = sources.get(name)
        return _q(v, unit, name, verified=src is not None,
                  citation=src or f"{entry} (UNVERIFIED)", confidence=confidence)

    return MaterialSpec(
        key=key,
        name=e.get("name", key),
        phase=e.get("phase", "crystalline"),
        k=field("k", "W/mK"),
        rho=field("rho", "kg/m3"),
        E=field("E", "GPa"),
        nu=field("nu", "-"),
        # Hardness is explicitly downgraded: it is the least reliable field
        # even when sourced (temper and surface preparation dominate it).
        Hv=field("Hv", "GPa", confidence=Confidence.LOW),
        v_long=field("v_long", "m/s"),
        v_trans=field("v_trans", "m/s"),
        theta_debye=field("theta_debye", "K"),
        number_density=_q(
            e.get("number_density"), "-", "number_density",
            verified=True, citation=f"{entry} (rho/M * N_A)", confidence=Confidence.HIGH,
        ),
        # c1/c2 carry their own citation when they are published fits.
        c1=_q(e.get("c1"), "GPa", "c1",
              verified=bool(e.get("hardness_coefficients_source")),
              citation=e.get("hardness_coefficients_source") or entry),
        c2=_q(e.get("c2"), "-", "c2",
              verified=bool(e.get("hardness_coefficients_source")),
              citation=e.get("hardness_coefficients_source") or entry),
        c1_temperature=e.get("c1_temperature"),
        notes=e.get("notes"),
        verified=verified,
    )


def load_gas(
    key: str,
    pressure_pa: float | None = None,
    flags: FlagSet | None = None,
) -> GasSpec:
    """Load a gas by key. `pressure_pa` is the gas pressure inside the gap."""
    data = _load_yaml("gases.yaml")
    if key not in data or key in ("accommodation_defaults", "accommodation_bands"):
        raise KeyError(
            f"unknown gas {key!r}; available: {', '.join(available_gases())}"
        )
    e = data[key]
    verified = bool(e.get("verified", False))
    cite = f"openkapitza/data/gases.yaml::{key}"

    if flags is not None and not verified:
        flags.add(
            Code.GAS_UNVERIFIED,
            Severity.CAUTION,
            f"Gas '{key}' uses bundled Sutherland constants that have not been "
            f"checked against primary sources.",
            remedy=(
                "Verify k(T) and mu(T) against a reference. Sutherland's form "
                "fits helium and hydrogen poorly."
            ),
        )

    p = None
    if pressure_pa is not None:
        p = Quantity(
            si=float(pressure_pa),
            unit="kPa",
            provenance=Provenance.USER_MEASURED,
            name="gas_pressure",
        )

    return GasSpec(
        key=key,
        name=e.get("name", key),
        M=_q(e["M"], "g/mol", "M", verified=verified, citation=cite,
             confidence=Confidence.HIGH),
        gamma=_q(e["gamma"], "-", "gamma", verified=verified, citation=cite,
                 confidence=Confidence.HIGH),
        Pr=_q(e["Pr"], "-", "Pr", verified=verified, citation=cite),
        mu_ref=_q(e["mu_ref"], "Pa.s", "mu_ref", verified=verified, citation=cite),
        mu_T_ref=_q(e["mu_T_ref"], "K", "mu_T_ref", verified=verified, citation=cite),
        mu_S=_q(e["mu_S"], "K", "mu_S", verified=verified, citation=cite),
        k_ref=_q(e["k_ref"], "W/mK", "k_ref", verified=verified, citation=cite),
        k_T_ref=_q(e["k_T_ref"], "K", "k_T_ref", verified=verified, citation=cite),
        k_S=_q(e["k_S"], "K", "k_S", verified=verified, citation=cite),
        pressure=p,
        notes=e.get("notes"),
        verified=verified,
    )


def accommodation_band(gas_key: str) -> dict | None:
    """Sourced (low, high) accommodation band for a gas, or None if unsourced."""
    bands = _load_yaml("gases.yaml").get("accommodation_bands", {})
    rec = bands.get(gas_key)
    return dict(rec) if rec else None


def default_accommodation(
    gas_key: str, solid_key: str, flags: FlagSet | None = None
) -> Quantity:
    """Look up a default thermal accommodation coefficient.

    These defaults are the weakest numbers in the gap model -- measured values
    for one gas/solid pair span an order of magnitude with surface condition.
    Always returned with LOW confidence and a flag.
    """
    table = _load_yaml("gases.yaml")["accommodation_defaults"]
    gas_table = table.get(gas_key, {})
    if isinstance(gas_table, dict):
        value = gas_table.get(solid_key, gas_table.get("default"))
    else:
        value = None
    if value is None:
        value = table.get("_fallback", 0.8)

    if flags is not None:
        flags.add(
            Code.GAP_ACCOMMODATION_DEFAULT,
            Severity.WARNING,
            f"Thermal accommodation coefficient for {gas_key} on {solid_key} "
            f"taken as a default of {value:.2f}.",
            remedy=(
                "Measured values vary by an order of magnitude with surface "
                "cleanliness, adsorbed layers and roughness. Supply a measured "
                "value, take one from a source matched to your actual surface "
                "condition, or run the MD gas-surface scattering calculation."
            ),
            alpha_default=float(value),
        )

    return Quantity(
        si=float(value),
        unit="-",
        provenance=Provenance.CORRELATION_DEFAULT,
        confidence=Confidence.LOW,
        citation="openkapitza/data/gases.yaml::accommodation_defaults (NEEDS VERIFICATION)",
        name="accommodation",
    )
