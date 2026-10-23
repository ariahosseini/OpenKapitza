"""Macroscale contact mechanics: hardness, regime, contact-spot conductance."""

from .cmy import (
    CMY_PRESSURE_MAX,
    CMY_PRESSURE_MIN,
    AreaModel,
    ContactResult,
    combined_roughness,
    combined_slope,
    contact_conductance_elastic,
    contact_conductance_plastic,
    harmonic_conductivity,
    light_load_check,
    mean_plane_separation,
    regime_branches,
)
from .hardness import (
    HardnessResult,
    contact_hardness,
    effective_modulus,
    softer_material,
)
from .regime import Regime, RegimeResult, select_regime

__all__ = [
    "CMY_PRESSURE_MAX",
    "CMY_PRESSURE_MIN",
    "AreaModel",
    "ContactResult",
    "combined_roughness",
    "combined_slope",
    "contact_conductance_elastic",
    "contact_conductance_plastic",
    "harmonic_conductivity",
    "light_load_check",
    "mean_plane_separation",
    "regime_branches",
    "HardnessResult",
    "contact_hardness",
    "effective_modulus",
    "softer_material",
    "Regime",
    "RegimeResult",
    "select_regime",
]
