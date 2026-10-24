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
    hot_hardness_factor,
    hot_hardness_note,
    softer_material,
)
from .macro import MacroResult, macro_flags, macro_resistance
from .regime import Regime, RegimeResult, select_regime
from .truncated import (
    REFERENCE_TRUNCATION,
    p_over_hc_truncated,
    reference_truncation,
    tg_conductance,
    tg_conductance_exact,
)

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
    "hot_hardness_factor",
    "hot_hardness_note",
    "effective_modulus",
    "softer_material",
    "Regime",
    "RegimeResult",
    "select_regime",
    "MacroResult",
    "macro_flags",
    "macro_resistance",
    "REFERENCE_TRUNCATION",
    "p_over_hc_truncated",
    "reference_truncation",
    "tg_conductance",
    "tg_conductance_exact",
]
