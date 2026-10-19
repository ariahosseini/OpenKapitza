"""Atomistic interfacial conductance.

Phase 0 shipped the DMM placeholder and the low-temperature quantum screen.
Phase 1 adds direct NEMD: geometry and input-deck generation, the extraction
of h_K from a temperature profile by extrapolated linear fits, and the
convergence harness.

The equilibrium cross-check (the temperature-fluctuation method, NOT
Green-Kubo) and Tier-2 machine-learned potentials arrive in Phase 4.
"""

from .convergence import (
    ConvergenceSeries,
    ConvergenceVerdict,
    check_flux_linearity,
    check_series,
    richardson_extrapolate,
)
from .dmm import (
    DMM_UNCERTAINTY_FACTOR,
    KapitzaEstimate,
    branch_cutoff_frequencies,
    common_cutoff_frequency,
    debye_heat_capacity,
    debye_integral,
    dmm_conductance,
    quantum_check,
    user_kapitza,
)
from .nemd import (
    TIER1_POTENTIALS,
    NEMDGeometry,
    NEMDSpec,
    build_input_deck,
    check_spec,
    complete_pairs,
    locate_potential,
    match_species,
    parse_profile,
    parse_reservoir_energy,
    potential_elements,
    potential_triples,
    resolve_pair_elements,
)
from .profile import (
    KapitzaExtraction,
    SideFit,
    TemperatureProfile,
    extract_kapitza,
    flux_from_reservoirs,
    naive_adjacent_bin_delta_T,
)

__all__ = [
    "ConvergenceSeries", "ConvergenceVerdict", "check_flux_linearity",
    "check_series", "richardson_extrapolate",
    "DMM_UNCERTAINTY_FACTOR", "KapitzaEstimate", "branch_cutoff_frequencies",
    "common_cutoff_frequency", "debye_heat_capacity", "debye_integral",
    "dmm_conductance", "quantum_check", "user_kapitza",
    "TIER1_POTENTIALS", "NEMDGeometry", "NEMDSpec", "build_input_deck",
    "check_spec",
    "complete_pairs",
    "locate_potential",
    "match_species",
    "potential_elements",
    "potential_triples",
    "resolve_pair_elements", "parse_profile", "parse_reservoir_energy",
    "KapitzaExtraction", "SideFit", "TemperatureProfile", "extract_kapitza",
    "flux_from_reservoirs", "naive_adjacent_bin_delta_T",
]
