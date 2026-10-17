"""Interstitial gas-gap conductance."""

from .gasprops import GasState, gas_state
from .rarefaction import (
    KN_FREE_MOLECULAR,
    KN_SLIP,
    KN_TRANSITION,
    GapResult,
    accommodation_term,
    gap_conductance,
    vacuum_gap,
)

__all__ = [
    "GasState",
    "gas_state",
    "KN_FREE_MOLECULAR",
    "KN_SLIP",
    "KN_TRANSITION",
    "GapResult",
    "accommodation_term",
    "gap_conductance",
    "vacuum_gap",
]
