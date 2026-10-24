"""Caveat flags.

The agent's contract with the user is that it never hands back a number
without saying what might be wrong with it. Flags are how the deterministic
layer states that; the LLM layer may only *render* flags, never invent or
suppress them.

Each flag carries a stable code so the agent layer can key explanations and
remedies off it rather than off prose.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Severity(enum.IntEnum):
    INFO = 0        # worth knowing, does not change the answer
    CAUTION = 1     # answer is usable but the stated uncertainty is optimistic
    WARNING = 2     # a model is being used outside its fitted range
    CRITICAL = 3    # the answer is probably wrong; do not report it unqualified

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Flag:
    code: str
    severity: Severity
    message: str
    remedy: str | None = None
    context: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        txt = f"[{self.severity.label.upper()}] {self.code}: {self.message}"
        if self.remedy:
            txt += f"\n    -> {self.remedy}"
        return txt


@dataclass
class FlagSet:
    """Accumulator passed down through a calculation."""

    flags: list[Flag] = field(default_factory=list)

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        remedy: str | None = None,
        **context: float,
    ) -> None:
        """Record a caveat. Identical (code, message) pairs are deduplicated.

        Deduplication matters because the same material can be loaded twice in
        a symmetric joint, and a report that says the same thing twice trains
        the reader to skim the caveats.
        """
        for existing in self.flags:
            if existing.code == code and existing.message == message:
                return
        self.flags.append(
            Flag(code=code, severity=severity, message=message, remedy=remedy, context=context)
        )

    def extend(self, other: "FlagSet") -> None:
        self.flags.extend(other.flags)

    @property
    def worst(self) -> Severity:
        return max((f.severity for f in self.flags), default=Severity.INFO)

    def at_least(self, severity: Severity) -> list[Flag]:
        return [f for f in self.flags if f.severity >= severity]

    def codes(self) -> set[str]:
        return {f.code for f in self.flags}

    def __iter__(self):
        return iter(self.flags)

    def __len__(self) -> int:
        return len(self.flags)

    def render(self) -> str:
        if not self.flags:
            return "No caveats raised."
        ordered = sorted(self.flags, key=lambda f: -f.severity)
        return "\n".join(str(f) for f in ordered)


# Stable codes used across the package. Keeping them in one place stops the
# agent layer from pattern-matching on message text.
class Code:
    CMY_PRESSURE_RANGE = "CMY_PRESSURE_RANGE"
    CMY_LIGHT_LOAD = "CMY_LIGHT_LOAD"
    TG_TRUNCATION_RANGE = "TG_TRUNCATION_RANGE"
    TG_TRUNCATION_SOURCE = "TG_TRUNCATION_SOURCE"
    TG_HARDNESS = "TG_HARDNESS"
    HARDNESS_TEMPERATURE = "HARDNESS_TEMPERATURE"
    MACRO_CONSTRICTION = "MACRO_CONSTRICTION"
    MACRO_RANGE = "MACRO_RANGE"
    FLATNESS_ASSUMED = "FLATNESS_ASSUMED"
    CLAMP_BOLTED = "CLAMP_BOLTED"
    STACK_VOIDS = "STACK_VOIDS"
    EMD_RATIO = "EMD_RATIO"
    EMD_SHORT = "EMD_SHORT"
    STACK_INCOMPLETE = "STACK_INCOMPLETE"
    STACK_THIN_TIM = "STACK_THIN_TIM"
    STACK_CLASS_RANGE = "STACK_CLASS_RANGE"
    CMY_SLOPE_ESTIMATED = "CMY_SLOPE_ESTIMATED"
    HARDNESS_FALLBACK = "HARDNESS_FALLBACK"
    REGIME_AMBIGUOUS = "REGIME_AMBIGUOUS"
    REGIME_ELASTIC = "REGIME_ELASTIC"
    GAP_RAREFIED = "GAP_RAREFIED"
    GAP_ACCOMMODATION_DEFAULT = "GAP_ACCOMMODATION_DEFAULT"
    GAP_ACCOMMODATION_BAND = "GAP_ACCOMMODATION_BAND"
    GAP_CONTINUUM_BREAKDOWN = "GAP_CONTINUUM_BREAKDOWN"
    KAPITZA_PLACEHOLDER = "KAPITZA_PLACEHOLDER"
    KAPITZA_NEGLIGIBLE = "KAPITZA_NEGLIGIBLE"
    KAPITZA_DOMINANT = "KAPITZA_DOMINANT"
    QUANTUM_EFFECTS = "QUANTUM_EFFECTS"
    MATERIAL_UNVERIFIED = "MATERIAL_UNVERIFIED"
    GAS_UNVERIFIED = "GAS_UNVERIFIED"
    AREA_FRACTION_MODEL = "AREA_FRACTION_MODEL"
    CONTACT_SATURATED = "CONTACT_SATURATED"
    # Phase 1: NEMD
    NEMD_POOR_FIT = "NEMD_POOR_FIT"
    NEMD_FEW_BINS = "NEMD_FEW_BINS"
    NEMD_FLUX_INCONSISTENT = "NEMD_FLUX_INCONSISTENT"
    NEMD_EXCLUSION_SENSITIVE = "NEMD_EXCLUSION_SENSITIVE"
    NEMD_CONDUCTIVITY_MISMATCH = "NEMD_CONDUCTIVITY_MISMATCH"
    NEMD_NONPHYSICAL_JUMP = "NEMD_NONPHYSICAL_JUMP"
    NEMD_FIT_WINDOW_GUESSED = "NEMD_FIT_WINDOW_GUESSED"
    NEMD_NOT_CONVERGED = "NEMD_NOT_CONVERGED"
    NEMD_FLUX_NONLINEAR = "NEMD_FLUX_NONLINEAR"
    NEMD_SIZE_LIMITED = "NEMD_SIZE_LIMITED"
    NEMD_LATTICE_STRAIN = "NEMD_LATTICE_STRAIN"
    NEMD_UNEQUILIBRATED = "NEMD_UNEQUILIBRATED"
    NEMD_STATIONARITY_UNRESOLVED = "NEMD_STATIONARITY_UNRESOLVED"
    NEMD_ERROR_UNDERSTATED = "NEMD_ERROR_UNDERSTATED"
    NEMD_SPOT_PRESSURE_MODEL = "NEMD_SPOT_PRESSURE_MODEL"
    NEMD_PHONON_ONLY_METAL = "NEMD_PHONON_ONLY_METAL"
    NEMD_CELL_STRESS = "NEMD_CELL_STRESS"
    NEMD_POTENTIAL_INCOMPLETE = "NEMD_POTENTIAL_INCOMPLETE"
    NEMD_POTENTIAL_UNVERIFIED = "NEMD_POTENTIAL_UNVERIFIED"
    # Section 11: MD-derived inputs.
    MD_FILLED = "MD_FILLED"
    MD_FILL_REFUSED = "MD_FILL_REFUSED"
    MD_OVERRIDES_MEASUREMENT = "MD_OVERRIDES_MEASUREMENT"
    MD_DISAGREES_WITH_MEASUREMENT = "MD_DISAGREES_WITH_MEASUREMENT"
    MD_ELASTIC_UNCONVERGED = "MD_ELASTIC_UNCONVERGED"
    MD_ACCOMMODATION_STATISTICS = "MD_ACCOMMODATION_STATISTICS"
    MD_ACCOMMODATION_CLEAN_SURFACE = "MD_ACCOMMODATION_CLEAN_SURFACE"
    MD_ACCOMMODATION_COVERAGE = "MD_ACCOMMODATION_COVERAGE"
    MD_ACCOMMODATION_SURFACE_HEATED = "MD_ACCOMMODATION_SURFACE_HEATED"
    MD_ACCOMMODATION_UNCALIBRATED = "MD_ACCOMMODATION_UNCALIBRATED"
    MD_ACCOMMODATION_QUANTUM_GAS = "MD_ACCOMMODATION_QUANTUM_GAS"
    MD_K_SIZE_EXTRAPOLATION = "MD_K_SIZE_EXTRAPOLATION"
    MD_K_METAL_PHONON_ONLY = "MD_K_METAL_PHONON_ONLY"
    MD_INDENT_NOT_MICROHARDNESS = "MD_INDENT_NOT_MICROHARDNESS"
    MD_INDENT_SIZE_EFFECT_REVERSED = "MD_INDENT_SIZE_EFFECT_REVERSED"
    # Phase 2: structures
    STRUCT_QUENCH_RATE = "STRUCT_QUENCH_RATE"
    STRUCT_VALIDATION_FAILED = "STRUCT_VALIDATION_FAILED"
    STRUCT_INTERFACE_BONDING = "STRUCT_INTERFACE_BONDING"
    # Phase 1: compute
    COMPUTE_COST_HIGH = "COMPUTE_COST_HIGH"
    COMPUTE_BACKEND_UNAVAILABLE = "COMPUTE_BACKEND_UNAVAILABLE"
