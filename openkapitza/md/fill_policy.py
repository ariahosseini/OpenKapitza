"""Section 11 of the agent instruction, as code.

When may an MD-computed number stand in for an experimental one? The
instruction answers per quantity; this module makes that answer enforceable,
so a calculator cannot hand the continuum model a value the rules forbid --
a metal's phonon-only conductivity, say, or an MD nanohardness posing as the
microhardness Song-Yovanovich needs.

The dividing line is that properties of the MATERIAL can be computed and
properties of the SURFACE FINISH and the TEST CONDITIONS cannot.

Three modes:

- fill     an input is missing; choose the best available candidate, where an
           MD value is eligible only if the policy allows it for this
           quantity and this material.
- compare  a measurement exists and MD runs alongside. The measurement is what
           the model uses; the MD value is reported with the ratio. MD never
           overwrites a measurement in this mode.
- request  the user explicitly asked for the MD value. It is used even over a
           measurement -- the user's call -- but that override is flagged.

Fill priority comes from provenance: measured, then literature, then MD, then
correlation default. Accommodation coefficients need no special case: their
database defaults are generic correlations, so an MD value outranks them by
the ordering itself, which is exactly what section 11 asks for.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace

from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Confidence, Provenance, Quantity

# Higher is better. MD sits between literature and correlation.
_RANK = {
    Provenance.USER_MEASURED: 6,
    Provenance.COMPUTED: 5,
    Provenance.LITERATURE: 4,
    Provenance.MD_COMPUTED: 3,
    Provenance.CORRELATION_DEFAULT: 2,
    Provenance.USER_ESTIMATED: 1,
    Provenance.PLACEHOLDER: 0,
}


class Mode(enum.Enum):
    FILL = "fill"
    COMPARE = "compare"
    REQUEST = "request"


@dataclass(frozen=True)
class QuantityPolicy:
    key: str
    computable: bool                 # can MD compute this at all?
    fillable: bool                   # may an MD value fill a missing input?
    fill_cap: Confidence | None      # confidence ceiling for an MD fill
    metals_fillable: bool = True     # may MD fill it for a metal?
    reason: str = ""


POLICIES: dict[str, QuantityPolicy] = {
    "accommodation": QuantityPolicy(
        "accommodation", computable=True, fillable=True,
        fill_cap=Confidence.MEDIUM,
        reason=("Highest-value fill: the generic default is the weakest input "
                "in the model. A clean-surface MD value is closer to a bound "
                "than to the answer for a contaminated technical surface."),
    ),
    "h_K": QuantityPolicy(
        "h_K", computable=True, fillable=False, fill_cap=None,
        reason=("MD's own output, not a stand-in for an experimental input. "
                "Compare against TDTR where measurements exist."),
    ),
    "elastic": QuantityPolicy(
        "elastic", computable=True, fillable=True,
        fill_cap=Confidence.MEDIUM,
        reason=("Cheap and well defined, but it is the POTENTIAL's stiffness. "
                "Capped at LOW unless the potential has passed the elastic "
                "gate against reference values."),
    ),
    "thermal_conductivity": QuantityPolicy(
        "thermal_conductivity", computable=True, fillable=True,
        fill_cap=Confidence.LOW, metals_fillable=False,
        reason=("Classical MD has no electrons. For a metal it gives the "
                "phonon part only -- a lower bound that misses roughly 60 % "
                "of k in SS 304 and roughly 90 % in nickel."),
    ),
    "nanohardness": QuantityPolicy(
        "nanohardness", computable=True, fillable=False, fill_cap=None,
        reason=("Compare against nanoindentation only. MD indents nanometres "
                "of perfect crystal at extreme strain rates."),
    ),
    "microhardness": QuantityPolicy(
        "microhardness", computable=False, fillable=False, fill_cap=None,
        reason=("c1, c2 are fitted to Vickers microhardness over micron-scale "
                "indents. MD reaches nanometres and typically returns several "
                "times the microhardness; it cannot supply these."),
    ),
    "roughness": QuantityPolicy(
        "roughness", computable=False, fillable=False, fill_cap=None,
        reason="Set by machining or polishing, not by material physics.",
    ),
    "slope": QuantityPolicy(
        "slope", computable=False, fillable=False, fill_cap=None,
        reason="Set by machining or polishing, not by material physics.",
    ),
    "pressure": QuantityPolicy(
        "pressure", computable=False, fillable=False, fill_cap=None,
        reason="A test condition.",
    ),
    "gas_pressure": QuantityPolicy(
        "gas_pressure", computable=False, fillable=False, fill_cap=None,
        reason="A test condition.",
    ),
    "temperature": QuantityPolicy(
        "temperature", computable=False, fillable=False, fill_cap=None,
        reason="A test condition.",
    ),
}


@dataclass(frozen=True)
class Selection:
    """What the model should use, and what else is worth reporting."""

    value: Quantity | None
    source: Provenance | None
    md_value: Quantity | None = None
    measured_value: Quantity | None = None
    ratio_md_to_measured: float | None = None


def policy_for(key: str) -> QuantityPolicy:
    try:
        return POLICIES[key]
    except KeyError:
        raise KeyError(
            f"no section-11 policy for {key!r}; known: {sorted(POLICIES)}"
        ) from None


def md_eligible(
    key: str, *, is_metal: bool = False
) -> tuple[bool, str]:
    """May an MD value fill this input for this material? And why not."""
    pol = policy_for(key)
    if not pol.computable:
        return False, pol.reason
    if not pol.fillable:
        return False, pol.reason
    if is_metal and not pol.metals_fillable:
        return False, pol.reason
    return True, ""


def cap_md(
    key: str, md: Quantity, *, potential_passes_gate: bool | None = None
) -> Quantity:
    """Apply the confidence ceiling for an MD fill of this quantity."""
    pol = policy_for(key)
    cap = pol.fill_cap or Confidence.LOW
    if key == "elastic" and potential_passes_gate is not True:
        # Unknown or failed gate: the potential's stiffness has not been shown
        # to resemble the material's. SiCGe.tersoff fails it outright.
        cap = Confidence.LOW
    current = md.confidence if md.confidence is not None else Confidence.LOW
    return replace(md, confidence=min(current, cap))


def select(
    key: str,
    candidates: list[Quantity],
    flags: FlagSet,
    *,
    mode: Mode = Mode.FILL,
    is_metal: bool = False,
    potential_passes_gate: bool | None = None,
    disagreement_threshold: float = 0.25,
) -> Selection:
    """Choose the value the continuum model should use.

    `candidates` may hold any mix of measured, literature, MD and correlation
    values. MD candidates are identified by provenance, never by position.
    """
    pol = policy_for(key)
    md = next((c for c in candidates if c.provenance is Provenance.MD_COMPUTED),
              None)
    others = [c for c in candidates if c.provenance is not Provenance.MD_COMPUTED]

    if md is not None and not pol.computable:
        raise ValueError(
            f"{key}: MD cannot compute this quantity. {pol.reason}"
        )

    measured = next((c for c in others
                     if c.provenance is Provenance.USER_MEASURED), None)
    ratio = None
    if md is not None and measured is not None and measured.si != 0.0:
        ratio = md.si / measured.si
        if abs(ratio - 1.0) > disagreement_threshold:
            flags.add(
                Code.MD_DISAGREES_WITH_MEASUREMENT,
                Severity.CAUTION,
                f"{key}: MD gives {ratio:.2f}x the measured value.",
                remedy=(
                    "A disagreement this size usually indicts the potential, "
                    "not the measurement. For elastic constants it is what the "
                    "potential gate exists to catch."
                ),
                ratio=ratio,
            )

    # --- request: the user's explicit choice of the MD value ---------------
    if mode is Mode.REQUEST:
        if md is None:
            raise ValueError(f"{key}: an MD value was requested but none given")
        # A request overrides the fill rules, not the physics. The user may
        # choose an MD conductivity for a metal, but it is still the phonon
        # part only, and saying so is not optional. (Found in review: this
        # branch used to return the value with no warning at all.)
        if key == "thermal_conductivity" and is_metal:
            flags.add(
                Code.MD_K_METAL_PHONON_ONLY,
                Severity.WARNING,
                f"{key}: the requested MD value for a metal is the lattice "
                f"(phonon) contribution only. {pol.reason}",
                remedy=("Add the electronic part (Wiedemann-Franz from the "
                        "electrical conductivity) before using it as total k."),
            )
        if measured is not None:
            flags.add(
                Code.MD_OVERRIDES_MEASUREMENT,
                Severity.WARNING,
                f"{key}: the MD value is being used in place of a measurement "
                f"at the user's request.",
                remedy="Report both, and say the substitution was requested.",
            )
        capped = cap_md(key, md, potential_passes_gate=potential_passes_gate)
        return Selection(capped, Provenance.MD_COMPUTED, md, measured, ratio)

    # --- compare: measurement is used, MD is reported beside it -------------
    if mode is Mode.COMPARE:
        if measured is None:
            raise ValueError(f"{key}: compare mode needs a measured value")
        return Selection(measured, Provenance.USER_MEASURED, md, measured, ratio)

    # --- fill: best available candidate by provenance -----------------------
    pool = list(others)
    if md is not None:
        ok, why = md_eligible(key, is_metal=is_metal)
        if ok:
            pool.append(cap_md(key, md,
                               potential_passes_gate=potential_passes_gate))
        else:
            flags.add(
                Code.MD_FILL_REFUSED,
                Severity.WARNING if is_metal else Severity.INFO,
                f"{key}: an MD value was available but may not fill this "
                f"input. {why}",
                remedy=("Use it for comparison only, or supply a measured or "
                        "literature value."),
            )

    if not pool:
        return Selection(None, None, md, measured, ratio)

    best = max(pool, key=lambda q: _RANK.get(q.provenance, -1))
    if best.provenance is Provenance.MD_COMPUTED:
        flags.add(
            Code.MD_FILLED,
            Severity.WARNING,
            f"{key} was filled from MD ({best.citation or 'potential not stated'})"
            f" at confidence {best.confidence.name if best.confidence is not None else 'LOW'}.",
            remedy=(
                "This is the potential's value standing in for the material's. "
                "Replace it with a measurement if one can be had."
            ),
        )
    return Selection(best, best.provenance, md, measured, ratio)
