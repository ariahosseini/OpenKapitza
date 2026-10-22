"""Quantities that carry their own epistemic status.

Every scalar in OpenKapitza is a `Quantity`: a value in SI, an optional
1-sigma uncertainty, where it came from, and how much we trust it. This is
what turns "flag any result you are not confident in" from a judgement call
into a mechanical property of the calculation.

`Quantity` deliberately does NOT implement arithmetic. Physics functions
take and return plain SI floats; the wrapper lives at the boundary. Making
callers write `.si` is a feature -- it keeps the unit-bearing object from
silently propagating through an expression where its unit no longer holds.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Iterable

from . import units


class Provenance(enum.Enum):
    """Where a number came from. Ordered roughly by trustworthiness."""

    USER_MEASURED = "user_measured"
    """Directly measured by the user (profilometry, indentation, TDTR...)."""

    COMPUTED = "computed"
    """Produced by this package from other quantities."""

    LITERATURE = "literature"
    """Tabulated value from a cited source."""

    MD_COMPUTED = "md_computed"
    """Computed by molecular dynamics with a stated interatomic potential.

    Sits between literature and correlation in trust, matching the fill
    priority of section 11 of the agent instruction: measurement, then
    database, then MD, then correlation. It is a property of the POTENTIAL as
    much as of the material -- a value that reproduces the potential
    faithfully can still be wrong about the material, as SiCGe.tersoff's
    elastic constants are -- so the default confidence is LOW and the fill
    policy may raise it only for quantities where MD is known to do well.
    The `citation` field should name the potential file and the `note` the
    temperature and run.
    """

    CORRELATION_DEFAULT = "correlation_default"
    """Filled in from an empirical correlation because nothing better exists."""

    USER_ESTIMATED = "user_estimated"
    """The user guessed. Treat with suspicion."""

    PLACEHOLDER = "placeholder"
    """Stand-in so the pipeline can run. Must not reach a final answer unflagged."""


class Confidence(enum.IntEnum):
    """How much weight a number can bear. IntEnum so `min()` works."""

    UNRELIABLE = 0   # known to be wrong or wildly model-dependent
    LOW = 1          # order of magnitude only
    MEDIUM = 2       # good to a factor of ~2
    HIGH = 3         # good to the stated uncertainty

    @property
    def label(self) -> str:
        return self.name.lower()


# Default confidence implied by each provenance, used when none is given.
_DEFAULT_CONFIDENCE = {
    Provenance.USER_MEASURED: Confidence.HIGH,
    Provenance.COMPUTED: Confidence.HIGH,
    Provenance.LITERATURE: Confidence.MEDIUM,
    Provenance.MD_COMPUTED: Confidence.LOW,
    Provenance.CORRELATION_DEFAULT: Confidence.LOW,
    Provenance.USER_ESTIMATED: Confidence.LOW,
    Provenance.PLACEHOLDER: Confidence.UNRELIABLE,
}


@dataclass(frozen=True)
class Quantity:
    """A scalar in SI, with uncertainty and provenance attached.

    Parameters
    ----------
    si
        The value, in base SI units.
    unit
        The unit the value is naturally reported in. Used only for display;
        `si` is always SI regardless of what this says.
    uncertainty
        1-sigma absolute uncertainty, in SI. None means "unquantified",
        which is not the same as zero and is reported as such.
    """

    si: float
    unit: str = ""
    uncertainty: float | None = None
    provenance: Provenance = Provenance.USER_ESTIMATED
    confidence: Confidence | None = None
    citation: str | None = None
    note: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not units.is_known(self.unit):
            raise units.UnitError(f"unknown unit {self.unit!r} for quantity {self.name!r}")
        if self.confidence is None:
            object.__setattr__(
                self, "confidence", _DEFAULT_CONFIDENCE[self.provenance]
            )
        if self.uncertainty is not None and self.uncertainty < 0:
            raise ValueError("uncertainty must be non-negative")

    # -- construction -------------------------------------------------------

    @classmethod
    def parse(
        cls,
        value: float,
        unit: str,
        *,
        rel_uncertainty: float | None = None,
        uncertainty: float | None = None,
        **kwargs,
    ) -> "Quantity":
        """Build from a value in `unit` (not SI). The IO-boundary constructor.

        `rel_uncertainty` is a convenience: 0.05 means +/-5 %.
        """
        si = units.to_si(value, unit)
        if uncertainty is not None:
            unc_si = units.to_si(uncertainty, unit) if unit not in ("C", "degC") else uncertainty
        elif rel_uncertainty is not None:
            unc_si = abs(si) * rel_uncertainty
        else:
            unc_si = None
        return cls(si=si, unit=unit, uncertainty=unc_si, **kwargs)

    def derived(
        self,
        si: float,
        *,
        unit: str | None = None,
        inputs: Iterable["Quantity"] = (),
        **kwargs,
    ) -> "Quantity":
        """Make a computed quantity whose confidence is capped by its inputs."""
        return make_derived(si, unit=unit or self.unit, inputs=[self, *inputs], **kwargs)

    # -- access -------------------------------------------------------------

    @property
    def display(self) -> float:
        """The value in `self.unit`, for printing."""
        return units.from_si(self.si, self.unit)

    @property
    def rel_uncertainty(self) -> float | None:
        if self.uncertainty is None or self.si == 0:
            return None
        return self.uncertainty / abs(self.si)

    def to(self, unit: str) -> float:
        return units.from_si(self.si, unit)

    def with_(self, **kwargs) -> "Quantity":
        return replace(self, **kwargs)

    def __str__(self) -> str:
        v = self.display
        txt = f"{v:.4g} {self.unit}".strip()
        if self.uncertainty is not None:
            txt += f" +/- {units.from_si(self.uncertainty, self.unit):.2g}"
        return f"{txt} [{self.provenance.value}/{self.confidence.label}]"


def make_derived(
    si: float,
    *,
    unit: str,
    inputs: Iterable[Quantity] = (),
    provenance: Provenance = Provenance.COMPUTED,
    confidence: Confidence | None = None,
    **kwargs,
) -> Quantity:
    """Construct a computed Quantity whose confidence cannot exceed its inputs.

    A result is never more trustworthy than the weakest number that fed it.
    An explicit `confidence` argument is treated as a further cap, not an
    override -- a model can only ever make things worse, not better.
    """
    inputs = list(inputs)
    caps = [q.confidence for q in inputs]
    if confidence is not None:
        caps.append(confidence)
    resolved = min(caps) if caps else Confidence.HIGH
    return Quantity(
        si=si, unit=unit, provenance=provenance, confidence=resolved, **kwargs
    )


@dataclass
class QuantityBundle:
    """A named group of quantities, e.g. all outputs of one model."""

    label: str
    items: dict[str, Quantity] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Quantity:
        return self.items[key]

    def __setitem__(self, key: str, value: Quantity) -> None:
        self.items[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.items

    @property
    def confidence(self) -> Confidence:
        if not self.items:
            return Confidence.HIGH
        return min(q.confidence for q in self.items.values())
