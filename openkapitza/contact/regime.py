"""Elastic / plastic deformation regime selection.

The original spec hardwired the plastic CMY correlation. That is wrong for a
large and important class of joints: two hard, smooth, low-slope surfaces --
Si on Si, Si on sapphire, polished ceramics -- deform elastically, and the
plastic correlation can be off by a large factor there.

The discriminant is a plasticity index built from the effective modulus, the
combined asperity slope and the contact microhardness:

    psi = E' * m / H_c

Large psi means the asperities yield; small psi means they spring back.

ON THE CONVENTION. Mikic's index is usually written the other way up,
gamma = H_c / (E' * m), with the published thresholds "gamma <= 0.33 deforms
plastically" and "gamma >= 3.0 deforms elastically". Since gamma = 1/psi,
those map exactly onto the thresholds used here: psi >= 3.0 plastic,
psi <= 0.33 elastic. The reciprocal form is used so that larger psi means
more plastic, which reads more naturally. The numbers are the published ones,
not invented.

Sources do differ on the precise form of the index, so the thresholds are
exposed as parameters, and the transition band is treated as genuinely
ambiguous: inside it the package computes BOTH branches and reports both
rather than silently picking one. Treat a joint in that band as having a
conductance uncertain by whatever the two branches disagree by. Note also
that the deformation mode is set by material properties and asperity shape,
not by the applied load, so psi does not move as pressure changes.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Quantity, make_derived

# Convention, not physics. See the module docstring.
PLASTIC_ABOVE = 3.0
ELASTIC_BELOW = 0.33


class Regime(enum.Enum):
    PLASTIC = "plastic"
    ELASTIC = "elastic"
    BOTH = "both"           # ambiguous: compute and report both branches


@dataclass
class RegimeResult:
    regime: Regime
    psi: Quantity
    forced: bool = False     # True when the user overrode the automatic choice


def select_regime(
    E_eff: Quantity,
    m_eff: Quantity,
    H_c: Quantity,
    flags: FlagSet,
    *,
    override: str | None = None,
    plastic_above: float = PLASTIC_ABOVE,
    elastic_below: float = ELASTIC_BELOW,
) -> RegimeResult:
    psi_si = E_eff.si * m_eff.si / H_c.si
    psi = make_derived(
        psi_si, unit="-", inputs=[E_eff, m_eff, H_c], name="plasticity_index"
    )

    if override is not None:
        regime = Regime(override)
        flags.add(
            Code.REGIME_AMBIGUOUS,
            Severity.INFO,
            f"Deformation regime forced to '{override}' by the user; the "
            f"plasticity index psi = {psi_si:.3g} would otherwise have "
            f"selected '{_auto(psi_si, plastic_above, elastic_below).value}'.",
            psi=psi_si,
        )
        return RegimeResult(regime=regime, psi=psi, forced=True)

    regime = _auto(psi_si, plastic_above, elastic_below)

    if regime is Regime.BOTH:
        flags.add(
            Code.REGIME_AMBIGUOUS,
            Severity.WARNING,
            f"Plasticity index psi = {psi_si:.3g} falls in the elastoplastic "
            f"transition band ({elastic_below} to {plastic_above}); neither the "
            f"plastic nor the elastic correlation is clearly applicable.",
            remedy=(
                "Both branches are computed and reported. Treat the spread "
                "between them as a lower bound on the model uncertainty. An "
                "elastoplastic model would be needed to do better."
            ),
            psi=psi_si,
        )
    elif regime is Regime.ELASTIC:
        flags.add(
            Code.REGIME_ELASTIC,
            Severity.INFO,
            f"Plasticity index psi = {psi_si:.3g} indicates elastic asperity "
            f"deformation; using the Mikic elastic correlation rather than "
            f"plastic CMY.",
            remedy=(
                "Note that the real contact area then scales with the elastic "
                "group sqrt(2)P/(m E') rather than with P/H_c, which changes "
                "the area scaling applied to the Kapitza resistance."
            ),
            psi=psi_si,
        )
    return RegimeResult(regime=regime, psi=psi)


def _auto(psi: float, plastic_above: float, elastic_below: float) -> Regime:
    if psi >= plastic_above:
        return Regime.PLASTIC
    if psi <= elastic_below:
        return Regime.ELASTIC
    return Regime.BOTH
