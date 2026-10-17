"""Convergence testing for a NEMD campaign.

Four axes, as Phase 0's instruction requires: cross-section, slab length, run
duration, and applied flux. The first three are "did the answer stop moving";
the fourth is different in kind and is the one most often skipped.

FLUX LINEARITY IS NOT A CONVERGENCE TEST, IT IS A VALIDITY TEST. The whole
extraction assumes linear response: h_K = q/dT only means something if dT is
proportional to q. Run two or three fluxes and regress dT on q. The line must
pass through the origin -- a significant intercept means something other than
the interface is contributing a fixed temperature offset, and the extracted
h_K is not a property of the interface.

This also catches the naive-extraction error from md.profile. Differencing
adjacent bins adds q*(dz_A/k_A + dz_B/k_B) to the jump, which is itself
proportional to q, so it inflates the slope while leaving the intercept near
zero. A flux-linearity test alone will not catch it -- which is exactly why
the extractor does the extrapolation properly rather than relying on this
test to notice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..schema.flags import Code, FlagSet, Severity

# Fractional change below which a series is taken as converged.
CONVERGENCE_TOLERANCE = 0.05
# Allowed intercept in the flux-linearity fit, relative to the dT range.
MAX_INTERCEPT_FRACTION = 0.10


@dataclass
class ConvergenceSeries:
    """h_K against one varied parameter."""

    axis: str                      # "cross_section" | "length" | "duration" | "flux"
    values: list[float]            # the parameter
    h_K: list[float]               # W/(m^2 K)
    units: str = ""

    def __post_init__(self) -> None:
        if len(self.values) != len(self.h_K):
            raise ValueError("values and h_K must be the same length")
        order = np.argsort(self.values)
        self.values = [self.values[i] for i in order]
        self.h_K = [self.h_K[i] for i in order]


@dataclass
class ConvergenceVerdict:
    axis: str
    converged: bool
    last_change: float
    total_spread: float
    recommendation: str = ""
    detail: dict = field(default_factory=dict)


def check_series(
    series: ConvergenceSeries,
    flags: FlagSet,
    *,
    tolerance: float = CONVERGENCE_TOLERANCE,
) -> ConvergenceVerdict:
    """Has h_K stopped moving along this axis?"""
    h = np.asarray(series.h_K, dtype=float)
    if h.size < 3:
        flags.add(
            Code.NEMD_NOT_CONVERGED,
            Severity.WARNING,
            f"Only {h.size} point(s) on the {series.axis} axis; convergence "
            f"cannot be judged from fewer than three.",
            remedy="Add points until the last two agree within the tolerance.",
            axis=series.axis,
        )
        return ConvergenceVerdict(series.axis, False, float("nan"),
                                  float("nan"),
                                  "add more points")

    last_change = abs(h[-1] - h[-2]) / abs(h[-2])
    spread = (h.max() - h.min()) / abs(np.median(h))
    converged = last_change <= tolerance

    verdict = ConvergenceVerdict(
        axis=series.axis,
        converged=converged,
        last_change=float(last_change),
        total_spread=float(spread),
        detail={"values": series.values, "h_K": series.h_K},
    )

    if not converged:
        trend = "rising" if h[-1] > h[-2] else "falling"
        verdict.recommendation = (
            f"extend the {series.axis} axis; h_K is still {trend}"
        )
        flags.add(
            Code.NEMD_NOT_CONVERGED,
            Severity.WARNING,
            f"h_K changed by {last_change * 100:.1f} % between the last two "
            f"{series.axis} points ({series.values[-2]:.3g} -> "
            f"{series.values[-1]:.3g} {series.units}), above the "
            f"{tolerance * 100:.0f} % tolerance.",
            remedy=(
                f"The answer is still {trend} with {series.axis}. Quoting the "
                f"largest box as the converged value understates the "
                f"uncertainty; extend the series or extrapolate explicitly."
            ),
            axis=series.axis, last_change=last_change,
        )
    return verdict


def check_flux_linearity(
    fluxes: list[float],
    delta_T: list[float],
    flags: FlagSet,
) -> dict:
    """Regress dT on q and confirm the response is linear through the origin.

    A non-zero intercept means a fixed temperature offset unrelated to the
    flux -- a thermostat artefact, an unconverged transient, or a second
    interface. Curvature means the gradient has left linear response.
    """
    q = np.asarray(fluxes, dtype=float)
    dT = np.asarray(delta_T, dtype=float)
    if q.size < 2:
        raise ValueError("need at least two fluxes to test linearity")

    slope, intercept = np.polyfit(q, dT, 1)
    pred = slope * q + intercept
    ss_res = float(np.sum((dT - pred) ** 2))
    ss_tot = float(np.sum((dT - dT.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    dT_range = float(dT.max() - dT.min()) or float(abs(dT.mean()))
    intercept_fraction = abs(intercept) / dT_range if dT_range else float("inf")

    # h_K from the slope through the origin is the flux-independent estimate.
    h_from_slope = 1.0 / slope if slope else float("inf")
    per_point = [f / d for f, d in zip(fluxes, delta_T)]
    spread = (max(per_point) - min(per_point)) / np.median(per_point)

    if intercept_fraction > MAX_INTERCEPT_FRACTION:
        flags.add(
            Code.NEMD_FLUX_NONLINEAR,
            Severity.WARNING,
            f"dT against flux has an intercept of {intercept:.3g} K, "
            f"{intercept_fraction * 100:.0f} % of the dT range, rather than "
            f"passing through the origin.",
            remedy=(
                "A flux-independent temperature offset is present. Check that "
                "the run reached steady state before averaging, that the "
                "reservoirs are not overlapping the fit region, and that "
                "there is only one interface in the box."
            ),
            intercept=float(intercept),
        )

    if spread > 2 * CONVERGENCE_TOLERANCE:
        flags.add(
            Code.NEMD_FLUX_NONLINEAR,
            Severity.WARNING,
            f"h_K computed point-by-point varies by {spread * 100:.1f} % across "
            f"the flux series, so it is not flux independent.",
            remedy=(
                "Outside linear response h_K is not a material property. "
                "Reduce the applied gradient until the point-by-point values "
                "agree, and use the slope-based value rather than any single "
                "point."
            ),
            spread=float(spread),
        )

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "intercept_fraction": float(intercept_fraction),
        "h_K_from_slope": float(h_from_slope),
        "h_K_per_point": per_point,
        "point_spread": float(spread),
        "linear": intercept_fraction <= MAX_INTERCEPT_FRACTION
                  and spread <= 2 * CONVERGENCE_TOLERANCE,
    }


def richardson_extrapolate(
    inverse_sizes: list[float], h_K: list[float]
) -> float:
    """Extrapolate h_K to infinite size by regressing on 1/L.

    Finite-size effects in NEMD go roughly as 1/L because the slab truncates
    the phonon mean free path spectrum, so a straight line in 1/L extrapolated
    to zero is the standard estimate of the bulk-limit value. Interfaces are
    less sensitive to this than bulk conductivity, but not immune.
    """
    x = np.asarray(inverse_sizes, dtype=float)
    y = np.asarray(h_K, dtype=float)
    if x.size < 3:
        raise ValueError("need at least three sizes to extrapolate")
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept)
