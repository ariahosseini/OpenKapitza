"""Is the reservoir power stationary? A drift test that knows its own noise.

WHY THIS REPLACED THE FIXED-TOLERANCE TEST (open issue P1-1). The first
version fitted dE/dt over each half of the production window and compared the
two slopes against a fixed 15 %. On the completed 400 ps relaxed Si/Ge run,
that ratio scattered from 3 % to 153 % across windows (mean 46 %, standard
deviation 36 %) with no trend at all: the metric's own noise was more than
twice the tolerance, so every run was reported "not steady" whatever its
physics. A test that fires on everything is as useless as one that fires on
nothing, and more dangerous because it looks conservative.

METHOD

1. Split the window into K equal sub-windows. The power in each is the
   least-squares slope of the accumulator inside it (see `window_powers` for
   why not the end-point difference).
2. Regress the K window powers on time. The slope's standard error uses the
   window-to-window noise estimated from SUCCESSIVE DIFFERENCES of the window
   powers (von Neumann). That noise includes autocorrelation inside each
   window, as long as windows are long against the correlation time, and it
   is not inflated by the curvature of a decaying transient, which an error
   from residuals about the line would be.
3. Express the trend as a relative drift across the whole window,
   drift = slope * T / |mean power|, with its standard error.

VERDICT, per reservoir, against `tolerance` (the drift that matters):

- drifting:   the trend is significant (|t| >= T_SIGNIFICANT) AND larger than
              the tolerance. The run was not at steady state.
- stationary: |drift| + 2 sigma <= tolerance. Certified steady to within the
              tolerance.
- unresolved: anything else. The run may be steady, but the data cannot show
              drift below the tolerance; this is a statistics problem, fixed
              by a longer window or a larger cross-section, not by a longer
              equilibration.

Keeping "unresolved" separate is the point: it tells the user which knob to
turn, where the old test said "not steady" in both cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

T_SIGNIFICANT = 3.0
DEFAULT_WINDOWS = 20
MIN_SAMPLES_PER_WINDOW = 3


class Verdict(str, Enum):
    STATIONARY = "stationary"
    DRIFTING = "drifting"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class DriftResult:
    verdict: Verdict
    mean_power: float          # same units as energy / time of the input
    drift: float               # relative change of power across the window
    drift_se: float            # its standard error
    t_stat: float              # slope / its standard error (inf if no noise)
    n_windows: int
    residual_lag1: float       # lag-1 autocorrelation of window residuals

    @property
    def resolution(self) -> float:
        """The smallest drift this data could have certified (2 sigma)."""
        return 2.0 * self.drift_se


def window_powers(t: np.ndarray, e: np.ndarray, n_windows: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """(window centre times, mean power per window) from a cumulative series.

    The power in each window is the least-squares slope of E(t) inside it.
    That is unbiased whether the noise sits on the cumulative tally (white
    noise on E) or on its increments (a random walk, as a Langevin tally's
    stochastic work is), and efficient in the first case. End-point
    differences, tried first, throw away every interior sample when the noise
    is on E. The between-window scatter, not the in-window fit, supplies the
    error, so the in-window fit's own error model does not matter.
    """
    edges = np.linspace(0, t.size, n_windows + 1).round().astype(int)
    centres, powers = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        tw, ew = t[a:b], e[a:b]
        if tw.size < 2:
            continue
        x = tw - tw.mean()
        sxx = float(x @ x)
        if sxx <= 0:
            continue
        centres.append(float(tw.mean()))
        powers.append(float(x @ (ew - ew.mean())) / sxx)
    return np.asarray(centres), np.asarray(powers)


def analyse_drift(t, e, *, tolerance: float = 0.15,
                  n_windows: int = DEFAULT_WINDOWS) -> DriftResult:
    """Trend test on one reservoir's cumulative energy series."""
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)
    k = min(n_windows, t.size // MIN_SAMPLES_PER_WINDOW)
    if k < 4:
        raise ValueError(f"{t.size} samples: too few for a trend over >= 4 windows")
    c, p = window_powers(t, e, k)
    mean_p = float(p.mean())
    if mean_p == 0.0:
        raise ValueError("mean reservoir power is zero; no heat flow to test")
    span = float(t[-1] - t[0])

    x = c - c.mean()
    sxx = float(x @ x)
    slope = float(x @ (p - mean_p)) / sxx
    resid = p - mean_p - slope * x
    # Noise from successive differences (von Neumann), NOT from residuals
    # about the line: a decaying transient is curved, and its curvature in
    # the residuals would inflate the noise estimate and hide the transient
    # itself. Successive differences of a smooth trend are small, so this
    # estimate stays honest whatever the trend's shape. (Residual-based
    # errors caught an exponential transient only 30 % of the time.)
    d = np.diff(p)
    s2 = float(d @ d) / (2.0 * d.size)
    slope_se = math.sqrt(s2 / sxx)
    lag1 = (float(resid[:-1] @ resid[1:]) / float(resid @ resid)
            if float(resid @ resid) > 0 else 0.0)

    drift = slope * span / abs(mean_p)
    drift_se = slope_se * span / abs(mean_p)
    t_stat = (slope / slope_se) if slope_se > 0 else (math.inf if slope != 0 else 0.0)

    if abs(t_stat) >= T_SIGNIFICANT and abs(drift) > tolerance:
        verdict = Verdict.DRIFTING
    elif abs(drift) + 2.0 * drift_se <= tolerance:
        verdict = Verdict.STATIONARY
    else:
        verdict = Verdict.UNRESOLVED
    return DriftResult(verdict, mean_p, drift, drift_se, t_stat, int(p.size), lag1)
