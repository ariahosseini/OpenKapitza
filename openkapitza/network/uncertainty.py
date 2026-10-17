"""Monte Carlo uncertainty propagation.

Analytic (linear) propagation is a poor fit here. The contact correlations are
power laws with exponents near one, the network is a series-parallel
combination, and several inputs carry uncertainties large enough that a
first-order expansion around the nominal point is not trustworthy. Sampling
handles all of that without special-casing.

Positive-definite quantities are sampled lognormally so that a 50 % relative
uncertainty cannot produce a negative roughness or hardness, and so that a
"factor of 2.5" band on a DMM estimate is represented symmetrically in log
space, which is how such a band is actually meant.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from ..schema.quantities import Quantity


@dataclass
class MonteCarloResult:
    nominal: float
    mean: float
    median: float
    p05: float
    p95: float
    samples: list[float]
    n_failed: int = 0

    @property
    def relative_band(self) -> float:
        return (self.p95 - self.p05) / self.median if self.median else float("nan")

    def summary(self, unit_scale: float = 1.0, unit: str = "") -> str:
        s = unit_scale
        return (
            f"nominal {self.nominal / s:.4g}{unit}, "
            f"median {self.median / s:.4g}{unit}, "
            f"90% band [{self.p05 / s:.4g}, {self.p95 / s:.4g}]{unit}"
        )


def sample_lognormal(q: Quantity, rng: random.Random) -> float:
    """Draw one sample of a positive quantity from its stated uncertainty.

    The distribution is MEDIAN-preserving: the nominal value is the median of
    the samples, so a stated "+/- 30 %" is symmetric in log space, which is
    how a relative uncertainty on a positive quantity is normally meant.

    The alternative -- preserving the arithmetic mean, mu = ln(x) - sigma^2/2 --
    puts the median BELOW the nominal for every input. Those shifts then
    propagate through 1/H_c and the other reciprocal terms and compound, so
    the reported median of the joint conductance drifts away from the nominal
    case (measured at +7.6 % for a representative stainless joint) with
    nothing in the report to explain the gap. Centring on the median keeps
    "nominal" and "median" the same quantity.

    A quantity with no uncertainty is returned unchanged -- "unquantified" is
    treated as "do not perturb", and the resulting band is therefore a lower
    bound on the true uncertainty. The report says so.
    """
    if q.uncertainty is None or q.uncertainty == 0 or q.si <= 0:
        return q.si
    rel = q.uncertainty / q.si
    # sigma of the underlying normal that reproduces this relative spread
    sigma = math.sqrt(math.log(1.0 + rel * rel))
    return math.exp(rng.gauss(math.log(q.si), sigma))


def propagate(
    evaluate: Callable[[random.Random], float],
    nominal: float,
    *,
    n: int = 2000,
    seed: int = 0,
) -> MonteCarloResult:
    """Run `evaluate` n times with a seeded RNG and summarise the output.

    The seed is fixed by default: a reproducible uncertainty band is worth
    more than an unbiased one here, because the band is reported to a user who
    will rerun the calculation and expect the same answer.
    """
    rng = random.Random(seed)
    samples: list[float] = []
    failed = 0
    for _ in range(n):
        try:
            v = evaluate(rng)
        except (ValueError, ZeroDivisionError, OverflowError):
            failed += 1
            continue
        if math.isfinite(v):
            samples.append(v)
        else:
            failed += 1

    if not samples:
        raise RuntimeError("all Monte Carlo samples failed to evaluate")

    samples.sort()

    def pct(p: float) -> float:
        idx = min(len(samples) - 1, max(0, int(round(p * (len(samples) - 1)))))
        return samples[idx]

    return MonteCarloResult(
        nominal=nominal,
        mean=sum(samples) / len(samples),
        median=pct(0.50),
        p05=pct(0.05),
        p95=pct(0.95),
        samples=samples,
        n_failed=failed,
    )
