"""How long before the heat flow is actually stationary?

Action item 2. `check_flux_stationarity` reports whether a run was at steady
state; this answers the prior question of how long that takes, so
`steady_steps` can be set from a measurement instead of a guess. Until drift
is under tolerance, every convergence rung above it measures transients, so
this comes before the convergence campaign rather than after.

Method
------
One long run instead of a sweep. `steady_steps` is set deliberately SMALL so
the transient lands inside the production window, and the reservoir
accumulators are then read in sliding windows: for a window starting at t0,
fit dE/dt over its first and second half and compare, exactly as
`check_flux_stationarity` does. The equilibration time is the smallest t0
whose window is stationary.

A sweep over `steady_steps` would answer the same question with one run per
point and no more information, because the accumulators already record the
whole history.

What to watch for
-----------------
The windowed drift is a ratio of two fitted slopes, so it is noisy when the
window is short or the power is near zero. Three guards: the window must hold
enough samples to fit, the drift must stay below tolerance for several
CONSECUTIVE window starts before the run is called equilibrated (one window
dipping under is noise), and the instantaneous power is printed alongside so a
drift that falls only because the power collapsed is visible rather than
mistaken for convergence.

Usage:
    python3 scripts/find_equilibration_time.py <run_dir> [--window-ps 100]
                                               [--tolerance 0.15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))          # pin to THIS tree

from openkapitza.md import parse_reservoir_series

PS = 1e-12
CONSECUTIVE_REQUIRED = 3


def _rate(ts: list[float], es: list[float]) -> float:
    n = len(ts)
    if n < 2:
        return 0.0
    mean_t = sum(ts) / n
    mean_e = sum(es) / n
    denom = sum((t - mean_t) ** 2 for t in ts)
    if denom == 0.0:
        return 0.0
    return sum((t - mean_t) * (e - mean_e) for t, e in zip(ts, es)) / denom


def windowed_drift(
    t: list[float], series: list[float], lo: int, hi: int
) -> tuple[float, float]:
    """(drift, mean power) for samples [lo, hi)."""
    mid = lo + (hi - lo) // 2
    first = _rate(t[lo:mid], series[lo:mid])
    second = _rate(t[mid:hi], series[mid:hi])
    mean_rate = (first + second) / 2.0
    if mean_rate == 0.0:
        return (float("inf"), 0.0)
    return (abs(second - first) / abs(mean_rate), mean_rate)


def analyse(run_dir: Path, window_ps: float, tolerance: float) -> int:
    path = run_dir / "reservoir_energy.dat"
    if not path.is_file():
        print(f"no reservoir_energy.dat in {run_dir}")
        return 2

    t, hot, cold = parse_reservoir_series(path)
    if len(t) < 40:
        print(f"only {len(t)} samples; need a longer run or finer sampling")
        return 2

    span_ps = (t[-1] - t[0]) / PS
    dt_ps = (t[1] - t[0]) / PS
    window_n = max(8, int(window_ps / dt_ps))
    if window_n >= len(t):
        window_ps = span_ps / 2.0
        window_n = len(t) // 2
        print(f"window reduced to {window_ps:.0f} ps to fit the run")

    print(f"{path.name}: {len(t)} samples over {span_ps:.0f} ps "
          f"(every {dt_ps:.2f} ps)")
    print(f"sliding window {window_ps:.0f} ps ({window_n} samples), "
          f"tolerance {tolerance * 100:.0f} %\n")

    step = max(1, window_n // 4)
    print(f"{'window start':>13} {'hot drift':>10} {'cold drift':>11} "
          f"{'hot power':>12} {'cold power':>12} {'imbal':>7} {'ok':>4}")

    equilibrated_at: float | None = None
    streak = 0
    imbalances: list[float] = []
    for lo in range(0, len(t) - window_n + 1, step):
        hi = lo + window_n
        d_hot, p_hot = windowed_drift(t, hot, lo, hi)
        d_cold, p_cold = windowed_drift(t, cold, lo, hi)
        worst = max(d_hot, d_cold)

        # Energy in must equal energy out at steady state. A persistent
        # imbalance is a DIFFERENT failure from drift: drift says the run has
        # not settled, imbalance says energy is going somewhere unaccounted
        # for -- a thermostat on the wrong group, walls that are not really
        # fixed, or stored strain energy still draining into the lattice.
        # Both must be satisfied before a window is a measurement.
        mean_mag = (abs(p_hot) + abs(p_cold)) / 2.0
        imbalance = (abs(abs(p_hot) - abs(p_cold)) / mean_mag
                     if mean_mag > 0.0 else float("inf"))
        imbalances.append(imbalance)

        ok = worst <= tolerance and imbalance <= tolerance
        start_ps = (t[lo] - t[0]) / PS
        print(f"{start_ps:11.0f} ps {d_hot * 100:9.1f}% {d_cold * 100:10.1f}% "
              f"{p_hot:12.3e} {p_cold:12.3e} {imbalance * 100:6.0f}% "
              f"{'yes' if ok else 'no':>4}")
        if ok:
            streak += 1
            if streak >= CONSECUTIVE_REQUIRED and equilibrated_at is None:
                # Credit the START of the streak, not the window that confirmed it.
                equilibrated_at = start_ps - (streak - 1) * step * dt_ps
        else:
            streak = 0

    print()
    if imbalances and min(imbalances) > tolerance:
        print(f"Reservoir power never balances: best imbalance "
              f"{min(imbalances) * 100:.0f} % against a {tolerance * 100:.0f} % "
              f"tolerance.")
        print("At steady state energy in must equal energy out. A persistent")
        print("imbalance is not a length-of-run problem and will NOT go away by")
        print("running longer -- look for stored strain energy draining into the")
        print("lattice (relax the cell first), a thermostat on the wrong group,")
        print("or walls that are not actually fixed.")
        print()
    if equilibrated_at is None:
        print(f"NOT EQUILIBRATED anywhere in {span_ps:.0f} ps.")
        print("Every window still drifts beyond tolerance, so no part of this")
        print("run is a steady-state measurement. Lengthen the run before")
        print("reading any h_K from it, and do not start the convergence")
        print("campaign on this cell yet.")
        return 1

    print(f"EQUILIBRATED from about {equilibrated_at:.0f} ps "
          f"({CONSECUTIVE_REQUIRED} consecutive stationary windows).")
    print(f"Set steady_steps to at least {equilibrated_at:.0f} ps of steps")
    print(f"(plus margin), and keep the production window at or above the")
    print(f"{window_ps:.0f} ps used here, or the stationarity test loses its")
    print(f"resolution.")
    print("\nThis is one cell. Equilibration scales with length, roughly as")
    print("L^2 for diffusive transport, so a longer slab needs longer -- do not")
    print("carry this number to a different geometry without rechecking.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window-ps", type=float, default=100.0)
    parser.add_argument("--tolerance", type=float, default=0.15)
    args = parser.parse_args(argv)
    return analyse(args.run_dir, args.window_ps, args.tolerance)


if __name__ == "__main__":
    raise SystemExit(main())
