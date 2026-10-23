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
accumulators are then read in sliding windows, each tested with the same
noise-aware trend test `check_flux_stationarity` uses (`md.stationarity`).
The equilibration time is the smallest t0 from which windows are CERTIFIED
stationary; windows the data cannot resolve do not count.

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
from openkapitza.md.stationarity import Verdict, analyse_drift

PS = 1e-12
CONSECUTIVE_REQUIRED = 3


def _window(t: list[float], series: list[float], lo: int, hi: int,
            tolerance: float):
    """Drift test on samples [lo, hi), or None if the window cannot be tested."""
    n = hi - lo
    k = min(20, n // 3)
    if k < 4:
        return None
    try:
        return analyse_drift(t[lo:hi], series[lo:hi], tolerance=tolerance, n_windows=k)
    except ValueError:
        return None


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
    window_n = max(12, int(window_ps / dt_ps))
    if window_n >= len(t):
        window_ps = span_ps / 2.0
        window_n = len(t) // 2
        print(f"window reduced to {window_ps:.0f} ps to fit the run")

    print(f"{path.name}: {len(t)} samples over {span_ps:.0f} ps "
          f"(every {dt_ps:.2f} ps)")
    print(f"sliding window {window_ps:.0f} ps ({window_n} samples), "
          f"tolerance {tolerance * 100:.0f} %\n")

    step = max(1, window_n // 4)
    print(f"{'window start':>13} {'hot drift':>15} {'cold drift':>15} "
          f"{'hot power':>12} {'cold power':>12} {'imbal':>7} {'verdict':>11}")

    equilibrated_at: float | None = None
    streak = 0
    imbalances: list[float] = []
    verdicts: list[str] = []
    for lo in range(0, len(t) - window_n + 1, step):
        hi = lo + window_n
        rh = _window(t, hot, lo, hi, tolerance)
        rc = _window(t, cold, lo, hi, tolerance)
        if rh is None or rc is None:
            verdicts.append("untestable")
            streak = 0
            continue

        # Energy in must equal energy out at steady state. A persistent
        # imbalance is a DIFFERENT failure from drift: drift says the run has
        # not settled, imbalance says energy is going somewhere unaccounted
        # for -- a thermostat on the wrong group, walls that are not really
        # fixed, or stored strain energy still draining into the lattice.
        # Both must be satisfied before a window is a measurement.
        p_hot, p_cold = rh.mean_power, rc.mean_power
        mean_mag = (abs(p_hot) + abs(p_cold)) / 2.0
        imbalance = (abs(abs(p_hot) - abs(p_cold)) / mean_mag
                     if mean_mag > 0.0 else float("inf"))
        imbalances.append(imbalance)

        if Verdict.DRIFTING in (rh.verdict, rc.verdict):
            verdict = "drifting"
        elif rh.verdict is Verdict.STATIONARY and rc.verdict is Verdict.STATIONARY:
            verdict = "stationary"
        else:
            verdict = "unresolved"
        verdicts.append(verdict)
        # Only a CERTIFIED stationary window counts toward equilibration:
        # "unresolved" means the data could not show it, not that it holds.
        ok = verdict == "stationary" and imbalance <= tolerance
        start_ps = (t[lo] - t[0]) / PS
        print(f"{start_ps:11.0f} ps "
              f"{rh.drift * 100:+6.1f}+/-{rh.drift_se * 100:4.1f}% "
              f"{rc.drift * 100:+6.1f}+/-{rc.drift_se * 100:4.1f}% "
              f"{p_hot:12.3e} {p_cold:12.3e} {imbalance * 100:6.0f}% "
              f"{verdict + ('' if ok or verdict != 'stationary' else '*'):>11}")
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
        if verdicts and "drifting" not in verdicts and "unresolved" in verdicts:
            print(f"STATISTICS-LIMITED over {span_ps:.0f} ps: no window shows a")
            print("significant drift, but none can be certified steady to within")
            print(f"{tolerance * 100:.0f} %. This is not an equilibration problem;")
            print("lengthen the window (--window-ps) or the cross-section to cut")
            print("the power noise. A longer equilibration will not help.")
            return 1
        print(f"NOT EQUILIBRATED anywhere in {span_ps:.0f} ps.")
        print("No window is certified stationary, so no part of this run is a")
        print("steady-state measurement. Lengthen the run before reading any")
        print("h_K from it, and do not start the convergence campaign on this")
        print("cell yet.")
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
