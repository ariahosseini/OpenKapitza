"""Known-answer validation of the accommodation-coefficient pipeline.

`fix wall/reflect/stochastic` imposes alpha, so a gas between a diffusive top
wall and an ideal bottom wall has a known answer: diffusive -> 1, specular -> 0,
Maxwell with diffuse fraction a -> a. Each wall gets an independent seed.
Reports the paired estimator (used for results) and the unpaired one (the
cross-check); both must land on the known answer.

Usage:
    python3 scripts/validate_accommodation.py --lmp <lmp> [--production-ps 1500]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openkapitza.md.accommodation import run_wall_validation  # noqa: E402
from openkapitza.schema import FlagSet  # noqa: E402

CASES = [("diffusive", None, 1.0, 11), ("specular", None, 0.0, 23),
         ("maxwell", 0.4, 0.4, 37), ("maxwell", 0.8, 0.8, 41)]


def _z(value: float, err: float, expected: float) -> float:
    # A specular wall returns every atom with its own energy, so the paired
    # estimate is exactly 0 with zero spread. That is a pass, not a division error.
    if abs(value - expected) < 1e-9:
        return 0.0
    return (value - expected) / err if err > 0 else float("inf")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lmp", required=True, type=Path)
    ap.add_argument("--production-ps", type=float, default=1500.0)
    ap.add_argument("--n-gas", type=int, default=600)
    ap.add_argument("--t-gas", type=float, default=600.0)
    args = ap.parse_args(argv)
    print(f"{'wall':14} {'expect':>6}  {'paired alpha':>17} {'z':>6}  "
          f"{'unpaired alpha':>17} {'z':>6}  {'Ei/2kTg':>7} {'pairs':>6}")
    worst = 0.0
    for model, frac, expected, seed in CASES:
        kw = dict(n_gas=args.n_gas, production_ps=args.production_ps,
                  t_gas=args.t_gas, seed=seed)
        if frac is not None:
            kw["diffuse_fraction"] = frac
        t = time.time()
        r = run_wall_validation(args.lmp, model, FlagSet(), **kw)
        zp = _z(r.alpha, r.stderr, expected)
        au, su = r.meta["alpha_unpaired"], r.meta["stderr_unpaired"]
        zu = _z(au, su, expected)
        worst = max(worst, abs(zp), abs(zu))
        name = model + (f"({frac})" if frac is not None else "")
        print(f"{name:14} {expected:6.2f}  {r.alpha:8.4f} +/- {r.stderr:.4f} {zp:+6.2f}  "
              f"{au:8.4f} +/- {su:.4f} {zu:+6.2f}  {r.incident_check:7.4f} "
              f"{r.meta.get('n_pairs', 0):6d}  {time.time() - t:.0f}s", flush=True)
    print(f"\nworst |z| = {worst:.2f}  ->  {'PASS' if worst < 3.0 else 'FAIL'}")
    return 0 if worst < 3.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
