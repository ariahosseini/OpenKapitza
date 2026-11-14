"""Analyse the drive-free check of the LJ x1, 1 GPa contact (S-5, KB 44).

    python3 scripts/analyze_drive_check.py ~/ok_jobs/drive_check [--json OUT]

For each finished equilibrium run: h from the slab-energy fluctuation decay
(one exponential, and two with the slow one taken as the interface), its
block error, the series length in relaxation times, and NEMD/EMD for each
drive measured on the same cell (KB 41, 43).

What it can and cannot decide:
- Bound: a correctly measured NEMD conductance is at least the equilibrium
  one. A drive whose h sits more than 2 sigma below EMD reads low.
- Expectation, not a bound: on a low-transmission contact (this one conducts
  ~10x less than the bonded interface) the non-equilibrium correction is
  small, so NEMD/EMD should be close to 1. A drive near EMD is favoured, but
  a ratio of ~1.7 is not excluded by the bound alone.
- The two slab lengths: an interface time doubles with slab length, so h
  should agree between them; if it does not, the fluctuation method is not
  measuring the interface here and the check is inconclusive.
- A series shorter than 500 tau reads h high by ~5-10 % (synthetic check in
  md/emd_kapitza.py); far smaller than the 1.7x being decided.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# NEMD on the same cell, MW/m^2K (KB 41, 43): (value, 1-sigma)
NEMD = {
    "HEX": (40.7, 0.5),
    "eHEX": (37.3, 0.5),          # 38.0 +/- 0.9 and 36.7 +/- 0.6, weighted
    "Langevin": (66.4, 2.4),      # 65.9 +/- 3.4, 65.4 +/- 2.4 (0.5 ps), 67.9 +/- 5.6
}
SIGMA = 2.0


def analyse(wd: Path) -> dict:
    from openkapitza.md.emd_kapitza import check, h_from_series, read_emd
    from openkapitza.schema.flags import FlagSet
    t, dT, n_a, n_b, area = read_emd(wd)
    out = {"n_a": n_a, "n_b": n_b, "area_nm2": area * 1e18, "ns": float(t[-1] - t[0]) / 1000}
    for model in ("single", "biexp"):
        try:
            r = h_from_series(t, dT, n_a, n_b, area, model=model)
        except ValueError as exc:
            out[model] = {"error": str(exc)}
            continue
        flags = FlagSet()
        check(r, flags)
        out[model] = {"h": r.h / 1e6, "err": (r.h_err or 0) / 1e6, "tau_ps": r.tau * 1e12,
                      "n_tau": r.duration / r.tau, "blocks": r.blocks,
                      "tau_fast_ps": r.tau_fast * 1e12 if r.tau_fast else None,
                      "slow_weight": r.slow_weight,
                      "flags": sorted({str(f.code) for f in flags.flags})}
    return out


def choose(r: dict) -> str:
    """The fit to quote. One exponential, unless the decay really has two parts:
    times >= 10x apart, each with >= 15 % of the variance (md/emd_kapitza's
    EMD_TWO_TIMESCALES rule); then the slow one, the interface (KB 23). A
    two-exponential fit to a clean single decay invents a fast part and
    inflates the error ~4x (synthetic check, tests/test_drive_check.py)."""
    from openkapitza.md.emd_kapitza import TWO_TIMESCALE_SEP, TWO_TIMESCALE_WEIGHT
    b = r.get("biexp", {})
    if "h" not in r.get("single", {}):
        return "biexp"
    if "h" in b and b.get("tau_fast_ps"):
        sep = b["tau_ps"] / b["tau_fast_ps"]
        w = b["slow_weight"]
        if sep >= TWO_TIMESCALE_SEP and min(w, 1 - w) >= TWO_TIMESCALE_WEIGHT:
            return "biexp"
    return "single"


def verdict(h: float, e: float) -> dict:
    rows = {}
    for drive, (hn, en) in NEMD.items():
        ratio = hn / h
        err = ratio * math.hypot(en / hn, e / h) if e > 0 else None
        below = err is not None and (1.0 - ratio) / err > SIGMA
        rows[drive] = {"ratio": ratio, "err": err,
                       "reads": "LOW: below the equilibrium bound" if below else
                       ("close to EMD (favoured)" if ratio < 1.25 else "above EMD (allowed)")}
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    root = a.root.expanduser()
    res = {}
    for wd in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("emd_")):
        done = wd / "job.done"
        if not done.is_file() or done.read_text().strip() not in ("0", ""):
            print(f"{wd.name:16} not finished")
            continue
        r = res[wd.name] = analyse(wd)
        print(f"\n{wd.name}: {r['n_a']}+{r['n_b']} mobile atoms, {r['ns']:.1f} ns")
        for model in ("single", "biexp"):
            m = r[model]
            if "error" in m:
                print(f"  {model:7} {m['error']}")
                continue
            extra = (f", fast {m['tau_fast_ps']:.2g} ps, slow weight {m['slow_weight']:.0%}"
                     if m["tau_fast_ps"] else "")
            print(f"  {model:7} h {m['h']:6.1f} +/- {m['err']:4.1f} MW/m2K   tau "
                  f"{m['tau_ps']:6.1f} ps ({m['n_tau']:.0f} tau){extra}  {', '.join(m['flags'])}")
        r["used"] = use = choose(r)
        best = r.get(use, {})
        if "h" in best:
            print(f"  using the {use} fit" + (" (one decay carries the variance)"
                                             if use == "single" else ""))
            r["verdict"] = verdict(best["h"], best["err"])
            for drive, v in r["verdict"].items():
                e = f" +/- {v['err']:.2f}" if v["err"] else ""
                print(f"  NEMD/EMD {drive:9} {v['ratio']:.2f}{e}  {v['reads']}")
    hs = [r[r["used"]]["h"] for r in res.values() if "h" in r.get(r.get("used", ""), {})]
    if len(hs) == 2:
        print(f"\nslab-length check: {hs[0]:.1f} vs {hs[1]:.1f} MW/m2K "
              f"({'agree' if abs(hs[0] - hs[1]) / max(hs) < 0.2 else 'DISAGREE: inconclusive'})")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
