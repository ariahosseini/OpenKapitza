"""Measure the weekend job set and answer its four questions (KB 28).

    python3 scripts/analyze_weekend.py ~/ok_jobs            # table on screen
    python3 scripts/analyze_weekend.py ~/ok_jobs --json docs/figures/data/weekend.json

Measures every finished case under ROOT/weekend/ with the same code the agent
uses (md.measure.measure_kapitza), then compares them in groups. Unfinished
cases are listed as such, so it can be run while the queue is still going.

Comparisons are ratios with a combined 1-sigma error, and a difference is
called only beyond 2 sigma:
  A. drives      Langevin 0.1 / 0.5 ps, eHEX, HEX on one cell; and the tally
                 flux against each drive's flux
  B. strain      Ge rescaled to a_Si against campaign r00 (483 +/- 20)
  C. damping     r00 at 0.1 ps against 0.5 and 1.0 ps
  D. He anomaly  he_ehex against ar_ehex; he_mass40 and he_nodirect against
                 he_ehex; ar_nodirect against ar_ehex
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R00 = (483.27, 20.13)        # campaign base cell, MW/m^2K (md_results.json)
SIGMA_CALL = 2.0

COMPARE = {
    "A. drives (LJ x1, 1 GPa)": [
        ("arb_langevin_d0.5", "arb_langevin_d0.1", "Langevin 0.5 ps / 0.1 ps"),
        ("arb_ehex", "arb_langevin_d0.1", "eHEX / Langevin 0.1 ps"),
        ("arb_ehex", "arb_langevin_d0.5", "eHEX / Langevin 0.5 ps"),
        ("arb_hex", "arb_ehex", "HEX / eHEX"),
    ],
    "B. strain (bonded Si/Ge, 16 nm)": [
        ("sige_ge_rescaled", "r00", "Ge rescaled to a_Si / strained (r00)"),
    ],
    "C. damping (bonded Si/Ge, 16 nm)": [
        ("sige_damp0.5", "r00", "0.5 ps / 0.1 ps (r00)"),
        ("sige_damp1.0", "r00", "1.0 ps / 0.1 ps (r00)"),
    ],
    "D. He anomaly (gas layer, 1 GPa, eHEX)": [
        ("he_ehex", "ar_ehex", "He layer / Ar layer"),
        ("he_mass40", "he_ehex", "He with Ar's mass / He"),
        ("he_nodirect", "he_ehex", "He, no Si-Ge LJ / He"),
        ("ar_nodirect", "ar_ehex", "Ar, no Si-Ge LJ / Ar"),
    ],
}


def measure(wd: Path) -> dict:
    """h_K and its context for one finished job folder."""
    from openkapitza.md.campaign import read_spec
    from openkapitza.md.measure import measure_kapitza
    from openkapitza.md.nemd import last_thermo_block, parse_log_thermo
    from openkapitza.schema.flags import FlagSet
    spec, _, extra = read_spec(wd)
    flags = FlagSet()
    m = measure_kapitza(wd, spec, flags)
    out = {"h": m.h_K / 1e6, "err": m.uncertainty / 1e6, "q_GW": m.flux / 1e9,
           "flux_mode": spec.flux_mode, "damping_ps": spec.damping_fs / 1000,
           "warn": sorted({str(f.code) for f in flags.flags if int(f.severity) >= 2})}
    rows = last_thermo_block(parse_log_thermo(wd / "log.lammps"))
    if rows and "Temp" in rows[0]:
        out["T_start"], out["T_end"] = rows[0]["Temp"], rows[-1]["Temp"]
    if m.tally:
        t = m.tally
        out["tally"] = {k: (v / 1e9 if k.startswith("q_") and v is not None else v)
                        for k, v in t.items() if k != "rows"}
        out["tally"].update({k: t[k] / 1e6 for k in ("h_into_b", "h_into_bs") if k in t})
    return out


def ratio(a: dict, b: dict) -> tuple[float, float]:
    r = a["h"] / b["h"]
    return r, r * math.hypot(a["err"] / a["h"], b["err"] / b["h"])


def verdict(r: float, e: float) -> str:
    if e <= 0:
        return ""
    z = (r - 1.0) / e
    if abs(z) < SIGMA_CALL:
        return f"same within {SIGMA_CALL:g} sigma ({z:+.1f})"
    return f"{'higher' if z > 0 else 'lower'} ({z:+.1f} sigma)"


def collect(root: Path) -> dict:
    base = root / "weekend"
    res, missing = {"r00": {"h": R00[0], "err": R00[1], "flux_mode": "langevin",
                            "damping_ps": 0.1, "warn": [], "note": "campaign r00"}}, {}
    for wd in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        done = wd / "job.done"
        if not done.is_file():
            missing[wd.name] = "running or queued"
            continue
        if done.read_text().strip() not in ("0", ""):
            missing[wd.name] = f"failed (exit {done.read_text().strip()})"
            continue
        try:
            res[wd.name] = measure(wd)
        except Exception as exc:                       # report, never hide
            missing[wd.name] = f"not measurable: {exc}"
    comps = {}
    for group, pairs in COMPARE.items():
        comps[group] = []
        for a, b, label in pairs:
            if a in res and b in res:
                r, e = ratio(res[a], res[b])
                comps[group].append({"label": label, "a": a, "b": b, "ratio": r, "err": e,
                                     "verdict": verdict(r, e)})
            else:
                comps[group].append({"label": label, "a": a, "b": b, "ratio": None,
                                     "verdict": "waiting for " + ", ".join(
                                         x for x in (a, b) if x not in res)})
    return {"results": res, "missing": missing, "comparisons": comps}


def render(d: dict) -> str:
    out = ["Case                 h_K (MW/m2K)    q (GW/m2)  drive        T start->end   warnings"]
    for name, r in d["results"].items():
        T = (f"{r['T_start']:.0f}->{r['T_end']:.0f} K" if "T_start" in r else "")
        drive = r["flux_mode"] + (f" {r['damping_ps']:g} ps" if r["flux_mode"] == "langevin"
                                  else "")
        q = f"{r['q_GW']:.2f}" if "q_GW" in r else "-"
        out.append(f"{name:20} {r['h']:7.1f} +/- {r['err']:5.1f}  {q:>9}  {drive:12} "
                   f"{T:14} {', '.join(r['warn'])}")
        if "tally" in r:
            t = r["tally"]
            out.append(f"{'':20}   tally: q_A-face {t['q_into_b']:.2f} +/- "
                       f"{t['q_into_b_se'] or 0:.2f} GW/m2 (h {t.get('h_into_b', 0):.0f}), "
                       f"q_B-face {t['q_into_bs']:.2f} +/- {t['q_into_bs_se'] or 0:.2f}")
    for name, why in d["missing"].items():
        out.append(f"{name:20} -- {why}")
    for group, rows in d["comparisons"].items():
        out.append(f"\n{group}")
        for c in rows:
            if c["ratio"] is None:
                out.append(f"  {c['label']:42} {c['verdict']}")
            else:
                out.append(f"  {c['label']:42} {c['ratio']:.3f} +/- {c['err']:.3f}  "
                           f"{c['verdict']}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", type=Path, help="also write the results here")
    a = ap.parse_args(argv)
    d = collect(a.root.expanduser())
    print(render(d))
    if a.json:
        a.json.write_text(json.dumps(d, indent=1, default=float))
        print(f"\nwritten: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
