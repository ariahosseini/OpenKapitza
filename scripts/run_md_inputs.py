"""Run the section-11 MD input calculators from the command line.

    python3 scripts/run_md_inputs.py accommodation --lmp L --potentials P --gas He
    python3 scripts/run_md_inputs.py kappa         --lmp L --potentials P --temperature 500
    python3 scripts/run_md_inputs.py indent        --lmp L --potentials P

Each prints the result and every flag raised. The library modules hold the
method; this file only wires arguments to them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openkapitza.md import accommodation as acc  # noqa: E402
from openkapitza.md import conductivity as kap  # noqa: E402
from openkapitza.md import indentation as ind  # noqa: E402
from openkapitza.md.elastic import CubicSpec  # noqa: E402
from openkapitza.schema import FlagSet  # noqa: E402

NI = CubicSpec("Ni", "fcc", 3.52, 58.6934, "eam", "Ni_u3.eam", None, "Ni")
SI = CubicSpec("Si", "diamond", 5.431, 28.0855, "sw", "Si.sw", "Si", "Si")


def cmd_accommodation(a, flags):
    spec = acc.AccommodationSpec(NI, a.gas, metal_lj="Ni", production_ps=a.production_ps,
                                 n_gas=a.n_gas, t_surface=a.t_surface, t_gas=a.t_gas)
    cal, scan = acc.calibrate_well_depth(a.lmp, spec, a.potentials, flags)
    print(f"calibration: LB well {scan.eps_pair * 1e3:.2f} meV sums to "
          f"{scan.e_ads * 1e3:.1f} meV; calibrated pair well {cal.eps_gs * 1e3:.3f} meV")
    scales = (tuple(a.scales) if a.scales else
              acc.target_range_scales(a.gas) if a.sensitivity else (1.0,))
    for s in scales:
        sub = acc.AccommodationSpec(**{**cal.__dict__, "eps_scale": s})
        r = acc.run_accommodation(a.lmp, sub, a.potentials, flags)
        m = r.meta
        print(f"{a.gas}/Ni eps x{s:.2f}: alpha {r.alpha:.4f} +/- {r.stderr:.4f} "
              f"(unpaired {m['alpha_unpaired']:.4f} +/- {m['stderr_unpaired']:.4f}); "
              f"pairs {m.get('n_pairs')}, incomplete {m.get('incomplete_fraction', 0):.3f}; "
              f"E_i check {r.incident_check:.4f}; T_s {m['t_surface_measured']:.1f} K; "
              f"coverage {m['coverage']:.3f}", flush=True)


def cmd_kappa(a, flags):
    spec = kap.KappaSpec(SI, temperature=a.temperature, cross_cells=a.cross_cells,
                         production_ps=a.production_ps, steady_ps=a.steady_ps,
                         theta_debye=645.0)
    pts = []
    for n in a.lengths:
        t = time.time()
        p = kap.run_length(a.lmp, kap.KappaSpec(**{**spec.__dict__, "length_cells": n}),
                           a.potentials, flags)
        pts.append(p)
        print(f"L {p.length / 10:6.2f} nm  k {p.k:7.2f} +/- {p.stderr:5.2f}  halves "
              f"{p.k_halves[0]:.1f}/{p.k_halves[1]:.1f}  imbalance {p.meta['imbalance']:.3f}  "
              f"atoms {p.meta['n_atoms']}  {time.time() - t:.0f}s", flush=True)
    if len(pts) >= 2:
        r = kap.extrapolate(pts, flags, label="Si (SW)")
        print(f"k_inf {r.k_inf:.1f} +/- {r.stderr:.1f} W/mK; ratio to longest "
              f"{r.extrapolation_ratio:.2f}; chi2/dof {r.meta['chi2_per_dof']:.2f}")


def cmd_indent(a, flags):
    spec = ind.IndentSpec(NI, radius=a.radius, lateral_cells=a.lateral_cells,
                          depth_cells=a.depth_cells, max_depth=a.max_depth, step=a.step)
    r = ind.run_indentation(a.lmp, spec, a.potentials, flags)
    for h, P, p in zip(r.depth, r.load, r.pressure):
        print(f"  h {h:5.2f} A   P {P:8.2f} nN   P/(pi a^2) {p:6.1f} GPa")
    print(f"pop-in {r.popin_depth} A at {r.popin_load} nN; H {r.hardness:.1f} +/- "
          f"{r.hardness_spread:.1f} GPa; Hertz E* {r.reduced_modulus:.0f} GPa; "
          f"box/contact {r.lateral_ratio:.1f}; atoms {r.meta['atoms']}")
    ind.compare_with_microhardness(r, flags, microhardness_GPa=a.microhardness,
                                   label=f"Ni R={a.radius:.0f} A")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lmp", required=True, type=Path)
    common.add_argument("--potentials", required=True, type=Path)
    p = sub.add_parser("accommodation", parents=[common])
    p.add_argument("--gas", default="He")
    p.add_argument("--production-ps", type=float, default=1000.0)
    p.add_argument("--n-gas", type=int, default=100)
    p.add_argument("--t-surface", type=float, default=300.0)
    p.add_argument("--t-gas", type=float, default=400.0)
    p.add_argument("--sensitivity", action="store_true")
    p.add_argument("--scales", type=float, nargs="+", default=None,
                   help="well-depth scales on the calibrated well (overrides --sensitivity)")
    p = sub.add_parser("kappa", parents=[common])
    p.add_argument("--temperature", type=float, default=500.0)
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--cross-cells", type=int, default=4)
    p.add_argument("--steady-ps", type=float, default=200.0)
    p.add_argument("--production-ps", type=float, default=500.0)
    p = sub.add_parser("indent", parents=[common])
    p.add_argument("--radius", type=float, default=25.0)
    p.add_argument("--lateral-cells", type=int, default=24)
    p.add_argument("--depth-cells", type=int, default=16)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--step", type=float, default=0.25)
    p.add_argument("--microhardness", type=float, default=None, help="GPa")
    a = ap.parse_args(argv)
    flags = FlagSet()
    {"accommodation": cmd_accommodation, "kappa": cmd_kappa, "indent": cmd_indent}[a.cmd](a, flags)
    print(flags.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
