"""Measure a potential's own lattice and elastic constants, and gate it.

A thin command-line wrapper over `openkapitza.md.elastic`, which holds the
method, the averaging and the gate. This script used to hold its own copy of
the measurement; that copy measured C11 and C12 only, at one strain set, and
the logic now lives in the library so the agent and this script cannot
disagree. (Knowledge base: "a fix that lives in a script is not a fix".)

What it reports, per species:
  - a0, C11, C12, C44 at 0 K, against reference single-crystal values;
  - the strain-magnitude spread -- the insensitivity check the official
    examples/ELASTIC names;
  - isotropic E and nu by Voigt-Reuss-Hill, which is what the continuum model
    takes;
  - whether the potential passes the elastic gate (each C_ij within 10 %).

For a pair of species it also compares the biaxial modulus
M = C11 + C12 - 2 C12^2 / C11, which decides how strain shares across a
coherent interface, and flags an inverted stiffness ordering. Measured for
SiCGe.tersoff: Si 138 / Ge 154 GPa from the potential against Si 180 / Ge 141
from experiment. Building the geometry differently cannot fix that; a relaxed
MD cell partitions strain by the potential's own C_ij.

Note on Poisson's ratio: the official examples/ELASTIC prints "Poisson Ratio"
= C12 / (C11 + C12), the SINGLE-CRYSTAL value for stress along [100] (0.335
for Si.sw). The value here is the isotropic POLYCRYSTAL nu (0.296 for Si.sw),
which is what the continuum model needs. They are different quantities.

Usage:
    python3 scripts/measure_potential_elastic.py --lmp <lmp> --potentials <dir>
        [--set sicge|sw|ni]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))          # pin to THIS tree

from openkapitza.md.elastic import (
    REFERENCE_CUBIC, CubicSpec, biaxial_modulus, gate, run_elastic,
    voigt_reuss_hill,
)
from openkapitza.schema import FlagSet

SETS = {
    "sicge": [
        CubicSpec("Si(D)", "diamond", 5.431, 28.0855, "tersoff",
                  "SiCGe.tersoff", "Si(D)", "Si"),
        CubicSpec("Ge", "diamond", 5.658, 72.63, "tersoff",
                  "SiCGe.tersoff", "Ge", "Ge"),
    ],
    "sw": [CubicSpec("Si", "diamond", 5.431, 28.0855, "sw", "Si.sw", "Si", "Si")],
    "ni": [CubicSpec("Ni", "fcc", 3.52, 58.6934, "eam", "Ni_u3.eam", None, "Ni")],
    # Si as described by the Munetoh SiO2 Tersoff (the crystal side of Si/SiO2).
    "sio": [CubicSpec("Si(SiO)", "diamond", 5.431, 28.0855, "tersoff", "SiO.tersoff",
                      "Si", "Si")],
    # Jelinek et al. 2012 MEAM (AlSiMgCuFe.meam): one fit with Al-Si and
    # Cu-Si cross terms, the candidate for die-level Si/Al and Si/Cu (P1-6).
    # Needs a LAMMPS build with the MEAM package.
    "alsicu": [
        CubicSpec(f"{el}(MEAM)", lat, a0, m, "meam", "library.meam", f"{el}S", el,
                  extra_files=("AlSiMgCuFe.meam",),
                  coeff_args=f"library.meam AlS SiS MgS CuS FeS AlSiMgCuFe.meam {el}S")
        for el, lat, a0, m in (("Si", "diamond", 5.431, 28.0855),
                               ("Al", "fcc", 4.05, 26.9815),
                               ("Cu", "fcc", 3.615, 63.546))
    ],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", required=True, type=Path)
    parser.add_argument("--potentials", required=True, type=Path)
    parser.add_argument("--set", default="sicge", choices=sorted(SETS))
    args = parser.parse_args(argv)

    results = []
    for spec in SETS[args.set]:
        flags = FlagSet()
        try:
            res = run_elastic(args.lmp, spec, args.potentials, flags)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"{spec.name}: measurement failed: {exc}")
            return 1
        results.append((spec, res, flags))

    print("=" * 96)
    print("Potential's own properties at 0 K, against reference single-crystal values")
    print("=" * 96)
    print(f"{'species':8} {'a0':>8} {'C11':>7} {'C12':>7} {'C44':>7} {'spread':>8}"
          f"  {'E':>7} {'nu':>6}  gate   C11/C12/C44 error")
    for spec, res, flags in results:
        iso = res.isotropic
        report = gate(res, spec.reference)
        e = report.errors
        print(f"{spec.name:8} {res.a0:8.4f} {res.c11:7.1f} {res.c12:7.1f} {res.c44:7.1f}"
              f" {res.spread * 100:7.3f}%  {iso.youngs:7.1f} {iso.poisson:6.3f}  "
              f"{'PASS' if report.passes else 'FAIL'}   "
              f"{e['c11'] * 100:+.0f}/{e['c12'] * 100:+.0f}/{e['c44'] * 100:+.0f} %")
        ref = REFERENCE_CUBIC[spec.reference]
        ri = voigt_reuss_hill(ref["c11"], ref["c12"], ref["c44"])
        print(f"{'':8} reference: a0 {ref['a0']:.3f}  C {ref['c11']}/{ref['c12']}"
              f"/{ref['c44']}  ->  E {ri.youngs:.1f}  nu {ri.poisson:.3f}"
              f"   [{ref['citation']}]")
        for flag in flags.flags:
            print(f"{'':8} [{flag.severity.name}] {flag.code}: {flag.message}")

    if len(results) == 2:
        (sa, ra, _), (sb, rb, _) = results
        ma, mb = biaxial_modulus(ra.c11, ra.c12), biaxial_modulus(rb.c11, rb.c12)
        fa, fb = REFERENCE_CUBIC[sa.reference], REFERENCE_CUBIC[sb.reference]
        xa, xb = biaxial_modulus(fa["c11"], fa["c12"]), biaxial_modulus(fb["c11"], fb["c12"])
        print("\nBiaxial modulus M -- decides strain sharing at a coherent interface")
        print(f"  {sa.name:8} potential {ma:6.1f} GPa   reference {xa:6.1f} GPa")
        print(f"  {sb.name:8} potential {mb:6.1f} GPa   reference {xb:6.1f} GPa")
        print(f"  ratio: potential {ma / mb:.2f}   reference {xa / xb:.2f}")
        if (ma > mb) != (xa > xb):
            print("\n  *** The potential INVERTS which material is stiffer. ***")
            print("  Strain partitioning at a coherent interface is decided by a")
            print("  property this potential gets qualitatively wrong. Building the")
            print("  geometry differently cannot fix this -- a relaxed MD cell")
            print("  partitions strain by the potential's own C_ij. Report it, or")
            print("  change potential.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
