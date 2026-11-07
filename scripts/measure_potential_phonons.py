"""Phonon (VDOS) part of the potential gate: optical and TA peaks vs measured (KB 37).

    python3 scripts/measure_potential_phonons.py --lmp LMP --potentials DIR [--set sicge|sw]

Each crystal: relax, equilibrate at 300 K, NVE with velocities dumped, DOS
from the velocity power spectrum (md/phonon_dos.py). Prints the peaks, their
errors against the reference and PASS/FAIL (optical within 5 %, TA within
10 %). Seconds per crystal on a laptop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openkapitza.md.elastic import CubicSpec                      # noqa: E402
from openkapitza.md.phonon_dos import VDOSSpec, gate, peaks, run_vdos   # noqa: E402

SETS = {
    "sicge": [(CubicSpec("Si(D)", "diamond", 5.431, 28.0855, "tersoff", "SiCGe.tersoff",
                         "Si(D)", "Si"), "Si"),
              (CubicSpec("Ge", "diamond", 5.658, 72.63, "tersoff", "SiCGe.tersoff",
                         "Ge", "Ge"), "Ge")],
    "sw": [(CubicSpec("Si", "diamond", 5.431, 28.0855, "sw", "Si.sw", "Si", "Si"), "Si")],
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmp", required=True)
    ap.add_argument("--potentials", type=Path, required=True)
    ap.add_argument("--set", default="sicge", choices=sorted(SETS))
    a = ap.parse_args(argv)
    for crystal, ref in SETS[a.set]:
        f, g = run_vdos(a.lmp, VDOSSpec(crystal), a.potentials)
        r = gate(peaks(f, g), ref)
        print(f"{crystal.name:8} {crystal.potential:16} optical {r.peaks['optical']:5.2f} THz "
              f"({r.errors['optical']:+.1%})  TA {r.peaks['ta']:4.2f} THz "
              f"({r.errors['ta']:+.1%})  {'PASS' if r.passes else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
