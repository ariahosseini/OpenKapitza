"""Interface bonding of a Si / a-SiO2 data file (types 1 = crystal Si,
2 = oxide Si, 3 = O): Si(crystal)-O bonds per nm^2, coverage against full
Si(100) oxidation, class, sub-oxides. Run it on the NEMD deck's
`relaxed.data` to know the class that was actually MEASURED, which can
differ from the class built (a weak cell's gap may close in relaxation).

Usage: python3 scripts/interface_chemistry.py <file.data> [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openkapitza.structure.analysis import read_lammps_data  # noqa: E402
from openkapitza.structure.interface import interface_chemistry  # noqa: E402


def main(argv=None) -> int:
    paths = (argv if argv is not None else sys.argv[1:])
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        c = read_lammps_data(p)
        c.periodic = (True, True, False)
        ch = interface_chemistry(c)
        sub = ", ".join(f"{k} {v * 100:.0f} %" for k, v in sorted(ch.suboxide.items()))
        print(f"{p}: {ch.si_o_bonds_per_nm2:.1f} Si(c)-O bonds/nm^2, "
              f"{ch.coverage * 100:.0f} % oxidised ({ch.bonding_class}); {sub}; "
              f"dangling O {ch.dangling_o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
