"""Build a Si(100) / a-SiO2 NEMD cell end to end (Phase 2).

    1. melt-quench a glass whose lateral size matches the Si supercell
       (or reuse one with --glass), and validate it;
    2. join it to a Si slab, optionally anneal the interface, report the
       bonding choices and the interface chemistry;
    3. write a ready NEMD directory (nemd.in, interface.data, potential)
       for the read-in deck.

Nothing about the glass is hidden: the validation table, the quench-rate
caveat and the interface-bonding flag are printed with the build.

Usage:
    python3 scripts/build_si_sio2.py --lmp <lmp> --potentials <dir> --out <dir>
        [--glass glass.data] [--quench-rate 1e13] [--bonding strong|intermediate|weak]
        [--bulk-nm 4] [--cells 5]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openkapitza.md.nemd import NEMDGeometry, NEMDSpec, build_input_deck  # noqa: E402
from openkapitza.schema import FlagSet  # noqa: E402
from openkapitza.structure.amorphous import (  # noqa: E402
    QuenchSpec, analyse_silica, judge, run_quench,
)
from openkapitza.structure.analysis import read_lammps_data  # noqa: E402
from openkapitza.structure.interface import InterfaceSpec, build_interface  # noqa: E402

SI_A0_SIO_TERSOFF = 5.4321      # A, measured: measure_potential_elastic.py --set sio


def nemd_spec(g: NEMDGeometry, production_ps: float = 500.0) -> NEMDSpec:
    """The read-in NEMD run for a Si/a-SiO2 cell built here (0.5 fs: light O)."""
    return NEMDSpec(geometry=g, element_a="Si", element_b="SiO2",
                    lattice_a=SI_A0_SIO_TERSOFF * 1e-10, lattice_b=SI_A0_SIO_TERSOFF * 1e-10,
                    temperature=300.0, structure_mode="read_data", data_file="interface.data",
                    pair_style="tersoff", pair_file="SiO.tersoff",
                    coeff_args="SiO.tersoff Si Si O", mass_a=28.0855, mass_b=28.0855,
                    timestep_fs=0.5, production_steps=int(production_ps * 2000),
                    equilibrate_steps=200_000, steady_steps=200_000, relax_steps=100_000,
                    profile_every=200, profile_frames=8,
                    types_a=(1,), types_b=(2, 3))     # crystal Si / glass Si + O


def geometry(bulk_nm: float, cells: int) -> NEMDGeometry:
    width = cells * SI_A0_SIO_TERSOFF
    return NEMDGeometry(wall_thickness=0.8e-9, reservoir_thickness=1.0e-9,
                        bulk_a_length=bulk_nm * 1e-9, bulk_b_length=bulk_nm * 1e-9,
                        cross_section=width * 1e-10, bin_width=0.4e-9)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lmp", required=True, type=Path)
    ap.add_argument("--potentials", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--glass", type=Path, help="reuse a quenched glass.data")
    ap.add_argument("--quench-rate", type=float, default=1e13)
    ap.add_argument("--bonding", choices=["strong", "intermediate", "weak"], default="strong",
                    help="interface bonding preset; the build reports what it reached")
    ap.add_argument("--bulk-nm", type=float, default=4.0)
    ap.add_argument("--cells", type=int, default=5, help="lateral Si cells")
    ap.add_argument("--production-ps", type=float, default=500.0)
    a = ap.parse_args(argv)

    flags = FlagSet()
    width = a.cells * SI_A0_SIO_TERSOFF
    g = geometry(a.bulk_nm, a.cells)
    oxide_len = (g.total_length - g.interface_z) * 1e10

    if a.glass:
        glass = read_lammps_data(a.glass)
        report = analyse_silica(glass)
        judge(report, QuenchSpec(quench_rate=a.quench_rate), flags)
    else:
        # Enough cristobalite cells along z that the glass covers the oxide side.
        nz = max(4, int(oxide_len / 7.16 * 1.05) + 1)
        qs = QuenchSpec(cells=(4, 4, nz), box=(width, width, oxide_len),
                        fix_lateral=True, quench_rate=a.quench_rate)
        q = run_quench(a.lmp, qs, a.potentials, a.out / "quench", flags)
        glass, report = q.cell, q.report
    print("a-SiO2 validation:")
    print(report.table())
    print("defects", {k: round(v, 4) for k, v in report.defects.items()})

    ispec = InterfaceSpec.from_preset(a.bonding, g, si_lattice=SI_A0_SIO_TERSOFF,
                                      lateral_cells=a.cells)
    res = build_interface(a.lmp, glass, ispec, a.potentials, a.out / "interface", flags)
    print(f"\njoin: {res.join.n_si_crystal} crystal Si + {res.join.n_glass} glass atoms; "
          f"cut crosses {res.join.broken_bonds} bonds ({res.join.broken_per_nm2:.1f}/nm^2)")
    print(f"interface: {res.chemistry.si_o_bonds_per_nm2:.1f} Si(c)-O bonds/nm^2 "
          f"({res.chemistry.coverage * 100:.0f} % oxidised, {res.chemistry.bonding_class}), "
          f"sub-oxides {res.chemistry.suboxide}, dangling O {res.chemistry.dangling_o}")

    nemd = a.out / "nemd"
    nemd.mkdir(parents=True, exist_ok=True)
    shutil.copy(res.data_file, nemd / "interface.data")
    shutil.copy(a.potentials / "SiO.tersoff", nemd / "SiO.tersoff")
    spec = nemd_spec(g, a.production_ps)
    (nemd / "nemd.in").write_text(build_input_deck(spec))
    print(f"\nNEMD directory ready: {nemd}")
    print(flags.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
