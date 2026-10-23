"""Build a crystalline A(hkl)/B(hkl) NEMD cell in any orientation (P2-1).

Finds the smallest rectangular commensurate supercell within the strain
tolerance (mismatch split between the two sides), cuts both slabs, sets the
gap from lateral registry, and writes a read-in NEMD directory:
interface.data + nemd.in (+ the potential files you name).

Materials are names from openkapitza.structure.crystalline.LIBRARY.

    python3 scripts/build_bicrystal.py --a Si --hkl-a 1 1 1 --b Al --hkl-b 1 1 1 \
        --max-strain 0.01 --pair-style meam --pair-file AlSiMgCuFe.meam \
        --coeff-args "library.meam Al Si Mg Cu Fe AlSiMgCuFe.meam Si Al" \
        --extra-files library.meam --potentials <dir> --out <dir>

Only the geometry is checked here; whether the potential describes the
pair is a separate question (see the elastic gate, measure_potential_elastic.py).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openkapitza.md.nemd import NEMDGeometry, NEMDSpec, build_input_deck  # noqa: E402
from openkapitza.structure.analysis import write_lammps_data  # noqa: E402
from openkapitza.structure.crystalline import LIBRARY, MASS, bicrystal_for_nemd  # noqa: E402


def _isotropic(c, hkl) -> bool:
    """Is the (hkl) surface elastically isotropic in-plane (couple x and y)?"""
    hkl = [int(x) for x in hkl]
    if abs(c.lattice[1] @ c.lattice[0]) > 1e-6:              # 120-degree basal cell
        return hkl[0] == hkl[1] == 0 and hkl[2] != 0         # basal (0001) only
    h = sorted(abs(x) for x in hkl)
    return h == [0, 0, 1] or (h[0] == h[1] == h[2] != 0)     # cubic {100}, {111}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, choices=sorted(LIBRARY))
    ap.add_argument("--b", required=True, choices=sorted(LIBRARY))
    ap.add_argument("--hkl-a", type=int, nargs=3, default=(1, 0, 0))
    ap.add_argument("--hkl-b", type=int, nargs=3, default=(1, 0, 0))
    ap.add_argument("--max-strain", type=float, default=0.02)
    ap.add_argument("--min-width", type=float, default=20.0, help="A, each lateral axis")
    ap.add_argument("--z-offset-a", type=float, default=0.0,
                    help="A: shift the cut through A to choose its top termination")
    ap.add_argument("--z-offset-b", type=float, default=0.0,
                    help="A: shift the cut through B to choose its bottom termination")
    ap.add_argument("--bulk-nm", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--timestep-fs", type=float, default=1.0)
    ap.add_argument("--production-ps", type=float, default=500.0)
    ap.add_argument("--plumbing", action="store_true",
                    help="tiny step counts: checks the deck runs end to end, measures nothing")
    ap.add_argument("--pair-style", default="tersoff")
    ap.add_argument("--pair-file", default="SiCGe.tersoff")
    ap.add_argument("--coeff-args", help="everything after 'pair_coeff * *'; required "
                    "when a side has more than one species")
    ap.add_argument("--extra-files", nargs="*", default=[])
    ap.add_argument("--potentials", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    A, B = LIBRARY[a.a](), LIBRARY[a.b]()
    g = NEMDGeometry(wall_thickness=0.8e-9, reservoir_thickness=1.0e-9,
                     bulk_a_length=a.bulk_nm * 1e-9, bulk_b_length=a.bulk_nm * 1e-9,
                     bin_width=0.4e-9)
    bc = bicrystal_for_nemd(A, tuple(a.hkl_a), B, tuple(a.hkl_b), g,
                            max_strain=a.max_strain, min_width=a.min_width,
                            z_offset_a=a.z_offset_a, z_offset_b=a.z_offset_b)
    r = bc.report
    multi = len(bc.types_a) > 1 or len(bc.types_b) > 1
    if multi and not a.coeff_args:
        ap.error("a side has several species: give --coeff-args with the type map "
                 f"{bc.type_species}")
    print(f"{A.name}{tuple(a.hkl_a)} / {B.name}{tuple(a.hkl_b)}")
    print(f"  matched cells A {tuple(round(x, 3) for x in r['cell_a_lengths'])}  "
          f"B {tuple(round(x, 3) for x in r['cell_b_lengths'])} A, repeat {r['repeat']}")
    print(f"  box {r['width'][0]:.3f} x {r['width'][1]:.3f} A, {r['atoms']} atoms")
    print(f"  strain A {tuple(round(100 * float(e), 2) for e in r['strain_a'])} %   "
          f"B {tuple(round(100 * float(e), 2) for e in r['strain_b'])} %")
    print(f"  terminations: A top {r['termination_a_top']}, B bottom {r['termination_b_bottom']}")
    print(f"  gap {bc.gap:.3f} A, closest A-B {r['closest_ab']:.3f} A, "
          f"interface {r['interface_offset']:+.2f} A from the geometry plane")
    print(f"  types {bc.type_species}")
    if not (_isotropic(A, a.hkl_a) and _isotropic(B, a.hkl_b)):
        print("  in-plane anisotropic surface: x and y relax independently (couple_xy off)")
    if multi:
        print("  NOTE: compound side(s): the terminations above are a CHOICE (polar faces "
              "differ); set them with --z-offset-a/-b and say which you ran.")

    a.out.mkdir(parents=True, exist_ok=True)
    write_lammps_data(bc.cell, a.out / "interface.data",
                      comment=f"OpenKapitza {A.name}{tuple(a.hkl_a)}/{B.name}{tuple(a.hkl_b)} "
                              f"types {bc.type_species}")
    spec = NEMDSpec(geometry=g, element_a=A.species[0], element_b=B.species[0],
                    lattice_a=float(np.linalg.norm(A.lattice[0])) * 1e-10,
                    lattice_b=float(np.linalg.norm(B.lattice[0])) * 1e-10, temperature=a.temperature,
                    structure_mode="read_data", data_file="interface.data",
                    pair_style=a.pair_style, pair_file=a.pair_file, coeff_args=a.coeff_args,
                    extra_files=tuple(a.extra_files), potential_dir=str(a.potentials) if a.potentials else None,
                    mass_a=MASS[A.species[0]], mass_b=MASS[B.species[0]],
                    timestep_fs=a.timestep_fs,
                    types_a=bc.types_a, types_b=bc.types_b,
                    couple_xy=_isotropic(A, a.hkl_a) and _isotropic(B, a.hkl_b),
                    **(dict(relax_steps=2000, equilibrate_steps=1000, steady_steps=1000,
                            production_steps=4000, profile_every=50, profile_frames=4)
                       if a.plumbing else
                       dict(production_steps=int(a.production_ps * 1000 / a.timestep_fs))))
    (a.out / "nemd.in").write_text(build_input_deck(spec))
    if a.potentials:
        for f in [a.pair_file, *a.extra_files]:
            shutil.copy(a.potentials / f, a.out / f)
    print(f"NEMD directory ready: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
