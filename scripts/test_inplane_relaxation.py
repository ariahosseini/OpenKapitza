"""Open issue P1-2: does clamping z explain the relaxed in-plane constant?

The NEMD cell builds both sides on ONE lattice constant (Si's) in all three
directions, then freezes the ends. The Ge half is therefore compressed along z
with nowhere to go, the bilayer carries sigma_zz, and Poisson coupling pushes
the shared in-plane constant toward Ge.

Linear elasticity with the potential's own constants (5a) predicts:
    sigma_zz = 0 (free)        a_par 51.8 % of the way from Si to Ge
    total z length clamped     a_par 63.4 %, sigma_zz -1.63 GPa
against the NEMD cell's 73.3 % and Pzz +1.51 GPa (LAMMPS: positive = compressive).

This script relaxes the same bilayer at 0 K, periodic in z (so no walls and
two interfaces), both ways: z free, and z held at the built length. If the
clamp is the cause, the two land near those two predictions.

Usage:
    python3 scripts/test_inplane_relaxation.py --lmp <lmp> --potentials <dir>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

A_SI, A_GE = 5.4320, 5.6567          # the potential's own a0 (5a)

DECK = """\
units           metal
boundary        p p p
atom_style      atomic
lattice         diamond 5.431
region          box block 0 {nx} 0 {nx} 0 {nz}
create_box      2 box
region          ge block INF INF INF INF {nsi} INF units lattice
create_atoms    1 box
group           ge region ge
set             group ge type 2
mass            1 28.0855
mass            2 72.63
pair_style      tersoff
pair_coeff      * * SiCGe.tersoff Si(D) Ge
thermo_style    custom step pe press pxx pyy pzz lx ly lz
{relax}
"""

RELAX_0K = """\
fix             br all box/relax {dims} couple xy vmax 0.001
min_modify      line quadratic
minimize        1.0e-14 1.0e-12 50000 500000
print           "RESULT $(lx/{nx}:%.6f) $(lz:%.4f) $(pxx/1e4:%.4f) $(pzz/1e4:%.4f)"
"""

# Finite T: NPT on the free dimensions, averaged over the second half.
RELAX_T = """\
minimize        1.0e-8 1.0e-10 10000 100000
# minimize advances the step counter; without this the ave/time window never
# lands on an output step and f_av reads as zero (first run did exactly that).
reset_timestep  0
velocity        all create {t2:.1f} 4928 mom yes rot yes dist gaussian
timestep        0.001
fix             md all npt temp {t:.1f} {t:.1f} 0.1 {npt_dims} couple xy
thermo          1000
run             {half}
variable        ap equal lx/{nx}
variable        lzv equal lz
variable        px equal pxx/1e4
variable        pz equal pzz/1e4
fix             av all ave/time 10 {nrep} {half} v_ap v_lzv v_px v_pz
run             {half}
print           "RESULT $(f_av[1]:%.6f) $(f_av[2]:%.4f) $(f_av[3]:%.4f) $(f_av[4]:%.4f)"
"""


def run(lmp: Path, pot: Path, dims: str, nx: int, nsi: int, nge: int,
        temperature: float = 0.0, ps: float = 60.0) -> tuple[float, ...]:
    with tempfile.TemporaryDirectory() as d:
        w = Path(d)
        (w / pot.name).write_bytes(pot.read_bytes())
        if temperature > 0:
            half = int(ps * 500)
            npt_dims = " ".join(f"{d} 0.0 0.0 1.0" for d in dims.split()[::2])
            relax = RELAX_T.format(t=temperature, t2=2 * temperature, npt_dims=npt_dims,
                                   nx=nx, half=half, nrep=half // 10)
        else:
            relax = RELAX_0K.format(dims=dims, nx=nx)
        (w / "in.relax").write_text(DECK.format(nx=nx, nz=nsi + nge, nsi=nsi, relax=relax))
        out = subprocess.run([str(lmp), "-in", "in.relax"], cwd=w, capture_output=True,
                             text=True, timeout=3600).stdout
    m = re.search(r"^RESULT (\S+) (\S+) (\S+) (\S+)", out, re.MULTILINE)
    if not m:
        raise RuntimeError(out[-600:])
    return tuple(float(v) for v in m.groups())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lmp", required=True, type=Path)
    ap.add_argument("--potentials", required=True, type=Path)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--layers", type=int, default=15, help="unit cells per side along z")
    ap.add_argument("--temperature", type=float, default=0.0, help="K; 0 = static")
    ap.add_argument("--ps", type=float, default=60.0, help="NPT length at finite T")
    a = ap.parse_args(argv)
    pot = a.potentials / "SiCGe.tersoff"
    print(f"T = {a.temperature:.0f} K")
    print(f"{'case':22} {'a_par':>8} {'toward Ge':>10} {'Lz':>9} {'Pxx GPa':>8} {'Pzz GPa':>8}")
    for name, dims in (("z free (sigma_zz=0)", "x 0.0 y 0.0 z 0.0"),
                       ("z clamped at build", "x 0.0 y 0.0")):
        ap_, lz, pxx, pzz = run(a.lmp, pot, dims, a.nx, a.layers, a.layers,
                                a.temperature, a.ps)
        frac = (ap_ - A_SI) / (A_GE - A_SI)
        print(f"{name:22} {ap_:8.4f} {frac * 100:9.1f}% {lz:9.3f} {pxx:8.3f} {pzz:8.3f}")
    print("\nLinear-elastic predictions: free 51.8 %, clamped 63.4 % (sigma_zz -1.63 GPa).")
    print("NEMD cell (walls, 300 K):   73.3 %, Pzz +1.51 GPa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
