"""Measure a potential's OWN lattice constants and elastic constants.

The agent instruction gates Tier-2 machine-learned potentials on phonon DOS,
elastic constants and bulk conductivity before any production run. Tier-1
classical potentials carry no such gate -- and this script exists because that
turned out to matter.

For `SiCGe.tersoff`, measured here against experiment:

    Si(D)   a0  5.4320 A  (expt 5.431)   C11 142.5 (166)   C12 75.4 (63.9)
    Ge      a0  5.6567 A  (expt 5.658)   C11 138.5 (128)   C12 44.4 (48.3)

The lattice constants are excellent -- within 0.03 %. The elastic constants
are 8-18 % off, and the error is not a uniform scaling: it **inverts the
ordering**. Biaxial modulus M = C11 + C12 - 2 C12^2 / C11 comes out

    potential:   Si 138 GPa,  Ge 154 GPa   -> Ge is the stiffer
    experiment:  Si 180 GPa,  Ge 141 GPa   -> Si is the stiffer

So any result that depends on how strain PARTITIONS between the two materials
is being decided by a property the potential gets backwards. That includes the
relaxed shared in-plane constant of a coherent interface, which sets the strain
on both sides, and h_K is strain-sensitive at the factor-of-two level. It is
also exactly what the Phase-2 strain-minimising supercell will compute.

A potential can therefore reproduce the quantity everyone checks (the lattice
constant) to 0.03 % while getting the quantity that actually decides the answer
qualitatively wrong. Checking a0 alone is not a validation.

Method
------
Lattice constant: `fix box/relax` + minimise, so the box finds the potential's
own equilibrium rather than the value it was built from.

Elastic constants: impose exx by `change_box x scale`, hold y and z, relax
internal coordinates, and read the stress. Under uniaxial STRAIN,
sxx = C11*exx and syy = szz = C12*exx. Diamond structure has no internal
degree of freedom to relax under [100] strain, so this is exact at 0 K.
Strains of +/-0.4 % are small enough to stay linear and large enough to
measure; the fit uses five points including zero and reports its residual.

Usage:
    python3 scripts/measure_potential_elastic.py --lmp <path> --potentials <dir>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

STRAINS = (-0.004, -0.002, 0.0, 0.002, 0.004)
BAR_PER_GPA = 1e4


@dataclass
class Species:
    name: str
    label: str              # label AS IN the potential file
    mass: float
    a0_guess: float         # starting lattice constant, Angstrom
    a0_ref: float           # reference (experimental) a0
    c11_ref: float
    c12_ref: float


# Reference values are room-temperature experimental figures; C_ij vary only
# a few percent between 0 and 300 K for these materials, so comparing a 0 K
# calculation against them is fair to within the quoted precision.
DEFAULT_SPECIES = [
    Species("Si(D)", "Si(D)", 28.0855, 5.431, 5.431, 165.7, 63.9),
    Species("Ge", "Ge", 72.63, 5.658, 5.658, 128.5, 48.3),
]

DECK = """\
units           metal
boundary        p p p
atom_style      atomic
lattice         diamond {a0}
region          box block 0 4 0 4 0 4
create_box      1 box
create_atoms    1 box
mass            1 {mass}
pair_style      {pair_style}
pair_coeff      * * {potential} {label}

# The potential's OWN equilibrium, not the value the cell was built from.
fix             br all box/relax iso 0.0 vmax 0.001
minimize        1.0e-12 1.0e-12 100000 1000000
unfix           br
variable        a_eq equal lx/4
print           "MEASURED a_eq ${{a_eq}}"
{strain_block}
"""

STRAIN_BLOCK = """\
# Uniaxial STRAIN: scale x, hold y and z, relax internal coordinates only.
change_box      all x scale {scale} remap units box
minimize        1.0e-12 1.0e-12 100000 1000000
variable        sxx equal -pxx/{bar_per_gpa}
variable        syy equal -pyy/{bar_per_gpa}
print           "MEASURED stress {eps} ${{sxx}} ${{syy}}"
"""


def _slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope through the origin-centred data, plus max residual."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    worst = max(abs(y - (slope * x + intercept)) for x, y in zip(xs, ys))
    return slope, worst


def biaxial_modulus(c11: float, c12: float) -> float:
    """M for (001) biaxial strain: the stiffness that decides strain sharing."""
    return c11 + c12 - 2.0 * c12 * c12 / c11


def _run(lmp: Path, workdir: Path, deck: str, tag: str) -> str:
    path = workdir / f"{tag}.in"
    path.write_text(deck)
    result = subprocess.run(
        [str(lmp), "-in", path.name],
        cwd=workdir, capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        errors = [l for l in result.stdout.splitlines() if "ERROR" in l]
        raise RuntimeError(f"{tag} failed: {errors or result.stdout[-400:]}")
    return result.stdout


def measure(lmp: Path, potentials: Path, pair_style: str, potential: str,
            species: list[Species]) -> int:
    source = potentials / potential
    if not source.is_file():
        print(f"potential not found: {source}")
        return 2

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / potential).write_bytes(source.read_bytes())

        for sp in species:
            strain_block = "".join(
                STRAIN_BLOCK.format(scale=1.0 + eps, eps=eps,
                                    bar_per_gpa=BAR_PER_GPA)
                for eps in STRAINS if eps != 0.0
            )
            # Each strain needs a fresh relaxed cell, so run them separately.
            a_eq = None
            stresses: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
            for eps in STRAINS:
                if eps == 0.0:
                    block = ""
                else:
                    block = STRAIN_BLOCK.format(
                        scale=1.0 + eps, eps=eps, bar_per_gpa=BAR_PER_GPA)
                out = _run(lmp, workdir, DECK.format(
                    a0=sp.a0_guess, mass=sp.mass, pair_style=pair_style,
                    potential=potential, label=sp.label, strain_block=block,
                ), f"{sp.name.replace('(', '_').replace(')', '')}_{eps}")
                m = re.search(r"MEASURED a_eq ([\d.eE+-]+)", out)
                if m:
                    a_eq = float(m.group(1))
                m = re.search(
                    r"MEASURED stress ([\d.eE+-]+) ([\d.eE+-]+) ([\d.eE+-]+)", out)
                if m:
                    stresses.append(tuple(float(g) for g in m.groups()))

            eps_list = [s[0] for s in stresses]
            c11, r11 = _slope(eps_list, [s[1] for s in stresses])
            c12, r12 = _slope(eps_list, [s[2] for s in stresses])
            rows.append((sp, a_eq, c11, c12, max(r11, r12)))

    print("=" * 78)
    print(f"{potential} [{pair_style}] -- the potential's own properties at 0 K")
    print("=" * 78)
    print(f"{'species':8} {'a0':>9} {'a0 ref':>8} {'err':>7}  "
          f"{'C11':>7} {'ref':>6} {'err':>7}  {'C12':>7} {'ref':>6} {'err':>7}")
    for sp, a_eq, c11, c12, resid in rows:
        da = (a_eq - sp.a0_ref) / sp.a0_ref * 100.0
        d11 = (c11 - sp.c11_ref) / sp.c11_ref * 100.0
        d12 = (c12 - sp.c12_ref) / sp.c12_ref * 100.0
        print(f"{sp.name:8} {a_eq:9.4f} {sp.a0_ref:8.3f} {da:+6.2f}%  "
              f"{c11:7.1f} {sp.c11_ref:6.1f} {d11:+6.1f}%  "
              f"{c12:7.1f} {sp.c12_ref:6.1f} {d12:+6.1f}%")
        if resid > 0.01:
            print(f"{'':8} (worst fit residual {resid:.4f} GPa -- "
                  f"check linearity)")

    if len(rows) == 2:
        print("\nBiaxial modulus M = C11 + C12 - 2 C12^2 / C11")
        print("This is the stiffness that decides how strain SHARES between two")
        print("coherently bonded materials, and so the relaxed in-plane constant.")
        (sa, _, c11a, c12a, _), (sb, _, c11b, c12b, _) = rows
        ma, mb = biaxial_modulus(c11a, c12a), biaxial_modulus(c11b, c12b)
        ra = biaxial_modulus(sa.c11_ref, sa.c12_ref)
        rb = biaxial_modulus(sb.c11_ref, sb.c12_ref)
        print(f"  {sa.name:8} potential {ma:6.1f} GPa   reference {ra:6.1f} GPa")
        print(f"  {sb.name:8} potential {mb:6.1f} GPa   reference {rb:6.1f} GPa")
        print(f"  ratio {sa.name}/{sb.name}: potential {ma / mb:.2f}   "
              f"reference {ra / rb:.2f}")
        if (ma > mb) != (ra > rb):
            print(f"\n  *** The potential INVERTS which material is stiffer. ***")
            print(f"  Strain partitioning, and therefore the relaxed in-plane")
            print(f"  constant of a coherent interface, is decided by a property")
            print(f"  this potential gets qualitatively wrong. Treat any")
            print(f"  strain-dependent result from it as suspect. Building the")
            print(f"  geometry differently cannot fix this -- a relaxed MD cell")
            print(f"  partitions strain by the potential's own C_ij whatever")
            print(f"  the starting structure. Report it, or change potential.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", required=True, type=Path)
    parser.add_argument("--potentials", required=True, type=Path)
    parser.add_argument("--pair-style", default="tersoff")
    parser.add_argument("--potential", default="SiCGe.tersoff")
    args = parser.parse_args(argv)
    try:
        return measure(args.lmp, args.potentials, args.pair_style,
                       args.potential, DEFAULT_SPECIES)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"measurement failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
