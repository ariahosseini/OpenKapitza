"""Cubic elastic constants from MD, and the isotropic E and nu the continuum needs.

Work package 5a, governed by section 11 of the agent instruction. Three jobs:

1. Measure a potential's own a0, C11, C12 and C44 at 0 K by finite strain,
   using the method of the official `examples/ELASTIC`: relax the box, impose a
   deformation with `change_box ... remap`, relax the atoms with the box held,
   and read the stress. Central differences at +/- strain cancel any residual
   stress left by the box relaxation.
2. Convert to isotropic E and nu by the Voigt-Reuss-Hill average, because the
   continuum model (the plasticity index, the elastic branch, the area fraction)
   takes a single E and nu per material, not a tensor.
3. Gate the potential against reference single-crystal constants, which decides
   the confidence cap any MD fill of E or nu may carry (`fill_policy.cap_md`).

Why C44 needs care. C11 and C12 come from a uniaxial strain along [100], under
which diamond-structure crystals have no internal degree of freedom. A SHEAR
strain is different: the two FCC sublattices of the diamond structure move
relative to one another (Kleinman's internal relaxation), and the relaxed C44
is substantially below the unrelaxed one. The minimisation after straining is
therefore not a convergence nicety for C44 -- it is the physics.

Why two strain magnitudes. The official script names the check: the elastic
constants should be insensitive to the strain magnitude, or the minimisation in
the deformed cell has not converged (or the strain has left the linear regime).
The measurement here is repeated at two magnitudes and flagged when they
disagree. The earlier `scripts/measure_potential_elastic.py` used one set and
did not meet that bar.
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Confidence, Provenance, Quantity
from .fill_policy import cap_md

BAR_PER_GPA = 1.0e4

# Reference single-crystal constants, GPa, room temperature. Used ONLY to gate a
# potential and to compare; never fed to the continuum model from here.
REFERENCE_CUBIC: dict[str, dict] = {
    "Si": {"c11": 165.7, "c12": 63.9, "c44": 79.6, "a0": 5.431,
           "citation": "McSkimin & Andreatch (1964), J. Appl. Phys. 35, 2161"},
    "Ge": {"c11": 128.5, "c12": 48.3, "c44": 66.8, "a0": 5.658,
           "citation": "McSkimin & Andreatch (1963), J. Appl. Phys. 34, 651"},
    "Ni": {"c11": 246.5, "c12": 147.3, "c44": 124.7, "a0": 3.524,
           "citation": "Simmons & Wang (1971), single-crystal elastic constants"},
    # Room temperature, as tabulated in arXiv:1605.09237 (Table of references).
    "Al": {"c11": 107.3, "c12": 60.08, "c44": 28.3, "a0": 4.0495,
           "citation": "Vallin et al. (1964), J. Appl. Phys. 35, 1825"},
    "Cu": {"c11": 168.4, "c12": 121.4, "c44": 75.4, "a0": 3.6149,
           "citation": "Ledbetter & Naimon (1974), J. Phys. Chem. Ref. Data 3, 897"},
}

# A potential passes the elastic gate when every one of C11, C12, C44 is within
# this fraction of reference. A stated convention, not a law: SiCGe.tersoff
# Si(D) misses C11 by 14 % and C12 by 18 % and fails, which is the point.
GATE_TOLERANCE = 0.10

# Insensitivity criterion across strain magnitudes.
STRAIN_SPREAD_TOLERANCE = 0.01


# --------------------------------------------------------------------------
# Isotropic averaging
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class IsotropicModuli:
    bulk: float          # B, GPa
    shear_voigt: float   # G_V, GPa  (upper bound)
    shear_reuss: float   # G_R, GPa  (lower bound)
    shear: float         # G_H = (G_V + G_R) / 2, GPa
    youngs: float        # E, GPa
    poisson: float       # nu
    zener: float         # A = 2 C44 / (C11 - C12); 1 for an isotropic crystal

    @property
    def voigt_reuss_spread(self) -> float:
        """(G_V - G_R) / G_H: how much the averaging choice itself matters."""
        return (self.shear_voigt - self.shear_reuss) / self.shear


def born_stable(c11: float, c12: float, c44: float) -> bool:
    """Born mechanical-stability criteria for a cubic crystal."""
    return c11 - c12 > 0.0 and c11 + 2.0 * c12 > 0.0 and c44 > 0.0


def voigt_reuss_hill(c11: float, c12: float, c44: float) -> IsotropicModuli:
    """Isotropic moduli of a randomly oriented polycrystal of a cubic material.

    Voigt assumes uniform strain (an upper bound), Reuss uniform stress (a lower
    bound); Hill's average of the two is the conventional estimate. For an
    isotropic crystal (C44 = (C11 - C12)/2) the bounds coincide and E, nu come
    back exactly -- tests/test_md_elastic.py pins that round trip.
    """
    if not born_stable(c11, c12, c44):
        raise ValueError(
            f"constants are not mechanically stable (C11={c11}, C12={c12}, "
            f"C44={c44}); a measurement this wrong should not be averaged"
        )
    bulk = (c11 + 2.0 * c12) / 3.0
    shear_v = (c11 - c12 + 3.0 * c44) / 5.0
    shear_r = 5.0 * (c11 - c12) * c44 / (4.0 * c44 + 3.0 * (c11 - c12))
    shear = 0.5 * (shear_v + shear_r)
    youngs = 9.0 * bulk * shear / (3.0 * bulk + shear)
    poisson = (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))
    zener = 2.0 * c44 / (c11 - c12)
    return IsotropicModuli(bulk, shear_v, shear_r, shear, youngs, poisson, zener)


def biaxial_modulus(c11: float, c12: float) -> float:
    """M for (001) biaxial strain: the stiffness that decides strain sharing."""
    return c11 + c12 - 2.0 * c12 * c12 / c11


# --------------------------------------------------------------------------
# The MD measurement
# --------------------------------------------------------------------------

@dataclass
class CubicSpec:
    """One cubic crystal, as the potential sees it."""

    name: str                 # reporting name, e.g. "Si(D)" or "Ni"
    lattice: str              # "diamond", "fcc" or "bcc"
    a0_guess: float           # Angstrom; only a starting point
    mass: float               # amu
    pair_style: str           # "tersoff", "sw", "eam", "eam/alloy", ...
    potential: str            # file name, resolved beside the deck
    label: str | None = None  # element label AS IN the file; None for funcfl eam
    reference: str | None = None   # key into REFERENCE_CUBIC
    cells: int = 4
    # Multi-file potentials (MEAM: a library file plus a parameter file) list
    # every file to stage beside the deck, and give the full pair_coeff
    # argument list after "* *". `potential` stays the primary file.
    extra_files: tuple[str, ...] = ()
    coeff_args: str | None = None
    # Variable-charge potentials (COMB) need atom_style charge; an elemental
    # crystal is neutral, so charges are set to 0 and no equilibration is run.
    charge: bool = False

    @property
    def files(self) -> tuple[str, ...]:
        return (self.potential, *self.extra_files)

    def pair_coeff(self) -> str:
        if self.coeff_args:
            return f"pair_coeff      * * {self.coeff_args}"
        # `pair_style eam` (funcfl) takes the file per type pair and no label;
        # every many-body and setfl style takes `* * file label`.
        if self.pair_style == "eam":
            return f"pair_coeff      1 1 {self.potential}"
        if not self.label:
            raise ValueError(f"{self.name}: pair_style {self.pair_style} needs a label")
        return f"pair_coeff      * * {self.potential} {self.label}"


def stage_potential(spec: "CubicSpec", potential_dir: Path, work: Path) -> None:
    """Copy every file the potential needs into the run directory."""
    for name in spec.files:
        source = Path(potential_dir) / name
        if not source.is_file():
            raise FileNotFoundError(f"potential file not found: {source}")
        (work / name).write_bytes(source.read_bytes())


DECK = """\
# OpenKapitza elastic constants -- {name}, {component} strain {eps:+.5f}
# Generated; see openkapitza/md/elastic.py.
units           metal
boundary        p p p
atom_style      {atom_style}
lattice         {lattice} {a0_guess:.6f}
region          box block 0 {n} 0 {n} 0 {n}
create_box      1 box
create_atoms    1 box
{charge_line}mass            1 {mass:.4f}
pair_style      {pair_style}
{pair_coeff}
thermo_style    custom step pe press pxx pyy pzz pyz lx ly lz

# The potential's OWN equilibrium, not the value the cell was built from.
fix             br all box/relax iso 0.0 vmax 0.001
minimize        1.0e-14 1.0e-14 100000 1000000
unfix           br
variable        a_eq equal lx/{n}
print           "ELASTIC_A0 ${{a_eq}}"
{strain_block}
variable        sxx equal -pxx/{bar_per_gpa}
variable        syy equal -pyy/{bar_per_gpa}
variable        szz equal -pzz/{bar_per_gpa}
variable        syz equal -pyz/{bar_per_gpa}
print           "ELASTIC_STRESS {component} {eps:.8f} ${{sxx}} ${{syy}} ${{szz}} ${{syz}}"
"""

STRAIN_XX = """\
# Uniaxial STRAIN along x; y and z held. sxx = C11*e, syy = szz = C12*e.
change_box      all x scale {scale:.10f} remap units box
minimize        1.0e-14 1.0e-14 100000 1000000
"""

STRAIN_YZ = """\
# Engineering shear yz: tilt = gamma * lz, so syz = C44 * gamma.
# The minimisation is the physics here, not a nicety: diamond-structure
# crystals relax internally under shear (Kleinman), lowering C44.
change_box      all triclinic
variable        tilt equal {eps:.10f}*lz
change_box      all yz delta ${{tilt}} remap units box
minimize        1.0e-14 1.0e-14 100000 1000000
"""


# Finite temperature (open issue P5-2). The 0 K constants above are compared
# with ROOM-temperature references (McSkimin 1964 at 298 K, Ledbetter 1974);
# this path makes that comparison like with like. NPT equilibration with the
# box set to its mean (as in the NEMD relaxation), the strain applied to the
# thermal box, NVT, and the stress averaged after a clock reset so the
# averaging window lines up with the run (fix ave/time fires on multiples of
# its Nfreq).
DECK_T = """\
# OpenKapitza elastic constants at {temperature:g} K -- {name}, {component} strain {eps:+.5f}
units           metal
boundary        p p p
atom_style      {atom_style}
lattice         {lattice} {a0_guess:.6f}
region          box block 0 {n} 0 {n} 0 {n}
create_box      1 box
create_atoms    1 box
{charge_line}mass            1 {mass:.4f}
pair_style      {pair_style}
{pair_coeff}
timestep        0.001
velocity        all create {t_init:.2f} {seed} mom yes rot yes dist gaussian
fix             eq all npt temp {temperature:.2f} {temperature:.2f} 0.1 iso 0.0 0.0 1.0
variable        lxv equal lx
fix             lav all ave/time 10 {nrep_eq} {n_eq} v_lxv
run             {n_eq}
unfix           eq
change_box      all x final $(xlo) $(xlo+f_lav) y final $(ylo) $(ylo+f_lav) z final $(zlo) $(zlo+f_lav) remap units box
unfix           lav
variable        a_eq equal lx/{n}
print           "ELASTIC_A0 ${{a_eq}}"
{strain_block}
fix             nvt all nvt temp {temperature:.2f} {temperature:.2f} 0.1
run             {n_settle}
reset_timestep  0
variable        sxx equal -pxx/{bar_per_gpa}
variable        syy equal -pyy/{bar_per_gpa}
variable        szz equal -pzz/{bar_per_gpa}
variable        syz equal -pyz/{bar_per_gpa}
fix             sav all ave/time 10 {nrep_avg} {n_avg} v_sxx v_syy v_szz v_syz
thermo          {n_avg}
run             {n_avg}
print           "ELASTIC_STRESS {component} {eps:.8f} $(f_sav[1]) $(f_sav[2]) $(f_sav[3]) $(f_sav[4])"
"""

STRAIN_XX_T = """\
change_box      all x scale {scale:.10f} remap units box
"""

STRAIN_YZ_T = """\
change_box      all triclinic
variable        tilt equal {eps:.10f}*lz
change_box      all yz delta ${{tilt}} remap units box
"""


def build_elastic_deck(spec: CubicSpec, component: str, eps: float, *,
                       temperature: float | None = None, equil_ps: float = 20.0,
                       settle_ps: float = 5.0, average_ps: float = 40.0,
                       seed: int = 4928) -> str:
    """One self-contained run: relax, strain, relax atoms, print stress.

    With `temperature`, the finite-T path (DECK_T) instead.
    """
    if temperature is not None:
        blocks = {"xx": STRAIN_XX_T.format(scale=1.0 + eps), "yz": STRAIN_YZ_T.format(eps=eps),
                  "none": ""}
        if component not in blocks:
            raise ValueError(f"unknown strain component {component!r}")
        n_eq, n_avg = int(equil_ps * 1000), int(average_ps * 1000)
        n_eq -= n_eq % 20; n_avg -= n_avg % 10
        return DECK_T.format(
            temperature=temperature, t_init=2.0 * temperature, seed=seed,
            name=spec.name, component=component, eps=eps, lattice=spec.lattice,
            a0_guess=spec.a0_guess, n=spec.cells, mass=spec.mass,
            pair_style=spec.pair_style, pair_coeff=spec.pair_coeff(),
            strain_block=blocks[component], bar_per_gpa=BAR_PER_GPA,
            atom_style="charge" if spec.charge else "atomic",
            charge_line="set             group all charge 0.0\n" if spec.charge else "",
            n_eq=n_eq, nrep_eq=n_eq // 20, n_settle=int(settle_ps * 1000),
            n_avg=n_avg, nrep_avg=n_avg // 10)
    if component == "xx":
        block = STRAIN_XX.format(scale=1.0 + eps)
    elif component == "yz":
        block = STRAIN_YZ.format(eps=eps)
    elif component == "none":
        block = ""
    else:
        raise ValueError(f"unknown strain component {component!r}")
    return DECK.format(
        name=spec.name, component=component, eps=eps, lattice=spec.lattice,
        a0_guess=spec.a0_guess, n=spec.cells, mass=spec.mass,
        pair_style=spec.pair_style, pair_coeff=spec.pair_coeff(),
        strain_block=block, bar_per_gpa=BAR_PER_GPA,
        atom_style="charge" if spec.charge else "atomic",
        charge_line="set             group all charge 0.0\n" if spec.charge else "",
    )


_A0 = re.compile(r"^ELASTIC_A0\s+([-\d.eE+]+)", re.MULTILINE)
_STRESS = re.compile(
    r"^ELASTIC_STRESS\s+(\w+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)"
    r"\s+([-\d.eE+]+)\s+([-\d.eE+]+)", re.MULTILINE)


def parse_elastic_output(text: str) -> tuple[float | None, list[tuple]]:
    """(a0, [(component, eps, sxx, syy, szz, syz), ...]) from LAMMPS stdout."""
    m = _A0.search(text)
    a0 = float(m.group(1)) if m else None
    rows = [(g[0], *(float(x) for x in g[1:])) for g in _STRESS.findall(text)]
    return a0, rows


@dataclass
class ElasticResult:
    spec_name: str
    a0: float
    c11: float
    c12: float
    c44: float
    by_strain: dict[float, tuple[float, float, float]] = field(default_factory=dict)
    potential: str = ""

    @property
    def spread(self) -> float:
        """Largest relative disagreement of any C_ij across strain magnitudes."""
        if len(self.by_strain) < 2:
            return float("nan")
        worst = 0.0
        values = list(self.by_strain.values())
        for i in range(3):
            column = [v[i] for v in values]
            ref = abs(sum(column) / len(column))
            if ref > 0.0:
                worst = max(worst, (max(column) - min(column)) / ref)
        return worst

    @property
    def isotropic(self) -> IsotropicModuli:
        return voigt_reuss_hill(self.c11, self.c12, self.c44)


def constants_from_rows(rows: list[tuple], magnitude: float) -> tuple[float, float, float]:
    """Central differences at +/- magnitude: (C11, C12, C44) in GPa."""
    def pick(component: str, sign: float) -> tuple:
        target = sign * magnitude
        for row in rows:
            if row[0] == component and math.isclose(row[1], target, rel_tol=1e-6):
                return row
        raise ValueError(f"missing {component} strain {target:+g}")

    xp, xm = pick("xx", 1.0), pick("xx", -1.0)
    yp, ym = pick("yz", 1.0), pick("yz", -1.0)
    c11 = (xp[2] - xm[2]) / (2.0 * magnitude)
    # C12 from both lateral components, averaged: they must agree by symmetry.
    c12 = 0.5 * ((xp[3] - xm[3]) + (xp[4] - xm[4])) / (2.0 * magnitude)
    c44 = (yp[5] - ym[5]) / (2.0 * magnitude)
    return c11, c12, c44


def assemble(spec: CubicSpec, a0: float, rows: list[tuple],
             magnitudes: tuple[float, ...], flags: FlagSet, *,
             spread_tolerance: float = STRAIN_SPREAD_TOLERANCE) -> ElasticResult:
    """Turn parsed stresses into constants, and run the insensitivity check."""
    by_strain = {m: constants_from_rows(rows, m) for m in magnitudes}
    # Report the SMALLEST magnitude: closest to the linear limit. The larger one
    # exists to test that the smallest is trustworthy.
    smallest = min(magnitudes)
    c11, c12, c44 = by_strain[smallest]
    result = ElasticResult(spec.name, a0, c11, c12, c44, by_strain,
                           potential=spec.potential)

    if len(magnitudes) >= 2 and result.spread > spread_tolerance:
        flags.add(
            Code.MD_ELASTIC_UNCONVERGED,
            Severity.WARNING,
            f"{spec.name}: elastic constants move by {result.spread * 100:.1f} % "
            f"between strain magnitudes {sorted(magnitudes)}.",
            remedy=(
                "The official examples/ELASTIC names this test: the constants "
                "should be insensitive to the strain magnitude. Tighten the "
                "minimisation, or shrink the strain if the larger one has left "
                "the linear regime."
            ),
            spread=result.spread,
        )
    if not born_stable(c11, c12, c44):
        flags.add(
            Code.MD_ELASTIC_UNCONVERGED,
            Severity.CRITICAL,
            f"{spec.name}: measured constants violate Born stability "
            f"(C11={c11:.1f}, C12={c12:.1f}, C44={c44:.1f} GPa).",
            remedy="The measurement or the potential is broken; do not use it.",
        )
    return result


def run_elastic(
    lmp: Path,
    spec: CubicSpec,
    potential_dir: Path,
    flags: FlagSet,
    *,
    magnitudes: tuple[float, ...] | None = None,
    timeout: int = 1800,
    temperature: float | None = None,
) -> ElasticResult:
    """Run every strain on a real engine and return the constants.

    `temperature` (K) runs the finite-T path: larger default strains (stress
    noise) and a 5 % strain-insensitivity tolerance instead of 1 %.
    """
    if magnitudes is None:
        magnitudes = (0.005, 0.01) if temperature else (0.002, 0.005)
    rows: list[tuple] = []
    a0: float | None = None
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        stage_potential(spec, potential_dir, work)
        for component in ("xx", "yz"):
            for m in magnitudes:
                for sign in (1.0, -1.0):
                    eps = sign * m
                    deck = work / f"el_{component}_{eps:+.5f}.in"
                    deck.write_text(build_elastic_deck(spec, component, eps,
                                                       temperature=temperature))
                    run = subprocess.run(
                        [str(lmp), "-in", deck.name], cwd=work,
                        capture_output=True, text=True, timeout=timeout,
                    )
                    if run.returncode != 0:
                        errors = [l for l in run.stdout.splitlines() if "ERROR" in l]
                        raise RuntimeError(
                            f"{spec.name} {component} {eps:+g}: "
                            f"{errors or run.stdout[-300:]}")
                    run_a0, run_rows = parse_elastic_output(run.stdout)
                    a0 = a0 if a0 is not None else run_a0
                    rows.extend(run_rows)
    if a0 is None:
        raise RuntimeError(f"{spec.name}: no lattice constant reported")
    return assemble(spec, a0, rows, magnitudes, flags,
                    spread_tolerance=0.05 if temperature else STRAIN_SPREAD_TOLERANCE)


# --------------------------------------------------------------------------
# Gate and hand-off to the continuum model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GateReport:
    passes: bool
    errors: dict[str, float]     # relative error per constant
    reference: str
    citation: str


def gate(result: ElasticResult, reference_key: str,
         tolerance: float = GATE_TOLERANCE) -> GateReport:
    """Compare against reference single-crystal constants."""
    ref = REFERENCE_CUBIC[reference_key]
    errors = {
        "a0": (result.a0 - ref["a0"]) / ref["a0"],
        "c11": (result.c11 - ref["c11"]) / ref["c11"],
        "c12": (result.c12 - ref["c12"]) / ref["c12"],
        "c44": (result.c44 - ref["c44"]) / ref["c44"],
    }
    passes = all(abs(errors[k]) <= tolerance for k in ("c11", "c12", "c44"))
    return GateReport(passes, errors, reference_key, ref["citation"])


def to_quantities(result: ElasticResult, *,
                  potential_passes_gate: bool | None) -> dict[str, Quantity]:
    """E and nu as MD_COMPUTED quantities, capped by the section-11 policy.

    The cap is applied HERE, at the boundary where an MD number becomes a
    candidate input, so no caller can forget it.
    """
    iso = result.isotropic
    note = (f"0 K static, Voigt-Reuss-Hill from C11={result.c11:.1f}, "
            f"C12={result.c12:.1f}, C44={result.c44:.1f} GPa; "
            f"VR spread {iso.voigt_reuss_spread * 100:.0f} %")
    common = dict(provenance=Provenance.MD_COMPUTED, confidence=Confidence.HIGH,
                  citation=result.potential, note=note)
    youngs = Quantity.parse(iso.youngs, "GPa", name=f"E_{result.spec_name}",
                            **common)
    poisson = Quantity.parse(iso.poisson, "-", name=f"nu_{result.spec_name}",
                             **common)
    return {
        "E": cap_md("elastic", youngs, potential_passes_gate=potential_passes_gate),
        "nu": cap_md("elastic", poisson, potential_passes_gate=potential_passes_gate),
    }
