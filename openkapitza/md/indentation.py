"""Nanoindentation by MD -- COMPARE ONLY (work package 5d).

Governed by section 11 of the agent instruction. The result never fills the
Vickers microhardness coefficients c1, c2: those are fitted over indentation
diagonals of microns, while MD indents nanometres of perfect crystal. Its
legitimate uses are comparison with nanoindentation data, a check that the
potential yields at all at a sane H/E, and the SIZE-EFFECT DIRECTION:
Song-Yovanovich hardness rises as contacts shrink, so MD should come out
well above the microhardness. Below it, suspect the potential.

METHOD (3D extension of the official examples/indent minimisation variant)

A slab with a frozen bottom layer, periodic laterally, shrink-wrapped along z.
A rigid spherical indenter (`fix indent ... sphere`, harmonic repulsion
K (r - R)^2) is lowered in steps; at each depth the system is energy-minimised
with the indenter energy included (`fix_modify energy yes`), as in the
official example. This is quasi-static and athermal (0 K), which removes
thermal noise and strain-rate effects but also thermally activated
nucleation, so the yield point is an upper bound.

What the run gives is LOAD, not hardness. Two things are added here:

- a CONTACT-AREA MODEL. The geometric spherical cap, a^2 = 2 R h - h^2, using
  the depth below the ORIGINAL surface. It ignores pile-up and sink-in, so it
  is stated with the result rather than hidden in it.
- the FULL CURVE. Load is not monotonic once the crystal yields (the official
  example's own load goes up and down), so "the hardness" depends on where the
  curve is read. Reported: the whole P(h), the first pop-in, and the mean
  pressure over the post-yield part of the curve.

The elastic part before pop-in is fitted to Hertz, P = (4/3) E* R^1/2 h^3/2,
which gives an indentation modulus to check against the elastic constants of
work package 5a -- an internal consistency check of the run.
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Confidence, Provenance, Quantity
from .elastic import CubicSpec

EV_PER_A_TO_NN = 1.602176634          # 1 eV/A = 1.602 nN
EV_PER_A3_TO_GPA = 160.2176634        # 1 eV/A^3 = 160.2 GPa


@dataclass
class IndentSpec:
    solid: CubicSpec
    radius: float = 25.0              # Angstrom
    k_indent: float = 10.0            # eV/A^3, fix indent force constant
    lateral_cells: int = 16
    depth_cells: int = 12
    a0: float | None = None           # potential's own lattice; None: a0_guess
    max_depth: float = 8.0            # Angstrom below the original surface
    step: float = 0.2                 # Angstrom per minimisation
    etol: float = 1.0e-10
    ftol: float = 1.0e-8
    maxiter: int = 20000

    @property
    def n_steps(self) -> int:
        return int(round(self.max_depth / self.step))


DECK = """\
# OpenKapitza nanoindentation -- {name}, sphere R = {radius} A, quasi-static
# 3D extension of examples/indent (minimisation variant). Generated; see
# openkapitza/md/indentation.py.
units           metal
boundary        p p s
atom_style      atomic
lattice         {lattice} {a0:.6f}
region          box block 0 {n} 0 {n} 0 {nz}
create_box      1 box
create_atoms    1 box
mass            1 {mass:.4f}
pair_style      {pair_style}
{pair_coeff}

region          bottom block INF INF INF INF INF 0.9 units lattice
group           bottom region bottom
group           mobile subtract all bottom
fix             freeze bottom setforce 0.0 0.0 0.0
thermo          1000
thermo_style    custom step pe fmax
min_modify      dmax 0.1
minimize        {etol} {ftol} {maxiter} {maxeval}

variable        zsurf equal $(bound(all,zmax))
variable        cx equal $(0.5*lx+xlo)
variable        cy equal $(0.5*ly+ylo)
print           "INDENT_SURFACE ${{zsurf}} $(lx) $(ly) $(atoms)"

variable        k loop {n_steps}
variable        h equal v_k*{step}
variable        zc equal v_zsurf+{radius}-v_h
fix             ind mobile indent {k_indent} sphere v_cx v_cy v_zc {radius} units box
fix_modify      ind energy yes
label           step
minimize        {etol} {ftol} {maxiter} {maxeval}
print           "INDENT $(v_h:%.6f) $(f_ind[1]:%.8f) $(f_ind[2]:%.8f) $(f_ind[3]:%.8f) $(bound(mobile,zmax):%.6f)"
next            k
jump            SELF step
"""


def build_indent_deck(spec: IndentSpec) -> str:
    return DECK.format(
        name=spec.solid.name, radius=spec.radius, lattice=spec.solid.lattice,
        a0=spec.a0 or spec.solid.a0_guess, n=spec.lateral_cells, nz=spec.depth_cells,
        mass=spec.solid.mass, pair_style=spec.solid.pair_style,
        pair_coeff=spec.solid.pair_coeff(), etol=spec.etol, ftol=spec.ftol,
        maxiter=spec.maxiter, maxeval=10 * spec.maxiter, n_steps=spec.n_steps,
        step=spec.step, k_indent=spec.k_indent,
    )


_SURF = re.compile(r"^INDENT_SURFACE ([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+) (\d+)", re.MULTILINE)
_STEP = re.compile(r"^INDENT ([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+)",
                   re.MULTILINE)


def parse_indent_output(text: str) -> tuple[tuple[float, float, float, int], np.ndarray]:
    """((z_surface, lx, ly, atoms), rows of [h, fx, fy, fz, zmax])."""
    m = _SURF.search(text)
    if not m:
        raise ValueError("no INDENT_SURFACE line in the LAMMPS output")
    rows = np.array([[float(v) for v in r] for r in _STEP.findall(text)])
    if rows.size == 0:
        raise ValueError("no INDENT steps in the LAMMPS output")
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))), rows


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def cap_contact_radius(radius: float, depth: np.ndarray) -> np.ndarray:
    """Geometric spherical-cap contact radius, a^2 = 2 R h - h^2 (h <= R)."""
    h = np.clip(depth, 0.0, radius)
    return np.sqrt(np.maximum(2.0 * radius * h - h * h, 0.0))


def hertz_modulus(depth: np.ndarray, load: np.ndarray, radius: float) -> float:
    """Least-squares E* (eV/A^3) for P = (4/3) E* R^1/2 h^3/2 through the origin."""
    x = (4.0 / 3.0) * math.sqrt(radius) * np.power(np.clip(depth, 0.0, None), 1.5)
    denom = float(x @ x)
    if denom == 0.0:
        raise ValueError("no positive depths to fit")
    return float(x @ load) / denom


@dataclass
class IndentResult:
    depth: np.ndarray            # A, below the original surface
    load: np.ndarray             # nN
    contact_radius: np.ndarray   # A, cap model
    pressure: np.ndarray         # GPa, P / (pi a^2)
    popin_depth: float | None    # A, first load drop
    popin_load: float | None     # nN
    hardness: float              # GPa, mean pressure past pop-in
    hardness_spread: float       # GPa, std of that pressure
    reduced_modulus: float       # GPa, Hertz E* before pop-in
    lateral_ratio: float         # box side / final contact diameter
    meta: dict = field(default_factory=dict)


def analyse_curve(rows: np.ndarray, radius: float, lateral: float,
                  flags: FlagSet, *, drop_fraction: float = 0.05,
                  label: str = "") -> IndentResult:
    depth = rows[:, 0]
    # The force ON the indenter from the atoms points up (+z) when pushing in.
    load = np.abs(rows[:, 3]) * EV_PER_A_TO_NN
    a = cap_contact_radius(radius, depth)
    with np.errstate(divide="ignore", invalid="ignore"):
        pressure = np.where(a > 0, load / EV_PER_A_TO_NN / (math.pi * a * a), 0.0) \
            * EV_PER_A3_TO_GPA

    # First pop-in: the first step where load falls by more than drop_fraction.
    popin = None
    for i in range(1, load.size):
        if load[i - 1] > 0 and (load[i - 1] - load[i]) / load[i - 1] > drop_fraction:
            popin = i - 1
            break
    elastic = slice(0, popin + 1 if popin is not None else load.size)
    e_star = hertz_modulus(depth[elastic], load[elastic] / EV_PER_A_TO_NN, radius) \
        * EV_PER_A3_TO_GPA

    if popin is None:
        flags.add(
            Code.MD_INDENT_NOT_MICROHARDNESS, Severity.WARNING,
            f"{label}: no pop-in up to {depth[-1]:.1f} A; the crystal never yielded, "
            f"so no hardness can be read from this curve.",
            remedy="Indent deeper or use a sharper (smaller) indenter.",
        )
        post = np.zeros(0, dtype=bool)
    else:
        post = np.arange(load.size) > popin
    if post.sum() >= 2:
        hardness = float(pressure[post].mean())
        spread = float(pressure[post].std(ddof=1))
    else:
        hardness, spread = float("nan"), float("nan")
    lateral_ratio = lateral / (2.0 * a[-1]) if a[-1] > 0 else float("inf")
    if lateral_ratio < 3.0:
        flags.add(
            Code.MD_INDENT_NOT_MICROHARDNESS, Severity.CAUTION,
            f"{label}: box is only {lateral_ratio:.1f} contact diameters wide at the "
            f"deepest step; periodic images constrain the plastic zone.",
            remedy="Widen the slab or stop at a shallower depth.",
            lateral_ratio=lateral_ratio,
        )
    return IndentResult(
        depth, load, a, pressure,
        float(depth[popin]) if popin is not None else None,
        float(load[popin]) if popin is not None else None,
        hardness, spread, e_star, lateral_ratio,
    )


def compare_with_microhardness(result: IndentResult, flags: FlagSet, *,
                               microhardness_GPa: float | None = None,
                               nanoindentation_GPa: float | None = None,
                               label: str = "") -> None:
    """The flags section 11 requires whenever an MD hardness is shown."""
    flags.add(
        Code.MD_INDENT_NOT_MICROHARDNESS, Severity.INFO,
        f"{label}: MD hardness {result.hardness:.1f} GPa is a nanoindentation-scale "
        f"value (0 K, quasi-static, perfect crystal, cap-model area). It is never "
        f"used for the microhardness c1, c2.",
        remedy="Compare with nanoindentation at 10-100 nm, not with Vickers data.",
    )
    if microhardness_GPa is not None and math.isfinite(result.hardness) \
            and result.hardness < microhardness_GPa:
        flags.add(
            Code.MD_INDENT_SIZE_EFFECT_REVERSED, Severity.WARNING,
            f"{label}: MD hardness {result.hardness:.1f} GPa is BELOW the measured "
            f"microhardness {microhardness_GPa:.1f} GPa. Hardness should rise as "
            f"contacts shrink (c2 < 0), so nanometre MD should sit well above it.",
            remedy="Suspect the potential's plasticity; check H/E against literature.",
        )
    if nanoindentation_GPa is not None and math.isfinite(result.hardness):
        ratio = result.hardness / nanoindentation_GPa
        result.meta["ratio_to_nanoindentation"] = ratio


def to_quantity(result: IndentResult, *, potential: str = "") -> Quantity:
    """Hardness as an MD_COMPUTED candidate. The policy marks it not fillable."""
    note = (f"COMPARE ONLY; quasi-static 0 K sphere indentation, cap-model area, "
            f"pop-in at {result.popin_depth} A, post-yield spread "
            f"{result.hardness_spread:.1f} GPa")
    return Quantity(si=result.hardness * 1e9, unit="Pa",
                    uncertainty=(result.hardness_spread * 1e9
                                 if math.isfinite(result.hardness_spread) else None),
                    provenance=Provenance.MD_COMPUTED, confidence=Confidence.LOW,
                    citation=potential, note=note, name="nanohardness")


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def run_indentation(lmp: Path, spec: IndentSpec, potential_dir: Path,
                    flags: FlagSet, *, timeout: int = 14400) -> IndentResult:
    source = Path(potential_dir) / spec.solid.potential
    if not source.is_file():
        raise FileNotFoundError(f"potential not found: {source}")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / spec.solid.potential).write_bytes(source.read_bytes())
        (work / "indent.in").write_text(build_indent_deck(spec))
        run = subprocess.run([str(lmp), "-in", "indent.in"], cwd=work,
                             capture_output=True, text=True, timeout=timeout)
        if run.returncode != 0:
            errors = [l for l in run.stdout.splitlines() if "ERROR" in l]
            raise RuntimeError(f"LAMMPS failed: {errors or run.stdout[-400:]}")
    (z_surf, lx, ly, atoms), rows = parse_indent_output(run.stdout)
    result = analyse_curve(rows, spec.radius, min(lx, ly), flags,
                           label=f"{spec.solid.name} R={spec.radius:.0f} A")
    result.meta.update(solid=spec.solid.name, potential=spec.solid.potential,
                       radius=spec.radius, atoms=atoms, z_surface=z_surf)
    return result
