"""Crystal / amorphous interface for NEMD: Si(100) joined to a-SiO2 (Phase 2).

Section 3 of the agent instruction: report the interface bonding choices
explicitly, because they change h_K more than most other decisions. This
module makes each one a named, recorded parameter rather than a side effect:

  * where the periodic glass is CUT. Every Si-O bond crossing the cut plane
    becomes a dangling bond at the interface. The cut is placed at the height
    crossed by the fewest bonds, and the number left per nm^2 is reported.
  * the GAP between the top Si plane and the lowest glass atom.
  * whether the interface is ANNEALED (a hot band around it lets Si(crystal)-O
    bonds form, then cools) or left as joined and only relaxed.
  * the resulting interface chemistry: Si(crystal)-O bonds per nm^2 and the
    sub-oxide distribution of the top crystal Si (Si+1, Si+2, Si+3 by number
    of O neighbours).

Atom types in the data file: 1 = crystal Si (side A), 2 = oxide Si, 3 = O
(side B). The NEMD deck takes it with structure_mode="read_data" and
coeff_args="SiO.tersoff Si Si O".
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..md.nemd import NEMDGeometry
from ..schema.flags import Code, FlagSet, Severity
from .analysis import Cell, pair_distances, read_lammps_data, write_lammps_data
from .crystals import MASS, diamond

SI_C, SI_OX, O_T = 1, 2, 3
SI_O_BOND = 2.0               # A, bond cutoff for counting


# Si(100): 2 surface atoms per a0^2 cell = 6.78 per nm^2. In a fully oxidised
# interface each bonds to two O, so ~13.6 Si(crystal)-O bonds per nm^2.
SI100_SITES_PER_NM2 = 2.0 / (0.5431 ** 2)
FULL_BONDS_PER_NM2 = 2.0 * SI100_SITES_PER_NM2

# Interface bonding classes (Lampin et al. 2012, arXiv:1210.0354: strongly
# coupled 0.83e-9 m^2K/W vs weakly coupled 4.27e-9 m^2K/W at 300 K, same
# potential and quench rate). Presets aim at a class; the build reports the
# coverage it actually reached and flags a miss.
# Measured on a 1e14 K/s test glass: annealing alone (1.0 A gap, 2000 K
# oxide-side band, 50 ps) reached only 35 % oxidation -- a stoichiometric glass
# surface cannot supply two O per surface Si where they are needed. "strong"
# therefore O-caps the Si surface before joining (an idealised, abrupt, fully
# bridged interface: every top Si is Si+2); "intermediate" is anneal-only.
_HOT = {"anneal": True, "anneal_T": 2000.0, "anneal_ps": 50.0, "cool_ps": 50.0,
        "anneal_band": 6.0, "anneal_band_si": 1.0}
BONDING_PRESETS = {
    "weak": {"gap": 2.5, "anneal": False},
    "intermediate": {"gap": 1.0, **_HOT},
    "strong": {"gap": 1.6, "o_terminate": True, **_HOT},
}


def bonding_class(coverage: float) -> str:
    return "weak" if coverage < 0.3 else ("strong" if coverage > 0.7 else "intermediate")


@dataclass
class InterfaceSpec:
    geometry: NEMDGeometry
    si_lattice: float = 5.431      # A, the potential's own a0 for Si
    lateral_cells: int = 5
    gap: float = 1.6               # A, top Si plane to lowest glass atom
    headroom: float = 10.0         # A of empty box above the cell, for the piston
    anneal: bool = True
    anneal_T: float = 1200.0       # K, hot band around the interface
    anneal_band: float = 6.0       # A above the interface (oxide side)
    anneal_band_si: float | None = None   # A below it (crystal side); None: = anneal_band
    anneal_ps: float = 20.0
    cool_ps: float = 20.0
    timestep_fs: float = 0.5
    seed: int = 5531
    potential: str = "SiO.tersoff"

    o_terminate: bool = False      # cap each top Si's two upward bonds with O
    bonding: str | None = None     # requested class, set by from_preset

    @property
    def width(self) -> float:
        return self.lateral_cells * self.si_lattice

    @classmethod
    def from_preset(cls, name: str, geometry: NEMDGeometry, **kw) -> "InterfaceSpec":
        if name not in BONDING_PRESETS:
            raise ValueError(f"unknown bonding preset {name!r}; use {sorted(BONDING_PRESETS)}")
        return cls(geometry, bonding=name, **{**BONDING_PRESETS[name], **kw})


def _bonds(glass: Cell, si_type: int, o_type: int) -> tuple[np.ndarray, np.ndarray]:
    """(z of the Si end, minimum-image dz to the O end) for every Si-O bond."""
    si = np.nonzero(glass.types == si_type)[0]
    o = np.nonzero(glass.types == o_type)[0]
    i, j, _ = pair_distances(glass, si, o, SI_O_BOND)
    Lz = glass.lengths[2]
    dz = glass.pos[j, 2] - glass.pos[i, 2]
    dz -= Lz * np.round(dz / Lz)
    return glass.pos[i, 2], dz


def _crossings(z_si: np.ndarray, dz: np.ndarray, z_cut: np.ndarray, Lz: float) -> np.ndarray:
    # With z measured from the cut (wrapped into [0, Lz)), a bond crosses the
    # plane exactly when its other end falls outside [0, Lz).
    zi = (z_si[None, :] - z_cut[:, None]) % Lz
    end = zi + dz[None, :]
    return np.count_nonzero((end < 0) | (end >= Lz), axis=1)


def bonds_crossing(glass: Cell, z_cut: float, *, si_type: int = 1, o_type: int = 2) -> int:
    """Si-O bonds in the periodic glass that cross the plane z = z_cut."""
    z_si, dz = _bonds(glass, si_type, o_type)
    return int(_crossings(z_si, dz, np.array([z_cut]), glass.lengths[2])[0])


def best_cut(glass: Cell, *, step: float = 0.1, si_type: int = 1,
             o_type: int = 2) -> tuple[float, int]:
    """Height in the periodic glass crossed by the fewest Si-O bonds."""
    z_si, dz = _bonds(glass, si_type, o_type)
    zs = np.arange(glass.lo[2], glass.hi[2], step)
    counts = _crossings(z_si, dz, zs, glass.lengths[2])
    k = int(np.argmin(counts))
    return float(zs[k]), int(counts[k])


@dataclass
class JoinReport:
    n_si_crystal: int
    n_glass: int
    cut_z: float
    broken_bonds: int
    broken_per_nm2: float
    glass_repeats: int
    meta: dict = field(default_factory=dict)


CAP_HEIGHT = 0.9            # A above the top Si plane for a bridging O


def capping_oxygen(si: Cell, keep: np.ndarray, a: float, width: float) -> np.ndarray:
    """One bridging O per vacant upper-neighbour site of the top Si plane.

    On Si(100) every vacant site above the cut is the shared neighbour of TWO
    top Si atoms, so one O there bridges them (Si-O-Si) and each top Si still
    gets two O neighbours: full oxidation, every top Si is Si+2. (Capping each
    dangling bond separately, the first version, put two O ~1.2 A apart at
    every site.) The two Si are 3.84 A apart, so an in-plane bridge has
    Si-O >= 1.92 A; the O starts CAP_HEIGHT above the plane and the anneal
    relaxes it. The coverage measured afterwards is the judge.
    """
    top = si.pos[keep, 2].max()
    top_idx = np.nonzero(keep & (np.abs(si.pos[:, 2] - top) < 1e-6))[0]
    above = np.nonzero(~keep & (np.abs(si.pos[:, 2] - top - a / 4.0) < 1e-6))[0]
    nn = math.sqrt(3.0) / 4.0 * a
    sites = []
    for k in above:
        d = si.pos[top_idx] - si.pos[k]
        d[:, :2] -= width * np.round(d[:, :2] / width)
        if np.count_nonzero(np.abs(np.linalg.norm(d, axis=1) - nn) < 1e-3) >= 2:
            sites.append(si.pos[k])
    caps = np.asarray(sites)
    caps[:, 2] = top + CAP_HEIGHT
    caps[:, :2] %= width
    return caps


def join(glass: Cell, spec: InterfaceSpec, flags: FlagSet) -> tuple[Cell, JoinReport]:
    """Si(100) slab from z = 0 to the interface, glass above it, as one cell.

    The glass must already have the Si supercell's lateral size (quench it
    with QuenchSpec(box=..., fix_lateral=True)).
    """
    g = spec.geometry
    w = spec.width
    if np.any(np.abs(glass.lengths[:2] - w) > 1e-3):
        raise ValueError(f"glass lateral size {glass.lengths[:2]} must equal the Si "
                         f"supercell width {w:.4f} A")
    iz = g.interface_z * 1e10
    total = g.total_length * 1e10
    a = spec.si_lattice
    nz = int(math.ceil(iz / a)) + 1
    si = diamond(a, (spec.lateral_cells, spec.lateral_cells, nz), type_id=SI_C)
    keep = si.pos[:, 2] < iz - 1e-6
    si_pos = si.pos[keep]
    top_si = float(si_pos[:, 2].max())
    cap = np.zeros((0, 3))
    if spec.o_terminate:
        cap = capping_oxygen(si, keep, a, w)
        top_si = float(cap[:, 2].max())          # the glass sits above the O layer

    # Glass types: incoming 1 = Si, 2 = O (amorphous.py convention).
    cut_z, broken = best_cut(glass)
    gz = (glass.pos[:, 2] - cut_z) % glass.lengths[2]
    need = total - (top_si + spec.gap)
    repeats = int(math.ceil(need / glass.lengths[2]))
    gpos, gtypes = [], []
    for r in range(repeats):
        p = glass.pos.copy()
        p[:, 2] = gz + r * glass.lengths[2]
        gpos.append(p); gtypes.append(glass.types)
    gpos = np.vstack(gpos); gtypes = np.concatenate(gtypes)
    gpos[:, 2] += top_si + spec.gap - gpos[:, 2].min()
    inside = gpos[:, 2] < total
    gpos, gtypes = gpos[inside], gtypes[inside]
    new_types = np.where(gtypes == 1, SI_OX, O_T)

    pos = np.vstack([si_pos, cap, gpos])
    types = np.r_[np.full(len(si_pos), SI_C), np.full(len(cap), O_T), new_types]
    cell = Cell(pos, types, np.zeros(3), np.array([w, w, total + spec.headroom]),
                {SI_C: MASS["Si"], SI_OX: MASS["Si"], O_T: MASS["O"]},
                periodic=(True, True, False))
    area_nm2 = (w / 10.0) ** 2
    rep = JoinReport(len(si_pos), len(gpos) + len(cap), cut_z, broken, broken / area_nm2,
                     repeats, meta={"top_si": top_si, "gap": spec.gap,
                                    "capping_O": len(cap)})
    if repeats > 1:
        flags.add(
            Code.STRUCT_INTERFACE_BONDING, Severity.CAUTION,
            f"The glass was stacked {repeats}x along z to fill the oxide side; its "
            f"structure repeats every {glass.lengths[2]:.1f} A.",
            remedy="Quench a glass at least as thick as the oxide side.",
        )
    return cell, rep


# --------------------------------------------------------------------------
# Interface chemistry
# --------------------------------------------------------------------------

@dataclass
class InterfaceChemistry:
    si_o_bonds_per_nm2: float
    suboxide: dict[str, float]            # fraction of crystal Si within `band` of the
                                          # topmost one with n O neighbours (~3 planes)
    dangling_o: int                       # O with one Si neighbour near the interface

    @property
    def coverage(self) -> float:
        """Si(crystal)-O bonds relative to a fully oxidised Si(100) interface."""
        return self.si_o_bonds_per_nm2 / FULL_BONDS_PER_NM2

    @property
    def bonding_class(self) -> str:
        return bonding_class(self.coverage)


def interface_chemistry(cell: Cell, *, band: float = 4.0) -> InterfaceChemistry:
    """Si(crystal)-O bonding across the interface."""
    si_c = np.nonzero(cell.types == SI_C)[0]
    top = cell.pos[si_c, 2].max()
    near = si_c[cell.pos[si_c, 2] > top - band]
    o = np.nonzero(cell.types == O_T)[0]
    i, j, _ = pair_distances(cell, near, o, SI_O_BOND)
    n_o = np.zeros(len(cell.types), dtype=int)
    np.add.at(n_o, i, 1)
    counts = n_o[near]
    area = (cell.lengths[0] / 10.0) * (cell.lengths[1] / 10.0)
    sub = {f"Si+{k}": float(np.mean(counts == k)) for k in range(1, 4)}
    sub["Si+0"] = float(np.mean(counts == 0))
    o_near = o[cell.pos[o, 2] < top + band]
    si_all = np.nonzero(cell.types != O_T)[0]
    oi, _, _ = pair_distances(cell, o_near, si_all, SI_O_BOND)
    n_si = np.zeros(len(cell.types), dtype=int)
    np.add.at(n_si, oi, 1)
    return InterfaceChemistry(len(i) / area, sub, int(np.sum(n_si[o_near] == 1)))


ANNEAL_DECK = """\
# OpenKapitza Si/a-SiO2 interface anneal. Generated; see structure/interface.py.
units           metal
boundary        p p f
atom_style      atomic
read_data       joined.data
pair_style      tersoff
pair_coeff      * * {potential} Si Si O
timestep        {dt_ps:.6f}
neighbor        2.0 bin
# Ends held fixed; a band either side of the interface is heated so Si-O
# bonds can form across it, the rest is held at 300 K.
region          lo block INF INF INF INF INF {z_lo_wall:.4f} units box
region          hi block INF INF INF INF {z_hi_wall:.4f} INF units box
region          band block INF INF INF INF {band_lo:.4f} {band_hi:.4f} units box
group           wlo region lo
group           whi region hi
group           ends union wlo whi
group           mobile subtract all ends
group           hot region band
group           hot intersect hot mobile
group           cold subtract mobile hot
fix             freeze ends setforce 0.0 0.0 0.0
velocity        ends set 0.0 0.0 0.0
minimize        1.0e-6 1.0e-8 5000 50000
reset_timestep  0
velocity        mobile create 300.0 {seed} mom yes rot yes dist gaussian
fix             tcold cold nvt temp 300.0 300.0 0.05
fix             thot hot nvt temp {t_hot:.1f} {t_hot:.1f} 0.05
thermo          {thermo}
run             {anneal_steps}
unfix           thot
fix             thot hot nvt temp {t_hot:.1f} 300.0 0.05
run             {cool_steps}
unfix           thot
unfix           tcold
minimize        1.0e-8 1.0e-10 10000 100000
write_data      interface.data
print           "ANNEAL_DONE"
"""


@dataclass
class InterfaceResult:
    data_file: Path
    join: JoinReport
    chemistry: InterfaceChemistry


def build_interface(lmp: Path, glass: Cell, spec: InterfaceSpec, potential_dir: Path,
                    workdir: Path, flags: FlagSet, *, timeout: int = 36000) -> InterfaceResult:
    """Join, optionally anneal, measure the interface chemistry, write the data file."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cell, rep = join(glass, spec, flags)
    write_lammps_data(cell, workdir / "joined.data", comment="Si(100) / a-SiO2, as joined")
    out_file = workdir / "joined.data"
    if spec.anneal:
        (workdir / spec.potential).write_bytes((Path(potential_dir) / spec.potential).read_bytes())
        g = spec.geometry
        iz = rep.meta["top_si"] + 0.5 * spec.gap
        steps = lambda ps: max(20, int(round(ps * 1000.0 / spec.timestep_fs)))
        deck = ANNEAL_DECK.format(
            potential=spec.potential, dt_ps=spec.timestep_fs * 1e-3,
            z_lo_wall=g.wall_thickness * 1e10, z_hi_wall=g.z_cold_end * 1e10,
            band_lo=iz - (spec.anneal_band if spec.anneal_band_si is None
                          else spec.anneal_band_si),
            band_hi=iz + spec.anneal_band,
            seed=spec.seed, t_hot=spec.anneal_T, thermo=steps(1.0),
            anneal_steps=steps(spec.anneal_ps), cool_steps=steps(spec.cool_ps))
        (workdir / "anneal.in").write_text(deck)
        run = subprocess.run([str(lmp), "-in", "anneal.in", "-log", "anneal.log"],
                             cwd=workdir, capture_output=True, text=True, timeout=timeout)
        if run.returncode != 0 or "ANNEAL_DONE" not in run.stdout:
            errors = [l for l in run.stdout.splitlines() if "ERROR" in l]
            raise RuntimeError(f"interface anneal failed: {errors or run.stdout[-400:]}")
        annealed = read_lammps_data(workdir / "interface.data")
        annealed.periodic = (True, True, False)
        cell = annealed
        out_file = workdir / "interface.data"
    chem = interface_chemistry(cell)
    if spec.bonding and chem.bonding_class != spec.bonding:
        flags.add(
            Code.STRUCT_INTERFACE_BONDING, Severity.CAUTION,
            f"Requested a {spec.bonding}ly bonded interface but reached "
            f"{chem.coverage * 100:.0f} % of full Si(100) oxidation "
            f"({chem.bonding_class}).",
            remedy=("Lampin et al. (2012) find a 5x spread in h_K between strong "
                    "and weak coupling with this potential. Adjust gap / anneal and "
                    "rebuild, or report the class actually reached."),
            coverage=chem.coverage,
        )
    flags.add(
        Code.STRUCT_INTERFACE_BONDING, Severity.INFO,
        f"Si/a-SiO2 interface: {'annealed at %.0f K' % spec.anneal_T if spec.anneal else 'joined, not annealed'}; "
        f"glass cut where {rep.broken_bonds} Si-O bonds cross ({rep.broken_per_nm2:.1f} per nm^2); "
        f"gap {spec.gap:.2f} A; {chem.si_o_bonds_per_nm2:.1f} Si(crystal)-O bonds per nm^2 "
        f"({chem.coverage * 100:.0f} % of full oxidation: {chem.bonding_class}); "
        f"crystal Si within 4 A of the interface: "
        + ", ".join(f"{k} {v * 100:.0f} %" for k, v in sorted(chem.suboxide.items())) + ".",
        remedy=("These choices change h_K more than most others (section 3). Quote them "
                "with the result; compare annealed and unannealed if h_K is sensitive."),
        bonds_per_nm2=chem.si_o_bonds_per_nm2,
    )
    return InterfaceResult(out_file, rep, chem)
