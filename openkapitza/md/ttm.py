"""Two-temperature (TTM-MD) NEMD for metal/non-metal interfaces (open issue P1-14).

Classical NEMD has no electrons. In a metal most of the heat is carried by
electrons, which must hand it to the lattice before it can cross into a
non-metal. This module adds that step to the NEMD deck with LAMMPS
`fix ttm/thermal`: a finite-difference electron grid, coupled to the metal's
atoms by a Langevin friction (Duffy & Rutherford), so the run measures the
phonon-phonon interface resistance AND the electron-phonon resistance in the
metal together (Wang, Ruan & Roy, PRB 85, 205311, 2012).

Design (knowledge base sections 7 and 8, all checked in the LAMMPS source):

* The fix requires periodic boundaries in all dimensions, so a TTM deck is
  `boundary p p p`. A vacuum gap above the top wall, wider than the potential
  cutoff, keeps the slab from seeing its own z-image; the deck measures that
  gap after relaxation and stops if it is too small.
* The fix forbids a changing box, so it is defined after the in-plane
  relaxation and equilibration, just before the gradient is imposed.
* Electrons live on the metal side only: every other grid cell has
  kappa_e = gamma_p = 0, which in `ttm/thermal` means no heat flows into it
  (face conductivity is the harmonic mean) and no force acts in it. C_vol
  stays positive everywhere (the update divides by it).
* Heat enters and leaves through the phonon Langevin reservoirs, as in Wang
  et al.; the TTM fix includes the metal atoms inside the reservoir, so heat
  carried by electrons reaches the thermostat through the lattice there.
* In steady state only kappa_e and g matter; C_e sets how fast the electrons
  settle, not the answer.

Units in the properties file are LAMMPS metal units: C_vol eV/(K A^3),
kappa_e eV/(ps A K), gamma_p g/mol/ps, with gamma_p = m g / (3 n k_B).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..schema.flags import Code, FlagSet, Severity
from .electron_phonon import COUPLING, LATTICE_K

EV = 1.602176634e-19          # J
KB = 1.380649e-23             # J/K
AMU = 1.66053906660e-27       # kg
NA = 6.02214076e23

# Electronic thermal conductivity near 300 K, W/mK (sourced only).
ELECTRON_K: dict[str, dict] = {
    "Al": {"value": 225.76, "citation": "Tong, Li, Ruan & Bao, arXiv:1907.01364 (DFT, 300 K)"},
    "Cu": {"value": 360.90, "citation": "Tong, Li, Ruan & Bao, arXiv:1907.01364 (DFT, 300 K)"},
}

# Sommerfeld coefficient (C_e = gamma T), experimental, mJ/mol/K^2, and the
# lattice constant used for the molar volume.
SOMMERFELD: dict[str, dict] = {
    "Al": {"mJ_mol_K2": 1.35, "a0": 4.0495, "atoms_per_cell": 4, "mass": 26.9815,
           "citation": "Kittel, Introduction to Solid State Physics, 8th ed., p. 139 "
                       "(Gupta et al., arXiv:1701.06999: experiments 1.32-1.46)"},
    "Cu": {"mJ_mol_K2": 0.695, "a0": 3.6149, "atoms_per_cell": 4, "mass": 63.546,
           "citation": "Kittel, Introduction to Solid State Physics, 8th ed., p. 139 "
                       "(Wang et al. 2012 use 96.8 J/m^3K^2)"},
}

COOLING_LENGTHS_EXCLUDED = 3.0      # metal-side fit starts this many l_c from the plane
MIN_ATOMS_PER_CELL = 1000           # Wang et al. 2012: local equilibrium per FD cell


def number_density(metal: str) -> float:
    """Atoms per m^3 from the tabulated lattice constant."""
    s = SOMMERFELD[metal]
    return s["atoms_per_cell"] / (s["a0"] * 1e-10) ** 3


def sommerfeld_volumetric(metal: str) -> float:
    """gamma in J/m^3/K^2."""
    s = SOMMERFELD[metal]
    molar_volume = NA / number_density(metal)          # m^3/mol
    return s["mJ_mol_K2"] * 1e-3 / molar_volume


def friction_gamma_p(g: float, mass_amu: float, n: float) -> float:
    """Langevin friction per atom, g/mol/ps, giving coupling g (W/m^3K).

    Per-atom power (3 k_B gamma / m)(T_e - T) times n atoms per volume = g (T_e - T).
    """
    gamma_si = mass_amu * AMU * g / (3.0 * n * KB)    # kg/s
    return gamma_si * 1e3 * NA * 1e-12


def to_metal_units(c_vol_si: float, k_e_si: float) -> tuple[float, float]:
    """C_vol J/m^3K -> eV/(K A^3); kappa W/mK -> eV/(ps A K)."""
    return c_vol_si / EV * 1e-30, k_e_si / EV * 1e-12 * 1e-10


def cooling_length(g: float, k_p: float) -> float:
    """sqrt(k_p/g), m: decay length of T_e - T_p next to the interface."""
    return math.sqrt(k_p / g)


@dataclass(frozen=True)
class TTMSpec:
    """Electron subsystem for one metal side of an NEMD cell."""

    metal: str                          # key into the sourced tables ("Al", "Cu")
    side: str = "b"                     # which NEMD side is the metal
    metal_types: tuple[int, ...] = (2,)
    g: float | None = None              # W/m^3K; default COUPLING[metal]
    k_e: float | None = None            # W/mK; default ELECTRON_K[metal]
    cell_ang: float = 10.0              # target electron-grid spacing along z
    box_z_ang: float | None = None      # required for read_data cells
    vacuum_min_ang: float = 12.0        # z-image separation the deck enforces
    output_frames: int | None = None    # T_e snapshots in production; default profile_frames
    exclusion_lc: float = COOLING_LENGTHS_EXCLUDED   # metal-side fit starts this many l_c out

    def resolved(self) -> dict:
        if self.side not in ("a", "b"):
            raise ValueError("ttm.side must be 'a' or 'b'")
        for table, name in ((COUPLING, "g"), (SOMMERFELD, "Sommerfeld gamma")):
            if self.metal not in table:
                raise ValueError(f"no sourced {name} for {self.metal}")
        if self.k_e is None and self.metal not in ELECTRON_K:
            raise ValueError(f"no sourced kappa_e for {self.metal}")
        return {
            "g": self.g if self.g is not None else COUPLING[self.metal]["g"],
            "k_e": self.k_e if self.k_e is not None else ELECTRON_K[self.metal]["value"],
            "gamma_e": sommerfeld_volumetric(self.metal),
            "n": number_density(self.metal),
            "mass": SOMMERFELD[self.metal]["mass"],
        }


def metal_extent(nemd_spec, ttm: TTMSpec) -> tuple[float, float]:
    """z-range (A) of the metal's MOBILE atoms, reservoir included, walls not."""
    g = nemd_spec.geometry
    a = 1e10
    if ttm.side == "b":
        return g.interface_z * a, g.z_cold_end * a
    return g.wall_thickness * a, g.interface_z * a


def grid(nemd_spec, ttm: TTMSpec, box_z_ang: float) -> tuple[int, np.ndarray]:
    """(Nz, metal flag per cell). A cell is metal if its centre is in the metal."""
    nz = max(3, int(round(box_z_ang / ttm.cell_ang)))
    dz = box_z_ang / nz
    lo, hi = metal_extent(nemd_spec, ttm)
    centres = (np.arange(nz) + 0.5) * dz
    return nz, (centres > lo) & (centres < hi)


def properties_file(nemd_spec, ttm: TTMSpec, box_z_ang: float) -> str:
    """The `ttm/thermal` properties grid: ix iy iz C_vol kappa_e gamma_p eta."""
    r = ttm.resolved()
    nz, metal = grid(nemd_spec, ttm, box_z_ang)
    c_vol, k_e = to_metal_units(r["gamma_e"] * nemd_spec.temperature, r["k_e"])
    gamma_p = friction_gamma_p(r["g"], r["mass"], r["n"])
    lines = [
        f"# OpenKapitza TTM properties: {ttm.metal} electrons on side {ttm.side}, "
        f"{int(metal.sum())} of {nz} cells along z (box {box_z_ang:.3f} A).",
        f"# g = {r['g']:.4g} W/m^3K, kappa_e = {r['k_e']:.4g} W/mK, "
        f"C_e = {r['gamma_e']:.4g} J/m^3K^2 x {nemd_spec.temperature:g} K.",
        "# Non-metal, wall and vacuum cells: kappa_e = gamma_p = 0 (no electrons);",
        "# C_vol kept positive because the update divides by it.",
        "# ix iy iz C_vol[eV/K/A^3] kappa_e[eV/ps/A/K] gamma_p[g/mol/ps] eta",
    ]
    for iz in range(nz):
        if metal[iz]:
            lines.append(f"1 1 {iz + 1} {c_vol:.6e} {k_e:.6e} {gamma_p:.6e} 0")
        else:
            lines.append(f"1 1 {iz + 1} {c_vol:.6e} 0 0 0")
    return "\n".join(lines) + "\n"


TTM_BLOCK = """\
# ---- electrons: two-temperature model on the {metal} side (P1-14) ----------
# fix ttm/thermal needs periodic z and a fixed box: the box was relaxed above
# and does not change from here. The vacuum above the top wall keeps the slab
# from interacting with its own z-image; stop if relaxation closed it.
variable        zgap equal (zhi-bound(all,zmax))+(bound(all,zmin)-zlo)
print           "OPENKAPITZA_VACUUM_GAP $(v_zgap:%.4f) min {vacuum_min:.2f}"
if "$(v_zgap) < {vacuum_min:.4f}" then "print 'ERROR: vacuum gap below the cutoff guard'" "quit 1"
group           gmetal_all type {metal_types}
group           gmetal intersect gmetal_all gmobile
fix             twotemp gmetal ttm/thermal {seed} ttm_props.in 1 1 {nz} set {temperature:.2f} outfile {out_every} T_e.out
"""


TTM_PRODUCTION = """\
# Mean electron temperature of the metal, averaged like the lattice profile.
# Instantaneous T_e is dominated by thermodynamic noise: the electrons' heat
# capacity is ~60x the lattice's smaller, so in a nm-scale cell a single
# snapshot scatters by ~T/sqrt(C_e V/k_B), tens of K (bulk-Al check). f_twotemp[1]
# sums T C V over ALL cells; the electron-free cells sit at their initial T.
variable        te_metal equal (f_twotemp[1]-{t0:.2f}*{c_vol:.6e}*{n_free}*lx*ly*lz/{nz})/({c_vol:.6e}*{n_metal}*lx*ly*lz/{nz})
fix             te_avg all ave/time {every} {nrep} {window} v_te_metal file T_e_mean.dat
"""


def production_block(nemd_spec, ttm: TTMSpec, box_z_ang: float, every: int,
                     nrep: int, window: int) -> str:
    r = ttm.resolved()
    nz, metal = grid(nemd_spec, ttm, box_z_ang)
    c_vol, _ = to_metal_units(r["gamma_e"] * nemd_spec.temperature, r["k_e"])
    return TTM_PRODUCTION.format(t0=nemd_spec.temperature, c_vol=c_vol,
                                 n_free=int((~metal).sum()), n_metal=int(metal.sum()),
                                 nz=nz, every=every, nrep=nrep, window=window)


def read_electron_mean(run_dir: Path | str) -> list[float]:
    """Per-frame mean metal T_e from T_e_mean.dat (fix ave/time output)."""
    out = []
    for ln in (Path(run_dir) / "T_e_mean.dat").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            out.append(float(ln.split()[1]))
    return out


def deck_block(nemd_spec, ttm: TTMSpec, box_z_ang: float, out_every: int) -> str:
    nz, _ = grid(nemd_spec, ttm, box_z_ang)
    return TTM_BLOCK.format(metal=ttm.metal, vacuum_min=ttm.vacuum_min_ang,
                            metal_types=" ".join(map(str, ttm.metal_types)),
                            seed=nemd_spec.seed + 7, nz=nz,
                            temperature=nemd_spec.temperature, out_every=out_every)


def check(nemd_spec, ttm: TTMSpec, flags: FlagSet) -> dict:
    """Caveats that travel with every TTM result; returns derived lengths (m)."""
    r = ttm.resolved()
    k_p = LATTICE_K.get(ttm.metal, {}).get("high")
    lc = cooling_length(r["g"], k_p) if k_p else None
    lo, hi = metal_extent(nemd_spec, ttm)
    metal_len = (hi - lo) * 1e-10
    flags.add(
        Code.NEMD_PHONON_ONLY_METAL, Severity.CAUTION,
        f"{ttm.metal}: electron-phonon coupling modelled with a two-temperature model "
        f"(g = {r['g']:.3g} W/m^3K, kappa_e = {r['k_e']:.3g} W/mK). Direct coupling "
        f"of metal electrons to the non-metal's phonons is NOT modelled.",
        remedy=("That channel acts in parallel and would raise h_K (Lu, Wang & Ruan "
                "2016: -18 % TBR on Cu/Si with an assumed coupling). g is the least "
                "certain input: run the sourced range for a band."),
    )
    if lc is not None and metal_len < 2 * ttm.exclusion_lc * lc:
        flags.add(
            Code.NEMD_FEW_BINS, Severity.WARNING,
            f"{ttm.metal} slab {metal_len * 1e9:.1f} nm is shorter than "
            f"{2 * ttm.exclusion_lc:g} cooling lengths "
            f"(l_c = {lc * 1e9:.1f} nm): the metal-side fit, which starts "
            f"{ttm.exclusion_lc:g} l_c from the interface, has little or "
            f"no region where electrons and lattice agree.",
            remedy="Lengthen the metal side (Wang et al. 2012: metal longer than the "
                   "cooling length, so a linear region exists).",
            cooling_length_m=lc,
        )
    area = (nemd_spec.snapped_area if nemd_spec.structure_mode == "build"
            else nemd_spec.geometry.area)
    per_cell = r["n"] * area * ttm.cell_ang * 1e-10
    if per_cell < MIN_ATOMS_PER_CELL:
        flags.add(
            Code.NEMD_FEW_BINS, Severity.WARNING,
            f"~{per_cell:.0f} metal atoms per electron cell (< {MIN_ATOMS_PER_CELL}): the "
            f"lattice temperature each cell couples to is noisy.",
            remedy=("Wang et al. 2012 need > ~1000 atoms per cell for local "
                    "equilibrium: widen the cross-section or coarsen cell_ang (keeping "
                    "it below the cooling length)."),
            atoms_per_cell=per_cell,
        )
    if lc is not None and ttm.cell_ang * 1e-10 > lc:
        flags.add(
            Code.NEMD_FEW_BINS, Severity.WARNING,
            f"Electron cell {ttm.cell_ang:g} A is longer than the cooling length "
            f"{lc * 1e10:.0f} A: the grid cannot resolve the non-equilibrium region.",
            remedy="Use cells shorter than the cooling length (Wang et al. 2012).",
        )
    if ttm.exclusion_lc < COOLING_LENGTHS_EXCLUDED:
        flags.add(
            Code.NEMD_FEW_BINS, Severity.WARNING,
            f"Metal-side fit starts {ttm.exclusion_lc:g} cooling lengths from the "
            f"interface (default {COOLING_LENGTHS_EXCLUDED:g}): the lattice there is "
            f"still out of equilibrium with the electrons by "
            f"~{math.exp(-ttm.exclusion_lc) * 100:.0f} % of the interface value.",
            remedy="A plumbing setting; use the default on a long enough metal slab.",
        )
    return {"cooling_length": lc, "exclusion": (ttm.exclusion_lc * lc) if lc else 0.0}


def fit_exclusion(ttm: TTMSpec) -> float:
    """Metal-side distance (m) from the interface left out of the bulk fit.

    Within a few cooling lengths of the plane the lattice and electron
    temperatures differ; the bulk extrapolation must start where they agree.
    The phonon-phonon-only lattice k is used (the larger, conservative l_c).
    """
    r = ttm.resolved()
    k_p = LATTICE_K.get(ttm.metal, {}).get("high")
    return ttm.exclusion_lc * cooling_length(r["g"], k_p) if k_p else 0.0


_TE_FILE = re.compile(r"^T_e\.out\.(\d+)$")


def read_electron_profile(run_dir: Path | str, box_z_ang: float,
                          frames: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean T_e(z) over the last `frames` snapshots (by write time).

    The deck restarts the step counter for production, so step numbers in the
    file names repeat across stages; modification time orders them.
    """
    files = [p for p in Path(run_dir).iterdir() if _TE_FILE.match(p.name)]
    if not files:
        raise ValueError("no T_e.out.* files: was the deck run with TTM?")
    files.sort(key=lambda p: (p.stat().st_mtime_ns, int(_TE_FILE.match(p.name).group(1))))
    if frames:
        files = files[-frames:]
    profiles = []
    for p in files:
        rows = [ln.split() for ln in p.read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
        data = np.array([[float(x) for x in r[:4]] for r in rows])
        order = np.argsort(data[:, 2])
        profiles.append(data[order, 3])
    T = np.mean(profiles, axis=0)
    nz = len(T)
    z = (np.arange(nz) + 0.5) * box_z_ang / nz
    return z, T, len(files)


@dataclass(frozen=True)
class EPComparison:
    """Phonon-only and TTM runs of the same cell, combined."""
    h_pp: float                 # W/m^2K, phonon-only NEMD
    h_pp_err: float
    h_ttm: float                # W/m^2K, with electrons
    h_ttm_err: float
    h_ep: float | None          # effective electron-phonon conductance, 1/h_ttm - 1/h_pp
    h_ep_err: float | None
    reduction: float            # 1 - h_ttm / h_pp


def compare_runs(h_pp: float, h_pp_err: float, h_ttm: float, h_ttm_err: float,
                 flags: FlagSet) -> EPComparison:
    """Effective electron-phonon conductance from a phonon-only/TTM pair.

    R_ep = 1/h_ttm - 1/h_pp, errors combined in quadrature (independent runs).
    If the difference is not resolved (R_ep within 2 sigma of zero or
    negative), h_ep is not quoted.
    """
    r_ep = 1.0 / h_ttm - 1.0 / h_pp
    r_err = math.hypot(h_ttm_err / h_ttm ** 2, h_pp_err / h_pp ** 2)
    red = 1.0 - h_ttm / h_pp
    if r_ep <= 2.0 * r_err:
        flags.add(
            Code.NEMD_ERROR_UNDERSTATED, Severity.CAUTION,
            f"Electron-phonon resistance not resolved: 1/h_ttm - 1/h_pp = "
            f"{r_ep * 1e9:+.2f} +/- {r_err * 1e9:.2f} x 1e-9 m^2K/W.",
            remedy="Longer production (both runs) to shrink the errors below the difference.",
        )
        return EPComparison(h_pp, h_pp_err, h_ttm, h_ttm_err, None, None, red)
    h_ep = 1.0 / r_ep
    return EPComparison(h_pp, h_pp_err, h_ttm, h_ttm_err, h_ep, h_ep * r_err / r_ep, red)
