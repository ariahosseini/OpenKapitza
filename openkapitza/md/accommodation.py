"""Thermal accommodation coefficients from gas-surface scattering MD.

Work package 5b, governed by section 11 of the agent instruction. The
accommodation coefficient is, by the project's own spec, the weakest input in
the whole model: measured values for one gas/solid pair span an order of
magnitude with surface cleanliness and adsorbed layers, and the defaults are
generic correlations hard-flagged LOW. An MD value is the one fill that beats
the default.

DEFINITION

    alpha = (E_r - E_i) / (E_s - E_i)

E_i and E_r are the mean energies per molecule of the incident and reflected
streams, and E_s = 2 k_B T_s is the mean energy a FLUX-WEIGHTED Maxwellian at
the surface temperature carries. "Flux-weighted" matters: the mean energy of
molecules crossing a plane is 2kT, not the 3/2 kT of molecules in a volume,
because fast molecules cross more often.

METHOD

A real atomistic slab of the solid sits at T_s. Above it, non-interacting gas
atoms (gas-gas pairs excluded from the neighbour list) are re-emitted
downward by a DIFFUSIVE stochastic wall at T_g, so every incident molecule
carries a flux-weighted Maxwellian at T_g -- which gives a built-in check:
<E_i> must come out at 2 k_B T_g, or the analysis is broken.

Above the slab's force range the gas flies freely at constant energy. Each
free-flight segment of an atom's trajectory is therefore one crossing event:
downward segments are incident, upward ones reflected. Counting segments,
not frames, is what makes the means flux-weighted rather than density-weighted.

`fix wall/reflect/stochastic` cannot MEASURE alpha -- it IMPOSES one. But that
makes it the right tool for validating this pipeline: replace the slab with an
idealised bottom wall and the answer is known. A diffusive wall must give
alpha = 1, a specular wall alpha = 0, and a Maxwell wall with diffuse fraction
a must give alpha = a (in Maxwell's model the energy accommodation equals the
diffuse fraction). `build_wall_validation_deck` builds exactly that.

WHAT THE NUMBER IS AND IS NOT

- It is a CLEAN-SURFACE value. Technical surfaces carry oxides and adsorbed
  layers that generally RAISE alpha, especially for light gases. Treat an MD
  value as closer to a bound than to the answer for a machined part.
- The gas-solid interaction dominates the uncertainty. It is usually a
  Lennard-Jones cross term from Lorentz-Berthelot mixing of the gas parameters
  with a metal's LJ parameters, which are not well founded for gas-metal
  physisorption. `run_sensitivity` scales the well depth and reports the range;
  quote that range, not one number.
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
from .fill_policy import cap_md

KB_EV = 8.617333262e-5          # Boltzmann constant, eV/K

# Gas Lennard-Jones parameters (eps in eV, sigma in Angstrom). Standard
# viscosity-fitted values; N2 treated as a sphere, which ignores its rotational
# degrees of freedom -- an MD alpha for N2 from this table is translational only.
GAS_LJ: dict[str, dict] = {
    "He": {"mass": 4.0026, "eps": 10.22 * KB_EV, "sigma": 2.556},
    "Ne": {"mass": 20.180, "eps": 35.6 * KB_EV, "sigma": 2.749},
    "Ar": {"mass": 39.948, "eps": 119.8 * KB_EV, "sigma": 3.405},
    "N2": {"mass": 28.014, "eps": 95.05 * KB_EV, "sigma": 3.698,
           "note": "spherical approximation; rotational accommodation absent"},
}

# Metal LJ parameters for mixing ONLY (Halicioglu & Pound 1975). These describe
# metal cohesion, not physisorption, and Lorentz-Berthelot mixing with them is
# known to misjudge gas-metal well depths -- which is why the well depth is
# scanned rather than trusted.
METAL_LJ: dict[str, dict] = {
    "Ni": {"eps": 0.5201, "sigma": 2.282,
           "citation": "Halicioglu & Pound (1975), Phys. Status Solidi A 30, 619"},
    "Cu": {"eps": 0.4093, "sigma": 2.338,
           "citation": "Halicioglu & Pound (1975)"},
}


# Target TOTAL adsorption energies (meV) for calibrating the gas-solid well.
# A gas atom feels many surface atoms at once, so its binding to the SURFACE is
# several times the pair well depth. Lorentz-Berthelot mixing with
# Halicioglu-Pound metal parameters gives an Ar-Ni pair well of 73 meV, which
# sums to a few hundred meV -- chemisorption strength. Run uncalibrated, the
# argon condensed: 202 adsorbed atoms on 72 surface sites, still climbing, and
# their collective pull lifted Ni atoms 2.6 A out of the surface. Hence the
# well is calibrated to a measured adsorption energy before any alpha run.
PHYSISORPTION_MEV: dict[str, dict] = {
    "Ar": {"value": 80.0, "range": (60.0, 110.0),
           "citation": ("Ar/Pt(111), experiment: Head-Gordon, Tully, Rettner, "
                        "Mullins & Auerbach, J. Chem. Phys. 94, 1516 (1991); "
                        "used as a transition-metal proxy"),
           "residence_300K_ps": (46.0, 57.0)},
    "He": {"value": 6.0, "range": (3.0, 10.0),
           "citation": ("He on noble metals: wells below 10 meV (Zaremba & Kohn "
                        "jellium); no Ni-specific value used -- an assumption, "
                        "covered by the well-depth scan")},
}


# Light gases whose energy exchange with a surface is dominated by elastic and
# single-phonon scattering, which classical MD does not capture.
QUANTUM_GASES = frozenset({"He", "Ne"})


def lorentz_berthelot(gas: str, metal: str) -> tuple[float, float]:
    """(eps_gs in eV, sigma_gs in Angstrom) by Lorentz-Berthelot mixing."""
    g, m = GAS_LJ[gas], METAL_LJ[metal]
    return math.sqrt(g["eps"] * m["eps"]), 0.5 * (g["sigma"] + m["sigma"])


AMU_A2_PS2_IN_EV = 1.0364269e-4    # 1 amu * (Angstrom/ps)^2 in eV


def suggest_dump_every(gas: str, t_max: float, region_height: float,
                       timestep_fs: float, *, frames_per_crossing: int = 6,
                       speed_multiple: float = 4.0) -> int:
    """Dump interval that keeps even fast molecules visible for several frames.

    A segment needs `min_frames` frames to count. A fixed dump interval would
    silently drop the fast tail of a light gas -- helium at 600 K crosses 40 A
    in well under a picosecond at four times its thermal speed -- and dropping
    fast molecules biases the mean energy DOWN. So the interval is set from the
    gas speed: a molecule at `speed_multiple` times the thermal speed
    sqrt(2kT/m) must spend at least `frames_per_crossing` frames in view.
    """
    mass = GAS_LJ[gas]["mass"]
    v_th = math.sqrt(2.0 * KB_EV * t_max / (mass * AMU_A2_PS2_IN_EV))   # A/ps
    crossing_ps = region_height / (speed_multiple * v_th)
    return max(1, int(crossing_ps / frames_per_crossing / (timestep_fs * 1e-3)))


def baule_goodman(gas_mass: float, solid_mass: float) -> float:
    """Goodman's modified Baule estimate, 2.4 mu / (1 + mu)^2.

    A classical binary-collision estimate, useful as a TREND check (heavy gases
    accommodate better on light solids) and not as a value.
    """
    mu = gas_mass / solid_mass
    return 2.4 * mu / (1.0 + mu) ** 2


# --------------------------------------------------------------------------
# Decks
# --------------------------------------------------------------------------

@dataclass
class AccommodationSpec:
    solid: CubicSpec
    gas: str
    t_surface: float = 300.0
    t_gas: float = 400.0
    metal_lj: str | None = None          # key into METAL_LJ for mixing
    eps_gs: float | None = None          # eV; default Lorentz-Berthelot
    sigma_gs: float | None = None        # Angstrom
    eps_scale: float = 1.0               # for the well-depth sensitivity scan
    calibrated: bool = False             # True once eps_gs came from calibrate_well_depth
    n_gas: int = 100
    lateral_cells: int = 6
    slab_cells: int = 6
    region_height: float = 40.0          # Angstrom of free flight
    timestep_fs: float | None = None     # None: 0.5 fs for He, 1 fs otherwise
    equil_ps: float = 20.0
    production_ps: float = 500.0
    dump_every: int | None = None        # None: chosen from the gas speed
    seed: int = 4928459

    def interaction(self) -> tuple[float, float]:
        if self.eps_gs is not None and self.sigma_gs is not None:
            eps, sigma = self.eps_gs, self.sigma_gs
        else:
            if not self.metal_lj:
                raise ValueError("give eps_gs and sigma_gs, or a metal_lj key to mix")
            eps, sigma = lorentz_berthelot(self.gas, self.metal_lj)
        return eps * self.eps_scale, sigma

    @property
    def dt_fs(self) -> float:
        if self.timestep_fs is not None:
            return self.timestep_fs
        # A light gas meets the steep repulsive wall fast: halve the step.
        return 0.5 if GAS_LJ[self.gas]["mass"] < 10.0 else 1.0

    @property
    def cutoff(self) -> float:
        return 2.5 * self.interaction()[1]

    @property
    def slab_top(self) -> float:
        return self.slab_cells * self.solid.a0_guess

    @property
    def z_free(self) -> float:
        """Above this the gas feels no force, so its energy is constant."""
        return self.slab_top + self.cutoff + 1.5

    @property
    def z_top(self) -> float:
        return self.z_free + self.region_height


SLAB_DECK = """\
# OpenKapitza accommodation coefficient -- {gas} on {solid}
# T_s = {t_surface} K (slab), T_g = {t_gas} K (diffusive wall above).
# Generated; see openkapitza/md/accommodation.py.
units           metal
dimension       3
boundary        p p f
atom_style      atomic
timestep        {dt_ps:.6f}

lattice         {lattice} {a0:.6f}
region          box  block 0 {lx:.6f} 0 {lx:.6f} 0 {z_top:.6f} units box
create_box      2 box
region          slab block INF INF INF INF 0 {slab_top_minus:.6f} units box
create_atoms    1 region slab
region          gasreg block INF INF INF INF {gas_lo:.6f} {gas_hi:.6f} units box
create_atoms    2 random {n_gas} {seed} gasreg
mass            1 {solid_mass:.4f}
mass            2 {gas_mass:.4f}

# Solid self-interaction from its potential; gas-solid by Lennard-Jones;
# gas-gas switched OFF so each gas atom is an independent probe.
pair_style      hybrid/overlay {pair_style} lj/cut {cutoff:.4f}
{pair_coeff}
pair_coeff      1 2 lj/cut {eps:.8f} {sigma:.6f} {cutoff:.4f}
pair_coeff      2 2 lj/cut 0.0 {sigma:.6f}
neigh_modify    exclude type 2 2

group           solid type 1
group           gas   type 2
region          rfix block INF INF INF INF INF {z_fix:.6f} units box
region          rsurf block INF INF INF INF {z_surf:.6f} INF units box
group           fixtmp  region rfix
group           fixed   intersect fixtmp solid
group           surftmp region rsurf
group           surface intersect surftmp solid
group           mobile  subtract solid fixed
group           bath    subtract mobile surface

velocity        fixed set 0.0 0.0 0.0
fix             hold fixed setforce 0.0 0.0 0.0
velocity        mobile create {t_init:.2f} {seed} mom yes rot yes dist gaussian
velocity        gas create {t_gas:.2f} {seed2} dist gaussian
fix             nves mobile nve
# Thermostat the slab interior only; the top layers stay NVE so the surface
# exchanges energy with the gas through real dynamics, not a thermostat.
fix             bath bath langevin {t_surface:.2f} {t_surface:.2f} 0.1 {seed3}
fix             nveg gas nve
fix             topwall gas wall/reflect/stochastic diffusive {seed4} zhi EDGE {t_gas:.2f} 0.0 0.0 0.0

# Top atomic layer (for counting adsorption sites) and the adsorption-well
# region above it, both fixed from the as-built surface.
variable        ztop0 equal bound(solid,zmax)
region          rtop  block INF INF INF INF $(v_ztop0-0.5) INF units box
group           toptmp region rtop
group           toplayer intersect toptmp solid
region          rwell block INF INF INF INF INF $(v_ztop0+{well_height:.4f}) units box
variable        nsites equal count(toplayer)
variable        nads   equal count(gas,rwell)

compute         zsurf surface reduce max z
compute         tsurf surface temp
thermo_style    custom step temp c_tsurf c_zsurf v_nads
thermo          {thermo_every}
run             {equil_steps}

# Production. The surface temperature and the coverage are MEASURED, because
# the hot gas heats the surface and the alpha formula needs the temperature
# the surface actually had, and because a piled-up adsorbed layer means the
# number is no longer a clean-surface value.
fix             tsa all ave/time 100 10 1000 c_tsurf v_nads ave running
compute         kea gas ke/atom
dump            traj gas custom {dump_every} gas.dump id z vz c_kea
dump_modify     traj sort id
run             {prod_steps}
variable        zs equal c_zsurf
variable        tsbar equal f_tsa[1]
variable        nabar equal f_tsa[2]
print           "ACCOM_GEOMETRY z_surface ${{zs}} z_free {z_free:.6f} z_top {z_top:.6f}"
print           "ACCOM_SURFACE t_surface ${{tsbar}} n_adsorbed ${{nabar}} n_sites ${{nsites}}"
"""


def build_accommodation_deck(spec: AccommodationSpec) -> str:
    eps, sigma = spec.interaction()
    if spec.solid.pair_style == "eam":
        pair_coeff = f"pair_coeff      1 1 eam {spec.solid.potential}"
    else:
        if not spec.solid.label:
            raise ValueError(f"{spec.solid.name}: a label is needed for {spec.solid.pair_style}")
        pair_coeff = (f"pair_coeff      * * {spec.solid.pair_style} "
                      f"{spec.solid.potential} {spec.solid.label} NULL")
    a0 = spec.solid.a0_guess
    dt_ps = spec.dt_fs * 1e-3
    return SLAB_DECK.format(
        gas=spec.gas, solid=spec.solid.name, t_surface=spec.t_surface,
        t_gas=spec.t_gas, dt_ps=dt_ps, lattice=spec.solid.lattice, a0=a0,
        lx=spec.lateral_cells * a0, z_top=spec.z_top,
        slab_top_minus=spec.slab_top - 0.1, gas_lo=spec.z_free + 1.0,
        gas_hi=spec.z_top - 1.0, n_gas=spec.n_gas, seed=spec.seed,
        seed2=spec.seed + 11, seed3=spec.seed + 23, seed4=spec.seed + 37,
        solid_mass=spec.solid.mass, gas_mass=GAS_LJ[spec.gas]["mass"],
        pair_style=spec.solid.pair_style, pair_coeff=pair_coeff,
        eps=eps, sigma=sigma, cutoff=spec.cutoff,
        z_fix=a0 - 0.1, z_surf=spec.slab_top - 2.0 * a0 - 0.1,
        t_init=2.0 * spec.t_surface, thermo_every=max(1, int(1000 / spec.dt_fs)),
        well_height=1.5 * sigma,
        equil_steps=int(round(spec.equil_ps / dt_ps)),
        prod_steps=int(round(spec.production_ps / dt_ps)),
        dump_every=(spec.dump_every or suggest_dump_every(
            spec.gas, max(spec.t_gas, spec.t_surface), spec.region_height,
            spec.dt_fs)),
        z_free=spec.z_free,
    )


WALL_DECK = """\
# OpenKapitza accommodation PIPELINE VALIDATION -- {gas}, bottom wall: {model}
# No solid: an idealised wall with a KNOWN answer. diffusive -> alpha = 1,
# specular -> alpha = 0, maxwell(a) -> alpha = a.
units           metal
dimension       3
boundary        p p f
atom_style      atomic
timestep        {dt_ps:.6f}
region          box block 0 30 0 30 0 {height:.4f} units box
create_box      1 box
create_atoms    1 random {n_gas} {seed} box
mass            1 {gas_mass:.4f}
pair_style      zero 3.0
pair_coeff      * *
velocity        all create {t_gas:.2f} {seed} dist gaussian
fix             nve all nve
fix             topwall all wall/reflect/stochastic diffusive {seed2} zhi EDGE {t_gas:.2f} 0.0 0.0 0.0
{bottom}
run             {equil_steps}
compute         kea all ke/atom
dump            traj all custom {dump_every} gas.dump id z vz c_kea
dump_modify     traj sort id
run             {prod_steps}
print           "ACCOM_GEOMETRY z_surface 0.0 z_free {z_free:.4f} z_top {height:.4f}"
"""


def build_wall_validation_deck(model: str, *, gas: str = "Ar",
                               t_surface: float = 300.0, t_gas: float = 400.0,
                               diffuse_fraction: float = 0.4, n_gas: int = 300,
                               equil_ps: float = 20.0, production_ps: float = 150.0,
                               dump_every: int | None = None,
                               seed: int = 772211) -> str:
    """Known-answer deck: gas between a diffusive top wall and an ideal bottom."""
    if model == "diffusive":
        bottom = (f"fix             botwall all wall/reflect/stochastic diffusive "
                  f"{seed + 5} zlo EDGE {t_surface:.2f} 0.0 0.0 0.0")
    elif model == "maxwell":
        if not 0.0 < diffuse_fraction <= 1.0:
            raise ValueError("diffuse_fraction must lie in (0, 1]")
        bottom = (f"fix             botwall all wall/reflect/stochastic maxwell "
                  f"{seed + 5} zlo EDGE {t_surface:.2f} 0.0 0.0 0.0 {diffuse_fraction:.4f}")
    elif model == "specular":
        bottom = "fix             botwall all wall/reflect zlo EDGE"
    else:
        raise ValueError(f"unknown wall model {model!r}")
    dt_ps = 1e-3
    return WALL_DECK.format(
        gas=gas, model=model, dt_ps=dt_ps, height=40.0, n_gas=n_gas, seed=seed,
        seed2=seed + 3, gas_mass=GAS_LJ[gas]["mass"], t_gas=t_gas, bottom=bottom,
        equil_steps=int(round(equil_ps / dt_ps)),
        prod_steps=int(round(production_ps / dt_ps)),
        dump_every=dump_every or suggest_dump_every(gas, max(t_gas, t_surface),
                                                    40.0, 1.0),
        z_free=1.0,
    )


# --------------------------------------------------------------------------
# Calibrating the gas-solid well
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WellScan:
    eps_pair: float              # eV, pair well depth used for the scan
    sigma: float                 # Angstrom
    e_ads: float                 # eV, summed adsorption energy (positive = bound)
    z_min: float                 # height of the minimum above the top layer, A
    site: tuple[float, float]    # lateral site, fractions of a0


def build_well_scan_deck(spec: AccommodationSpec, *, grid: int = 4,
                         dz: float = 0.1) -> str:
    """Static scan: one gas atom over a frozen slab, a grid of sites x heights.

    The adsorption energy is taken over a grid of lateral sites covering one
    surface cell, so the deepest site is FOUND rather than assumed -- which
    keeps this correct for fcc, bcc and diamond surfaces alike. Energies are
    total potential energy minus its value with the atom out of range, which
    avoids relying on per-atom energy splitting under a hybrid pair style.
    """
    eps, sigma = spec.interaction()
    a0 = spec.solid.a0_guess
    lx = spec.lateral_cells * a0
    if spec.solid.pair_style == "eam":
        pair_coeff = f"pair_coeff      1 1 eam {spec.solid.potential}"
    else:
        pair_coeff = (f"pair_coeff      * * {spec.solid.pair_style} "
                      f"{spec.solid.potential} {spec.solid.label} NULL")
    lines = [
        f"# OpenKapitza well-depth scan -- {spec.gas} over {spec.solid.name}",
        "units           metal", "boundary        p p f",
        "atom_style      atomic",
        f"lattice         {spec.solid.lattice} {a0:.6f}",
        f"region          box block 0 {lx:.6f} 0 {lx:.6f} 0 {spec.z_top:.6f} units box",
        "create_box      2 box",
        f"region          slab block INF INF INF INF 0 {spec.slab_top - 0.1:.6f} units box",
        "create_atoms    1 region slab",
        f"create_atoms    2 single {lx / 2:.6f} {lx / 2:.6f} {spec.z_top - 2.0:.6f} units box",
        f"mass            1 {spec.solid.mass:.4f}",
        f"mass            2 {GAS_LJ[spec.gas]['mass']:.4f}",
        f"pair_style      hybrid/overlay {spec.solid.pair_style} lj/cut {2.5 * sigma:.4f}",
        pair_coeff,
        f"pair_coeff      1 2 lj/cut {eps:.8f} {sigma:.6f} {2.5 * sigma:.4f}",
        f"pair_coeff      2 2 lj/cut 0.0 {sigma:.6f}",
        "group           solid type 1", "group           gas type 2",
        "thermo_style    custom step pe",
        "variable        zt equal bound(solid,zmax)",
        "run             0",
        'print           "WELLREF $(pe:%.10f) $(v_zt:%.6f)"',
    ]
    # Lateral origin at the box centre, which is a lattice point.
    heights = [1.0 + dz * k for k in range(int((2.5 * sigma + 1.5) / dz) + 1)]
    for i in range(grid):
        for j in range(grid):
            fx, fy = i / grid, j / grid
            x, y = lx / 2 + fx * a0, lx / 2 + fy * a0
            for h in heights:
                lines += [
                    f"set             group gas x {x:.6f} y {y:.6f} z $(v_zt+{h:.4f})",
                    "run             0",
                    f'print           "WELLSCAN {fx:.4f} {fy:.4f} {h:.4f} $(pe:%.10f)"',
                ]
    return "\n".join(lines) + "\n"


_WREF = re.compile(r"^WELLREF ([-\d.eE+]+) ([-\d.eE+]+)", re.MULTILINE)
_WSCAN = re.compile(r"^WELLSCAN ([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+)",
                    re.MULTILINE)


def analyse_well_scan(text: str, eps_pair: float, sigma: float) -> WellScan:
    ref = _WREF.search(text)
    if not ref:
        raise ValueError("no WELLREF line: the scan did not run")
    pe_far = float(ref.group(1))
    best = None
    for fx, fy, h, pe in _WSCAN.findall(text):
        e = float(pe) - pe_far
        if best is None or e < best[0]:
            best = (e, float(h), (float(fx), float(fy)))
    if best is None or best[0] >= 0.0:
        raise ValueError("no bound state found in the scan")
    return WellScan(eps_pair, sigma, -best[0], best[1], best[2])


def calibrate_well_depth(lmp: Path, spec: AccommodationSpec, potential_dir: Path,
                         flags: FlagSet, *, target_meV: float | None = None,
                         timeout: int = 1800) -> tuple[AccommodationSpec, WellScan]:
    """Rescale the pair well so the SUMMED adsorption energy hits the target.

    Lennard-Jones energy is linear in epsilon at fixed sigma, and the height of
    the minimum depends on sigma alone, so one scan fixes the scale:
        eps_calibrated = eps_scan * E_target / E_ads(eps_scan).
    """
    if target_meV is None:
        if spec.gas not in PHYSISORPTION_MEV:
            raise ValueError(f"no physisorption target for {spec.gas}; pass target_meV")
        target_meV = PHYSISORPTION_MEV[spec.gas]["value"]
    base = AccommodationSpec(**{**spec.__dict__, "eps_scale": 1.0})
    eps0, sigma = base.interaction()
    source = Path(potential_dir) / spec.solid.potential
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / spec.solid.potential).write_bytes(source.read_bytes())
        out = _run(lmp, work, build_well_scan_deck(base), timeout)
    scan = analyse_well_scan(out, eps0, sigma)
    eps_cal = eps0 * (target_meV * 1e-3) / scan.e_ads
    calibrated = AccommodationSpec(**{**spec.__dict__, "eps_gs": eps_cal,
                                      "sigma_gs": sigma, "calibrated": True})
    flags.add(
        Code.MD_ACCOMMODATION_UNCALIBRATED, Severity.INFO,
        f"{spec.gas}/{spec.solid.name}: mixing-rule pair well {eps0 * 1e3:.1f} meV "
        f"sums to an adsorption energy of {scan.e_ads * 1e3:.0f} meV; rescaled to "
        f"{eps_cal * 1e3:.2f} meV to give {target_meV:.0f} meV.",
        remedy="Recorded for provenance; the calibrated well is what was used.",
        e_ads_uncalibrated_meV=scan.e_ads * 1e3, eps_calibrated_meV=eps_cal * 1e3,
    )
    return calibrated, scan


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

_GEOM = re.compile(r"ACCOM_GEOMETRY z_surface ([-\d.eE+]+) z_free ([-\d.eE+]+) "
                   r"z_top ([-\d.eE+]+)")


_SURF = re.compile(r"ACCOM_SURFACE t_surface ([-\d.eE+]+) n_adsorbed ([-\d.eE+]+) "
                   r"n_sites ([-\d.eE+]+)")


def parse_surface(text: str) -> tuple[float, float, int]:
    """(measured surface temperature, mean adsorbed count, number of sites)."""
    m = _SURF.search(text)
    if not m:
        raise ValueError("no ACCOM_SURFACE line in the LAMMPS output")
    return float(m.group(1)), float(m.group(2)), int(float(m.group(3)))


def parse_geometry(text: str) -> tuple[float, float, float]:
    m = _GEOM.search(text)
    if not m:
        raise ValueError("no ACCOM_GEOMETRY line in the LAMMPS output")
    return tuple(float(g) for g in m.groups())


def read_dump(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(z, vz, ke) arrays of shape [frames, atoms] from a sorted custom dump."""
    zs, vzs, kes = [], [], []
    with open(path, "r", encoding="utf-8") as fh:
        lines = iter(fh)
        for line in lines:
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                n = int(next(lines))
            elif line.startswith("ITEM: ATOMS"):
                block = np.array([next(lines).split() for _ in range(n)], dtype=float)
                order = np.argsort(block[:, 0])
                block = block[order]
                zs.append(block[:, 1]); vzs.append(block[:, 2]); kes.append(block[:, 3])
    if not zs:
        raise ValueError(f"no frames in {path}")
    return np.array(zs), np.array(vzs), np.array(kes)


def _segments(z: np.ndarray, vz: np.ndarray, ke: np.ndarray, z_free: float,
              z_top: float, min_frames: int) -> list[list[tuple[int, int, int, float]]]:
    """Per atom, the ordered free-flight segments (start, end, direction, energy).

    A segment is a run of consecutive frames in which the atom is in the
    force-free region and its normal velocity keeps its sign. One segment is
    one crossing.

    Selection rule: keep every segment whose START is observed (frame > 0),
    including ones the run ends before they finish. A segment straddles a run
    boundary with probability proportional to its duration, so dropping
    boundary-touching segments at BOTH ends would preferentially remove slow
    molecules and push the mean energy up. Counting each crossing by its start
    time is unbiased, and a truncated segment's energy is still exact because
    free-flight energy is constant. (The first draft dropped both ends; caught
    in review before any run.)
    """
    n_frames, n_atoms = z.shape
    margin = 0.5
    out: list[list[tuple[int, int, int, float]]] = []
    for a in range(n_atoms):
        segs: list[tuple[int, int, int, float]] = []
        inside = (z[:, a] > z_free) & (z[:, a] < z_top - margin)
        direction = np.sign(vz[:, a])
        start = None
        for f in range(n_frames + 1):
            continuing = (f < n_frames and inside[f] and direction[f] != 0
                          and start is not None and direction[f] == direction[start])
            if continuing:
                continue
            if start is not None:
                end = f - 1
                if end - start + 1 >= min_frames and start > 0:
                    segs.append((start, end, int(direction[start]),
                                 float(np.median(ke[start:end + 1, a]))))
                start = None
            if f < n_frames and inside[f] and direction[f] != 0:
                start = f
        out.append(segs)
    return out


def free_flight_segments(z: np.ndarray, vz: np.ndarray, ke: np.ndarray,
                         z_free: float, z_top: float, *, min_frames: int = 3
                         ) -> tuple[list[float], list[float]]:
    """Energies of incident (down) and reflected (up) free-flight segments.

    Averaging over segments rather than frames is what makes the means
    flux-weighted. See `_segments` for the selection rule.
    """
    down: list[float] = []
    up: list[float] = []
    for segs in _segments(z, vz, ke, z_free, z_top, min_frames):
        for _, _, d, e in segs:
            (up if d > 0 else down).append(e)
    return down, up


@dataclass(frozen=True)
class PairedEvents:
    e_in: np.ndarray            # incident energy of each completed collision
    e_out: np.ndarray           # the same atom's reflected energy
    n_eligible: int             # incidents started early enough to count
    n_incomplete: int           # of those, no reflection seen before the end

    @property
    def incomplete_fraction(self) -> float:
        return self.n_incomplete / self.n_eligible if self.n_eligible else float("nan")


def paired_events(z: np.ndarray, vz: np.ndarray, ke: np.ndarray, z_free: float,
                  z_top: float, *, min_frames: int = 3,
                  tail_fraction: float = 0.1) -> PairedEvents:
    """Each incident segment paired with the SAME atom's next reflected one.

    Why pair: alpha is a small difference of two large, noisy means when the
    gas barely accommodates (He on a metal). But per collision, a near-elastic
    reflection carries almost the incident energy back, so E_out - E_in has a
    far smaller variance than either energy. Same estimand, much less noise.

    Bias control: an incident whose atom is still trapped when the run ends has
    no partner, and the trapped ones are preferentially the well-accommodated
    ones. Incidents starting in the final `tail_fraction` of the run are not
    counted at all, and the fraction of the rest left without a partner is
    returned so the caller can flag it.
    """
    n_frames = z.shape[0]
    last_start = int(n_frames * (1.0 - tail_fraction))
    e_in, e_out = [], []
    n_eligible = n_incomplete = 0
    for segs in _segments(z, vz, ke, z_free, z_top, min_frames):
        for k, (start, _, d, e) in enumerate(segs):
            if d > 0 or start >= last_start:
                continue
            n_eligible += 1
            if k + 1 < len(segs) and segs[k + 1][2] > 0:
                e_in.append(e); e_out.append(segs[k + 1][3])
            else:
                n_incomplete += 1
    return PairedEvents(np.asarray(e_in), np.asarray(e_out), n_eligible, n_incomplete)


@dataclass
class AccommodationResult:
    alpha: float
    stderr: float
    e_incident: float          # eV, measured mean
    e_reflected: float         # eV
    e_incident_expected: float # 2 k T_g, eV
    e_surface: float           # 2 k T_s, eV
    n_incident: int
    n_reflected: int
    meta: dict = field(default_factory=dict)

    @property
    def incident_check(self) -> float:
        """Measured / expected incident energy; 1.0 when the pipeline is sound."""
        return self.e_incident / self.e_incident_expected


def accommodation_from_energies(down: list[float], up: list[float],
                                t_surface: float, t_gas: float, flags: FlagSet,
                                *, n_boot: int = 400, seed: int = 1,
                                min_events: int = 500,
                                pairs: PairedEvents | None = None) -> AccommodationResult:
    """alpha with a bootstrap standard error, plus the incident-energy check.

    With `pairs`, alpha is the paired estimate <E_out - E_in> / (E_s - <E_in>)
    and the unpaired one is kept in meta as a cross-check; both estimate the
    same quantity, so they must agree within their errors.
    """
    if len(down) < 2 or len(up) < 2:
        raise ValueError(f"too few events: {len(down)} incident, {len(up)} reflected")
    e_s = 2.0 * KB_EV * t_surface
    e_g = 2.0 * KB_EV * t_gas
    d = np.asarray(down); u = np.asarray(up)
    e_i, e_r = float(d.mean()), float(u.mean())
    if abs(e_s - e_i) < 1e-12:
        raise ValueError("T_s and T_g give equal energies; alpha is undefined")
    alpha = (e_r - e_i) / (e_s - e_i)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        bi = d[rng.integers(0, d.size, d.size)].mean()
        br = u[rng.integers(0, u.size, u.size)].mean()
        boots.append((br - bi) / (e_s - bi))
    stderr = float(np.std(boots, ddof=1))

    result = AccommodationResult(alpha, stderr, e_i, e_r, e_g, e_s, d.size, u.size)
    result.meta.update(alpha_unpaired=alpha, stderr_unpaired=stderr)

    if pairs is not None and pairs.e_in.size >= 2:
        pi, po = pairs.e_in, pairs.e_out
        a_p = float((po - pi).mean() / (e_s - pi.mean()))
        pb = []
        for _ in range(n_boot):
            idx = rng.integers(0, pi.size, pi.size)
            pb.append((po[idx] - pi[idx]).mean() / (e_s - pi[idx].mean()))
        se_p = float(np.std(pb, ddof=1))
        result.alpha, result.stderr = a_p, se_p
        result.meta.update(estimator="paired", n_pairs=int(pi.size),
                           incomplete_fraction=pairs.incomplete_fraction)
        alpha, stderr = a_p, se_p
        if pairs.incomplete_fraction > 0.01:
            flags.add(
                Code.MD_ACCOMMODATION_STATISTICS, Severity.CAUTION,
                f"{pairs.incomplete_fraction * 100:.1f} % of counted incidents had "
                f"no reflection before the run ended (trapped atoms).",
                remedy=("Trapped atoms are the well-accommodated ones, so losing them "
                        "biases alpha low. Run longer, or raise tail_fraction."),
                incomplete=pairs.incomplete_fraction,
            )
        joint = math.hypot(se_p, result.meta["stderr_unpaired"])
        if abs(a_p - result.meta["alpha_unpaired"]) > 3.0 * joint:
            flags.add(
                Code.MD_ACCOMMODATION_STATISTICS, Severity.WARNING,
                f"Paired alpha {a_p:.3f} and unpaired {result.meta['alpha_unpaired']:.3f} "
                f"disagree beyond 3 sigma.",
                remedy="The pairing is miscounting events; do not use this value.",
            )
    else:
        result.meta.update(estimator="unpaired")

    if abs(result.incident_check - 1.0) > max(0.03, 4.0 * d.std() / e_g / math.sqrt(d.size)):
        flags.add(
            Code.MD_ACCOMMODATION_STATISTICS,
            Severity.WARNING,
            f"Incident energy {e_i:.5f} eV against the 2kT_g = {e_g:.5f} eV a "
            f"diffusive wall must emit (ratio {result.incident_check:.3f}).",
            remedy=("The segment analysis or the wall setup is wrong; the alpha "
                    "built on it is not trustworthy. Check the free-flight "
                    "region is truly force-free."),
            ratio=result.incident_check,
        )
    if min(d.size, u.size) < min_events or stderr > 0.05:
        flags.add(
            Code.MD_ACCOMMODATION_STATISTICS,
            Severity.CAUTION,
            f"alpha = {alpha:.3f} +/- {stderr:.3f} from {d.size} incident and "
            f"{u.size} reflected events.",
            remedy="Run longer or use more gas atoms to tighten the estimate.",
            stderr=stderr,
        )
    return result


def _run(lmp: Path, work: Path, deck: str, timeout: int) -> str:
    path = work / "accom.in"
    path.write_text(deck)
    run = subprocess.run([str(lmp), "-in", path.name], cwd=work,
                         capture_output=True, text=True, timeout=timeout)
    if run.returncode != 0:
        errors = [l for l in run.stdout.splitlines() if "ERROR" in l]
        raise RuntimeError(f"LAMMPS failed: {errors or run.stdout[-400:]}")
    return run.stdout


def run_wall_validation(lmp: Path, model: str, flags: FlagSet, *,
                        timeout: int = 3600, **kwargs) -> AccommodationResult:
    """Run the known-answer deck and return the measured alpha."""
    t_s = kwargs.get("t_surface", 300.0); t_g = kwargs.get("t_gas", 400.0)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        out = _run(lmp, work, build_wall_validation_deck(model, **kwargs), timeout)
        _, z_free, z_top = parse_geometry(out)
        z, vz, ke = read_dump(work / "gas.dump")
    down, up = free_flight_segments(z, vz, ke, z_free, z_top)
    pairs = paired_events(z, vz, ke, z_free, z_top)
    result = accommodation_from_energies(down, up, t_s, t_g, flags, pairs=pairs)
    result.meta.update(model=model, **{k: v for k, v in kwargs.items()})
    return result


def run_accommodation(lmp: Path, spec: AccommodationSpec, potential_dir: Path,
                      flags: FlagSet, *, timeout: int = 7200) -> AccommodationResult:
    """Gas-surface scattering on a real slab; returns alpha with its error."""
    source = Path(potential_dir) / spec.solid.potential
    if not source.is_file():
        raise FileNotFoundError(f"potential not found: {source}")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / spec.solid.potential).write_bytes(source.read_bytes())
        out = _run(lmp, work, build_accommodation_deck(spec), timeout)
        z_surface, z_free, z_top = parse_geometry(out)
        t_surf, n_ads, n_sites = parse_surface(out)
        z, vz, ke = read_dump(work / "gas.dump")
    if z_surface + spec.cutoff >= z_free:
        flags.add(
            Code.MD_ACCOMMODATION_STATISTICS, Severity.WARNING,
            f"The surface rose to z = {z_surface:.2f} A, within the force range "
            f"of the free-flight boundary at {z_free:.2f} A.",
            remedy="Raise z_free; segments may not be force-free.",
        )
    down, up = free_flight_segments(z, vz, ke, z_free, z_top)
    pairs = paired_events(z, vz, ke, z_free, z_top)
    # The formula needs the temperature the surface ACTUALLY had. The hot gas
    # heats it, and using the thermostat setpoint would silently bias alpha.
    result = accommodation_from_energies(down, up, t_surf, spec.t_gas, flags, pairs=pairs)
    eps, sigma = spec.interaction()
    coverage = n_ads / n_sites if n_sites else float("nan")
    result.meta.update(gas=spec.gas, solid=spec.solid.name, eps_gs=eps,
                       sigma_gs=sigma, eps_scale=spec.eps_scale,
                       potential=spec.solid.potential, z_surface=z_surface,
                       t_surface_set=spec.t_surface, t_surface_measured=t_surf,
                       n_adsorbed=n_ads, n_sites=n_sites, coverage=coverage,
                       calibrated=spec.calibrated)
    if not spec.calibrated:
        flags.add(
            Code.MD_ACCOMMODATION_UNCALIBRATED, Severity.WARNING,
            f"{spec.gas}/{spec.solid.name}: the gas-solid well was not calibrated "
            f"to a measured adsorption energy.",
            remedy=("Run calibrate_well_depth first. Mixing-rule wells summed over "
                    "the surface can reach chemisorption strength: uncalibrated "
                    "Ar on Ni condensed onto the surface and lifted Ni atoms out."),
        )
    if spec.gas in QUANTUM_GASES:
        flags.add(
            Code.MD_ACCOMMODATION_QUANTUM_GAS, Severity.CAUTION,
            f"{spec.gas}: classical scattering over-estimates alpha for light gases. "
            f"He and Ne exchange energy mainly through elastic and single-phonon "
            f"events, which classical dynamics does not reproduce.",
            remedy=("Read the MD value as an UPPER estimate for the clean surface "
                    "(Fan & Manson, arXiv:0804.2269, report classical values 'much "
                    "too large for He and Ne'). Technical surfaces still sit higher."),
        )
    if coverage > 0.1:
        flags.add(
            Code.MD_ACCOMMODATION_COVERAGE, Severity.WARNING,
            f"{n_ads:.1f} gas atoms adsorbed on average on {n_sites} surface sites "
            f"({coverage:.2f} per site).",
            remedy=("With gas-gas forces off, adsorbed atoms can stack on one site, "
                    "and incoming atoms then scatter off a perturbed surface. "
                    "Reduce n_gas; this is not a clean-surface value as it stands."),
            coverage=coverage,
        )
    if abs(t_surf - spec.t_surface) / spec.t_surface > 0.03:
        flags.add(
            Code.MD_ACCOMMODATION_SURFACE_HEATED, Severity.CAUTION,
            f"Surface ran at {t_surf:.1f} K against a {spec.t_surface:.0f} K "
            f"setpoint; alpha was computed with the measured value.",
            remedy="Reduce the gas flux or thicken the thermostatted layer.",
            t_surface=t_surf,
        )
    flags.add(
        Code.MD_ACCOMMODATION_CLEAN_SURFACE, Severity.CAUTION,
        f"{spec.gas} on {spec.solid.name}: alpha = {result.alpha:.3f} is a "
        f"CLEAN-SURFACE value.",
        remedy=("Oxides and adsorbed layers on technical surfaces generally raise "
                "alpha, most of all for light gases. Treat this as closer to a "
                "bound than to the value for a machined part."),
    )
    return result


def target_range_scales(gas: str) -> tuple[float, float, float]:
    """Well-depth scales spanning the literature range of the adsorption energy.

    Applied to a CALIBRATED spec, (low/value, 1, high/value) reruns alpha at the
    ends of the measured physisorption range, which is the honest uncertainty
    of the gas-solid interaction -- better founded than an arbitrary x0.5/x2.
    """
    t = PHYSISORPTION_MEV[gas]
    lo, hi = t["range"]
    return (lo / t["value"], 1.0, hi / t["value"])


def run_sensitivity(lmp: Path, spec: AccommodationSpec, potential_dir: Path,
                    flags: FlagSet, scales: tuple[float, ...] = (0.5, 1.0, 2.0),
                    **kwargs) -> list[AccommodationResult]:
    """alpha at several gas-solid well depths. Quote the range, not one value."""
    results = []
    for s in scales:
        sub = AccommodationSpec(**{**spec.__dict__, "eps_scale": s})
        results.append(run_accommodation(lmp, sub, potential_dir, flags, **kwargs))
    return results


def to_quantity(result: AccommodationResult, *,
                spread: tuple[float, float] | None = None) -> Quantity:
    """alpha as an MD_COMPUTED candidate, capped by the section-11 policy.

    If a well-depth `spread` (low, high) is supplied, it becomes the stated
    uncertainty, because it -- not the bootstrap error -- dominates.
    """
    unc = result.stderr
    note = (f"clean surface, {result.meta.get('gas')} on {result.meta.get('solid')}, "
            f"T_s {result.e_surface / (2 * KB_EV):.0f} K, "
            f"{result.n_reflected} reflections, bootstrap +/- {result.stderr:.3f}")
    if spread is not None:
        unc = max(unc, 0.5 * (spread[1] - spread[0]))
        note += f"; well-depth range {spread[0]:.3f}-{spread[1]:.3f}"
    q = Quantity(si=result.alpha, unit="-", uncertainty=unc,
                 provenance=Provenance.MD_COMPUTED, confidence=Confidence.HIGH,
                 citation=str(result.meta.get("potential", "")), note=note,
                 name="accommodation")
    return cap_md("accommodation", q)
