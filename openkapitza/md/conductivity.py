"""Bulk lattice thermal conductivity by direct NEMD (work package 5c).

Governed by section 11 of the agent instruction. The value can FILL k_s for a
non-metal at LOW confidence. For a metal it is the phonon part only -- a lower
bound that misses most of the heat, since classical MD has no electrons -- and
the fill policy refuses it as total k.

METHOD (the official examples/KAPPA Langevin layout)

A fully periodic cell with no interface and no free surface. Two thin slabs
half a box apart are thermostatted hot and cold by `fix langevin` restricted
to their region through a `temp/region` bias (`fix_modify ... temp`), exactly
as in examples/KAPPA/in.langevin. Heat flows from hot to cold through BOTH
halves of the periodic box, so

    J = (dE/dt) / (2 A)        k = J / |dT/dz|

with dE/dt taken from the cumulative energy each reservoir exchanges
(`tally yes`), fitted against time. The hot and cold rates must agree, or the
system is not in steady state. Each half of the box gives an independent
gradient; their disagreement is part of the stated error.

SIZE EFFECTS

In a crystal, phonon mean free paths exceed any affordable box, and the
reservoirs scatter phonons. k grows with length. The standard correction
(Schelling, Phillpot & Keblinski, PRB 65, 144306, 2002) is

    1/k(L) = 1/k_inf + c / L

fitted over several lengths. It assumes one effective mean free path; for
materials whose heat is carried by a broad spread of mean free paths, notably
silicon, the linear form can UNDER-estimate k_inf when L is short against the
dominant mean free paths (Sellan et al., PRB 81, 214305, 2010). Lengths should
therefore be as long as compute allows, and the extrapolation ratio
k_inf / k(L_max) is reported and flagged.

TEMPERATURE

Classical MD populates every mode with k_B T. Below the Debye temperature that
overpopulates high-frequency modes. Section 11 says to report the uncorrected
value with the flag; no correction is applied here.
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
from .elastic import CubicSpec, stage_potential
from .fill_policy import cap_md

# eV/ps per A^2, divided by K/A, to W/(m K): 1.602177e-19 / 1e-12 / 1e-10.
EV_PS_A_TO_W_MK = 1.602176634e-19 / 1e-12 / 1e-10

# Reservoir slab thickness and the margin excluded next to each reservoir, as
# fractions of the box length. The profile bends near a thermostat (the
# official example's README says the same), so those bins are not fitted.
RESERVOIR_FRACTION = 0.05
MARGIN_FRACTION = 0.05


@dataclass
class KappaSpec:
    solid: CubicSpec
    temperature: float = 300.0
    delta_t: float = 40.0            # K, hot minus cold reservoir setpoint
    a0: float | None = None          # potential's own lattice; None: a0_guess
    cross_cells: int = 4             # lateral size in unit cells
    length_cells: int = 40           # box length along z in unit cells
    bins: int | None = None          # None: one bin per two unit cells
    timestep_fs: float = 1.0
    npt_ps: float = 20.0
    nvt_ps: float = 20.0
    steady_ps: float = 200.0
    production_ps: float = 500.0
    damp_ps: float = 0.1             # reservoir Langevin damping
    thermo_every: int = 1000
    theta_debye: float | None = None  # K, for the quantum flag only
    seed: int = 74581

    @property
    def n_bins(self) -> int:
        if self.bins is not None:
            return self.bins
        return max(20, self.length_cells // 2)

    @property
    def t_hot(self) -> float:
        return self.temperature + 0.5 * self.delta_t

    @property
    def t_cold(self) -> float:
        return self.temperature - 0.5 * self.delta_t


DECK = """\
# OpenKapitza bulk thermal conductivity -- {name}, {length_cells} cells along z
# Periodic Langevin NEMD after examples/KAPPA/in.langevin. Generated; see
# openkapitza/md/conductivity.py.
units           metal
boundary        p p p
atom_style      atomic
lattice         {lattice} {a0:.6f}
region          box block 0 {nx} 0 {nx} 0 {nz}
create_box      1 box
create_atoms    1 box
mass            1 {mass:.4f}
pair_style      {pair_style}
{pair_coeff}
timestep        {dt_ps:.6f}
neighbor        2.0 bin

# ---- thermal expansion: relax the box at T, then fix it at the MEAN size ----
velocity        all create {t_init:.2f} {seed} mom yes rot yes dist gaussian
variable        Lx equal lx
variable        Ly equal ly
variable        Lz equal lz
fix             npt all npt temp {temperature:.2f} {temperature:.2f} 0.1 aniso 0.0 0.0 1.0
fix             box all ave/time 10 100 1000 v_Lx v_Ly v_Lz ave running
thermo          {thermo_every}
run             {npt_steps}
unfix           npt
change_box      all x final 0 $(f_box[1]) y final 0 $(f_box[2]) z final 0 $(f_box[3]) remap units box
unfix           box
fix             nvt all nvt temp {temperature:.2f} {temperature:.2f} 0.1
run             {nvt_steps}
unfix           nvt
print           "KAPPA_BOX $(lx:%.6f) $(ly:%.6f) $(lz:%.6f) $(atoms)"

# ---- reservoirs: hot at the bottom, cold half a box away ----
region          hot  block INF INF INF INF 0 $({res}*lz) units box
region          cold block INF INF INF INF $(0.5*lz) $((0.5+{res})*lz) units box
compute         Thot  all temp/region hot
compute         Tcold all temp/region cold
fix             nve all nve
fix             hot  all langevin {t_hot:.2f} {t_hot:.2f} {damp_ps:.4f} {seed_hot} tally yes
fix             cold all langevin {t_cold:.2f} {t_cold:.2f} {damp_ps:.4f} {seed_cold} tally yes
fix_modify      hot  temp Thot
fix_modify      cold temp Tcold
thermo_style    custom step temp c_Thot c_Tcold f_hot f_cold
run             {steady_steps}

# ---- production: restart the clock and the energy tallies ----
reset_timestep  0
unfix           hot
unfix           cold
fix             hot  all langevin {t_hot:.2f} {t_hot:.2f} {damp_ps:.4f} {seed_hot2} tally yes
fix             cold all langevin {t_cold:.2f} {t_cold:.2f} {damp_ps:.4f} {seed_cold2} tally yes
fix_modify      hot  temp Thot
fix_modify      cold temp Tcold
compute         layers all chunk/atom bin/1d z lower {bin_frac:.8f} units reduced
fix             prof all ave/chunk 10 {n_samples} {production_steps} layers temp file profile.dat
print           "KAPPA_PRODUCTION_BEGIN"
run             {production_steps}
print           "KAPPA_PRODUCTION_END"
"""


def build_kappa_deck(spec: KappaSpec) -> str:
    dt_ps = spec.timestep_fs * 1e-3
    steps = lambda ps: max(1, int(round(ps / dt_ps)))
    prod = steps(spec.production_ps)
    # Make the production run a whole number of thermo intervals and of
    # 10-step samples, so the single ave/chunk frame lands on its last step.
    prod = max(spec.thermo_every, prod - prod % spec.thermo_every)
    return DECK.format(
        name=spec.solid.name, length_cells=spec.length_cells,
        lattice=spec.solid.lattice, a0=spec.a0 or spec.solid.a0_guess,
        nx=spec.cross_cells, nz=spec.length_cells, mass=spec.solid.mass,
        pair_style=spec.solid.pair_style, pair_coeff=spec.solid.pair_coeff(),
        dt_ps=dt_ps, t_init=2.0 * spec.temperature, seed=spec.seed,
        temperature=spec.temperature, thermo_every=spec.thermo_every,
        npt_steps=steps(spec.npt_ps), nvt_steps=steps(spec.nvt_ps),
        res=RESERVOIR_FRACTION, t_hot=spec.t_hot, t_cold=spec.t_cold,
        damp_ps=spec.damp_ps, seed_hot=spec.seed + 1, seed_cold=spec.seed + 2,
        seed_hot2=spec.seed + 3, seed_cold2=spec.seed + 4,
        steady_steps=steps(spec.steady_ps), production_steps=prod,
        n_samples=prod // 10, bin_frac=1.0 / spec.n_bins,
    )


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_BOX = re.compile(r"^KAPPA_BOX ([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+) (\d+)", re.MULTILINE)


def parse_box(text: str) -> tuple[float, float, float, int]:
    m = _BOX.search(text)
    if not m:
        raise ValueError("no KAPPA_BOX line in the LAMMPS output")
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))


def parse_production_thermo(text: str) -> np.ndarray:
    """Rows (step, temp, T_hot, T_cold, E_hot, E_cold) from the production run."""
    try:
        block = text.split("KAPPA_PRODUCTION_BEGIN", 1)[1].split("KAPPA_PRODUCTION_END", 1)[0]
    except IndexError:
        raise ValueError("production markers not found in the LAMMPS output") from None
    rows, header = [], False
    for line in block.splitlines():
        parts = line.split()
        if parts[:2] == ["Step", "Temp"]:
            header = True
            continue
        if header:
            if line.startswith("Loop time"):
                break
            try:
                rows.append([float(p) for p in parts[:6]])
            except ValueError:
                continue
    if len(rows) < 3:
        raise ValueError("too few production thermo rows to fit a flux")
    return np.asarray(rows)


def read_profile(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(reduced z centre, temperature, atom count) of the last ave/chunk frame."""
    frames: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 3:              # frame header: step nchunks total
                if current:
                    frames.append(current)
                current = []
            elif len(parts) >= 4:            # chunk coord ncount temp
                current.append((float(parts[1]), float(parts[3]), float(parts[2])))
    if current:
        frames.append(current)
    if not frames:
        raise ValueError(f"no profile frames in {path}")
    arr = np.asarray(frames[-1])
    return arr[:, 0], arr[:, 1], arr[:, 2]


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

@dataclass
class KappaPoint:
    length: float                # box length along z, Angstrom
    k: float                     # W/(m K)
    stderr: float
    k_halves: tuple[float, float]
    flux: float                  # W/m^2
    gradient: float              # K/m, mean magnitude
    rate_hot: float              # eV/ps, energy injected by the hot reservoir
    rate_cold: float             # eV/ps, energy removed by the cold reservoir
    meta: dict = field(default_factory=dict)


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and its standard error."""
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    dof = max(1, x.size - 2)
    resid = y - A @ coef
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(coef[0]), float(math.sqrt(cov[0, 0]))


def kappa_from_run(thermo: np.ndarray, z: np.ndarray, temp: np.ndarray,
                   counts: np.ndarray, box: tuple[float, float, float],
                   timestep_fs: float, flags: FlagSet, *,
                   label: str = "") -> KappaPoint:
    """k for one length from the reservoir tallies and the temperature profile."""
    lx, ly, lz = box
    area = lx * ly
    t_ps = thermo[:, 0] * timestep_fs * 1e-3
    s_hot, _ = _linfit(t_ps, thermo[:, 4])
    s_cold, _ = _linfit(t_ps, thermo[:, 5])
    # One reservoir adds energy and the other removes it; whichever sign
    # convention the tally uses, the magnitudes must match in steady state.
    r_hot, r_cold = abs(s_hot), abs(s_cold)
    rate = 0.5 * (r_hot + r_cold)
    if (s_hot > 0) == (s_cold > 0):
        raise ValueError("both reservoirs moved energy the same way; not a gradient run")
    imbalance = abs(r_hot - r_cold) / rate
    if imbalance > 0.05:
        flags.add(
            Code.MD_K_SIZE_EXTRAPOLATION, Severity.CAUTION,
            f"{label}: hot and cold reservoirs exchanged energy at rates "
            f"differing by {imbalance * 100:.0f} %; steady state not reached.",
            remedy="Lengthen the steady-state run before production.",
            imbalance=imbalance,
        )
    flux_ev = rate / (2.0 * area)                    # eV/ps/A^2, both halves

    res, mar = RESERVOIR_FRACTION, MARGIN_FRACTION
    ok = counts > 0
    halves = []
    for lo, hi in ((res + mar, 0.5 - mar), (0.5 + res + mar, 1.0 - mar)):
        sel = ok & (z > lo) & (z < hi)
        if sel.sum() < 4:
            raise ValueError("too few profile bins in a bulk half; use more bins")
        g, ge = _linfit(z[sel] * lz, temp[sel])     # K/A
        halves.append((abs(g), ge))
    ks = tuple(flux_ev / g * EV_PS_A_TO_W_MK for g, _ in halves)
    k = 0.5 * (ks[0] + ks[1])
    fit_rel = math.hypot(*(ge / g for g, ge in halves)) / 2.0
    stderr = max(0.5 * abs(ks[0] - ks[1]), k * fit_rel, k * imbalance / 2.0)
    grad = 0.5 * (halves[0][0] + halves[1][0]) * 1e10
    return KappaPoint(lz, k, stderr, ks, flux_ev * 1.602176634e-19 / 1e-12 / 1e-20,
                      grad, r_hot, r_cold, meta={"imbalance": imbalance})


@dataclass
class KappaResult:
    k_inf: float
    stderr: float
    points: list[KappaPoint]
    slope: float                 # c in 1/k = 1/k_inf + c/L, (m K/W) * A
    extrapolation_ratio: float   # k_inf / k(L_max)
    meta: dict = field(default_factory=dict)


def extrapolate(points: list[KappaPoint], flags: FlagSet, *,
                label: str = "") -> KappaResult:
    """Weighted fit of 1/k against 1/L; k_inf from the intercept."""
    if len(points) < 2:
        raise ValueError("need at least two lengths to extrapolate")
    pts = sorted(points, key=lambda p: p.length)
    x = np.array([1.0 / p.length for p in pts])
    y = np.array([1.0 / p.k for p in pts])
    sy = np.array([p.stderr / p.k ** 2 for p in pts])
    w = 1.0 / np.maximum(sy, 1e-12) ** 2
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    a, c = cov @ (A.T @ (w * y))
    chi2 = float(np.sum(w * (y - (a + c * x)) ** 2))
    dof = len(pts) - 2
    if dof > 0 and chi2 / dof > 1.0:
        cov = cov * (chi2 / dof)           # inflate for scatter beyond the bars
    if a <= 0:
        raise ValueError("non-positive intercept: 1/k does not extrapolate; "
                         "lengths too short or too noisy")
    k_inf = 1.0 / a
    stderr = math.sqrt(cov[0, 0]) / a ** 2
    ratio = k_inf / pts[-1].k
    result = KappaResult(k_inf, stderr, pts, float(c), ratio,
                         meta={"chi2_per_dof": chi2 / dof if dof > 0 else float("nan")})
    sev = Severity.INFO
    reasons = []
    if len(pts) < 3:
        sev = Severity.WARNING
        reasons.append("only two lengths, so linearity in 1/L is untested")
    if ratio > 2.0:
        sev = Severity.WARNING
        reasons.append(f"k_inf is {ratio:.1f}x the longest cell's k")
    if c < 0:
        sev = Severity.WARNING
        reasons.append("k falls with length, which boundary scattering cannot "
                       "produce: the runs are unconverged or the cells too short")
    if stderr > 0.5 * k_inf:
        sev = Severity.CRITICAL
        reasons.append(f"the error ({stderr:.0f}) is more than half the value, so "
                       f"k_inf is undetermined; do not use it")
    if dof > 0 and chi2 / dof > 4.0:
        sev = Severity.WARNING
        reasons.append(f"1/k is not linear in 1/L (chi2/dof {chi2 / dof:.1f})")
    flags.add(
        Code.MD_K_SIZE_EXTRAPOLATION, sev,
        f"{label}: k_inf = {k_inf:.1f} +/- {stderr:.1f} W/mK extrapolated from "
        f"{len(pts)} lengths ({pts[0].length / 10:.1f}-{pts[-1].length / 10:.1f} nm; "
        f"longest gives {pts[-1].k:.1f})"
        + (": " + "; ".join(reasons) if reasons else "."),
        remedy=("The linear 1/L form can under-estimate k_inf for materials with "
                "a broad spread of phonon mean free paths (Sellan et al. 2010). "
                "Add longer cells, or cross-check with Green-Kubo."),
        ratio=ratio,
    )
    return result


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def _run(lmp: Path, work: Path, deck: str, timeout: int) -> str:
    (work / "kappa.in").write_text(deck)
    run = subprocess.run([str(lmp), "-in", "kappa.in"], cwd=work,
                         capture_output=True, text=True, timeout=timeout)
    if run.returncode != 0:
        errors = [l for l in run.stdout.splitlines() if "ERROR" in l]
        raise RuntimeError(f"LAMMPS failed: {errors or run.stdout[-400:]}")
    return run.stdout


def run_length(lmp: Path, spec: KappaSpec, potential_dir: Path, flags: FlagSet,
               *, timeout: int = 14400) -> KappaPoint:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        stage_potential(spec.solid, potential_dir, work)
        out = _run(lmp, work, build_kappa_deck(spec), timeout)
        lx, ly, lz, n_atoms = parse_box(out)
        thermo = parse_production_thermo(out)
        z, temp, counts = read_profile(work / "profile.dat")
    point = kappa_from_run(thermo, z, temp, counts, (lx, ly, lz), spec.timestep_fs,
                           flags, label=f"{spec.solid.name} L={lz / 10:.1f} nm")
    point.meta.update(n_atoms=n_atoms, length_cells=spec.length_cells,
                      t_mean_profile=float(np.mean(temp[counts > 0])))
    return point


def run_conductivity(lmp: Path, spec: KappaSpec, potential_dir: Path,
                     flags: FlagSet, *, lengths: tuple[int, ...] = (20, 40, 80),
                     is_metal: bool = False, **kwargs) -> KappaResult:
    """k(L) at several lengths (unit cells) and the 1/L extrapolation."""
    points = []
    for n in lengths:
        sub = KappaSpec(**{**spec.__dict__, "length_cells": n})
        points.append(run_length(lmp, sub, potential_dir, flags, **kwargs))
    result = extrapolate(points, flags, label=spec.solid.name)
    result.meta.update(solid=spec.solid.name, potential=spec.solid.potential,
                       temperature=spec.temperature, is_metal=is_metal)
    if spec.theta_debye and spec.temperature < spec.theta_debye:
        flags.add(
            Code.QUANTUM_EFFECTS, Severity.CAUTION,
            f"{spec.solid.name}: T = {spec.temperature:.0f} K is below the Debye "
            f"temperature {spec.theta_debye:.0f} K; classical MD overpopulates "
            f"high-frequency modes. The value is reported uncorrected.",
            remedy="Treat as indicative; no quantum correction has been applied.",
        )
    if is_metal:
        flags.add(
            Code.MD_K_METAL_PHONON_ONLY, Severity.WARNING,
            f"{spec.solid.name}: {result.k_inf:.1f} W/mK is the lattice (phonon) "
            f"part only; classical MD has no conduction electrons.",
            remedy=("Not a total k. Add the electronic part from the electrical "
                    "conductivity (Wiedemann-Franz) before any use."),
        )
    return result


def to_quantity(result: KappaResult) -> Quantity:
    """k_inf as an MD_COMPUTED candidate, capped by the section-11 policy.

    For a metal it is still returned -- for COMPARE mode -- but named as the
    lattice part, and the fill policy refuses it as total k.
    """
    metal = bool(result.meta.get("is_metal"))
    longest = result.points[-1]
    note = (f"{'LATTICE PART ONLY; ' if metal else ''}NEMD 1/L extrapolation over "
            f"{len(result.points)} lengths up to {longest.length / 10:.1f} nm "
            f"(k = {longest.k:.1f} there), T {result.meta.get('temperature', 0):.0f} K, "
            f"classical, uncorrected")
    q = Quantity(si=result.k_inf, unit="W/mK", uncertainty=result.stderr,
                 provenance=Provenance.MD_COMPUTED, confidence=Confidence.HIGH,
                 citation=str(result.meta.get("potential", "")), note=note,
                 name="thermal_conductivity_lattice" if metal else "thermal_conductivity")
    return cap_md("thermal_conductivity", q)
