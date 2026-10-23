"""Amorphous SiO2 by melt-and-quench, with the validation gate (Phase 2).

Governed by section 3 of the agent instruction: always report the quench
rate and state it against experiment, and validate density, RDF peaks, bond
angles and coordination before the glass is used.

PROTOCOL

  1. Start from ideal beta-cristobalite (2.17 g/cm^3), scaled to the target
     box -- optionally with x, y fixed to a crystalline Si supercell so the
     glass can later be joined to it.
  2. Melt at constant volume (glass density 2.20 g/cm^3) at T_melt.
  3. Quench linearly in NVT from T_melt to T_final at `quench_rate`.
  4. Anneal at zero pressure (z only when x, y are fixed), set the box to its
     MEAN over the second half (the lesson from the NEMD relaxation: freezing
     a barostatted box mid-oscillation stores strain), then minimise.

THE QUENCH RATE IS A REAL LIMITATION. MD reaches 1e11-1e13 K/s; laboratory
glasses form near 1 K/s. Faster quenches freeze in more small rings and
defects and a lower fictive-temperature-like disorder. Every downstream h_K
inherits this, so the rate is stated in a flag that is always raised.

TARGETS (experimental, as collected in arXiv:1109.2852): density 2.20 g/cm^3;
Si-O 1.61 A, O-O ~2.63 A, Si-Si ~3.10 A (Johnson, Wright & Sinclair 1983);
Si-O-Si 140-150 deg (Dupree & Pettifer 1991; 144 deg, Mozzi & Warren 1969);
O-Si-O 109.4 deg; coordination 4 (Si) and 2 (O).
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..schema.flags import Code, FlagSet, Severity
from .analysis import (
    Cell, angle_peak, bond_angles, coordination, partial_rdf, read_lammps_data,
    write_lammps_data,
)
from .crystals import beta_cristobalite

SI, O = 1, 2
GLASS_DENSITY = 2.20                 # g/cm^3
SIO2_FORMULA_MASS = 60.0843          # amu
AVOGADRO = 6.02214076e23
LAB_QUENCH_RATE = 1.0                # K/s, order of magnitude for a real glass

# metric: (target, tolerance, unit, citation)
TARGETS: dict[str, tuple[float, float, str, str]] = {
    "density": (2.20, 0.11, "g/cm3", "a-SiO2, standard value"),
    "r_SiO": (1.61, 0.04, "A", "Johnson, Wright & Sinclair (1983)"),
    "r_OO": (2.63, 0.08, "A", "Johnson, Wright & Sinclair (1983)"),
    "r_SiSi": (3.10, 0.08, "A", "Johnson, Wright & Sinclair (1983)"),
    "angle_SiOSi": (145.0, 10.0, "deg", "140-150, Dupree & Pettifer (1991); 144, Mozzi & Warren (1969)"),
    "angle_OSiO": (109.4, 3.0, "deg", "tetrahedral"),
    "frac_Si4": (1.0, 0.05, "-", "fourfold Si"),
    "frac_O2": (1.0, 0.05, "-", "twofold O"),
}
SI_O_CUTOFF = 2.0                    # A, inside the first Si-O minimum


@dataclass
class QuenchSpec:
    cells: tuple[int, int, int] = (4, 4, 4)       # cristobalite cells to start from
    box: tuple[float, float, float] | None = None # target box (A); None: cubic at 2.20
    fix_lateral: bool = False                      # keep x, y at box[0:2] throughout
    t_melt: float = 6000.0
    melt_ps: float = 20.0
    t_final: float = 300.0
    quench_rate: float = 1.0e13                    # K/s
    anneal_ps: float = 50.0
    timestep_fs: float = 0.5
    seed: int = 81235
    potential: str = "SiO.tersoff"
    n_segments: int = 10                           # restart checkpoints along the quench

    @property
    def quench_ps(self) -> float:
        return (self.t_melt - self.t_final) / self.quench_rate * 1e12

    def steps(self, ps: float) -> int:
        n = max(20, int(round(ps * 1000.0 / self.timestep_fs)))
        return -(-n // 20) * 20          # multiple of 20 for the averaging windows


def start_cell(spec: QuenchSpec) -> Cell:
    """Cristobalite scaled to the target box at glass density."""
    c = beta_cristobalite(spec.cells, si_type=SI, o_type=O)
    n_formula = int((c.types == SI).sum())
    vol = n_formula * SIO2_FORMULA_MASS / AVOGADRO / GLASS_DENSITY / 1e-24   # A^3
    if spec.box is None:
        side = vol ** (1.0 / 3.0)
        target = np.array([side, side, side])
    else:
        target = np.array(spec.box, dtype=float)
        if spec.fix_lateral:
            # x, y as given; z chosen for glass density.
            target[2] = vol / (target[0] * target[1])
    scale = target / c.lengths
    return Cell(c.pos * scale, c.types, np.zeros(3), target, dict(c.masses))


# The deck is built from stages so a killed run can resume. The melt and each
# quench segment end with `write_restart` and a progress marker; a resumed
# deck reads the last completed segment and carries on. Each segment ramps
# only between its own end temperatures, so a resumed quench follows exactly
# the schedule an uninterrupted one would. (A 3-hour quench was lost at 68 %
# to a container restart before this existed; cluster walltime limits are
# the same risk.)
HEAD = """\
# OpenKapitza melt-quench a-SiO2. Generated; see openkapitza/structure/amorphous.py.
# Quench rate {rate:.3g} K/s ({t_melt:.0f} -> {t_final:.0f} K over {quench_ps:.1f} ps,
# {n_seg} segments).
units           metal
boundary        p p p
atom_style      atomic
{structure}
pair_style      tersoff
pair_coeff      * * {potential} Si O
timestep        {dt_ps:.6f}
neighbor        2.0 bin
thermo          {thermo}
thermo_style    custom step temp pe press lx ly lz density
"""

MELT = """\
velocity        all create {t_init:.1f} {seed} mom yes rot yes dist gaussian
fix             md all nvt temp {t_melt:.1f} {t_melt:.1f} 0.05
run             {melt_steps}
unfix           md
print           "QUENCH_MELTED"
write_restart   seg00.restart
print           "0" file progress.txt screen no
"""

SEGMENT = """\
fix             md all nvt temp {t0:.2f} {t1:.2f} 0.05
run             {steps}
unfix           md
write_restart   seg{k:02d}.restart
print           "{k}" file progress.txt screen no
"""

ANNEAL = """\
print           "QUENCH_QUENCHED"
# Anneal at zero pressure; set the box to its second-half mean before the
# barostat comes off, then minimise.
reset_timestep  0
variable        lxv equal lx
variable        lyv equal ly
variable        lzv equal lz
fix             md all npt temp {t_final:.1f} {t_final:.1f} 0.05 {baro}
fix             bav all ave/time 10 {nrep} {anneal_steps} v_lxv v_lyv v_lzv
run             {anneal_steps}
unfix           md
change_box      all x final $(xlo) $(xlo+f_bav[1]) y final $(ylo) $(ylo+f_bav[2]) z final $(zlo) $(zlo+f_bav[3]) remap units box
unfix           bav
minimize        1.0e-8 1.0e-10 10000 100000
write_data      glass.data
print           "QUENCH_DONE density $(density:%.5f) lx $(lx:%.4f) ly $(ly:%.4f) lz $(lz:%.4f) pe $(pe:%.4f)"
"""


def segment_temperatures(spec: QuenchSpec) -> list[float]:
    return [spec.t_melt + (spec.t_final - spec.t_melt) * k / spec.n_segments
            for k in range(spec.n_segments + 1)]


def build_quench_deck(spec: QuenchSpec, *, resume_from: int | None = None) -> str:
    """The whole schedule, or its remainder after completed segment `resume_from`."""
    anneal = spec.steps(spec.anneal_ps)
    baro = "z 0.0 0.0 0.5" if spec.fix_lateral else "iso 0.0 0.0 0.5"
    structure = ("read_data       start.data" if resume_from is None
                 else f"read_restart    seg{resume_from:02d}.restart")
    out = [HEAD.format(rate=spec.quench_rate, t_melt=spec.t_melt, t_final=spec.t_final,
                       quench_ps=spec.quench_ps, n_seg=spec.n_segments, structure=structure,
                       potential=spec.potential, dt_ps=spec.timestep_fs * 1e-3,
                       thermo=max(100, spec.steps(1.0)))]
    if resume_from is None:
        out.append(MELT.format(t_init=2.0 * spec.t_melt, seed=spec.seed,
                               t_melt=spec.t_melt, melt_steps=spec.steps(spec.melt_ps)))
        start = 1
    else:
        out.append(f"print           \"QUENCH_RESUMED after segment {resume_from}\"\n")
        start = resume_from + 1
    temps = segment_temperatures(spec)
    seg_steps = spec.steps(spec.quench_ps / spec.n_segments)
    for k in range(start, spec.n_segments + 1):
        out.append(SEGMENT.format(t0=temps[k - 1], t1=temps[k], steps=seg_steps, k=k))
    out.append(ANNEAL.format(t_final=spec.t_final, baro=baro, nrep=anneal // 20,
                             anneal_steps=anneal))
    return "".join(out)


def last_completed_segment(workdir: Path) -> int | None:
    """The progress marker, if it points at a restart file that exists."""
    marker = Path(workdir) / "progress.txt"
    try:
        k = int(marker.read_text().strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return None
    return k if (Path(workdir) / f"seg{k:02d}.restart").is_file() else None


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@dataclass
class Metric:
    name: str
    value: float
    target: float
    tolerance: float
    unit: str
    citation: str

    @property
    def passes(self) -> bool:
        return abs(self.value - self.target) <= self.tolerance


@dataclass
class SilicaReport:
    metrics: dict[str, Metric]
    defects: dict[str, float] = field(default_factory=dict)

    @property
    def passes(self) -> bool:
        return all(m.passes for m in self.metrics.values())

    def table(self) -> str:
        rows = [f"{'metric':12} {'value':>9} {'target':>9} {'tol':>6}  ok"]
        for m in self.metrics.values():
            rows.append(f"{m.name:12} {m.value:9.3f} {m.target:9.3f} {m.tolerance:6.2f}  "
                        f"{'yes' if m.passes else 'NO'}   [{m.citation}]")
        return "\n".join(rows)


def analyse_silica(cell: Cell, *, density: float | None = None) -> SilicaReport:
    """The section-3 validation numbers for an SiO2 cell (types 1 = Si, 2 = O)."""
    r_max = min(6.0, 0.49 * float(cell.lengths.min()))
    vals = {
        "density": density if density is not None else cell.density(),
        "r_SiO": partial_rdf(cell, SI, O, r_max=r_max).first_peak(1.3, 2.0),
        "r_OO": partial_rdf(cell, O, O, r_max=r_max).first_peak(2.2, 3.0),
        "r_SiSi": partial_rdf(cell, SI, SI, r_max=r_max).first_peak(2.6, 3.6),
        "angle_SiOSi": angle_peak(bond_angles(cell, O, SI, SI_O_CUTOFF)),
        "angle_OSiO": angle_peak(bond_angles(cell, SI, O, SI_O_CUTOFF)),
    }
    cn_si = coordination(cell, SI, O, SI_O_CUTOFF)
    cn_o = coordination(cell, O, SI, SI_O_CUTOFF)
    vals["frac_Si4"] = float(np.mean(cn_si == 4))
    vals["frac_O2"] = float(np.mean(cn_o == 2))
    metrics = {k: Metric(k, v, *TARGETS[k]) for k, v in vals.items()}
    defects = {
        "Si3": float(np.mean(cn_si == 3)), "Si5": float(np.mean(cn_si >= 5)),
        "O1": float(np.mean(cn_o == 1)), "O3": float(np.mean(cn_o >= 3)),
    }
    return SilicaReport(metrics, defects)


def judge(report: SilicaReport, spec: QuenchSpec, flags: FlagSet) -> None:
    """Raise the flags section 3 requires."""
    flags.add(
        Code.STRUCT_QUENCH_RATE, Severity.CAUTION,
        f"Glass quenched at {spec.quench_rate:.1e} K/s, about "
        f"{math.log10(spec.quench_rate / LAB_QUENCH_RATE):.0f} orders of magnitude "
        f"faster than a laboratory glass (~1 K/s).",
        remedy=("Fast quenches freeze in more small rings and coordination defects; "
                "every h_K computed on this glass inherits that. Quote the rate with "
                "the result, and compare two rates if h_K matters at that level."),
        quench_rate=spec.quench_rate,
    )
    failed = [m for m in report.metrics.values() if not m.passes]
    if failed:
        detail = "; ".join(f"{m.name} {m.value:.3g} vs {m.target:.3g} +/- {m.tolerance:.2g} {m.unit}"
                           for m in failed)
        flags.add(
            Code.STRUCT_VALIDATION_FAILED, Severity.WARNING,
            f"a-SiO2 fails validation: {detail}.",
            remedy=("Do not use this glass for h_K without stating the deviation. "
                    "Slower quench, longer anneal, or a different potential."),
            n_failed=len(failed),
        )


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

_DONE = re.compile(r"^QUENCH_DONE density (\S+) lx (\S+) ly (\S+) lz (\S+) pe (\S+)", re.M)


@dataclass
class QuenchResult:
    cell: Cell
    report: SilicaReport
    density: float
    data_file: Path


def run_quench(lmp: Path, spec: QuenchSpec, potential_dir: Path, workdir: Path,
               flags: FlagSet, *, timeout: int = 36000) -> QuenchResult:
    """Run the melt-quench in `workdir` (kept: glass.data, logs, restarts).

    Resumes automatically from the last completed segment if `workdir` holds
    one, so calling this again after an interruption continues the run.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / spec.potential).write_bytes((Path(potential_dir) / spec.potential).read_bytes())
    done = last_completed_segment(workdir)
    if done is None:
        write_lammps_data(start_cell(spec), workdir / "start.data",
                          comment="beta-cristobalite scaled to glass density")
    (workdir / "quench.in").write_text(build_quench_deck(spec, resume_from=done))
    log = "log.lammps" if done is None else f"log.resume{done:02d}.lammps"
    run = subprocess.run([str(lmp), "-in", "quench.in", "-log", log], cwd=workdir,
                         capture_output=True, text=True, timeout=timeout)
    if run.returncode != 0:
        errors = [l for l in run.stdout.splitlines() if "ERROR" in l]
        raise RuntimeError(f"LAMMPS failed: {errors or run.stdout[-400:]}")
    m = _DONE.search(run.stdout)
    if not m:
        raise RuntimeError("quench did not report QUENCH_DONE")
    cell = read_lammps_data(workdir / "glass.data")
    report = analyse_silica(cell, density=float(m.group(1)))
    judge(report, spec, flags)
    return QuenchResult(cell, report, float(m.group(1)), workdir / "glass.data")
