"""Vibrational density of states: the phonon part of the potential gate (KB 37).

The agent instruction (section 4) gates a potential on phonon DOS, elastic
constants and bulk k before production. Elastic constants (md/elastic.py)
and bulk k (md/conductivity.py) exist; this is the third.

METHOD. A bulk crystal relaxed to the potential's own lattice constant,
equilibrated at T, then run NVE with velocities dumped every few steps. The
vibrational DOS is the power spectrum of the atomic velocities
(Wiener-Khinchin: the Fourier transform of the velocity autocorrelation),
summed over atoms and directions, with a Hann window:

    g(f) ~ sum_i,a | FFT[ v_i,a(t) w(t) ] |^2

It contains every mode, anharmonicity included, at the run temperature.

THE GATE compares two features with measured values:
- the optical peak, the highest-frequency maximum, against the zone-centre
  optical (Raman) frequency (Si 15.6 THz, 520.7 cm^-1; Ge 9.0 THz,
  ~301 cm^-1 at room temperature); it must agree within 5 %. A potential
  that puts the optical band in the wrong place misplaces exactly the modes
  that decide how much of the spectrum the two sides share;
- the transverse-acoustic peak, the strongest maximum below a third of the
  band top (Si ~4.5 THz, Ge ~2.4 THz from neutron-derived DOS), with a wider
  10 % band, because a short run broadens it.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .elastic import CubicSpec, stage_potential

# Measured reference features, THz.
REFERENCE_PHONON = {
    "Si": {"optical": 15.6, "ta": 4.5,
           "citation": "Raman 520.7 cm^-1 (zone-centre optical); TA DOS peak from neutron data"},
    "Ge": {"optical": 9.0, "ta": 2.4,
           "citation": "Raman ~301 cm^-1 at 300 K (zone-centre optical); TA DOS peak from neutron data"},
}
OPTICAL_TOLERANCE = 0.05
TA_TOLERANCE = 0.10

DECK = """\
# OpenKapitza vibrational DOS -- {name} (md/phonon_dos.py)
units           metal
boundary        p p p
atom_style      atomic
lattice         {lattice} {a0_guess:.6f}
region          box block 0 {n} 0 {n} 0 {n}
create_box      1 box
create_atoms    1 box
mass            1 {mass:.4f}
pair_style      {pair_style}
{pair_coeff}
timestep        {dt_ps:.6f}
fix             br all box/relax iso 0.0 vmax 0.001
minimize        1.0e-12 1.0e-12 10000 100000
unfix           br
velocity        all create {t2:.2f} {seed} mom yes rot yes dist gaussian
fix             eq all nvt temp {T:.2f} {T:.2f} {damp:.4f}
thermo          1000
run             {equil}
unfix           eq
fix             nve all nve
dump            v all custom {every} velocities.dump id vx vy vz
dump_modify     v sort id
run             {prod}
print           "VDOS_DONE"
"""


@dataclass
class VDOSSpec:
    crystal: CubicSpec
    temperature: float = 300.0
    timestep_fs: float = 1.0
    equil_steps: int = 5000
    prod_steps: int = 10000
    dump_every: int = 4
    seed: int = 4711


def build_vdos_deck(spec: VDOSSpec) -> str:
    c = spec.crystal
    return DECK.format(name=c.name, lattice=c.lattice, a0_guess=c.a0_guess, n=c.cells,
                       mass=c.mass, pair_style=c.pair_style, pair_coeff=c.pair_coeff(),
                       dt_ps=spec.timestep_fs * 1e-3, t2=2 * spec.temperature,
                       T=spec.temperature, damp=0.1, seed=spec.seed, equil=spec.equil_steps,
                       every=spec.dump_every, prod=spec.prod_steps)


def read_velocities(path: Path) -> np.ndarray:
    """(frames, atoms, 3) velocities from a sorted `dump custom id vx vy vz`."""
    frames, cur, n, reading = [], [], None, False
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("ITEM: NUMBER OF ATOMS"):
            reading = "n"
            continue
        if reading == "n":
            n = int(ln)
            reading = False
            continue
        if ln.startswith("ITEM: ATOMS"):
            cur = []
            frames.append(cur)
            reading = True
            continue
        if ln.startswith("ITEM:"):
            reading = False
            continue
        if reading is True:
            p = ln.split()
            cur.append((float(p[1]), float(p[2]), float(p[3])))
    frames = [f for f in frames if len(f) == n]
    return np.asarray(frames)


def vdos(v: np.ndarray, sample_fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(frequency THz, normalised g(f)) from (frames, atoms, 3) velocities."""
    nt = v.shape[0]
    w = np.hanning(nt)[:, None, None]
    spec = np.abs(np.fft.rfft((v - v.mean(axis=0)) * w, axis=0)) ** 2
    g = spec.sum(axis=(1, 2))
    f = np.fft.rfftfreq(nt, d=sample_fs * 1e-15) / 1e12
    g[0] = 0.0
    g = g / np.trapezoid(g, f) if hasattr(np, "trapezoid") else g / np.trapz(g, f)
    return f, g


def _smooth(g: np.ndarray, width: int) -> np.ndarray:
    k = np.ones(width) / width
    return np.convolve(g, k, mode="same")


def peaks(f: np.ndarray, g: np.ndarray, smooth_thz: float = 0.3) -> dict[str, float]:
    """In THz: the TA peak (the strongest maximum below a third of the band
    top) and the optical peak (the highest-frequency maximum holding at least
    30 % of the tallest); f_max, where the smoothed DOS ends."""
    df = f[1] - f[0]
    gs = _smooth(g, max(1, int(round(smooth_thz / df))))
    top = gs.max()
    f_max = float(f[np.nonzero(gs > 0.02 * top)[0][-1]])
    idx = [i for i in range(1, len(gs) - 1) if gs[i] >= gs[i - 1] and gs[i] >= gs[i + 1]]
    low = [i for i in idx if f[i] < f_max / 3]
    strong = [i for i in idx if gs[i] >= 0.3 * top]
    if not low or not strong:
        raise ValueError("no peaks in the DOS")
    return {"ta": float(f[max(low, key=lambda i: gs[i])]), "optical": float(f[strong[-1]]),
            "f_max": f_max}


@dataclass
class PhononGate:
    passes: bool
    peaks: dict
    errors: dict
    reference: str
    citation: str


def gate(pk: dict, reference_key: str) -> PhononGate:
    ref = REFERENCE_PHONON[reference_key]
    err = {k: (pk[k] - ref[k]) / ref[k] for k in ("optical", "ta")}
    ok = abs(err["optical"]) <= OPTICAL_TOLERANCE and abs(err["ta"]) <= TA_TOLERANCE
    return PhononGate(ok, pk, err, reference_key, ref["citation"])


def run_vdos(lmp: Path | str, spec: VDOSSpec, potential_dir: Path | str, *,
             timeout: int = 3600, keep: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run the deck and return (f THz, g)."""
    work = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="vdos_"))
    work.mkdir(parents=True, exist_ok=True)
    stage_potential(spec.crystal, Path(potential_dir), work)
    (work / "vdos.in").write_text(build_vdos_deck(spec))
    out = subprocess.run([str(lmp), "-in", "vdos.in", "-log", "vdos.log"], cwd=work,
                         capture_output=True, text=True, timeout=timeout)
    if "VDOS_DONE" not in out.stdout:
        raise RuntimeError(f"LAMMPS did not finish:\n{out.stdout[-2000:]}{out.stderr[-1000:]}")
    v = read_velocities(work / "velocities.dump")
    return vdos(v, spec.timestep_fs * spec.dump_every)
