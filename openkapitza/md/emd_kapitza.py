"""Equilibrium cross-check of the interface conductance (open issue P1-8).

Temperature-fluctuation method (Rajabpour & Volz; Merabia & Termentzidis,
arXiv:1209.3485). In an isolated two-slab system at equilibrium the
temperature difference dT = T_A - T_B fluctuates and relaxes through the
interface. For a resistance dominated by the interface,

    <dT(0) dT(t)> ~ <dT^2> exp(-t / tau),   tau = C_r / (A h),
    C_r = C_A C_B / (C_A + C_B)

so tau from the autocorrelation gives h. C_i = 3 N_i k_B (classical solid).

This is NOT expected to equal the NEMD value. EMD gives the equilibrium
(Landauer) conductance, NEMD the non-equilibrium one; Merabia & Termentzidis
find NEMD > EMD by ~5x for typical semiconductor-like interfaces, >10x near
mass ratio 1, and estimate ~3x for Si/Ge. The cross-check is therefore the
RATIO and its direction, not agreement.

Valid when the interface dominates the relaxation: L_i / k_i << 1/h on both
sides (checked); otherwise bulk conduction inside the slabs slows the decay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..schema.flags import Code, FlagSet, Severity

KB = 1.380649e-23
EXPECTED_RATIO = (1.0, 10.0)        # NEMD/EMD: Merabia & Termentzidis range (>1)

EMD_BLOCK = """\
# ---- equilibrium fluctuation run (P1-8) ------------------------------------
# NVE on the mobile atoms, no reservoirs: the slab temperatures fluctuate and
# relax through the interface. dT(t) is logged for its autocorrelation.
fix             nve gmobile nve
group           emd_a_all type {types_a}
group           emd_b_all type {types_b}
group           emd_a intersect emd_a_all gmobile
group           emd_b intersect emd_b_all gmobile
compute         t_a emd_a temp
compute         t_b emd_b temp
variable        ta equal c_t_a
variable        tb equal c_t_b
variable        tnow equal time
print           "OPENKAPITZA_EMD n_a $(count(emd_a)) n_b $(count(emd_b)) area $(lx*ly:%.6f)"
run             {settle_steps}
reset_timestep  0
fix             emd_log all print {every} "${{tnow}} ${{ta}} ${{tb}}" file emd_temperatures.dat screen no title "# time[ps] T_a[K] T_b[K]"
thermo          {thermo_every}
thermo_style    custom step temp pe etotal
run             {production_steps}
print           "OPENKAPITZA_EMD_DONE"
"""

MARKER = "# ---- impose the gradient"


@dataclass(frozen=True)
class EMDSpec:
    settle_ps: float = 20.0          # NVE settle after the thermostat comes off
    production_ps: float = 1000.0
    sample_fs: float = 10.0          # dT logging interval


def build_emd_deck(nemd_spec, emd: EMDSpec = EMDSpec()) -> str:
    """The NEMD deck up to the end of equilibration, then an NVE fluctuation run.

    Same structure, potential, relaxation and piston as the NEMD run it is
    compared with, so the two measure the same cell.
    """
    from .nemd import build_input_deck
    if nemd_spec.ttm is not None:
        raise ValueError("the fluctuation method is phonon-only; use a spec without ttm")
    deck = build_input_deck(nemd_spec)
    i = deck.index(MARKER)
    dt_fs = nemd_spec.timestep_fs
    every = max(1, int(round(emd.sample_fs / dt_fs)))
    return deck[:i] + EMD_BLOCK.format(
        types_a=" ".join(map(str, nemd_spec.types_a)),
        types_b=" ".join(map(str, nemd_spec.types_b)),
        settle_steps=int(emd.settle_ps * 1000 / dt_fs),
        production_steps=int(emd.production_ps * 1000 / dt_fs),
        every=every, thermo_every=max(1, int(10_000 / dt_fs)),
    )


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Unbiased autocorrelation of a zero-mean series (FFT), lags 0..n-1."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / np.arange(n, 0, -1)


# Synthetic check (Ornstein-Uhlenbeck with known tau, 8 seeds): the fit reads
# tau LOW by ~10 % with 143 tau of data and ~5 % with 714 tau (log of a noisy,
# mean-subtracted ACF), with 15 % and 7 % scatter. Fitting down to 0.5 ACF(0)
# gave the least scatter of the windows tried (0.2-0.6).
MIN_DURATION_TAU = 500.0


def fit_tau(t: np.ndarray, acf: np.ndarray, *, lo: float = 1.0, hi: float = 0.5) -> float:
    """Exponential decay time from ln ACF between lo*ACF(0) and hi*ACF(0)."""
    c0 = acf[0]
    if c0 <= 0:
        raise ValueError("zero-variance temperature difference")
    below = np.nonzero(acf < hi * c0)[0]
    end = int(below[0]) if len(below) else len(acf)
    start = int(np.nonzero(acf[:end] < lo * c0)[0][0]) if np.any(acf[:end] < lo * c0) else 0
    if end - start < 3:
        raise ValueError("too few points in the decay window; sample faster or run longer")
    slope = np.polyfit(t[start:end], np.log(acf[start:end]), 1)[0]
    if slope >= 0:
        raise ValueError("autocorrelation does not decay")
    return -1.0 / slope


@dataclass
class EMDResult:
    h: float                 # W/m^2K
    h_err: float | None      # block standard error
    tau: float               # s
    duration: float          # s, length of the series
    c_r: float               # J/K
    area: float              # m^2
    blocks: int
    ratio_nemd: float | None = None


def h_from_series(time_ps: np.ndarray, dT: np.ndarray, n_a: int, n_b: int,
                  area_m2: float, *, blocks: int = 5) -> EMDResult:
    """h from a dT(t) series: whole-series tau, and a block standard error."""
    t = (np.asarray(time_ps) - time_ps[0]) * 1e-12
    c_a, c_b = 3 * n_a * KB, 3 * n_b * KB
    c_r = c_a * c_b / (c_a + c_b)
    tau = fit_tau(t, autocorrelation(dT))
    h = c_r / (area_m2 * tau)
    hs = []
    m = len(dT) // blocks
    for k in range(blocks):
        try:
            seg = dT[k * m:(k + 1) * m]
            hs.append(c_r / (area_m2 * fit_tau(t[:m], autocorrelation(seg))))
        except ValueError:
            pass
    err = float(np.std(hs, ddof=1) / math.sqrt(len(hs))) if len(hs) >= 3 else None
    return EMDResult(h, err, tau, float(t[-1]), c_r, area_m2, len(hs))


def read_emd(run_dir: Path | str) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """(time ps, dT K, n_a, n_b, area m^2) from a finished EMD run."""
    wd = Path(run_dir)
    data = np.loadtxt(wd / "emd_temperatures.dat", comments="#")
    n_a = n_b = None; area = None
    for ln in (wd / "log.lammps").read_text().splitlines():
        if ln.startswith("OPENKAPITZA_EMD n_a"):
            p = ln.split()
            n_a, n_b, area = int(float(p[2])), int(float(p[4])), float(p[6]) * 1e-20
    if n_a is None:
        raise ValueError("no OPENKAPITZA_EMD line in log.lammps")
    return data[:, 0], data[:, 1] - data[:, 2], n_a, n_b, area


def check(result: EMDResult, flags: FlagSet, *, h_nemd: float | None = None,
          bulk: tuple[tuple[float, float], tuple[float, float]] | None = None) -> EMDResult:
    """Validity and the NEMD comparison. `bulk` = ((L_a, k_a), (L_b, k_b)) in SI."""
    n_tau = result.duration / result.tau
    if n_tau < MIN_DURATION_TAU:
        flags.add(
            Code.EMD_SHORT, Severity.CAUTION,
            f"Fluctuation series is {n_tau:.0f} relaxation times long: tau reads low "
            f"(h high) by ~5-10 % at 140-700 tau in the synthetic check.",
            remedy=f"Run >= {MIN_DURATION_TAU:.0f} tau of production.",
            duration_in_tau=n_tau,
        )
    if bulk is not None:
        r_bulk = max(L / k for L, k in bulk)
        if r_bulk > 0.2 / result.h:
            flags.add(
                Code.NEMD_POOR_FIT, Severity.WARNING,
                f"Slab bulk resistance L/k = {r_bulk:.3g} m^2K/W is not small against "
                f"1/h = {1 / result.h:.3g}: bulk conduction slows the dT decay, so h is "
                f"under-estimated.",
                remedy="Use shorter slabs for the fluctuation method.",
            )
    if h_nemd is not None:
        ratio = h_nemd / result.h
        result.ratio_nemd = ratio
        lo, hi = EXPECTED_RATIO
        ok = lo <= ratio <= hi
        flags.add(
            Code.EMD_RATIO,
            Severity.WARNING if not ok else Severity.INFO,
            f"NEMD / EMD = {ratio:.2f} ({'outside' if not ok else 'inside'} the expected "
            f"{lo:g}-{hi:g}: NEMD measures the non-equilibrium conductance and exceeds the "
            f"equilibrium one, ~3x for Si/Ge per Merabia & Termentzidis).",
            remedy=("A ratio below 1 means one of the two runs is wrong (unconverged, "
                    "wrong area or wrong slab heat capacities); a ratio far above the "
                    "range suggests the NEMD cell is ballistic-limited."),
            ratio=ratio,
        )
    return result
