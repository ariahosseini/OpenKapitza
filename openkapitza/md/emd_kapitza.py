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
# Slab ENERGIES (kinetic + potential). The kinetic temperature alone swaps
# energy with the slab's own potential energy on vibrational time scales, so
# its dT autocorrelation collapses to ~0.3 within 0.05 ps (first Si/Ge run)
# and buries the interfacial decay; total energy only moves through the
# interface (frozen walls do no work).
compute         ke_at all ke/atom
compute         pe_at all pe/atom
compute         e_a emd_a reduce sum c_ke_at c_pe_at
compute         e_b emd_b reduce sum c_ke_at c_pe_at
variable        ea equal c_e_a[1]+c_e_a[2]
variable        eb equal c_e_b[1]+c_e_b[2]
variable        tnow equal time
print           "OPENKAPITZA_EMD n_a $(count(emd_a)) n_b $(count(emd_b)) area $(lx*ly:%.6f)"
run             {settle_steps}
reset_timestep  0
fix             emd_log all print {every} "${{tnow}} ${{ta}} ${{tb}} ${{ea}} ${{eb}}" file emd_temperatures.dat screen no title "# time[ps] T_a[K] T_b[K] E_a[eV] E_b[eV]"
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
# Two decays this far apart, each carrying this much of the variance, leave h
# undetermined (first Si/Ge run: 1.3 ps / 77 ps, 23 % / 73 %).
TWO_TIMESCALE_SEP = 10.0
TWO_TIMESCALE_WEIGHT = 0.15


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


def fit_biexponential(t: np.ndarray, acf: np.ndarray, *, end_fraction: float = 0.15,
                      n_grid: int = 60) -> tuple[float, float, float, float]:
    """ACF/ACF(0) ~ a1 exp(-t/tau1) + a2 exp(-t/tau2), tau1 < tau2.

    The slab-energy dT of a real cell decays in two stages: a fast partial
    drop (the per-atom potential-energy split across the interface jitters on
    vibrational time scales) and the slow interfacial relaxation. Only tau2 is
    the interface. Grid over (tau1, tau2), linear least squares for the
    amplitudes; no SciPy. Returns (tau1, tau2, a1, a2), times in t's units.
    """
    y = acf / acf[0]
    below = np.nonzero(y < end_fraction)[0]
    n = int(below[0]) if len(below) else len(y)
    tt, yy = t[:n], y[:n]
    if n < 10:
        raise ValueError("too few points before the ACF decays; sample faster or run longer")
    span = tt[-1]
    best = None
    for t1 in np.geomspace(span * 1e-4, span * 0.3, n_grid):
        for t2 in np.geomspace(t1 * 1.5, span * 20, n_grid):
            X = np.c_[np.exp(-tt / t1), np.exp(-tt / t2)]
            coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
            if np.any(coef < 0):
                continue
            res = float(np.sum((X @ coef - yy) ** 2))
            if best is None or res < best[0]:
                best = (res, t1, t2, coef[0], coef[1])
    if best is None:
        raise ValueError("no positive two-exponential fit")
    return best[1], best[2], best[3], best[4]


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
    tau_fast: float | None = None    # s, biexponential fit only
    slow_weight: float | None = None

    @property
    def h_fast(self) -> float | None:
        """h if the FAST biexponential component were the interface."""
        return self.c_r / (self.area * self.tau_fast) if self.tau_fast else None


def h_from_series(time_ps: np.ndarray, dT: np.ndarray, n_a: int, n_b: int,
                  area_m2: float, *, blocks: int = 5, model: str = "single") -> EMDResult:
    """h from a dT(t) series: whole-series tau, and a block standard error.

    model "single": one exponential (synthetic, or clean energy series);
    "biexp": two exponentials, the slow one taken as the interface -- use for
    slab-energy series from a real cell.
    """
    t = (np.asarray(time_ps) - time_ps[0]) * 1e-12
    c_a, c_b = 3 * n_a * KB, 3 * n_b * KB
    c_r = c_a * c_b / (c_a + c_b)
    tau_fast = weight = None
    if model == "biexp":
        tau_fast, tau, a1, a2 = fit_biexponential(t, autocorrelation(dT))
        weight = a2 / (a1 + a2)
        one = lambda tt, x: fit_biexponential(tt, autocorrelation(x))[1]
    elif model == "single":
        tau = fit_tau(t, autocorrelation(dT))
        one = lambda tt, x: fit_tau(tt, autocorrelation(x))
    else:
        raise ValueError("model must be 'single' or 'biexp'")
    h = c_r / (area_m2 * tau)
    hs = []
    m = len(dT) // blocks
    for k in range(blocks):
        try:
            hs.append(c_r / (area_m2 * one(t[:m], dT[k * m:(k + 1) * m])))
        except ValueError:
            pass
    err = float(np.std(hs, ddof=1) / math.sqrt(len(hs))) if len(hs) >= 3 else None
    return EMDResult(h, err, tau, float(t[-1]), c_r, area_m2, len(hs),
                     tau_fast=tau_fast, slow_weight=weight)


EV_OVER_KB = 1.602176634e-19 / KB    # K per eV


def read_emd(run_dir: Path | str) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """(time ps, dT K, n_a, n_b, area m^2) from a finished EMD run.

    dT is the ENERGY temperature difference E_a/C_a - E_b/C_b (C = 3 N k_B)
    when the energy columns are present; the kinetic-temperature difference
    only for older files (its fast intra-slab component hides the decay).
    """
    wd = Path(run_dir)
    data = np.loadtxt(wd / "emd_temperatures.dat", comments="#")
    n_a = n_b = None; area = None
    for ln in (wd / "log.lammps").read_text().splitlines():
        if ln.startswith("OPENKAPITZA_EMD n_a"):
            p = ln.split()
            n_a, n_b, area = int(float(p[2])), int(float(p[4])), float(p[6]) * 1e-20
    if n_a is None:
        raise ValueError("no OPENKAPITZA_EMD line in log.lammps")
    if data.shape[1] >= 5:
        dT = (data[:, 3] / (3 * n_a) - data[:, 4] / (3 * n_b)) * EV_OVER_KB
        return data[:, 0], dT, n_a, n_b, area
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
    if result.tau_fast and result.slow_weight is not None:
        sep = result.tau / result.tau_fast
        w = result.slow_weight
        if sep > TWO_TIMESCALE_SEP and TWO_TIMESCALE_WEIGHT < w < 1 - TWO_TIMESCALE_WEIGHT:
            flags.add(
                Code.EMD_TWO_TIMESCALES, Severity.WARNING,
                f"The dT autocorrelation has two well-separated decays "
                f"({result.tau_fast * 1e12:.2g} ps, weight {1 - w:.0%}; "
                f"{result.tau * 1e12:.3g} ps, weight {w:.0%}). The lumped two-slab model "
                f"behind h = C_r/(A tau) has one; h lies between "
                f"{result.h / 1e6:.3g} (slow) and {result.h_fast / 1e6:.3g} MW/m^2K (fast) "
                f"and is not determined by this run. In the first Si/Ge run (KB section 17) "
                f"the slow part was not energy leakage and did not scale like an interface "
                f"time; slow-relaxing phonon modes that span both slabs are the leading "
                f"suspect.",
                remedy="Treat EMD as unresolved. Diagnose with a run whose slabs are twice "
                       "as long (an interface time doubles, a mode lifetime does not) or "
                       "at a higher temperature (mode lifetimes shorten ~1/T).",
                h_slow=result.h, h_fast=result.h_fast, separation=sep, slow_weight=w,
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
