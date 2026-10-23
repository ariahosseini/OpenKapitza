"""One NEMD run directory in, one h_K measurement with an honest error out.

Every script used to assemble this itself -- health check, stationarity,
profile, flux, extraction -- and each had its own copy (knowledge base: "a
fix that lives in a script is not a fix"). This is the single entry point the
smoke script, the relaxation comparison and the convergence-campaign collector
all call.

THE ERROR BAR. `extract_kapitza` reports the uncertainty of the extrapolated
linear fits, which treats every temperature bin as independent and the
profile as a single sample in time. Neither holds: neighbouring bins share
phonons, and the profile is correlated over the run. The profile file already
holds the remedy: its frames are independent averages over successive slices
of the production window. Extracting h_K from each frame, with that frame's
own flux, gives block estimates whose scatter is an error that includes both
correlations. The stated uncertainty is the larger of the two, and
`NEMD_ERROR_UNDERSTATED` says when the fit error alone would have been
optimistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Quantity
from .nemd import (
    FS_TO_PS, NEMDSpec, check_flux_stationarity, check_run_health,
    M_TO_ANG, parse_log_thermo, parse_profile, parse_profile_frames, parse_relaxed_box,
    parse_relaxed_positions, parse_reservoir_energy, parse_reservoir_series,
)
from .profile import (KapitzaExtraction, extract_kapitza, flux_from_reservoirs,
                      raw_delta_T)

# Block error this many times the fit error, or more, is worth saying.
UNDERSTATED_RATIO = 2.0


@dataclass
class KapitzaMeasurement:
    extraction: KapitzaExtraction       # all frames, full-window flux
    h_K: float                          # W/(m^2 K)
    uncertainty: float                  # W/(m^2 K), max(fit, block)
    fit_se: float
    block_se: float | None              # None with fewer than 3 usable frames
    per_frame_delta_T: list[float]      # K, unguarded (may be <= 0 in noise)
    per_frame_flux: list[float]         # W/m^2, each frame's own window
    flux: float                         # W/m^2
    drifts: dict = field(default_factory=dict)

    @property
    def relative_uncertainty(self) -> float:
        return self.uncertainty / self.h_K

    def quantity(self) -> Quantity:
        q = self.extraction.h_K
        return Quantity(si=self.h_K, unit=q.unit, uncertainty=self.uncertainty,
                        provenance=q.provenance, confidence=q.confidence,
                        name="h_K", note=(q.note or "") +
                        f"; error = max(fit, {len(self.per_frame_delta_T)}-frame jackknife)")


def frame_fluxes(reservoir_path: Path, frame_ends_steps: list[int], window_steps: int,
                 timestep_fs: float, area: float, sample_every: int,
                 production_steps: int) -> list[float]:
    """Flux over each profile frame's own time window, from the accumulators.

    Alignment uses neither LAMMPS's `time` nor an assumption about the first
    sample. `fix print` writes on multiples of `sample_every` and the
    production run ends on one, so the LAST sample is at `production_steps`
    and sample i sits at production_steps - (n-1-i)*sample_every. That holds
    whether or not a step-0 line is written (current LAMMPS writes one,
    "0 0 0"; the first version of this function assumed it did not and was off
    by one sample). Both tallies are zero at step 0, since the deck
    re-creates the reservoir fixes there, so that point is added if absent.
    """
    t, hot, cold = (np.asarray(v) for v in parse_reservoir_series(reservoir_path))
    dt_s = timestep_fs * FS_TO_PS * 1e-12
    n = t.size
    steps = production_steps - (n - 1 - np.arange(n)) * float(sample_every)
    if steps[0] > 0.5:
        steps = np.concatenate([[0.0], steps])
        hot = np.concatenate([[0.0], hot])
        cold = np.concatenate([[0.0], cold])
    # The direction of each reservoir's transfer comes from the WHOLE run. Over
    # a short window the net transfer can be smaller than the thermal noise in
    # the tally, and taking each window's magnitude then inflates the flux --
    # by 2.6x on a 16 ps test run with 2 ps frames. With a fixed sign the
    # frame fluxes average exactly to the full-window flux.
    sign_hot = 1.0 if hot[-1] - hot[0] >= 0 else -1.0
    sign_cold = 1.0 if cold[-1] - cold[0] >= 0 else -1.0
    out = []
    for end in frame_ends_steps:
        s0, s1 = end - window_steps, end
        if s0 < steps[0] - 0.5 or s1 > steps[-1] + 0.5:
            raise ValueError("profile frame lies outside the reservoir series")
        dh = np.interp(s1, steps, hot) - np.interp(s0, steps, hot)
        dc = np.interp(s1, steps, cold) - np.interp(s0, steps, cold)
        out.append(0.5 * (sign_hot * dh + sign_cold * dc) / ((s1 - s0) * dt_s * area))
    return out


def measure_kapitza(run_dir: Path | str, spec: NEMDSpec, flags: FlagSet, *,
                    k_expected_a: float | None = None,
                    k_expected_b: float | None = None) -> KapitzaMeasurement:
    """Health, stationarity, all-frames extraction and block error for one run."""
    wd = Path(run_dir)
    g = spec.geometry
    area = spec.snapped_area if spec.structure_mode == "build" else g.area
    log = wd / "log.lammps"
    # The relaxed cross-section, not the built one: the in-plane barostat
    # changes it (Si/Ge +5 %), and every h_K from a relaxed cell used to be
    # high by that ratio.
    box = parse_relaxed_box(log)
    if box:
        area = box[0] * box[1] / M_TO_ANG ** 2
    if log.exists():
        check_run_health(parse_log_thermo(log), spec.temperature, flags,
                         expected_pzz_GPa=spec.spot_pressure_GPa or 0.0,
                         lateral_constrained=(bool(spec.spot_pressure_GPa)
                                              and spec.spot_pressure_mode == "strain"))

    energy = wd / "reservoir_energy.dat"
    profile_path = wd / "temperature_profile.dat"
    drifts = check_flux_stationarity(energy, flags)
    profile = parse_profile(profile_path, g, flags, frames="all")

    # After a piston relaxation the interface and the cold reservoir are no
    # longer where the geometry put them (Si/Ge: interface 2.1 A lower, top
    # wall 3.1 A higher). Bins are assigned to a side by z < interface_z, so
    # the measured plane is used, and side B's range follows the wall.
    interface_z, bulk_a, bulk_b = g.interface_z, g.bulk_a, g.bulk_b
    relaxed = parse_relaxed_positions(log)
    if relaxed:
        interface_z = 0.5 * (relaxed["top_a"] + relaxed["bottom_b"]) / M_TO_ANG
        wall_shift = relaxed["wall_hi_zmin"] / M_TO_ANG - g.z_cold_end
        bulk_a = (g.bulk_a[0], interface_z)
        bulk_b = (interface_z, g.bulk_b[1] + wall_shift)
        profile = replace(profile, interface_z=interface_z)
    frames_interface = interface_z
    if spec.ttm is not None:
        # Two-temperature run: next to the interface the metal lattice and its
        # electrons are out of equilibrium over a few cooling lengths, so the
        # metal's bulk fit starts beyond them (md.ttm.fit_exclusion).
        from .ttm import fit_exclusion
        cut = fit_exclusion(spec.ttm)
        if spec.ttm.side == "b":
            bulk_b = (bulk_b[0] + cut, bulk_b[1])
        else:
            bulk_a = (bulk_a[0], bulk_a[1] - cut)
        if (bulk_b if spec.ttm.side == "b" else bulk_a)[1] <= \
                (bulk_b if spec.ttm.side == "b" else bulk_a)[0]:
            raise ValueError("metal slab shorter than the TTM fit exclusion "
                             f"({cut * 1e9:.1f} nm): lengthen the metal side")

    e_hot, e_cold, duration = parse_reservoir_energy(energy)
    flux = flux_from_reservoirs(e_hot, e_cold, duration, area, flags)
    extraction = extract_kapitza(profile, flux, flags, bulk_a=bulk_a, bulk_b=bulk_b,
                                 k_expected_a=k_expected_a, k_expected_b=k_expected_b)
    h = extraction.h_K.si
    fit_se = extraction.h_K.uncertainty or 0.0

    # Block estimates: the jump and the flux per frame, each frame with its own
    # flux window. Per-frame JUMPS, not per-frame h_K: a short slice can give a
    # negative jump in noise, which is fine to average but has no h_K, and
    # averaging reciprocals would bias the mean anyway.
    frames = parse_profile_frames(profile_path, g, FlagSet())
    window = max(spec.profile_every,
                 (spec.production_steps // spec.profile_frames // spec.profile_every)
                 * spec.profile_every)
    per_dT: list[float] = []
    per_q: list[float] = []
    block_failure = None
    try:
        fluxes = frame_fluxes(energy, [s for s, _ in frames], window,
                              spec.timestep_fs, area, spec.profile_every,
                              spec.production_steps)
        for (_, prof), q in zip(frames, fluxes):
            prof = replace(prof, interface_z=frames_interface)
            per_dT.append(raw_delta_T(prof, bulk_a=bulk_a, bulk_b=bulk_b))
            per_q.append(q)
    except ValueError as exc:
        # Say so: a block error that silently vanishes leaves the optimistic
        # fit error standing as if it were the whole story.
        block_failure = str(exc)
        per_dT, per_q = [], []

    # Is the JUMP itself still trending across production? This is the
    # steady-state question that matters for h_K, and far better resolved
    # than reservoir-power drift (see md.stationarity.analyse_block_trend).
    if len(per_dT) >= 4:
        from .stationarity import Verdict, analyse_block_trend
        try:
            tr = analyse_block_trend(per_dT)
        except ValueError:
            tr = None
        if tr is not None and tr.verdict is Verdict.DRIFTING:
            flags.add(
                Code.NEMD_UNEQUILIBRATED,
                Severity.WARNING,
                f"The temperature jump trends by {tr.drift * 100:+.0f} +/- "
                f"{tr.drift_se * 100:.0f} % across the production frames; the "
                f"interface had not reached steady state.",
                remedy=("Lengthen the steady-state stage; h_K from a trending "
                        "jump depends on which part of the run is averaged."),
                jump_drift=tr.drift,
            )
        elif tr is not None and tr.verdict is Verdict.UNRESOLVED:
            flags.add(
                Code.NEMD_STATIONARITY_UNRESOLVED,
                Severity.INFO,
                f"Per-frame jump: no significant trend ({tr.drift * 100:+.0f} +/- "
                f"{tr.drift_se * 100:.0f} %), but not certified flat to 10 %.",
                remedy="The block error on h_K already carries this scatter.",
            )

    block_se = None
    n = len(per_dT)
    if n >= 3:
        # Leave-one-out jackknife on h = mean(q) / mean(dT).
        dT_arr, q_arr = np.asarray(per_dT), np.asarray(per_q)
        loo = []
        for k in range(n):
            keep = np.arange(n) != k
            d = dT_arr[keep].mean()
            loo.append(q_arr[keep].mean() / d if d > 0 else math.inf)
        loo = np.asarray(loo)
        if np.all(np.isfinite(loo)):
            block_se = float(math.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
        else:
            block_failure = ("leaving out one frame drives the mean jump to zero or "
                             "below; the run is too noisy for a block error")
    uncertainty = max(fit_se, block_se or 0.0)
    if block_se is not None and fit_se > 0 and block_se > UNDERSTATED_RATIO * fit_se:
        flags.add(
            Code.NEMD_ERROR_UNDERSTATED,
            Severity.CAUTION,
            f"Frame-to-frame scatter gives a standard error of "
            f"{block_se / h * 100:.1f} % on h_K, {block_se / fit_se:.1f}x the fit "
            f"error of {fit_se / h * 100:.1f} %. The fit treats bins and time as "
            f"independent; they are not.",
            remedy=("The larger (block) error is the one reported. Lengthen the "
                    "production run or widen the cross-section to reduce it."),
            ratio=block_se / fit_se,
        )
    elif block_se is None:
        reason = block_failure or f"only {n} profile frames"
        flags.add(
            Code.NEMD_ERROR_UNDERSTATED,
            Severity.CAUTION if block_failure else Severity.INFO,
            f"No block error on h_K ({reason}); the stated error is the fit "
            f"error alone, which treats bins and time as independent.",
            remedy="Write at least 4 profile frames and run long enough that each "
                   "frame's jump is well above its noise.",
        )
    return KapitzaMeasurement(extraction, h, uncertainty, fit_se, block_se,
                              per_dT, per_q, flux, drifts)
