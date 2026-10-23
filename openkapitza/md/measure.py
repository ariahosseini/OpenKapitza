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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Quantity
from .nemd import (
    FS_TO_PS, NEMDSpec, check_flux_stationarity, check_run_health,
    parse_log_thermo, parse_profile, parse_profile_frames,
    parse_reservoir_energy, parse_reservoir_series,
)
from .profile import KapitzaExtraction, extract_kapitza, flux_from_reservoirs

# Block error this many times the fit error, or more, is worth saying.
UNDERSTATED_RATIO = 2.0


@dataclass
class KapitzaMeasurement:
    extraction: KapitzaExtraction       # all frames, full-window flux
    h_K: float                          # W/(m^2 K)
    uncertainty: float                  # W/(m^2 K), max(fit, block)
    fit_se: float
    block_se: float | None              # None with fewer than 3 frames
    per_frame_h_K: list[float]
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
                        f"; error = max(fit, {len(self.per_frame_h_K)}-frame block)")


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
    out = []
    for end in frame_ends_steps:
        s0, s1 = end - window_steps, end
        if s0 < steps[0] - 0.5 or s1 > steps[-1] + 0.5:
            raise ValueError("profile frame lies outside the reservoir series")
        dh = np.interp(s1, steps, hot) - np.interp(s0, steps, hot)
        dc = np.interp(s1, steps, cold) - np.interp(s0, steps, cold)
        out.append(0.5 * (abs(dh) + abs(dc)) / ((s1 - s0) * dt_s * area))
    return out


def measure_kapitza(run_dir: Path | str, spec: NEMDSpec, flags: FlagSet, *,
                    k_expected_a: float | None = None,
                    k_expected_b: float | None = None) -> KapitzaMeasurement:
    """Health, stationarity, all-frames extraction and block error for one run."""
    wd = Path(run_dir)
    g = spec.geometry
    area = spec.snapped_area if spec.structure_mode == "build" else g.area
    log = wd / "log.lammps"
    if log.exists():
        check_run_health(parse_log_thermo(log), spec.temperature, flags)

    energy = wd / "reservoir_energy.dat"
    profile_path = wd / "temperature_profile.dat"
    drifts = check_flux_stationarity(energy, flags)
    profile = parse_profile(profile_path, g, flags, frames="all")
    e_hot, e_cold, duration = parse_reservoir_energy(energy)
    flux = flux_from_reservoirs(e_hot, e_cold, duration, area, flags)
    extraction = extract_kapitza(profile, flux, flags, bulk_a=g.bulk_a, bulk_b=g.bulk_b,
                                 k_expected_a=k_expected_a, k_expected_b=k_expected_b)
    h = extraction.h_K.si
    fit_se = extraction.h_K.uncertainty or 0.0

    # Block estimates: one h_K per frame, each with its own window's flux.
    frames = parse_profile_frames(profile_path, g, FlagSet())
    window = max(spec.profile_every,
                 (spec.production_steps // spec.profile_frames // spec.profile_every)
                 * spec.profile_every)
    per_frame: list[float] = []
    try:
        fluxes = frame_fluxes(energy, [s for s, _ in frames], window,
                              spec.timestep_fs, area, spec.profile_every,
                              spec.production_steps)
        for (_, prof), q in zip(frames, fluxes):
            r = extract_kapitza(prof, q, FlagSet(), bulk_a=g.bulk_a, bulk_b=g.bulk_b)
            per_frame.append(r.h_K.si)
    except ValueError:
        per_frame = []

    block_se = None
    if len(per_frame) >= 3:
        block_se = float(np.std(per_frame, ddof=1) / math.sqrt(len(per_frame)))
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
        flags.add(
            Code.NEMD_ERROR_UNDERSTATED,
            Severity.INFO,
            f"Only {len(per_frame)} usable profile frames; the error on h_K is the "
            f"fit error alone, which treats bins and time as independent.",
            remedy="Write at least 4 profile frames (spec.profile_frames).",
        )
    return KapitzaMeasurement(extraction, h, uncertainty, fit_se, block_se,
                              per_frame, flux, drifts)
