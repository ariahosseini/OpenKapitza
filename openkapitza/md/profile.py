"""Extracting Kapitza conductance from a NEMD temperature profile.

THE CORRECTION THIS MODULE EXISTS FOR.

In a steady direct-NEMD run the temperature profile looks like

    [wall][hot reservoir][ bulk A ]|[ bulk B ][cold reservoir][wall]

with a linear gradient on each side, slope q/k, and a genuine discontinuity
at the interface plane. The quantity wanted is that discontinuity.

The tempting shortcut -- difference the two bins either side of the interface
-- is wrong, and wrong in a specific direction. Those bins sit a finite
distance from the interface, so their difference contains the real jump PLUS
the bulk gradient across that distance:

    T_naive = dT_true + q*(dz_A/k_A + dz_B/k_B)

Since both terms are positive, the naive estimate OVERSTATES the jump and
therefore UNDERSTATES h_K. Worse, the error grows with the applied flux, so
it masquerades as a physical flux dependence and survives a careless
convergence check.

The correct procedure is to fit a straight line to the linear region on each
side and extrapolate BOTH fits to the interface plane:

    dT = T_A(z_interface) - T_B(z_interface)
    h_K = q / dT

Two regions must be excluded before fitting. The thermostatted reservoirs do
not follow the bulk gradient at all. And near the interface the phonon
distribution is out of local equilibrium over some distance, bending the
profile; including that region biases the extrapolation. The width of that
second zone is a judgement, so this module sweeps it and reports how much the
answer moves -- a result that depends strongly on where you stopped fitting
is not a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Confidence, Provenance, Quantity

# Defaults for the fit windows, as fractions of each side's bulk length.
DEFAULT_RESERVOIR_EXCLUSION = 0.15
DEFAULT_INTERFACE_EXCLUSION = 0.15
# Reported for diagnosis, but NOT used as a gate -- see _fit's docstring for
# why R^2 is the wrong criterion for a shallow gradient.
MIN_FIT_R2 = 0.98
# The real gate: standard error of the extrapolated jump, relative to it.
MAX_DELTA_T_REL_SE = 0.10
# Minimum bins per side for the fit to mean anything.
MIN_FIT_BINS = 8
# If sweeping the interface exclusion moves dT by more than this, the fit
# window is contaminated.
MAX_EXCLUSION_SENSITIVITY = 0.10
# Hot and cold reservoir fluxes should agree this well at steady state.
MAX_FLUX_IMBALANCE = 0.05


@dataclass
class SideFit:
    """Least-squares line through one side's linear region."""

    slope: float            # K/m (signed, along +z)
    intercept: float        # K
    r2: float
    n_bins: int
    z_min: float
    z_max: float
    T_at_interface: float   # the extrapolated value -- the whole point
    se_at_interface: float  # standard error of that extrapolation, K

    def conductivity(self, flux: float) -> float:
        """k implied by this side's gradient, k = -q/(dT/dz).

        A useful independent check: if the fitted conductivity is far from the
        potential's known bulk value, the run is not what it is assumed to be
        (too short, not at steady state, or size-limited).
        """
        return float("inf") if self.slope == 0 else abs(flux / self.slope)


@dataclass
class TemperatureProfile:
    """Binned steady-state temperature profile along the transport direction."""

    z: np.ndarray               # bin centres, m
    T: np.ndarray               # bin temperature, K
    interface_z: float          # interface plane, m
    area: float                 # cross-section, m^2
    counts: np.ndarray | None = None   # atoms per bin

    def __post_init__(self) -> None:
        self.z = np.asarray(self.z, dtype=float)
        self.T = np.asarray(self.T, dtype=float)
        if self.z.shape != self.T.shape:
            raise ValueError("z and T must have the same shape")
        if self.z.size < 2 * MIN_FIT_BINS:
            raise ValueError(
                f"need at least {2 * MIN_FIT_BINS} bins to fit both sides, "
                f"got {self.z.size}"
            )
        if not np.all(np.diff(self.z) > 0):
            raise ValueError("z must be strictly increasing")
        if not (self.z[0] < self.interface_z < self.z[-1]):
            raise ValueError("interface_z must lie inside the profile")

    @property
    def side_a_mask(self) -> np.ndarray:
        return self.z < self.interface_z

    @property
    def side_b_mask(self) -> np.ndarray:
        return self.z > self.interface_z


@dataclass
class KapitzaExtraction:
    delta_T: Quantity
    h_K: Quantity               # W/(m^2 K), per REAL contact area
    R_K: Quantity               # m^2 K/W
    flux: Quantity              # W/m^2
    fit_a: SideFit
    fit_b: SideFit
    naive_delta_T: float        # what the adjacent-bin shortcut would give
    sensitivity: dict = field(default_factory=dict)

    @property
    def naive_error(self) -> float:
        """Fractional error the adjacent-bin shortcut would have introduced."""
        return (self.naive_delta_T - self.delta_T.si) / self.delta_T.si


def _fit(z: np.ndarray, T: np.ndarray, interface_z: float) -> SideFit:
    """Least squares, plus the standard error of the extrapolated value.

    R^2 alone is a poor gate here. When the bulk gradient is shallow compared
    with the bin-to-bin scatter -- common at low flux, which is exactly where
    you want to be for linear response -- R^2 is low even though the slope is
    perfectly well determined and the extrapolation is sound. R^2 is measuring
    signal-to-noise, not linearity.

    The quantity that actually matters is how well the extrapolated value at
    the interface is pinned down:

        se(z0) = s * sqrt(1/n + (z0 - zbar)^2 / Sxx),  s^2 = SSres/(n-2)

    This is the textbook standard error of the mean response, and it grows
    with the distance extrapolated -- correctly penalising a fit window that
    stops far short of the interface. Propagated through both sides it gives
    an honest uncertainty on dT, and hence on h_K, instead of a pass/fail.
    """
    slope, intercept = np.polyfit(z, T, 1)
    pred = slope * z + intercept
    resid = T - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((T - np.mean(T)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n = z.size
    zbar = float(np.mean(z))
    sxx = float(np.sum((z - zbar) ** 2))
    if n > 2 and sxx > 0:
        s = math.sqrt(ss_res / (n - 2))
        se = s * math.sqrt(1.0 / n + (interface_z - zbar) ** 2 / sxx)
    else:
        se = float("inf")

    return SideFit(
        slope=float(slope),
        intercept=float(intercept),
        r2=r2,
        n_bins=int(n),
        z_min=float(z.min()),
        z_max=float(z.max()),
        T_at_interface=float(slope * interface_z + intercept),
        se_at_interface=float(se),
    )


def _select(
    profile: TemperatureProfile,
    side: str,
    reservoir_exclusion: float,
    interface_exclusion: float,
    bulk: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bins of one side, with the reservoir and near-interface zones removed.

    `bulk` gives the explicit z-range of the unthermostatted slab, which the
    NEMD driver always knows from the geometry it built. Supplying it is
    strongly preferred: the fractional fallback is a guess, and a reservoir
    even slightly thicker than the guessed exclusion leaves thermostatted
    bins in the fit, which drags the extrapolation toward the reservoir
    temperature and understates the jump.
    """
    mask = profile.side_a_mask if side == "a" else profile.side_b_mask
    z, T = profile.z[mask], profile.T[mask]
    if z.size == 0:
        raise ValueError(f"no bins on side {side}")

    if bulk is not None:
        lo, hi = min(bulk), max(bulk)
        inside = (z >= lo) & (z <= hi)
        z, T = z[inside], T[inside]
        if z.size == 0:
            raise ValueError(f"no bins on side {side} inside bulk range {bulk}")
        length = abs(profile.interface_z - (lo if side == "a" else hi))
        res_cut = 0.0            # the bulk range already excludes reservoirs
    else:
        length = abs(profile.interface_z - (z.min() if side == "a" else z.max()))
        res_cut = reservoir_exclusion * length

    int_cut = interface_exclusion * length

    if side == "a":
        keep = (z >= z.min() + res_cut) & (z <= profile.interface_z - int_cut)
    else:
        keep = (z <= z.max() - res_cut) & (z >= profile.interface_z + int_cut)
    return z[keep], T[keep]


def naive_adjacent_bin_delta_T(profile: TemperatureProfile) -> float:
    """The WRONG estimator, provided only for comparison.

    Differences the bins immediately either side of the interface. Retained so
    the error it introduces can be quantified and reported rather than argued
    about. Never used to produce a result.
    """
    a = profile.T[profile.side_a_mask]
    b = profile.T[profile.side_b_mask]
    return float(a[-1] - b[0])


def extract_kapitza(
    profile: TemperatureProfile,
    flux: float,
    flags: FlagSet,
    *,
    reservoir_exclusion: float = DEFAULT_RESERVOIR_EXCLUSION,
    interface_exclusion: float = DEFAULT_INTERFACE_EXCLUSION,
    bulk_a: tuple[float, float] | None = None,
    bulk_b: tuple[float, float] | None = None,
    k_expected_a: float | None = None,
    k_expected_b: float | None = None,
) -> KapitzaExtraction:
    """Extract h_K from a steady profile by extrapolated linear fits.

    `flux` is the magnitude of the heat flux through the system in W/m^2,
    measured from the reservoir energy rates (see `flux_from_reservoirs`),
    never from a nominal setpoint.

    `bulk_a` and `bulk_b` give the z-range of each unthermostatted slab. Pass
    them whenever the geometry is known -- the NEMD driver always does. The
    fractional fallback guesses where the reservoirs end, and a reservoir
    thicker than the guess leaves pinned bins in the fit, biasing the jump
    low while R^2 collapses.
    """
    if flux <= 0:
        raise ValueError("flux must be positive; pass the magnitude")

    if bulk_a is None or bulk_b is None:
        flags.add(
            Code.NEMD_FIT_WINDOW_GUESSED,
            Severity.CAUTION,
            f"Fit windows were guessed as fixed fractions "
            f"({reservoir_exclusion:.0%} reservoir, "
            f"{interface_exclusion:.0%} interface) rather than taken from the "
            f"run geometry.",
            remedy=(
                "Pass bulk_a and bulk_b with the actual slab bounds. If the "
                "reservoirs are thicker than the guessed exclusion, "
                "thermostatted bins stay in the fit and the jump comes out "
                "too small. Check R^2 on both sides before trusting this."
            ),
        )

    za, Ta = _select(profile, "a", reservoir_exclusion, interface_exclusion, bulk_a)
    zb, Tb = _select(profile, "b", reservoir_exclusion, interface_exclusion, bulk_b)

    for side, z in (("A", za), ("B", zb)):
        if z.size < MIN_FIT_BINS:
            flags.add(
                Code.NEMD_FEW_BINS,
                Severity.CRITICAL,
                f"Only {z.size} bins remain on side {side} after excluding the "
                f"reservoir and near-interface zones (minimum {MIN_FIT_BINS}).",
                remedy=(
                    "Lengthen the slabs, use finer binning, or narrow the "
                    "exclusion zones. A fit through this few bins cannot "
                    "support an extrapolation to the interface."
                ),
                n_bins=z.size,
            )
            raise ValueError(f"too few bins on side {side} to fit ({z.size})")

    fit_a = _fit(za, Ta, profile.interface_z)
    fit_b = _fit(zb, Tb, profile.interface_z)

    delta_T = fit_a.T_at_interface - fit_b.T_at_interface
    se_delta_T = math.hypot(fit_a.se_at_interface, fit_b.se_at_interface)
    if delta_T <= 0:
        flags.add(
            Code.NEMD_NONPHYSICAL_JUMP,
            Severity.CRITICAL,
            f"Extrapolated temperature jump is {delta_T:.4g} K, i.e. not "
            f"positive in the direction of heat flow.",
            remedy=(
                "Heat must flow from hot to cold across the interface. Check "
                "the sign convention on the flux, that side A is the hot side, "
                "and that the interface position is correct."
            ),
            delta_T=delta_T,
        )
        raise ValueError(f"non-physical temperature jump: {delta_T:.4g} K")

    # Fitted conductivities as an independent sanity check.
    for side, fit, k_exp in (("A", fit_a, k_expected_a), ("B", fit_b, k_expected_b)):
        if k_exp is None:
            continue
        k_fit = fit.conductivity(flux)
        ratio = k_fit / k_exp
        if not (0.5 <= ratio <= 2.0):
            flags.add(
                Code.NEMD_CONDUCTIVITY_MISMATCH,
                Severity.WARNING,
                f"Side {side} gradient implies k = {k_fit:.3g} W/m.K against "
                f"an expected {k_exp:.3g} (ratio {ratio:.2f}).",
                remedy=(
                    "The slab is probably shorter than the dominant phonon "
                    "mean free paths, so the apparent conductivity is "
                    "size-limited. That also affects the near-interface "
                    "region, so treat the extracted h_K as length dependent "
                    "and run the length convergence test."
                ),
                k_fit=k_fit, k_expected=k_exp,
            )

    # The extrapolation uncertainty, not R^2, is the meaningful quality gate.
    rel_se = se_delta_T / abs(delta_T) if delta_T else float("inf")
    if rel_se > MAX_DELTA_T_REL_SE:
        flags.add(
            Code.NEMD_POOR_FIT,
            Severity.WARNING,
            f"The extrapolated jump is {delta_T:.3g} +/- {se_delta_T:.3g} K "
            f"({rel_se * 100:.0f} % standard error), so h_K is poorly "
            f"determined by this run.",
            remedy=(
                "Either the bulk gradient is too shallow against the bin "
                "scatter, or the fit window stops too far from the interface. "
                "Average over more timesteps, raise the flux while staying in "
                "linear response, or widen the fit window toward the "
                "interface if the profile is linear there."
            ),
            rel_se=rel_se,
        )

    sens = _exclusion_sensitivity(profile, flux, reservoir_exclusion, bulk_a, bulk_b)
    if sens["relative_spread"] > MAX_EXCLUSION_SENSITIVITY:
        flags.add(
            Code.NEMD_EXCLUSION_SENSITIVE,
            Severity.WARNING,
            f"The extracted jump moves by {sens['relative_spread'] * 100:.1f} % "
            f"as the near-interface exclusion is swept over "
            f"{sens['swept_range'][0]:.0%}-{sens['swept_range'][1]:.0%} of the "
            f"slab length.",
            remedy=(
                "A result that depends on where you stopped fitting is not a "
                "measurement. The non-equilibrium region near the interface is "
                "probably wider than the exclusion zone; widen it and lengthen "
                "the slabs so enough linear region remains."
            ),
            spread=sens["relative_spread"],
        )

    naive = naive_adjacent_bin_delta_T(profile)
    h = flux / delta_T

    conf = Confidence.MEDIUM
    if flags.worst >= Severity.WARNING:
        conf = Confidence.LOW

    return KapitzaExtraction(
        delta_T=Quantity(si=delta_T, unit="K", uncertainty=se_delta_T,
                         provenance=Provenance.COMPUTED,
                         confidence=conf, name="delta_T"),
        h_K=Quantity(si=h, unit="MW/m2K", uncertainty=h * rel_se,
                     provenance=Provenance.COMPUTED,
                     confidence=conf, name="h_K",
                     note="direct NEMD, extrapolated fits; per REAL contact area"),
        R_K=Quantity(si=1.0 / h, unit="m2K/W", provenance=Provenance.COMPUTED,
                     confidence=conf, name="R_K"),
        flux=Quantity(si=flux, unit="W/m2", provenance=Provenance.COMPUTED,
                      name="flux"),
        fit_a=fit_a,
        fit_b=fit_b,
        naive_delta_T=naive,
        sensitivity=sens,
    )


def _exclusion_sensitivity(
    profile: TemperatureProfile,
    flux: float,
    reservoir_exclusion: float,
    bulk_a: tuple[float, float] | None = None,
    bulk_b: tuple[float, float] | None = None,
) -> dict:
    """Sweep the near-interface exclusion and see how much dT moves."""
    widths = [0.05, 0.10, 0.15, 0.20, 0.25]
    values: list[float] = []
    used: list[float] = []
    for w in widths:
        try:
            za, Ta = _select(profile, "a", reservoir_exclusion, w, bulk_a)
            zb, Tb = _select(profile, "b", reservoir_exclusion, w, bulk_b)
            if za.size < MIN_FIT_BINS or zb.size < MIN_FIT_BINS:
                continue
            fa = _fit(za, Ta, profile.interface_z)
            fb = _fit(zb, Tb, profile.interface_z)
            values.append(fa.T_at_interface - fb.T_at_interface)
            used.append(w)
        except ValueError:
            continue
    if len(values) < 2:
        return {"relative_spread": 0.0, "swept_range": (0.0, 0.0),
                "values": values, "widths": used, "insufficient": True}
    med = float(np.median(values))
    spread = (max(values) - min(values)) / abs(med) if med else float("inf")
    return {
        "relative_spread": spread,
        "swept_range": (min(used), max(used)),
        "values": values,
        "widths": used,
        "h_K_range": (flux / max(values), flux / min(values)),
    }


def flux_from_reservoirs(
    energy_added: float,
    energy_removed: float,
    duration: float,
    area: float,
    flags: FlagSet,
) -> float:
    """Heat flux from the reservoir energy accumulators, with a balance check.

    Energies in J, duration in s, area in m^2. At steady state the hot
    reservoir's input and the cold reservoir's output must match; a persistent
    imbalance means the system is still heating up, energy is leaking through
    the walls, or the thermostat is doing work that is not being accounted.
    Using a nominal setpoint instead of the measured rates hides all of that.
    """
    if duration <= 0 or area <= 0:
        raise ValueError("duration and area must be positive")
    q_hot = abs(energy_added) / (duration * area)
    q_cold = abs(energy_removed) / (duration * area)
    mean = 0.5 * (q_hot + q_cold)
    if mean <= 0:
        raise ValueError("no net energy flow through the reservoirs")

    imbalance = abs(q_hot - q_cold) / mean
    if imbalance > MAX_FLUX_IMBALANCE:
        flags.add(
            Code.NEMD_FLUX_INCONSISTENT,
            Severity.WARNING,
            f"Hot and cold reservoir fluxes differ by {imbalance * 100:.1f} % "
            f"({q_hot:.4g} vs {q_cold:.4g} W/m^2).",
            remedy=(
                "At steady state these must agree. Check that the averaging "
                "window starts after the transient, that the walls are truly "
                "fixed, and that no other fix is adding or removing energy."
            ),
            imbalance=imbalance,
        )
    return mean
