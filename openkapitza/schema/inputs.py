"""Input models for a joint-conductance calculation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .flags import Code, FlagSet, Severity
from .quantities import Confidence, Provenance, Quantity


@dataclass
class MaterialSpec:
    """Bulk and surface-mechanical properties of one of the two solids."""

    key: str
    name: str
    phase: str                      # "crystalline" | "amorphous"
    k: Quantity                     # thermal conductivity, W/(m*K)
    rho: Quantity                   # density, kg/m^3
    E: Quantity                     # Young's modulus, Pa
    nu: Quantity                    # Poisson ratio, -
    Hv: Quantity                    # Vickers microhardness, Pa
    v_long: Quantity | None = None  # longitudinal sound speed, m/s
    v_trans: Quantity | None = None # transverse sound speed, m/s
    theta_debye: Quantity | None = None   # K
    number_density: Quantity | None = None  # atoms/m^3
    c1: Quantity | None = None      # Song-Yovanovich coefficient, Pa
    c2: Quantity | None = None      # Song-Yovanovich exponent, -
    # Hot-hardness fit {C_T [1/K], T_ref [K], T_max [K]}; informational (see
    # contact.hardness.hot_hardness_factor), never applied by default.
    c1_temperature: dict | None = None
    # Heat-carrier mean free path data for the ballistic-spot check
    # (contact.ballistic, P0-11): {carrier, mfp_nm | mfp_long_fraction, ...}.
    ballistic: dict | None = None
    notes: str | None = None
    verified: bool = False

    @property
    def has_sy_coefficients(self) -> bool:
        return self.c1 is not None and self.c2 is not None


@dataclass
class SurfaceSpec:
    """Measured topography of one surface.

    `sigma` and `m` are per-surface values. The CMY model uses the combined
    pair sigma = sqrt(s1^2 + s2^2), m = sqrt(m1^2 + m2^2); that combination
    happens in the contact module, not here, so the provenance of each
    individual measurement stays visible.
    """

    sigma: Quantity                 # RMS roughness, m
    m: Quantity | None = None       # mean absolute asperity slope, -
    # How the slope was measured: "stylus" (the CMY calibration practice),
    # "optical", "afm", or None (not stated). See check_slope_instrument.
    slope_instrument: str | None = None

    def check_slope_instrument(self, flags: FlagSet, side: str = "") -> None:
        """Flag a measured slope from an instrument finer than CMY's calibration.

        CMY's constants were fitted to stylus-profilometer slopes (~1 cm
        traces). RMS slope is set by the shortest wavelengths an instrument
        resolves (Persson et al. 2010), so AFM or high-resolution optical
        profiles report a larger m and over-predict h_c in proportion
        (P0-12, KB 24).
        """
        inst = (self.slope_instrument or "").lower()
        if self.m is None or not inst or inst.startswith("stylus") or inst == "unknown":
            return
        fine = any(t in inst for t in ("afm", "optical", "interfer", "confocal", "white"))
        flags.add(
            Code.SLOPE_INSTRUMENT,
            Severity.WARNING if fine else Severity.CAUTION,
            f"Surface {side} slope m = {self.m.si:.4g} was measured by "
            f"'{self.slope_instrument}'"
            + (", which resolves finer roughness than the stylus profilometry CMY "
               "was calibrated with: expect m, and so h_c, to be over-predicted, "
               "possibly several-fold." if fine else
               ", not a stylus profilometer; CMY was calibrated on stylus slopes."),
            remedy="Re-measure the slope with a stylus profilometer on ~1 cm traces, or "
                   "filter the profile to a stylus-like bandwidth before taking m.",
            instrument=self.slope_instrument)

    def slope_or_estimate(self, flags: FlagSet) -> Quantity:
        """Return the measured slope, or Antonetti's correlation if absent.

        The correlation m ~= 0.125 * (sigma * 1e6)^0.402 (sigma in m) is a fit
        to a limited set of machined metal surfaces. It is weak, it was never
        intended for ceramics or polished semiconductors, and the resulting
        conductance carries its error directly, since h_c is linear in m.
        """
        if self.m is not None:
            return self.m
        sigma_um = self.sigma.si * 1e6
        m_est = 0.125 * (sigma_um ** 0.402)
        m_alt = 0.076 * (sigma_um ** 0.52)
        flags.add(
            Code.CMY_SLOPE_ESTIMATED,
            Severity.WARNING,
            f"Asperity slope not supplied; estimated m = {m_est:.4f} from "
            f"Antonetti's roughness correlation at sigma = {sigma_um:.3f} um "
            f"(fitted for 0.216-9.6 um). A second published correlation "
            f"(Bahrami et al. 2004, m ~ 0.076 sigma^0.52) gives {m_alt:.4f}: "
            f"h_c is uncertain by that ratio ({m_est / m_alt:.2f}x).",
            remedy=(
                "Measure the mean absolute profile slope by STYLUS profilometry "
                "on traces of ~1 cm, the practice behind the CMY calibration "
                "data. Do not take m from AFM or high-resolution optical "
                "profiles: slope is dominated by the shortest wavelengths "
                "resolved (Persson et al. 2010), so a finer instrument reports "
                "a larger m and over-predicts h_c in proportion (KB 24). The "
                "correlation was fitted to machined metals and should not be "
                "trusted for polished semiconductors, ceramics or lapped "
                "surfaces."
            ),
            m_estimated=m_est,
            m_alternative=m_alt,
            sigma_um=sigma_um,
        )
        return Quantity(
            si=m_est,
            unit="-",
            provenance=Provenance.CORRELATION_DEFAULT,
            confidence=Confidence.LOW,
            citation="Antonetti et al. roughness-slope correlation, as given in Yovanovich, Culham & Teertstra, Electronics Cooling (1997); range 0.216-9.6 um",
            name="asperity_slope",
        )


@dataclass
class GasSpec:
    """Interstitial gas. `None` anywhere a GasSpec is expected means vacuum."""

    key: str
    name: str
    M: Quantity                     # molar mass, kg/mol
    gamma: Quantity                 # ratio of specific heats, -
    Pr: Quantity                    # Prandtl number, -
    mu_ref: Quantity
    mu_T_ref: Quantity
    mu_S: Quantity
    k_ref: Quantity
    k_T_ref: Quantity
    k_S: Quantity
    pressure: Quantity | None = None  # gas pressure in the gap, Pa
    notes: str | None = None
    verified: bool = False


@dataclass
class JointInputs:
    """Everything needed for a Phase-0 continuum joint-conductance run."""

    material_a: MaterialSpec
    material_b: MaterialSpec
    surface_a: SurfaceSpec
    surface_b: SurfaceSpec
    pressure: Quantity              # apparent contact pressure, Pa
    temperature: Quantity           # K
    gas: GasSpec | None = None      # None => vacuum
    accommodation_a: Quantity | None = None
    accommodation_b: Quantity | None = None
    regime_override: str | None = None   # "plastic" | "elastic" | None (auto)
    # Filled (TIM) or bonded (solder, sinter, die attach) joint: a series stack
    # replaces the contact-spot + gas-gap network (network.stack, open issue P0-1).
    fill: object | None = None
    # Truncation level of the asperity-height distribution, in units of sigma
    # (open issue P0-7). None = fully Gaussian CMY. Set, the plastic branch and
    # P/H_c use the truncated-Gaussian model (contact.truncated).
    z_trunc: Quantity | None = None
    # Macro geometry (contact.macro): combined maximum out-of-flatness of the two
    # faces, joint radius (sqrt(A/pi) for a non-circular footprint) and how the
    # joint is clamped: "uniform" | "bolted" | "unknown" (None = unknown).
    # Highest pressure the joint has carried before (contact.unloading): if
    # above `pressure`, the plastic branch follows the unloading curve.
    max_pressure: Quantity | None = None
    flatness: Quantity | None = None
    joint_radius: Quantity | None = None
    clamping: str | None = None
    label: str = ""
    # Dopant concentration (cm^-3) of a silicon side, for the doped-Si check
    # (P0-13). None = undoped or not stated.
    doping_a_cm3: float | None = None
    doping_b_cm3: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def is_vacuum(self) -> bool:
        return self.gas is None

    def summary(self) -> str:
        lines = [
            f"Joint: {self.label or (self.material_a.key + '/' + self.material_b.key)}",
            f"  T            = {self.temperature}",
            f"  P (apparent) = {self.pressure}",
            f"  sigma_a      = {self.surface_a.sigma}",
            f"  sigma_b      = {self.surface_b.sigma}",
            f"  medium       = "
            + (f"{self.fill.kind} stack (filled/bonded; no gas path)" if self.fill is not None
               else ('vacuum' if self.is_vacuum else self.gas.name)),
        ]
        if self.gas is not None and self.gas.pressure is not None:
            lines.append(f"  P_gas        = {self.gas.pressure}")
        if self.max_pressure is not None:
            lines.append(f"  P_max before = {self.max_pressure} (unloading curve)")
        if self.flatness is not None:
            lines.append(f"  flatness     = {self.flatness} (combined), joint radius "
                         f"{self.joint_radius}")
        if self.clamping:
            lines.append(f"  clamping     = {self.clamping}")
        if self.z_trunc is not None:
            lines.append(f"  z_trunc      = {self.z_trunc.si:g} sigma (truncated Gaussian)")
        return "\n".join(lines)


DOPED_SI_ONSET_CM3 = 1e19


def check_doped_si(inputs: "JointInputs", flags: FlagSet) -> None:
    """Flag a degenerately doped silicon side (P0-13, KB 24).

    Electron-phonon scattering lowers silicon's lattice conductivity above
    ~1e19 cm^-3, by up to ~42 % for p-type near 1e21 (Liao et al., PRL 114,
    115901, 2015), while the electronic part stays below ~1 W/mK. The handbook
    k then over-states the constriction conductance. No correction is applied:
    the reduction depends on dopant type and concentration, which the sourced
    data here do not tabulate.
    """
    for side, mat, n in (("A", inputs.material_a, inputs.doping_a_cm3),
                         ("B", inputs.material_b, inputs.doping_b_cm3)):
        if n is None or n < DOPED_SI_ONSET_CM3 or not mat.key.lower().startswith("si") \
                or mat.key in ("SiO2",):
            continue
        flags.add(
            Code.DOPED_SI_CONDUCTIVITY, Severity.CAUTION,
            f"Side {side} ({mat.key}) is doped at {n:.1e} cm^-3: silicon's lattice "
            f"conductivity falls above ~1e19 cm^-3 (up to ~42 % for p-type near 1e21; "
            f"Liao et al. 2015), but k = {mat.k.si:.0f} W/mK is the undoped value.",
            remedy="Supply a measured conductivity for this wafer, or treat h_c as an "
                   "upper estimate.",
            doping_cm3=n)

