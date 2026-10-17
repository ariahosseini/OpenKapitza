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
        flags.add(
            Code.CMY_SLOPE_ESTIMATED,
            Severity.WARNING,
            f"Asperity slope not supplied; estimated m = {m_est:.4f} from "
            f"Antonetti's roughness correlation at sigma = {sigma_um:.3f} um.",
            remedy=(
                "Measure the mean absolute profile slope by profilometry. "
                "h_c is directly proportional to m, so an error here passes "
                "straight into the answer. The correlation was fitted to "
                "machined metals and should not be trusted for polished "
                "semiconductors, ceramics or lapped surfaces."
            ),
            m_estimated=m_est,
            sigma_um=sigma_um,
        )
        return Quantity(
            si=m_est,
            unit="-",
            provenance=Provenance.CORRELATION_DEFAULT,
            confidence=Confidence.LOW,
            citation="Antonetti roughness-slope correlation (NEEDS VERIFICATION)",
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
    label: str = ""
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
            f"  medium       = {'vacuum' if self.is_vacuum else self.gas.name}",
        ]
        if self.gas is not None and self.gas.pressure is not None:
            lines.append(f"  P_gas        = {self.gas.pressure}")
        return "\n".join(lines)
