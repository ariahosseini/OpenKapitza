"""Phase-0 placeholder for the Kapitza conductance.

This is NOT the MD calculation. It is a Debye diffuse-mismatch estimate whose
only job is to let the resistance network close before NEMD exists, and --
more usefully -- to answer the question "would running MD change the answer?"
before anyone spends node-hours finding out.

Diffuse mismatch model, Debye approximation. For each side define the
flux-weighted and density-weighted velocity sums over one longitudinal and
two transverse branches:

    S2 = 1/v_L^2 + 2/v_T^2        (phonon flux weight)
    S3 = 1/v_L^3 + 2/v_T^3        (mode density weight)

A diffusely scattered phonon forgets where it came from, so the transmission
probability is the ratio of the available flux on the far side:

    alpha_12 = S2_2 / (S2_1 + S2_2)

THE ELASTIC CUTOFF. Scattering here is elastic, so a phonon can only cross if
its frequency exists on BOTH sides. Phonons above the softer material's
maximum frequency have no states to transmit into and carry nothing. The
frequency integral must therefore stop at the highest frequency common to all
branches in both solids -- set by the most restrictive (slowest) branch, in
practice the transverse branch of the softer material.

Omitting that cutoff is not a small error. For Si/Ge at 300 K it inflates the
answer by roughly a factor of 4.5, because only about a fifth of silicon's
phonon spectrum lies below germanium's transverse cutoff.

Each branch gets its own Debye cutoff from the material's own zone boundary:

    k_D,i      = (6 pi^2 n_i)^(1/3)
    omega_D,ij = v_ij * k_D,i
    omega_cut  = min over every branch of both materials

and the conductance from side 1 becomes

    h = (k_B / 8 pi^2) * (k_B T / hbar)^3 * S2_1 * I(x_cut) * alpha_12
    I(x)  = integral_0^x u^4 e^u / (e^u - 1)^2 du
    x_cut = hbar * omega_cut / (k_B T)

HOW WRONG IS THIS, EVEN WHEN RIGHT? DMM remains a severe idealisation:
elastic, diffuse scattering of a Debye phonon gas with no inelastic channels,
no interfacial bonding chemistry, no disorder or intermixing, and no optical
branches at all -- which for Ge/Si can carry over 40 % of the conductance at
300 K. The error is also DIRECTIONAL: the Debye construction overestimates
the maximum frequency of each branch, inflating the incident phonon flux, so
Debye-DMM systematically OVER-predicts. Against real dispersion it can be off
by a large factor in strongly mismatched systems.

Everything this module returns is therefore marked PLACEHOLDER with LOW
confidence, and the network module refuses to present it as a final answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..schema.flags import Code, FlagSet, Severity
from ..schema.inputs import MaterialSpec
from ..schema.quantities import Confidence, Provenance, Quantity, make_derived
from ..schema.units import HBAR, K_B

# Factor-of-N band reported as the uncertainty on a DMM estimate. This is a
# statement about how far DMM typically sits from measurement, not a
# propagated uncertainty. The band is deliberately wide and the bias is known
# to be upward (see the module docstring).
DMM_UNCERTAINTY_FACTOR = 2.5


@dataclass
class KapitzaEstimate:
    h_K: Quantity           # W/(m^2 K), per REAL contact area
    transmission: Quantity  # alpha_12, -
    source: str             # "dmm_debye" | "user" | "md"


def debye_heat_capacity(
    theta_debye: Quantity, number_density: Quantity, temperature: Quantity
) -> Quantity:
    """Volumetric heat capacity from the Debye model, J/(m^3 K).

        C_V = 9 n k_B (T/theta)^3 * integral_0^(theta/T) x^4 e^x / (e^x-1)^2 dx
    """
    T = temperature.si
    theta = theta_debye.si
    n = number_density.si
    x_max = theta / T

    # Simpson integration; the integrand is smooth and bounded (-> x^2 at
    # small x, exponentially damped at large x), so a modest grid is ample.
    steps = 2000
    h = x_max / steps
    total = 0.0
    for i in range(steps + 1):
        x = i * h
        if x < 1e-8:
            f = 0.0
        else:
            ex = math.exp(x) if x < 700 else float("inf")
            f = 0.0 if ex == float("inf") else (x ** 4) * ex / (ex - 1.0) ** 2
        w = 1 if i in (0, steps) else (4 if i % 2 else 2)
        total += w * f
    integral = total * h / 3.0

    C = 9.0 * n * K_B * (T / theta) ** 3 * integral
    return make_derived(
        C, unit="J/m3K",
        inputs=[theta_debye, number_density, temperature], name="C_V",
    )


def _velocity_sums(mat: MaterialSpec) -> tuple[float, float]:
    vl, vt = mat.v_long.si, mat.v_trans.si
    s2 = 1.0 / vl ** 2 + 2.0 / vt ** 2
    s3 = 1.0 / vl ** 3 + 2.0 / vt ** 3
    return s2, s3


def debye_wavevector(number_density: float) -> float:
    """Debye zone-boundary wavevector, k_D = (6 pi^2 n)^(1/3)."""
    return (6.0 * math.pi ** 2 * number_density) ** (1.0 / 3.0)


def branch_cutoff_frequencies(mat: MaterialSpec) -> list[float]:
    """Debye cutoff angular frequency of each branch: one LA, two TA."""
    k_D = debye_wavevector(mat.number_density.si)
    return [mat.v_long.si * k_D, mat.v_trans.si * k_D, mat.v_trans.si * k_D]


def common_cutoff_frequency(a: MaterialSpec, b: MaterialSpec) -> tuple[float, str]:
    """Highest frequency present in every branch of both solids.

    Elastic scattering admits only frequencies that exist on both sides, so
    the limiting branch is the slowest one of either material. Returns the
    frequency and which material set it, for reporting.
    """
    cuts_a = branch_cutoff_frequencies(a)
    cuts_b = branch_cutoff_frequencies(b)
    w_a, w_b = min(cuts_a), min(cuts_b)
    return (w_a, a.key) if w_a <= w_b else (w_b, b.key)


def debye_integral(x_max: float, steps: int = 4000) -> float:
    """I(x) = integral_0^x u^4 e^u / (e^u - 1)^2 du, by Simpson's rule.

    The integrand tends to u^2 as u -> 0 and is exponentially damped at large
    u, so it is smooth and bounded and a modest grid is ample.
    """
    if x_max <= 0:
        return 0.0
    if steps % 2:
        steps += 1
    h = x_max / steps
    total = 0.0
    for i in range(steps + 1):
        u = i * h
        if u < 1e-8:
            f = 0.0
        else:
            eu = math.exp(u) if u < 700 else float("inf")
            f = 0.0 if eu == float("inf") else (u ** 4) * eu / (eu - 1.0) ** 2
        w = 1 if i in (0, steps) else (4 if i % 2 else 2)
        total += w * f
    return total * h / 3.0


def dmm_conductance(
    material_a: MaterialSpec,
    material_b: MaterialSpec,
    temperature: Quantity,
    flags: FlagSet,
) -> KapitzaEstimate:
    """Debye-DMM estimate of h_K, per unit REAL contact area."""
    for m in (material_a, material_b):
        if m.v_long is None or m.v_trans is None or m.theta_debye is None \
                or m.number_density is None:
            raise ValueError(
                f"material '{m.key}' lacks the sound velocities, Debye "
                f"temperature or number density needed for a DMM estimate"
            )

    s2_a, _ = _velocity_sums(material_a)
    s2_b, _ = _velocity_sums(material_b)

    alpha = s2_b / (s2_a + s2_b)

    # Elastic cutoff: only frequencies present on both sides can cross.
    omega_cut, limiting = common_cutoff_frequency(material_a, material_b)
    T = temperature.si
    x_cut = HBAR * omega_cut / (K_B * T)

    h_si = (
        (K_B / (8.0 * math.pi ** 2))
        * (K_B * T / HBAR) ** 3
        * s2_a
        * debye_integral(x_cut)
        * alpha
    )

    flags.add(
        Code.KAPITZA_PLACEHOLDER,
        Severity.WARNING,
        f"Kapitza conductance is a Debye-DMM placeholder, not an MD result: "
        f"h_K = {h_si / 1e6:.1f} MW/m^2K with transmission {alpha:.3f}, "
        f"cut off at the slowest branch of '{limiting}' "
        f"(hbar*omega/kT = {x_cut:.2f}).",
        remedy=(
            f"DMM assumes elastic diffuse scattering of a Debye phonon gas and "
            f"ignores interfacial bonding, disorder, inelastic channels and "
            f"optical branches. Its error is directional: the Debye "
            f"construction overestimates each branch's maximum frequency, so "
            f"it systematically OVER-predicts. Treat it as good to roughly a "
            f"factor of {DMM_UNCERTAINTY_FACTOR:g}, biased high. Run the "
            f"Phase-1 NEMD calculation before quoting h_K, and check the "
            f"sensitivity report first to see whether it would move the joint "
            f"conductance at all."
        ),
        h_K_MW=h_si / 1e6,
        transmission=alpha,
        x_cut=x_cut,
    )

    quantum_check(material_a, material_b, temperature, flags)

    h_K = Quantity(
        si=h_si,
        unit="MW/m2K",
        uncertainty=h_si * (DMM_UNCERTAINTY_FACTOR - 1.0),
        provenance=Provenance.PLACEHOLDER,
        confidence=Confidence.LOW,
        name="h_K",
        note="Debye DMM; per unit REAL contact area",
    )
    return KapitzaEstimate(
        h_K=h_K,
        transmission=make_derived(
            alpha, unit="-", inputs=[temperature], confidence=Confidence.LOW,
            provenance=Provenance.PLACEHOLDER, name="transmission",
        ),
        source="dmm_debye",
    )


def quantum_check(
    material_a: MaterialSpec,
    material_b: MaterialSpec,
    temperature: Quantity,
    flags: FlagSet,
    *,
    ratio_threshold: float = 0.5,
) -> None:
    """Warn when T is low enough that classical MD would misrepresent C_V.

    Below roughly half the Debye temperature a classical simulation populates
    high-frequency modes that are in reality frozen out, overestimating the
    heat capacity and therefore, generally, the conductance.

    The package deliberately does NOT apply a quantum correction automatically.
    The usual fix -- rescaling the MD temperature so classical and quantum
    total energies match, then scaling the conductance by dT_quantum/dT_classical
    -- is not variationally justified, is known to be unreliable for transport
    properties, and can move the answer in the wrong direction. It is offered
    only as an explicitly-labelled bracket, never as the primary result.
    """
    T = temperature.si
    worst = None
    for m in (material_a, material_b):
        if m.theta_debye is None:
            continue
        r = T / m.theta_debye.si
        if worst is None or r < worst[1]:
            worst = (m, r)
    if worst is None:
        return
    mat, ratio = worst
    if ratio >= ratio_threshold:
        return

    severity = Severity.WARNING if ratio > 0.25 else Severity.CRITICAL
    flags.add(
        Code.QUANTUM_EFFECTS,
        severity,
        f"T = {T:.0f} K is {ratio:.2f} of the Debye temperature of "
        f"'{mat.key}' ({mat.theta_debye.to('K'):.0f} K). Classical MD will "
        f"overpopulate high-frequency modes and overestimate the heat capacity.",
        remedy=(
            "Report the uncorrected value as the primary result. A quantum "
            "correction can be computed as a bracket, but the standard "
            "temperature-rescaling approach is not reliable for interfacial "
            "transport and should be labelled as such, not folded silently "
            "into the answer."
        ),
        T_over_theta=ratio,
    )


def user_kapitza(
    h_K_si: float,
    *,
    uncertainty: float | None = None,
    note: str = "user-supplied",
) -> KapitzaEstimate:
    """Wrap a measured or externally-computed h_K (per REAL contact area)."""
    return KapitzaEstimate(
        h_K=Quantity(
            si=h_K_si, unit="MW/m2K", uncertainty=uncertainty,
            provenance=Provenance.USER_MEASURED, confidence=Confidence.MEDIUM,
            name="h_K", note=note + "; per unit REAL contact area",
        ),
        transmission=Quantity(
            si=float("nan"), unit="-", provenance=Provenance.USER_MEASURED,
            name="transmission",
        ),
        source="user",
    )
