"""Gas-gap conductance with temperature-jump (rarefaction) correction.

    h_g = k_g / (Y + M)

Y is the mean separation of the two mean planes. M is the rarefaction
parameter, which accounts for the temperature jump at each gas-solid boundary
when the mean free path is not negligible against the gap:

    M     = alpha * beta * Lambda
    alpha = (2 - a1)/a1 + (2 - a2)/a2        (thermal accommodation)
    beta  = 2*gamma / ((gamma + 1) * Pr)

The model is a lumped simplification of the full gap integral: it replaces
the distribution of local gap thicknesses across the joint with a single mean
separation. That approximation is good when the gap is large compared with
the mean free path and degrades as M grows relative to Y, which is exactly
the condition flagged below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Confidence, Provenance, Quantity, make_derived
from .gasprops import GasState

# Knudsen number thresholds. Conventional boundaries for the flow regimes.
KN_SLIP = 0.01
KN_TRANSITION = 0.1
KN_FREE_MOLECULAR = 10.0


@dataclass
class GapResult:
    h_g: Quantity           # W/(m^2 K), per apparent area
    M: Quantity             # rarefaction parameter, m
    Y: Quantity             # mean plane separation, m
    knudsen: Quantity       # Lambda / Y, -
    alpha_term: Quantity    # (2-a1)/a1 + (2-a2)/a2, -
    state: GasState


def accommodation_term(a1: Quantity, a2: Quantity) -> Quantity:
    """alpha = (2 - a1)/a1 + (2 - a2)/a2.

    Diverges as either coefficient approaches zero -- a perfectly specular
    surface exchanges no energy with the gas and the gap stops conducting.
    """
    for a in (a1, a2):
        if not (0.0 < a.si <= 1.0):
            raise ValueError(
                f"accommodation coefficient must lie in (0, 1], got {a.si}"
            )
    term = (2.0 - a1.si) / a1.si + (2.0 - a2.si) / a2.si
    return make_derived(term, unit="-", inputs=[a1, a2], name="alpha_term")


def gap_conductance(
    state: GasState,
    Y: Quantity,
    a1: Quantity,
    a2: Quantity,
    flags: FlagSet,
) -> GapResult:
    """Compute the gas-gap conductance per unit apparent area."""
    alpha = accommodation_term(a1, a2)
    M_si = alpha.si * state.beta.si * state.mfp.si
    M = make_derived(
        M_si, unit="um", inputs=[alpha, state.beta, state.mfp], name="M",
    )

    ratio = M_si / Y.si
    kn = state.mfp.si / Y.si
    cap = None

    if ratio > 1.0:
        cap = Confidence.LOW
        flags.add(
            Code.GAP_RAREFIED,
            Severity.WARNING,
            f"Rarefaction parameter M = {M.to('um'):.3g} um exceeds the mean "
            f"plane separation Y = {Y.to('um'):.3g} um (M/Y = {ratio:.2g}); the "
            f"gap conductance is dominated by the temperature jump rather than "
            f"by conduction through the gas.",
            remedy=(
                "In this regime h_g is controlled almost entirely by the "
                "accommodation coefficients, which are the least certain "
                "numbers in the model. Either measure them or treat h_g as an "
                "order-of-magnitude estimate."
            ),
            M_over_Y=ratio,
        )

    if kn > KN_TRANSITION:
        cap = Confidence.LOW if cap is None else cap
        regime = (
            "free-molecular" if kn > KN_FREE_MOLECULAR else "transition"
        )
        flags.add(
            Code.GAP_CONTINUUM_BREAKDOWN,
            Severity.WARNING,
            f"Knudsen number Lambda/Y = {kn:.3g} puts the gap in the {regime} "
            f"regime.",
            remedy=(
                "The lumped h_g = k_g/(Y+M) form is a continuum-plus-jump "
                "model. Deep in the free-molecular regime the heat flux no "
                "longer scales with a gas conductivity at all, and a kinetic "
                "treatment is needed. Raising the gas pressure or using a "
                "heavier gas moves the joint back toward continuum."
            ),
            knudsen=kn,
        )

    h_si = state.k_g.si / (Y.si + M_si)
    h_g = make_derived(
        h_si, unit="W/m2K", inputs=[state.k_g, Y, M], confidence=cap, name="h_g",
    )

    return GapResult(
        h_g=h_g,
        M=M,
        Y=Y,
        knudsen=make_derived(kn, unit="-", inputs=[state.mfp, Y], name="knudsen"),
        alpha_term=alpha,
        state=state,
    )


def gap_conductance_integral(
    state: GasState,
    Y: Quantity,
    sigma: Quantity,
    a1: Quantity,
    a2: Quantity,
    flags: FlagSet,
    n_quad: int = 2000,
) -> GapResult:
    """Gas-gap conductance from the Gaussian gap integral.

    OPT-IN ALTERNATIVE to :func:`gap_conductance`. The default remains the
    lumped ``h_g = k_g / (Y + M)`` because that is the form the project
    specifies and the form validated in the literature correlations.

    Rationale
    ---------
    The local gap thickness is not uniform. For two conforming rough surfaces
    it is Gaussian about the mean plane separation Y with standard deviation
    equal to the combined roughness sigma. Because the local conductance goes
    as 1/(t + M), and 1/x is convex, averaging over that distribution gives a
    LARGER conductance than evaluating at the mean thickness:

        E[1/(t+M)] > 1/(E[t]+M)                      (Jensen's inequality)

    so the lumped form systematically UNDERpredicts, and the deficit is
    largest where M is small compared with Y -- the continuum limit, where
    the 1/t convexity is not softened by the M offset.

        h_g = k_g * integral_0^inf phi((t-Y)/sigma)/sigma * dt/(t + M)

    The lower limit is 0, not -inf, and the Gaussian is deliberately NOT
    renormalised over t > 0. The t < 0 tail is the region where the surfaces
    touch; that area fraction belongs to the contact path h_c, so leaving it
    out of the gap integral is what correctly weights h_g by the gap's own
    share of the apparent area.

    Numerics
    --------
    In reduced variables u = t/sigma, Ys = Y/sigma, Ms = M/sigma the integral
    is (k_g/sigma) * (1/sqrt(2*pi)) * I with

        I = integral_0^U exp(-(u - Ys)^2 / 2) / (u + Ms) du

    which is logarithmically divergent as Ms -> 0: the Gaussian leaves finite
    density at t -> 0 while 1/t blows up. Rather than cut the singularity off
    by hand, substitute w = ln(u + Ms), so u = e^w - Ms and du = e^w dw. The
    1/(u + Ms) factor cancels EXACTLY against du and the integrand becomes

        I = integral_{ln Ms}^{ln(U + Ms)} exp(-(e^w - Ms - Ys)^2 / 2) dw

    which is bounded by 1 everywhere and smooth, so fixed-step Simpson is
    accurate and no singularity handling is needed. The weak divergence then
    shows up honestly as an I that grows like ln(1/Ms), not as a quadrature
    failure.
    """
    if n_quad % 2:
        n_quad += 1                       # Simpson needs an even interval count
    if sigma.si <= 0.0:
        raise ValueError(f"combined roughness must be positive, got {sigma.si}")

    alpha = accommodation_term(a1, a2)
    M_si = alpha.si * state.beta.si * state.mfp.si
    M = make_derived(
        M_si, unit="um", inputs=[alpha, state.beta, state.mfp], name="M",
    )

    Ys = Y.si / sigma.si
    Ms = M_si / sigma.si
    if Ms <= 0.0:
        # A real gas always has M > 0; guard only against a degenerate input.
        Ms = 1e-12

    upper = Ys + 8.0                      # 8 sigma above the mean plane
    w_lo, w_hi = math.log(Ms), math.log(upper + Ms)
    step = (w_hi - w_lo) / n_quad
    total = 0.0
    for i in range(n_quad + 1):
        w = w_lo + i * step
        weight = 1 if i in (0, n_quad) else (4 if i % 2 else 2)
        total += weight * math.exp(-0.5 * (math.exp(w) - Ms - Ys) ** 2)
    total *= step / 3.0

    h_si = (state.k_g.si / sigma.si) * total / math.sqrt(2.0 * math.pi)

    ratio = M_si / Y.si
    kn = state.mfp.si / Y.si
    cap = Confidence.LOW if (ratio > 1.0 or kn > KN_TRANSITION) else None

    lumped = state.k_g.si / (Y.si + M_si)
    h_g = make_derived(
        h_si,
        unit="W/m2K",
        inputs=[state.k_g, Y, sigma, M],
        confidence=cap,
        name="h_g",
        note=(
            f"Gaussian gap integral, Y/sigma = {Ys:.2f}, M/sigma = {Ms:.3g}; "
            f"{h_si / lumped:.3f}x the lumped k_g/(Y+M)"
        ),
    )

    return GapResult(
        h_g=h_g,
        M=M,
        Y=Y,
        knudsen=make_derived(kn, unit="-", inputs=[state.mfp, Y], name="knudsen"),
        alpha_term=alpha,
        state=state,
    )


def vacuum_gap() -> Quantity:
    """h_g = 0 in vacuum. Radiation across the gap is not modelled.

    Radiative exchange across a microscopic gap is negligible at the
    temperatures this package targets -- it becomes comparable to solid
    conduction only at very low contact pressure and high temperature. If
    either applies, add it explicitly rather than relying on this.
    """
    return Quantity(
        si=0.0,
        unit="W/m2K",
        provenance=Provenance.COMPUTED,
        confidence=Confidence.HIGH,
        name="h_g",
        note="vacuum: no gas path; radiation not modelled",
    )
