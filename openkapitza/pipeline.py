"""End-to-end Phase-0 continuum calculation.

One entry point, `run_continuum`, takes a JointInputs and returns everything
the agent layer needs to report: the joint conductance, the breakdown by
path, every caveat raised along the way, and -- importantly -- whether an MD
campaign would change the answer.

When the deformation regime is ambiguous, both branches are computed and
returned. Nothing here silently picks one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .contact import (
    ContactResult,
    HardnessResult,
    Regime,
    RegimeResult,
    combined_roughness,
    combined_slope,
    contact_conductance_elastic,
    contact_conductance_plastic,
    contact_conductance_unloading,
    contact_hardness,
    hot_hardness_note,
    macro_flags,
    macro_resistance,
    effective_modulus,
    harmonic_conductivity,
    light_load_check,
    mean_plane_separation,
    select_regime,
)
from .contact.macro import crowned_gap_conductance
from .gap import GapResult, gap_conductance, gas_state, vacuum_gap
from .md.dmm import KapitzaEstimate, dmm_conductance
from .network import MonteCarloResult, NetworkResult, combine, propagate, sample_lognormal
from .network.stack import StackResult, as_network, stack_resistance
from .schema import (
    Confidence,
    FlagSet,
    JointInputs,
    Quantity,
    Severity,
    accommodation_band,
    default_accommodation,
)
from .schema.flags import Code
from .schema.quantities import make_derived


@dataclass
class JointResult:
    inputs: JointInputs
    flags: FlagSet
    hardness: HardnessResult | None       # None for a filled/bonded joint
    regime: RegimeResult | None
    E_eff: Quantity | None
    sigma_eff: Quantity
    m_eff: Quantity | None
    k_s: Quantity | None
    Y: Quantity | None
    kapitza: KapitzaEstimate | None
    gap: GapResult | None
    branches: dict[str, NetworkResult] = field(default_factory=dict)
    uncertainty: dict[str, MonteCarloResult] = field(default_factory=dict)
    # P0-8: h_g and h_joint over the sourced accommodation band, when any
    # alpha was a default: {"alpha": (lo, hi), "h_g": (lo, hi), "h_joint": (lo, hi)}
    accommodation_band: dict | None = None
    stack: StackResult | None = None      # filled/bonded joint (P0-1)
    macro: object | None = None           # contact.macro.MacroResult, when flatness given

    @property
    def primary(self) -> NetworkResult:
        """The branch to quote when a single number is wanted.

        With an ambiguous regime this returns the plastic branch, but the
        REGIME_AMBIGUOUS flag is raised and both branches are in `branches`.
        A caller presenting only this number without the flag is misusing it.
        """
        if self.stack is not None:
            return self.branches[self.stack.kind]
        if self.regime.regime is Regime.ELASTIC:
            return self.branches["elastic"]
        return self.branches["plastic"]

    @property
    def confidence(self) -> Confidence:
        if self.stack is not None:
            return self.primary.h_joint.confidence
        return min(
            self.primary.h_joint.confidence,
            self.hardness.H_c.confidence,
        )


def run_continuum(
    inputs: JointInputs,
    *,
    kapitza: KapitzaEstimate | None = None,
    flags: FlagSet | None = None,
    with_uncertainty: bool = True,
    n_samples: int = 2000,
) -> JointResult:
    """Run the full Phase-0 continuum calculation."""
    flags = flags or FlagSet()
    a, b = inputs.material_a, inputs.material_b
    if inputs.fill is not None:
        return _run_stack(inputs, flags, with_uncertainty, n_samples)

    # -- surface parameters -------------------------------------------------
    m_a = inputs.surface_a.slope_or_estimate(flags)
    m_b = inputs.surface_b.slope_or_estimate(flags)
    sigma_eff = combined_roughness(inputs.surface_a.sigma, inputs.surface_b.sigma)
    m_eff = combined_slope(m_a, m_b)
    k_s = harmonic_conductivity(a.k, b.k)

    # -- hardness and regime ------------------------------------------------
    z_t = inputs.z_trunc
    hardness = contact_hardness(a, b, inputs.pressure, sigma_eff, m_eff, flags, z_trunc=z_t)
    hot_hardness_note(hardness.softer, inputs.temperature.si, flags)
    E_eff = effective_modulus(a, b)
    regime = select_regime(
        E_eff, m_eff, hardness.H_c, flags, override=inputs.regime_override
    )

    # -- contact branches ---------------------------------------------------
    P_max = inputs.max_pressure
    unloading = P_max is not None and P_max.si > inputs.pressure.si
    if P_max is not None and not unloading:
        flags.add(Code.UNLOADING, Severity.INFO,
                  "The earlier maximum pressure is not above the working pressure: first-"
                  "loading model used.")
    if z_t is None and not unloading:
        light_load_check(hardness.P_over_Hc, flags)   # P0-7: CMY under-predicts here
    contacts: dict[str, ContactResult] = {}
    unload = None
    if regime.regime in (Regime.PLASTIC, Regime.BOTH):
        if unloading:
            h_max = contact_hardness(a, b, P_max, sigma_eff, m_eff, FlagSet(), z_trunc=z_t)
            contacts["plastic"], unload = contact_conductance_unloading(
                k_s, sigma_eff, m_eff, inputs.pressure, P_max, h_max.H_c, flags, z_trunc=z_t)
            first = contact_conductance_plastic(k_s, sigma_eff, m_eff, hardness.P_over_Hc,
                                                FlagSet(), z_trunc=z_t)
            ratio = contacts["plastic"].h_c.si / first.h_c.si
            flags.add(Code.UNLOADING, Severity.INFO,
                      f"Unloading from {P_max.si / 1e6:.3g} MPa: contact conductance is "
                      f"{ratio:.2f}x the first-loading value at {inputs.pressure.si / 1e6:.3g} MPa "
                      f"(plastic branch; Mikic / Milanez et al.).",
                      remedy="The unloading magnitudes are checked for internal consistency "
                             "only; no tabulated unloading data were available to test them.",
                      ratio=ratio)
        else:
            contacts["plastic"] = contact_conductance_plastic(
                k_s, sigma_eff, m_eff, hardness.P_over_Hc, flags, z_trunc=z_t
            )
    if z_t is not None and regime.regime in (Regime.ELASTIC, Regime.BOTH):
        flags.add(Code.TG_TRUNCATION_SOURCE, Severity.CAUTION,
                  "z_trunc applies to the plastic branch only; the elastic (Mikic) branch "
                  "is still fully Gaussian.",
                  remedy="Milanez, Yovanovich & Mantelli built the truncated model on the "
                         "plastic CMY model. At light load the elastic branch will read low "
                         "for the same reason plain CMY does.")
    if regime.regime in (Regime.ELASTIC, Regime.BOTH):
        contacts["elastic"] = contact_conductance_elastic(
            k_s, sigma_eff, m_eff, inputs.pressure, E_eff, flags
        )

    # -- gap ----------------------------------------------------------------
    if unload is not None:
        # Separation at P_max, opened by Z^2 sigma on unloading (our reading of the
        # height variable u = lambda_max + X^2; affects only the gas path).
        p_max_ratio = make_derived(P_max.si / h_max.H_c.si, unit="-",
                                   inputs=[P_max, h_max.H_c], name="P_over_Hmax")
        Y0 = mean_plane_separation(sigma_eff, p_max_ratio, flags, z_trunc=z_t)
        Y = make_derived(Y0.si * unload.separation_ratio, unit="um", inputs=[Y0],
                         name="Y", note="separation at P_max opened by the unloading "
                                        "parameter (our reading; gas path only)")
    else:
        Y = mean_plane_separation(sigma_eff, hardness.P_over_Hc, flags, z_trunc=z_t)
    gap: GapResult | None = None
    h_g_override: Quantity | None = None
    if inputs.is_vacuum:
        h_g_override = vacuum_gap()
    else:
        state = gas_state(inputs.gas, inputs.temperature)
        alpha_a = inputs.accommodation_a or default_accommodation(
            inputs.gas.key, a.key, flags
        )
        alpha_b = inputs.accommodation_b or default_accommodation(
            inputs.gas.key, b.key, flags
        )
        gap = gap_conductance(state, Y, alpha_a, alpha_b, flags)
        defaulted = inputs.accommodation_a is None or inputs.accommodation_b is None

    # -- interfacial term ---------------------------------------------------
    if kapitza is None:
        kapitza = dmm_conductance(a, b, inputs.temperature, flags)
    # reference only: where the h_K in use sits against measured values for the pair
    from .md.measured import check_kapitza
    check_kapitza(a.key, b.key, inputs.temperature.si, kapitza.h_K.si, kapitza.source, flags)

    # -- out-of-flatness (macro) constriction ----------------------------------
    macro = None
    if inputs.flatness is not None:
        macro = macro_resistance(inputs.pressure, E_eff, sigma_eff, inputs.flatness,
                                 inputs.joint_radius, k_s, flags)
        if gap is not None:
            gap = _crowned_gap(gap, macro)

    # -- network ------------------------------------------------------------
    branches: dict[str, NetworkResult] = {}
    for name, contact in contacts.items():
        branches[name] = combine(
            contact, kapitza, gap, hardness.H_c, flags,
            gap_conductance_override=h_g_override,
            R_macro=macro.R_L_area if macro is not None else None,
        )
    first = branches.get("plastic") or next(iter(branches.values()))
    R_rest = first.R_constriction.si + first.R_kapitza.si
    macro_flags(macro, R_rest, inputs.clamping, inputs.flatness is not None,
                not inputs.is_vacuum, flags)

    result = JointResult(
        inputs=inputs, flags=flags, hardness=hardness, regime=regime,
        E_eff=E_eff, sigma_eff=sigma_eff, m_eff=m_eff, k_s=k_s, Y=Y,
        kapitza=kapitza, gap=gap, branches=branches, macro=macro,
    )
    if gap is not None and defaulted:
        result.accommodation_band = _accommodation_band(
            result, state, alpha_a, alpha_b, contacts, flags)

    if with_uncertainty:
        result.uncertainty = _propagate(result, n_samples)
    return result


def _stack_kapitza(inputs: JointInputs, flags: FlagSet):
    """DMM h_K at each bond face when both neighbours are in the database."""
    from .schema import load_material
    fill = inputs.fill
    if fill.kind != "bond" or not fill.layers:
        return None, None
    out = []
    for outer, layer in ((inputs.material_a, fill.layers[0]),
                         (inputs.material_b, fill.layers[-1])):
        if layer.material_key is None:
            out.append(None); continue
        try:
            inner = load_material(layer.material_key, FlagSet())
            est = dmm_conductance(outer, inner, inputs.temperature, flags)
            out.append(est.h_K)
        except (KeyError, ValueError, FileNotFoundError):
            out.append(None)
    return out[0], out[1]


def _run_stack(inputs: JointInputs, flags: FlagSet, with_uncertainty: bool,
               n_samples: int) -> JointResult:
    """Filled or bonded joint: a series stack, no contact spots or gas path."""
    sigma_eff = combined_roughness(inputs.surface_a.sigma, inputs.surface_b.sigma)
    ka, kb = _stack_kapitza(inputs, flags)
    stack = stack_resistance(inputs.fill, sigma_eff, flags, kapitza_a=ka, kapitza_b=kb)
    if inputs.flatness is not None:
        _fill_flatness_check(inputs, flags)
    result = JointResult(
        inputs=inputs, flags=flags, hardness=None, regime=None, E_eff=None,
        sigma_eff=sigma_eff, m_eff=None, k_s=None, Y=None, kapitza=None, gap=None,
        branches={stack.kind: as_network(stack, inputs.pressure)}, stack=stack,
    )
    if with_uncertainty:
        result.uncertainty = {"h_joint": propagate(
            lambda rng: _perturbed_stack(inputs, ka, kb, rng), stack.h_joint.si, n=n_samples)}
    return result


def _fill_flatness_check(inputs: JointInputs, flags: FlagSet) -> None:
    """A fill absorbs out-of-flatness, but its thickness then varies by delta."""
    delta = inputs.flatness.si
    blt = sum(layer.thickness.si for layer in inputs.fill.layers)
    if blt < delta:
        flags.add(Code.MACRO_CONSTRICTION, Severity.WARNING,
                  f"{inputs.fill.kind.upper()} thickness {blt * 1e6:.3g} um is less than the "
                  f"combined out-of-flatness {delta * 1e6:.3g} um: the faces can touch at the "
                  f"crown and the layer thickens toward the edges.",
                  remedy="R from a single thickness is then wrong in both directions (contact "
                         "at the crown, thicker layer elsewhere). Use the area-averaged "
                         "thickness, or measure R on the assembly.")
    else:
        flags.add(Code.MACRO_CONSTRICTION, Severity.INFO,
                  f"The {inputs.fill.kind} fills the out-of-flatness ({delta * 1e6:.3g} um): "
                  f"its thickness varies by up to that much across the face.",
                  remedy="Give the AREA-AVERAGED thickness (minimum + about delta/2 for a "
                         "crown), not the minimum or the spacer size.")


def _perturbed_stack(inputs: JointInputs, ka, kb, rng: random.Random) -> float:
    import copy
    from .network.stack import FillSpec, Layer
    f = inputs.fill
    def draw(q):
        return None if q is None else q.with_(si=sample_lognormal(q, rng))
    layers = [Layer(l.label, draw(l.thickness), draw(l.k), l.void_fraction, l.material_key)
              for l in f.layers]
    g = FillSpec(f.kind, layers, f.tim_class, draw(f.R_c_total), draw(f.R_measured),
                 draw(f.h_K_a), draw(f.h_K_b))
    return stack_resistance(g, None, FlagSet(), kapitza_a=draw(ka),
                            kapitza_b=draw(kb)).h_joint.si


def _crowned_gap(gap: GapResult, macro) -> GapResult:
    """Gap conductance averaged over a crowned face (contact.macro)."""
    import dataclasses
    h = crowned_gap_conductance(gap.state.k_g.si, gap.Y.si, gap.M.si, macro.rho, macro.a_L,
                                macro.a_L / macro.B)
    q = make_derived(h, unit="W/m2K", inputs=[gap.h_g, macro.R_L_area], name="h_g",
                     confidence=Confidence.LOW,
                     note="lumped gap model averaged over the crown profile (extension)")
    return dataclasses.replace(gap, h_g=q)


def _accommodation_band(result, state, alpha_a, alpha_b, contacts, flags) -> dict | None:
    """h_g and h_joint with every DEFAULTED alpha at the band's low and high end.

    A user-supplied alpha is kept as given. The band is widened to include the
    default itself when the default lies outside it.
    """
    inputs = result.inputs
    band = accommodation_band(inputs.gas.key)
    if band is None:
        return None
    defaults = [x.si for x, given in ((alpha_a, inputs.accommodation_a),
                                      (alpha_b, inputs.accommodation_b)) if given is None]
    lo = min([band["low"], *defaults]); hi = max([band["high"], *defaults])
    branch = result.primary.branch
    out = {"alpha": (lo, hi), "h_g": [], "h_joint": []}
    for value in (lo, hi):
        quiet = FlagSet()
        aa = alpha_a if inputs.accommodation_a is not None else alpha_a.with_(si=value)
        ab = alpha_b if inputs.accommodation_b is not None else alpha_b.with_(si=value)
        g = gap_conductance(state, result.Y, aa, ab, quiet)
        mac = result.macro
        if mac is not None:
            g = _crowned_gap(g, mac)
        net = combine(contacts[branch], result.kapitza, g, result.hardness.H_c, quiet,
                      R_macro=mac.R_L_area if mac is not None else None)
        out["h_g"].append(g.h_g.si); out["h_joint"].append(net.h_joint.si)
    out["h_g"] = tuple(out["h_g"]); out["h_joint"] = tuple(out["h_joint"])
    hj = result.primary.h_joint.si
    span = (out["h_joint"][1] - out["h_joint"][0]) / hj
    flags.add(
        Code.GAP_ACCOMMODATION_BAND,
        Severity.CAUTION if span > 0.05 else Severity.INFO,
        f"Defaulted accommodation for {inputs.gas.key}: over alpha {lo:.2f}-{hi:.2f} "
        f"h_g = {out['h_g'][0]:.4g}-{out['h_g'][1]:.4g} W/m^2K and h_joint = "
        f"{out['h_joint'][0]:.4g}-{out['h_joint'][1]:.4g} W/m^2K "
        f"({span * 100:.0f} % of the quoted {hj:.4g}).",
        remedy=("The spread is surface state: clean machined or polished metal sits at "
                "the low end, bead-blasted or air-exposed surfaces at the high end "
                f"({band['source']}). A measured alpha, fitted with this same gap "
                "model, removes it."
                + (f" Air uses the {band['proxy']} band." if band.get("proxy") else "")),
        alpha_low=lo, alpha_high=hi, h_joint_low=out["h_joint"][0],
        h_joint_high=out["h_joint"][1],
    )
    return out


def _propagate(result: JointResult, n_samples: int) -> dict[str, MonteCarloResult]:
    """Monte Carlo the joint conductance over the uncertain inputs.

    Only quantities that carry an explicit uncertainty are perturbed. Where
    the user supplied none, the quantity is held fixed and the resulting band
    therefore UNDERSTATES the true uncertainty -- the report says so rather
    than inventing a spread.
    """
    inputs = result.inputs
    branch = result.primary.branch

    def evaluate(rng: random.Random) -> float:
        perturbed = _perturb(inputs, rng, result.kapitza)
        sub, kap = perturbed
        quiet = FlagSet()
        r = run_continuum(
            sub, kapitza=kap, flags=quiet, with_uncertainty=False
        )
        target = r.branches.get(branch) or next(iter(r.branches.values()))
        return target.h_joint.si

    out = {
        "h_joint": propagate(
            evaluate, result.primary.h_joint.si, n=n_samples
        )
    }
    return out


def _perturb(inputs: JointInputs, rng: random.Random, kapitza: KapitzaEstimate):
    """Draw one perturbed copy of the inputs and the Kapitza estimate."""
    import copy

    sub = copy.copy(inputs)
    sub.surface_a = copy.copy(inputs.surface_a)
    sub.surface_b = copy.copy(inputs.surface_b)
    sub.material_a = copy.copy(inputs.material_a)
    sub.material_b = copy.copy(inputs.material_b)

    sub.surface_a.sigma = inputs.surface_a.sigma.with_(
        si=sample_lognormal(inputs.surface_a.sigma, rng)
    )
    sub.surface_b.sigma = inputs.surface_b.sigma.with_(
        si=sample_lognormal(inputs.surface_b.sigma, rng)
    )
    if inputs.surface_a.m is not None:
        sub.surface_a.m = inputs.surface_a.m.with_(
            si=sample_lognormal(inputs.surface_a.m, rng)
        )
    if inputs.surface_b.m is not None:
        sub.surface_b.m = inputs.surface_b.m.with_(
            si=sample_lognormal(inputs.surface_b.m, rng)
        )
    sub.pressure = inputs.pressure.with_(si=sample_lognormal(inputs.pressure, rng))
    if inputs.max_pressure is not None and inputs.max_pressure.uncertainty:
        sub.max_pressure = inputs.max_pressure.with_(
            si=sample_lognormal(inputs.max_pressure, rng))
    if inputs.flatness is not None and inputs.flatness.uncertainty:
        sub.flatness = inputs.flatness.with_(si=sample_lognormal(inputs.flatness, rng))
    if inputs.z_trunc is not None and inputs.z_trunc.uncertainty:
        # Additive spread on z_t, kept above 2.5 sigma (below that no CMY-family model holds)
        z = rng.gauss(inputs.z_trunc.si, inputs.z_trunc.uncertainty)
        sub.z_trunc = inputs.z_trunc.with_(si=max(z, 2.5))
    sub.material_a.Hv = inputs.material_a.Hv.with_(
        si=sample_lognormal(inputs.material_a.Hv, rng)
    )
    sub.material_b.Hv = inputs.material_b.Hv.with_(
        si=sample_lognormal(inputs.material_b.Hv, rng)
    )

    kap = KapitzaEstimate(
        h_K=kapitza.h_K.with_(si=sample_lognormal(kapitza.h_K, rng)),
        transmission=kapitza.transmission,
        source=kapitza.source,
    )
    return sub, kap
