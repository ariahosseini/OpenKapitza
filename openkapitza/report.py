"""Rendering a JointResult as text.

This module is deliberately dumb. It formats numbers that already exist in
the result object and prints the flags the deterministic layer raised. It
computes nothing and it suppresses nothing.

That matters for the agent layer in Phase 3: the LLM's report-writing node
must be constrained to numbers present in the structured result, and the
cheapest way to enforce that is to have a reference rendering that is known
to contain exactly those numbers and no others.
"""

from __future__ import annotations

from .pipeline import JointResult
from .schema.flags import Severity


def render(result: JointResult, *, show_samples: bool = False) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add("OpenKapitza -- joint thermal conductance (Phase 0, continuum)")
    add("=" * 72)
    add("")
    add(result.inputs.summary())
    add("")

    add("-- derived surface and material parameters " + "-" * 29)
    add(f"  sigma_eff (combined RMS)   {result.sigma_eff.to('um'):10.4g} um")
    add(f"  m_eff (combined slope)     {result.m_eff.si:10.4g}")
    add(f"  k_s (harmonic mean)        {result.k_s.si:10.4g} W/m.K")
    add(f"  E' (effective modulus)     {result.E_eff.to('GPa'):10.4g} GPa")
    add(f"  H_c (contact hardness)     {result.hardness.H_c.to('GPa'):10.4g} GPa"
        f"   [{result.hardness.method}]")
    add(f"  P/H_c                      {result.hardness.P_over_Hc.si:10.4g}")
    add(f"  psi (plasticity index)     {result.regime.psi.si:10.4g}"
        f"   -> {result.regime.regime.value}")
    add(f"  Y (mean plane separation)  {result.Y.to('um'):10.4g} um")
    add("")

    if result.gap is not None:
        add("-- gas gap " + "-" * 61)
        add(f"  k_g                        {result.gap.state.k_g.si:10.4g} W/m.K")
        add(f"  mean free path             {result.gap.state.mfp.to('nm'):10.4g} nm")
        add(f"  M (rarefaction)            {result.gap.M.to('um'):10.4g} um")
        add(f"  Knudsen (Lambda/Y)         {result.gap.knudsen.si:10.4g}")
        add(f"  h_g                        {result.gap.h_g.si:10.4g} W/m^2.K")
        add("")
    else:
        add("-- gas gap: vacuum, no gas path (radiation not modelled) " + "-" * 15)
        add("")

    add("-- interfacial (Kapitza) term " + "-" * 42)
    add(f"  h_K  [{result.kapitza.source}]        "
        f"{result.kapitza.h_K.to('MW/m2K'):10.4g} MW/m^2.K  (per REAL contact area)")
    add("")

    for name, net in result.branches.items():
        add(f"-- network, {name} branch " + "-" * (48 - len(name)))
        add(f"  xi = A_real/A_apparent     {net.xi.si:10.4g}")
        add(f"  R_constriction             {net.R_constriction.si:10.4g} m^2.K/W")
        add(f"  R_Kapitza (area-scaled)    {net.R_kapitza.si:10.4g} m^2.K/W")
        add(f"    (unscaled, per real area {net.R_kapitza_raw.si:10.4g} m^2.K/W -- "
            f"NOT what enters the network)")
        add(f"  h_solid (series)           {net.h_solid.si:10.4g} W/m^2.K")
        add(f"  h_gap   (parallel)         {net.h_gap.si:10.4g} W/m^2.K")
        add("")
        add(f"  h_JOINT                    {net.h_joint.si:10.4g} W/m^2.K"
            f"   [{net.h_joint.confidence.label}]")
        add(f"  R_TOTAL                    {net.R_total.si:10.4g} m^2.K/W")
        add("")
        add("  resistance breakdown:")
        for c in net.contributions:
            add(f"    {c.label:36s} {c.fraction * 100:6.2f} %   {c.note}")
        add("")
        s = net.sensitivity
        add("  would MD change this?")
        add(f"    h_K swept by a factor of {s['factor_swept']:.2f} moves h_joint by "
            f"{s['relative_span'] * 100:.1f} %")
        add(f"    a perfect interface (h_K -> inf) would give "
            f"{s['h_joint_perfect_interface']:.4g} W/m^2.K "
            f"(+{s['max_possible_gain'] * 100:.1f} %)")
        add(f"    NEMD should be run at a spot pressure of ~"
            f"{net.spot_pressure.to('GPa'):.3g} GPa, NOT the apparent "
            f"{result.inputs.pressure.to('MPa'):.3g} MPa")
        add("")

    if "h_joint" in result.uncertainty:
        mc = result.uncertainty["h_joint"]
        add("-- uncertainty (Monte Carlo over quantities with stated errors) " + "-" * 8)
        add(f"  {mc.summary(unit=' W/m^2.K')}")
        add(f"  90% band spans a factor of "
            f"{mc.p95 / mc.p05:.2f} ({len(mc.samples)} samples"
            + (f", {mc.n_failed} failed" if mc.n_failed else "") + ")")
        add("  NOTE: inputs with no stated uncertainty were held fixed, so this")
        add("        band is a LOWER BOUND on the true uncertainty.")
        add("")

    add("-- caveats " + "-" * 61)
    worst = result.flags.worst
    add(f"  worst severity: {worst.label.upper()}   ({len(result.flags)} flags)")
    add("")
    add(result.flags.render())
    add("")
    if worst >= Severity.CRITICAL:
        add("!! A CRITICAL flag is set. Do not quote the total without resolving it.")
        add("")
    return "\n".join(lines)
