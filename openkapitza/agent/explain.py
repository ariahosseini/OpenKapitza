"""Plain-language summary of a run: the number, what limits it, how far to
trust it, and what to do next. The full technical report stays in
openkapitza.report.render; this is the conversational layer's answer."""

from __future__ import annotations

from ..schema.flags import Severity
from .planner import Plan

_SEV = {Severity.CRITICAL: "CRITICAL", Severity.WARNING: "WARNING",
        Severity.CAUTION: "caution", Severity.INFO: "note"}


def _fmt_h(h: float) -> str:
    if h >= 1e9:
        return f"{h / 1e9:.3g} GW/m²K"
    if h >= 1e6:
        return f"{h / 1e6:.3g} MW/m²K"
    return f"{h / 1e3:.3g} kW/m²K" if h >= 1e3 else f"{h:.3g} W/m²K"


def _fmt_r(r: float) -> str:
    return f"{r * 1e6:.3g} mm²K/W"


def confidence_word(result) -> str:
    worst = min((f.severity for f in result.flags.flags),
                key=lambda s: [Severity.CRITICAL, Severity.WARNING, Severity.CAUTION,
                               Severity.INFO].index(s), default=Severity.INFO)
    return {Severity.CRITICAL: "low — a critical issue is flagged",
            Severity.WARNING: "limited — see the warnings",
            Severity.CAUTION: "moderate", Severity.INFO: "good"}[worst]


def explain(result, plan: Plan) -> str:
    net = result.primary
    h, R = net.h_joint.si, net.R_total.si
    lines = [f"Joint conductance: {_fmt_h(h)}  (resistance {_fmt_r(R)})"]
    mc = result.uncertainty.get("h_joint") if result.uncertainty else None
    if mc is not None:
        lines.append(f"  90 % band from the input uncertainties: {_fmt_h(mc.p05)} – "
                     f"{_fmt_h(mc.p95)} (inputs without a stated uncertainty held fixed)")
    if result.accommodation_band:
        lo, hi = result.accommodation_band["h_joint"]
        lines.append(f"  over the accommodation-coefficient band: {_fmt_h(lo)} – {_fmt_h(hi)}")
    lines.append(f"  Confidence: {confidence_word(result)}")

    # what carries the resistance
    lines.append("")
    gas = next((c for c in net.contributions if c.label.startswith("gas gap")), None)
    if gas is not None:
        lines.append(f"Heat splits: {1 - gas.fraction:.0%} through the solid contact, "
                     f"{gas.fraction:.0%} through the gas.")
        lines.append("Resistance of the solid path, by part:")
    else:
        lines.append("Where the resistance is:")
    parts = [c for c in net.contributions
             if c.label not in ("solid path (series total)",) and not c.label.startswith("gas gap")]
    for c in sorted(parts, key=lambda c: -c.fraction)[:4]:
        if c.fraction >= 0.005:
            lines.append(f"  - {c.label}: {c.fraction:.0%}")
    if result.stack is None and result.regime is not None and len(result.branches) > 1:
        other = [n for n in result.branches if n != net.branch][0]
        lines.append(f"  (regime ambiguous: the {other} branch gives "
                     f"{_fmt_h(result.branches[other].h_joint.si)})")

    # atomistic route
    lines.append("")
    lines.append(f"Atomistic (MD) calculation: {plan.md} — {plan.md_reason}.")
    if plan.md != "not needed":
        if plan.route_status in ("ready", "caveat"):
            lines.append(f"  Route: {plan.route_potential}; {plan.route_note}")
            lines.append(f"  Prepared cluster jobs that cover it: {', '.join(plan.cluster_jobs)}"
                         " (say 'prepare jobs <folder>').")
            if result.stack is None:
                lines.append(f"  Run it at the spot pressure {net.spot_pressure.si / 1e9:.2g} GPa, "
                             "not the apparent pressure.")
        else:
            lines.append(f"  No route yet: {plan.route_note}")

    if plan.caveats:
        lines.append("")
        lines.append("What to watch:")
        for f in plan.caveats:
            lines.append(f"  [{_SEV[f.severity]}] {f.message}")
        if plan.caveats_hidden:
            lines.append(f"  (+{plan.caveats_hidden} more; 'report' shows everything)")
    if plan.next_measurements:
        lines.append("")
        lines.append("Best next measurements:")
        for m in plan.next_measurements[:4]:
            lines.append(f"  - {m}")
    return "\n".join(lines)
