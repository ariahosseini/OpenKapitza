"""Decide what a joint needs after the continuum run.

The continuum result already says how much the atomic-interface term could
move the answer (the h_K sweep in network.combine). The planner turns that
into a recommendation -- no MD, optional, or recommended -- and, when MD is
worth it, says whether this package has a validated route for the pair and
which prepared cluster jobs cover it. It also ranks the next measurements by
the flags the run raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schema.flags import Severity

# Below this joint-level change from sweeping h_K, MD cannot change the answer.
MD_NOT_NEEDED = 0.02
# Above this, MD is the critical path.
MD_RECOMMENDED = 0.10

# What the MD side can do per pair today (tracker: P1-5, P1-6, P1-14, Phase 2).
# key: frozenset of material keys -> (status, potential, note, prepare_cluster jobs)
MD_ROUTES: dict[frozenset, tuple[str, str, str, tuple[str, ...]]] = {
    frozenset({"Si", "Ge"}): (
        "caveat", "Tier 1: SiCGe.tersoff",
        "fails the elastic gate and inverts the Si/Ge biaxial-stiffness ordering (P1-5); "
        "results are reported with that stated. A gated Tier-2 MLIP fixes it (Phase 4).",
        ("campaign", "sige_spot_stress", "sige_spot_strain", "sige_emd")),
    frozenset({"Si", "Al"}): (
        "ready", "Tier 1: AlSiMgCuFe.meam (MEAM, Jelinek)",
        "passes the elastic gate on Si and Al (P1-6). Electron-phonon resistance on the Al "
        "side needs the two-temperature pair (P1-14).",
        ("sial_pp", "sial_ttm")),
    frozenset({"Si", "Cu"}): (
        "ready", "Tier 1: AlSiMgCuFe.meam (MEAM, Jelinek)",
        "passes the elastic gate on Si and Cu (P1-6); electron-phonon estimate is a band "
        "(Cu coupling spans ~5x).",
        ("sicu_pp",)),
    frozenset({"Si", "SiO2"}): (
        "ready", "Tier 1: SiO.tersoff + melt-quench a-SiO2",
        "amorphous route (Phase 2): glass by melt-and-quench, strong and weak interface "
        "bonding as a bracket; the quench rate is flagged.",
        ("sio2_strong", "sio2_weak")),
}
NO_ROUTE_NOTE = ("no shipped cross potential validated for this pair; a Tier-2 MLIP "
                 "trained on the interface is needed (Phase 4). Alloys (SS 304, Al 6061) "
                 "have no atomistic route at all.")


@dataclass
class Plan:
    md: str                          # "not needed" | "optional" | "recommended"
    md_reason: str
    route_status: str | None = None  # "ready" | "caveat" | "none"
    route_potential: str | None = None
    route_note: str | None = None
    cluster_jobs: tuple[str, ...] = ()
    next_measurements: list[str] = field(default_factory=list)
    caveats: list = field(default_factory=list)       # Flag objects, ranked
    caveats_hidden: int = 0
    md_caveats: list = field(default_factory=list)    # the ingested MD run's own, ranked
    md_caveats_hidden: int = 0


_ORDER = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.CAUTION: 2, Severity.INFO: 3}

# Flag code -> the measurement that would remove it (ordered by usual payoff).
_MEASURE = {
    "CLAMP_BOLTED": "Measure h on the assembled, bolted joint, or at least the face flatness "
                    "and the pressure map (pressure film).",
    "FLATNESS_ASSUMED": "Measure each face's out-of-flatness (dial gauge / optical flat) and "
                        "give the joint area.",
    "MACRO_CONSTRICTION": "Lap the faces flatter or raise the load; or fill the macro gap "
                          "with a TIM.",
    "CMY_SLOPE_ESTIMATED": "Measure the asperity slope (profilometer Δa) on both faces.",
    "HARDNESS_FALLBACK": "Run load-dependent Vickers indentation (10-200 gf) to fit c1, c2 "
                         "for the softer face, then 'set c1_a_GPa=...' and 'set c2_a=...' "
                         "(or _b).",
    "CMY_LIGHT_LOAD": "At this light load, a measured h at the working pressure (or a "
                      "z_trunc fitted to light-load data) is worth more than refined inputs.",
    "STACK_INCOMPLETE": "Measure the missing term: the TIM boundary resistance (ASTM D5470 "
                        "intercept), a bonded face's interface conductance, or the total R.",
    "GAP_ACCOMMODATION_BAND": "If the gas path matters, the accommodation coefficient of "
                              "these surfaces sets it; it is surface-state dependent.",
    "STACK_VOIDS": "Image the bond (X-ray / acoustic) to get the void fraction.",
}


def plan(result, md_used=None) -> Plan:
    """Build the plan from a JointResult (md_used: the ingest.MDResult applied)."""
    inputs = result.inputs
    net = result.primary
    pair = frozenset({inputs.material_a.key, inputs.material_b.key})

    if result.stack is not None:
        share = result.stack.R_kapitza / result.stack.R_total.si if result.stack.R_total.si else 0
        missing = next((f for f in result.flags.flags if f.code == "STACK_INCOMPLETE"
                        and "kapitza_share_pessimistic" in f.context), None)
        if missing is not None:
            # Faces with no estimate at all: judge by what they could carry at the
            # low end of measured interfaces (the stack flag's pessimistic share).
            share = max(share, missing.context["kapitza_share_pessimistic"])
        if share == 0:
            md, why = "not needed", "the stack has no solid-solid atomic interface in the model"
        elif missing is not None and share >= MD_NOT_NEEDED:
            md = "recommended" if share >= MD_RECOMMENDED else "optional"
            why = (f"some faces have no interface estimate; at 100 MW/m²K they would carry "
                   f"{share:.0%} of the stack resistance")
        elif share < MD_NOT_NEEDED:
            md, why = "not needed", f"atomic interfaces are {share:.1%} of the stack resistance"
        elif share < MD_RECOMMENDED:
            md, why = "optional", f"atomic interfaces are {share:.1%} of the stack resistance"
        else:
            md, why = "recommended", (f"atomic interfaces are {share:.0%} of the stack "
                                      "resistance, on a DMM placeholder")
        faces = _stack_face_pairs(result)
        pair = next((p for p in faces if p in MD_ROUTES), next(iter(faces), pair))
    else:
        span = net.sensitivity.get("relative_span", 0.0)
        gain = net.sensitivity.get("max_possible_gain", 0.0)
        if span < MD_NOT_NEEDED:
            md, why = "not needed", (f"sweeping the interface term moves the answer by "
                                     f"{span:.1%}; contact spots, flatness or gas dominate")
        elif span < MD_RECOMMENDED:
            md, why = "optional", (f"the interface term could move the answer by {span:.0%} "
                                   f"(perfect interface: +{gain:.0%})")
        else:
            md, why = "recommended", (f"the interface term could move the answer by "
                                      f"{span:.0%} and rests on a DMM placeholder")

    if md_used is not None:
        conv = {True: "converged", False: "not converged", None: "single run"}[md_used.converged]
        md, why = "in use", (f"h_K = {md_used.h_K / 1e6:.4g} ± {md_used.uncertainty / 1e6:.3g} "
                             f"MW/m²K from {md_used.potential} ({conv}); before it, {why}")
    p = Plan(md=md, md_reason=why)
    if md not in ("not needed", "in use"):
        route = MD_ROUTES.get(pair)
        if route:
            p.route_status, p.route_potential, p.route_note, p.cluster_jobs = route
        else:
            p.route_status, p.route_note = "none", NO_ROUTE_NOTE

    flags = sorted(result.flags.flags, key=lambda f: _ORDER.get(f.severity, 9))
    md_own = [f for f in flags if f.message.startswith("[MD]")]
    flags = [f for f in flags if not f.message.startswith("[MD]")]
    # The MD run's caveats get their own short list: its summary codes first.
    md_own.sort(key=lambda f: (not str(f.code).startswith("MD_"), _ORDER.get(f.severity, 9)))
    md_keep = [f for f in md_own if f.severity != Severity.INFO]
    p.md_caveats, p.md_caveats_hidden = md_keep[:3], max(0, len(md_keep) - 3)
    seen, ranked = set(), []
    for f in flags:
        if f.code in seen:
            continue
        seen.add(f.code)
        ranked.append(f)
    shown = [f for f in ranked if f.severity != Severity.INFO][:6]
    if len(shown) < 6:
        shown += [f for f in ranked if f.severity == Severity.INFO][: 6 - len(shown)]
    p.caveats = shown
    p.caveats_hidden = len(ranked) - len(shown)
    acting = {f.code for f in ranked if f.severity != Severity.INFO}
    p.next_measurements = [_MEASURE[c] for c in _MEASURE if c in acting]
    return p


def _stack_face_pairs(result) -> list[frozenset]:
    """Material pairs at the solid faces of a bonded stack (where keys are known)."""
    keys = [result.inputs.material_a.key]
    keys += [l.material_key for l in result.inputs.fill.layers]
    keys.append(result.inputs.material_b.key)
    return [frozenset({a, b}) for a, b in zip(keys, keys[1:]) if a and b and a != b]
