"""Filled and bonded joints: a 1D series stack (open issue P0-1).

Most package and board joints are not bare metal-on-metal with gas in the
gaps. They are FILLED (grease, gel, pad, phase-change material) or BONDED
(solder, sintered Ag, die attach). For those, the contact-spot plus gas-gap
network does not apply; the joint is a series stack:

  filled:  R = R_c1 + BLT / k_TIM + R_c2                (Prasher; Mahajan 2004)
  bonded:  R = R_K(a|1) + sum_i t_i / (k_i (1 - f_i)) + R_K(n|b)

R_c are the TIM-substrate boundary resistances (trapped air in the valleys
the TIM does not wet, capillarity, filler packing); R_K are atomic-interface
(Kapitza) resistances at full contact, where the MD part of this project
enters directly, with no contact-area scaling. f_i is a void area fraction.

Sources and limits (knowledge base, P0-1 survey):
  * Class ranges for R_TIM (Blazej, Electronics Cooling, Nov 2003; K cm^2/W):
    grease 0.2-1, pad 1-3, PCM 0.3-0.7, gel 0.4-0.8, adhesive < 1 (best 0.15),
    solder < 0.05. Used only to flag a result outside its class.
  * BLT depends on the TIM's yield stress and the pressure (Prasher, IEEE
    TCPT 28, 230, 2005); the correlation's constants were not read, so BLT is
    an INPUT (measured, or the datasheet's at the working pressure).
  * Voids: no closed-form model; small distributed voids barely matter, one
    large void dominates, and from 7.5 % void area every simulated IGBT case
    exceeded a 20 % R_th increase (Alavi et al., Electronics 13, 2188, 2024).
    The area-reduction form used here is an estimate with that caveat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Confidence, Quantity, make_derived

MM2K_W = 1e-6                    # m^2 K/W per mm^2 K/W

# Blazej 2003, total R_TIM per class, converted from K cm^2/W to m^2 K/W.
# Grease lower bound lowered 20 -> 10 mm^2K/W: DeVoto et al. (NREL/CP-5400-68014,
# ITherm 2017, Tables 2-3, ASTM D5470 at 139 kPa) measured R = BLT/k + R_c with
# k 3.6-3.8 W/mK and R_c 4.2-12.2 mm^2K/W for three commercial greases, i.e.
# 12-18 mm^2K/W at 29-50 um BLT -- good current greases sit below Blazej's 20.
TIM_CLASS_RANGE = {
    "grease": (0.1e-4, 1.0e-4),
    "pad": (1.0e-4, 3.0e-4),
    "pcm": (0.3e-4, 0.7e-4),
    "gel": (0.4e-4, 0.8e-4),
    "adhesive": (0.15e-4, 1.0e-4),
    "solder": (0.0, 0.05e-4),
}
TIM_CLASS_SOURCE = "Blazej, 'Thermal Interface Materials', Electronics Cooling, Nov 2003"

# Measured solid-solid interface conductances in the survey (knowledge base
# sections 7-8: Si/Al 0.2-0.47, GaN/SiC 0.17-0.23, Si/SiO2 > 0.6 GW/m^2K) sit at
# ~0.1-1 GW/m^2K. The LOW end decides whether an omitted face could matter.
H_K_PESSIMISTIC = 1e8            # W/m^2K
KAPITZA_MATTERS = 0.05           # share of R above which an omitted face is a WARNING
VOID_WARNING = 0.05              # above this, device-level R_th can rise > 20 %
BLT_OVER_ROUGHNESS_MIN = 2.0     # a TIM thinner than ~2 sigma_eff does not fill


@dataclass
class Layer:
    """One bulk layer of the stack."""
    label: str
    thickness: Quantity          # m
    k: Quantity                  # W/(m K), in-plane-averaged through-thickness value
    void_fraction: float = 0.0   # area fraction of voids, 0..1
    material_key: str | None = None   # for a DMM Kapitza estimate at its faces


@dataclass
class FillSpec:
    """A filled (TIM) or bonded joint between material_a and material_b."""
    kind: str                                # "tim" | "bond"
    layers: list[Layer]
    tim_class: str | None = None             # grease | pad | pcm | gel | adhesive | solder
    R_c_total: Quantity | None = None        # TIM: R_c1 + R_c2, m^2K/W (measured, D5470 intercept)
    R_measured: Quantity | None = None       # TIM: datasheet/measured total at the working pressure
    h_K_a: Quantity | None = None            # bond: per-area conductance at the a|layer face
    h_K_b: Quantity | None = None            # bond: at the layer|b face


@dataclass
class StackTerm:
    label: str
    resistance: float            # m^2 K/W
    fraction: float = 0.0
    note: str = ""


@dataclass
class StackResult:
    kind: str
    h_joint: Quantity
    R_total: Quantity
    terms: list[StackTerm] = field(default_factory=list)
    R_kapitza: float = 0.0       # sum of atomic-interface terms, m^2K/W
    complete: bool = True        # False if a known term is missing (bound only)


def as_network(stack: StackResult, pressure: Quantity):
    """The stack as a NetworkResult, so consumers of `primary` keep working.

    Full contact: xi = 1, no constriction, no gas path; the Kapitza term is
    the stack's atomic-interface terms, unscaled.
    """
    from .combine import Contribution, NetworkResult
    zero = make_derived(0.0, unit="m2K/W", name="R_constriction",
                        note="no contact spots in a filled or bonded joint")
    rk = make_derived(stack.R_kapitza, unit="m2K/W", name="R_kapitza",
                      note="atomic interfaces at full contact")
    return NetworkResult(
        h_joint=stack.h_joint, R_total=stack.R_total, h_solid=stack.h_joint,
        R_constriction=zero, R_kapitza=rk, R_kapitza_raw=rk,
        xi=make_derived(1.0, unit="-", name="xi", note="full contact"),
        h_gap=make_derived(0.0, unit="W/m2K", name="h_gap", note="no gas path"),
        spot_pressure=pressure, branch=stack.kind,
        contributions=[Contribution(t.label, t.resistance, t.fraction, t.note)
                       for t in stack.terms],
        sensitivity={},
    )


def _layer_resistance(layer: Layer, flags: FlagSet) -> tuple[float, str]:
    if not 0.0 <= layer.void_fraction < 1.0:
        raise ValueError(f"{layer.label}: void fraction must be in [0, 1)")
    r = layer.thickness.si / (layer.k.si * (1.0 - layer.void_fraction))
    note = f"t/k = {layer.thickness.si * 1e6:.3g} um / {layer.k.si:.3g} W/mK"
    if layer.void_fraction > 0.0:
        note += f", voids {layer.void_fraction * 100:.1f} % (area reduction)"
        flags.add(
            Code.STACK_VOIDS,
            Severity.WARNING if layer.void_fraction > VOID_WARNING else Severity.CAUTION,
            f"{layer.label}: {layer.void_fraction * 100:.1f} % voids treated as lost "
            f"conducting area, t/(k(1-f)).",
            remedy=("This 1D form OVER-states the penalty of many small, spread "
                    "voids (heat spreads around them; Alavi et al. 2024 find them "
                    "negligible) and UNDER-states one large void under a hot spot. "
                    "From ~5-7.5 % void area, device-level R_th rises can exceed 20 %. "
                    "Use an X-ray void map and a 3D model for a real joint."),
            void_fraction=layer.void_fraction,
        )
    return r, note


def stack_resistance(fill: FillSpec, sigma_eff: Quantity | None, flags: FlagSet, *,
                     kapitza_a: Quantity | None = None,
                     kapitza_b: Quantity | None = None) -> StackResult:
    """Series resistance of a filled or bonded joint, per apparent area."""
    if fill.kind not in ("tim", "bond"):
        raise ValueError("fill.kind must be 'tim' or 'bond'")
    if not fill.layers and fill.R_measured is None:
        raise ValueError("a stack needs at least one layer or a measured total")
    terms: list[StackTerm] = []
    complete = True
    r_kap = 0.0
    inputs: list[Quantity] = []

    if fill.kind == "tim":
        if fill.R_measured is not None:
            terms.append(StackTerm("TIM (measured total)", fill.R_measured.si,
                                   note="measured or datasheet R_TIM at the working pressure"))
            inputs.append(fill.R_measured)
        else:
            for layer in fill.layers:
                r, note = _layer_resistance(layer, flags)
                terms.append(StackTerm(f"{layer.label} (bulk)", r, note=note))
                inputs += [layer.thickness, layer.k]
            if fill.R_c_total is not None:
                terms.append(StackTerm("TIM boundaries R_c1 + R_c2", fill.R_c_total.si,
                                       note="measured (e.g. ASTM D5470 intercept)"))
                inputs.append(fill.R_c_total)
            else:
                complete = False
                flags.add(
                    Code.STACK_INCOMPLETE, Severity.WARNING,
                    "TIM boundary resistances R_c1 + R_c2 not supplied: R_TIM is the "
                    "BULK term only, so h_joint is an UPPER bound.",
                    remedy=("Supply R_c (ASTM D5470: intercept of R vs BLT) or a measured "
                            "R_TIM at the working pressure. For scale only: a worked example "
                            "in Blazej (2003) puts the two boundaries at ~17 mm^2K/W, "
                            "comparable to the bulk of a 0.1 mm grease."),
                )
        if sigma_eff is not None and fill.layers and fill.R_measured is None:
            blt = sum(layer.thickness.si for layer in fill.layers)
            if blt < BLT_OVER_ROUGHNESS_MIN * sigma_eff.si:
                flags.add(
                    Code.STACK_THIN_TIM, Severity.WARNING,
                    f"BLT {blt * 1e6:.3g} um is under {BLT_OVER_ROUGHNESS_MIN:g}x the "
                    f"combined roughness ({sigma_eff.si * 1e6:.3g} um): asperities "
                    f"bridge the TIM.",
                    remedy=("Solid-solid contact spots then act in parallel with the "
                            "TIM; that path is not modelled. Measure R_TIM at this "
                            "pressure on these surfaces."),
                )
    else:  # bond
        for layer in fill.layers:
            r, note = _layer_resistance(layer, flags)
            terms.append(StackTerm(f"{layer.label} (bulk)", r, note=note))
            inputs += [layer.thickness, layer.k]
        missing = []
        for side, q in (("a", fill.h_K_a or kapitza_a), ("b", fill.h_K_b or kapitza_b)):
            if q is None:
                missing.append(side)
                continue
            terms.append(StackTerm(f"interface {side} (Kapitza)", 1.0 / q.si,
                                   note="atomic interface at full contact, unscaled"))
            r_kap += 1.0 / q.si
            inputs.append(q)

    R = sum(t.resistance for t in terms)
    if R <= 0:
        raise ValueError("stack resistance must be positive")
    if fill.kind == "bond" and missing:
        worst = len(missing) / H_K_PESSIMISTIC
        share = worst / (R + worst)
        matters = share > KAPITZA_MATTERS
        complete = complete and not matters
        flags.add(
            Code.STACK_INCOMPLETE, Severity.WARNING if matters else Severity.INFO,
            f"No interface (Kapitza) conductance at face(s) {', '.join(missing)}. Even at "
            f"{H_K_PESSIMISTIC / 1e6:.0f} MW/m^2K per face (the low end of measured "
            f"solid-solid interfaces) they would carry {share * 100:.1f} % of R"
            + ("; h_joint is an UPPER bound." if matters else ": negligible for this bond."),
            remedy=("Give h_K (MD, measured, or DMM when both neighbours are in the "
                    "materials database). At full contact it enters unscaled; it matters "
                    "for thin bonds and thin films (open issue P0-5)."),
            kapitza_share_pessimistic=share,
        )
    for t in terms:
        t.fraction = t.resistance / R

    if fill.tim_class is not None and complete:
        if fill.tim_class not in TIM_CLASS_RANGE:
            raise ValueError(f"unknown tim_class {fill.tim_class!r}")
        lo, hi = TIM_CLASS_RANGE[fill.tim_class]
        if not lo <= R <= hi:
            flags.add(
                Code.STACK_CLASS_RANGE, Severity.CAUTION,
                f"R = {R / MM2K_W:.3g} mm^2K/W is outside the typical {fill.tim_class} range "
                f"{lo / MM2K_W:.3g}-{hi / MM2K_W:.3g} mm^2K/W.",
                remedy=(f"Check BLT, k and units against the datasheet ({TIM_CLASS_SOURCE}). "
                        "Datasheet R is often in K in^2/W (x 6.45 for K cm^2/W)."),
                R_total=R,
            )
    cap = None if complete else Confidence.LOW
    return StackResult(
        kind=fill.kind,
        h_joint=make_derived(1.0 / R, unit="W/m2K", inputs=inputs, confidence=cap, name="h_joint"),
        R_total=make_derived(R, unit="m2K/W", inputs=inputs, confidence=cap, name="R_total"),
        terms=terms, R_kapitza=r_kap, complete=complete,
    )
