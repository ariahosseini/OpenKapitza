"""The input schema: every question the agent can ask, in asking order.

Each Field knows its question, why it matters (shown on "why"), its unit and
accepted units, when it applies, whether it is required, and its default with
the reason for that default. Nothing here runs physics; it only turns typed
text into values the builder accepts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..schema.db import _load_yaml, available_gases, available_materials
from ..schema.units import UnitError, to_si

Answers = dict[str, Any]

# Units the core table lacks but people type for these quantities.
_EXTRA_UNITS = {
    "mm2K/W": 1e-6, "mm^2K/W": 1e-6, "Kmm2/W": 1e-6, "K mm2/W": 1e-6,
    "Kcm2/W": 1e-4, "K cm2/W": 1e-4, "cm2K/W": 1e-4,
    "Kin2/W": 6.4516e-4, "K in2/W": 6.4516e-4,
    "mm2": 1e-6, "mm^2": 1e-6, "cm2": 1e-4, "cm^2": 1e-4, "m2": 1.0, "m^2": 1.0,
    "in2": 6.4516e-4, "in^2": 6.4516e-4,
    "mil": 25.4e-6, "mils": 25.4e-6, "µm": 1e-6, "μm": 1e-6,
}


class FieldError(ValueError):
    """A typed answer that cannot be used; the message is shown to the user."""


@dataclass
class Field:
    name: str
    question: str
    why: str
    kind: str = "float"                  # float | choice | text
    unit: str | None = None              # the unit the builder wants (lab unit)
    units: tuple[str, ...] = ()          # units accepted from the user
    choices: tuple[str, ...] | Callable[[], tuple[str, ...]] = ()
    applies: Callable[[Answers], bool] = lambda a: True
    required: Callable[[Answers], bool] = lambda a: True
    default: Any = None                  # value, or callable(answers) -> value
    default_reason: str = ""
    lo: float | None = None              # exclusive lower bound, in `unit`
    hi: float | None = None
    advanced: bool = False               # never asked; set with "set name=value"

    def choice_list(self) -> tuple[str, ...]:
        return self.choices() if callable(self.choices) else self.choices

    def default_for(self, answers: Answers) -> Any:
        return self.default(answers) if callable(self.default) else self.default

    # ---- parsing ---------------------------------------------------------
    def parse(self, text: str) -> Any:
        text = text.strip()
        if self.kind == "choice":
            return self._parse_choice(text)
        if self.kind == "text":
            if not text:
                raise FieldError("an empty answer is not usable here")
            return text
        return self._parse_float(text)

    def _parse_choice(self, text: str) -> str:
        opts = self.choice_list()
        low = text.lower().replace(" ", "")
        for o in opts:
            if low == o.lower().replace(" ", ""):
                return o
        alias = _ALIASES.get(self.name, {}).get(low)
        if alias in opts:
            return alias
        hits = [o for o in opts if o.lower().startswith(low)] if low else []
        if len(hits) == 1:
            return hits[0]
        raise FieldError(f"choose one of: {', '.join(opts)}")

    def _parse_float(self, text: str) -> float:
        dims = re.fullmatch(r"\s*(\d+\.?\d*)\s*(?:x|×|\*)\s*(\d+\.?\d*)\s*(mm|cm|in|m)?\s*", text)
        if dims and self.unit == "mm2":
            u = dims.group(3) or "mm"
            text = f"{float(dims.group(1)) * float(dims.group(2))} {u}2"
        m = re.fullmatch(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([^\s].*)?", text)
        if not m:
            raise FieldError(f"expected a number{(' in ' + self.unit) if self.unit else ''}, "
                             f"e.g. '1.2{(' ' + self.unit) if self.unit else ''}'")
        value = float(m.group(1))
        unit = (m.group(2) or "").strip()
        if unit and self.unit is None:
            raise FieldError("this is a plain number, without a unit")
        if unit and unit != self.unit:
            if self.units and unit not in self.units:
                raise FieldError(f"unit {unit!r} not accepted here; use one of "
                                 f"{', '.join(self.units)}")
            value = _convert(value, unit, self.unit)
        if self.lo is not None and value <= self.lo:
            raise FieldError(f"must be greater than {self.lo:g} {self.unit or ''}".strip())
        if self.hi is not None and value > self.hi:
            raise FieldError(f"must be at most {self.hi:g} {self.unit or ''}".strip())
        return value


def _si(value: float, unit: str) -> float:
    if unit in _EXTRA_UNITS:
        return value * _EXTRA_UNITS[unit]
    return to_si(value, unit)


def _convert(value: float, unit: str, target: str) -> float:
    try:
        if unit in ("C", "degC", "°C") and target == "K":
            return value + 273.15
        return _si(value, unit) / _si(1.0, target)
    except (UnitError, KeyError) as e:
        raise FieldError(f"unknown unit {unit!r}") from e


def material_phase(key: str) -> str:
    return _load_yaml("materials.yaml")[key].get("phase", "crystalline")


_ALIASES = {
    "material_a": {"silicon": "Si", "germanium": "Ge", "aluminium": "Al", "aluminum": "Al",
                   "6061": "Al6061", "al6061-t6": "Al6061", "copper": "Cu", "silica": "SiO2",
                   "stainless": "SS304", "ss": "SS304", "nickel": "Ni200", "ni": "Ni200"},
    "medium": {"none": "vacuum", "nitrogen": "N2", "helium": "He", "argon": "Ar",
               "hydrogen": "H2"},
    "joint_type": {"contact": "bare", "dry": "bare", "grease": "tim", "paste": "tim",
                   "pad": "tim", "solder": "bond", "sinter": "bond", "attach": "bond"},
}
_ALIASES["material_b"] = _ALIASES["material_a"]

_LEN = ("um", "µm", "μm", "micron", "nm", "mm", "mil", "mils")
_PRESS = ("MPa", "kPa", "Pa", "GPa", "bar", "mbar", "atm", "psi", "torr")
_RES = ("mm2K/W", "mm^2K/W", "Kmm2/W", "Kcm2/W", "cm2K/W", "Kin2/W", "m2K/W")

bare = lambda a: a.get("joint_type", "bare") == "bare"
tim = lambda a: a.get("joint_type") == "tim"
bond = lambda a: a.get("joint_type") == "bond"
no = lambda a: False

FIELDS: list[Field] = [
    Field("material_a", "What is the first solid (side A)?",
          "The two materials set the conductivities, hardness and modulus of the joint, and "
          "decide whether an atomistic (MD) calculation of the interface is possible.",
          kind="choice", choices=lambda: tuple(available_materials())),
    Field("phase_a", "Is side A crystalline or amorphous?",
          "Only matters for the atomistic route: crystals are cut and stacked, glasses are "
          "made by melt-and-quench. You decide; the default is the database's.",
          kind="choice", choices=("crystalline", "amorphous"),
          applies=lambda a: "material_a" in a,
          default=lambda a: material_phase(a["material_a"]),
          default_reason="the database's phase for this material"),
    Field("material_b", "What is the second solid (side B)?",
          "As for side A.", kind="choice", choices=lambda: tuple(available_materials())),
    Field("phase_b", "Is side B crystalline or amorphous?",
          "As for side A.", kind="choice", choices=("crystalline", "amorphous"),
          applies=lambda a: "material_b" in a,
          default=lambda a: material_phase(a["material_b"]),
          default_reason="the database's phase for this material"),
    Field("temperature_K", "Temperature of the joint?",
          "Sets the conductivities and gas properties; below the Debye temperature it also "
          "triggers the quantum-effects warning for any MD result.",
          unit="K", units=("K", "C", "degC", "°C"), lo=0.0, hi=2000.0),
    Field("joint_type", "Is it a bare contact, a thermal interface material (grease, pad, "
          "PCM), or a bonded joint (solder, sinter, die attach)?",
          "A bare contact conducts through asperity spots and the gas around them; a filled "
          "or bonded joint is a series stack of layers instead.",
          kind="choice", choices=("bare", "tim", "bond"), default="bare",
          default_reason="most common case for two metal parts pressed together"),
    Field("pressure_MPa", "Apparent contact pressure (clamping force / nominal area)?",
          "Sets how much of the surface is in real contact; conductance rises almost "
          "linearly with it in a bare joint.",
          unit="MPa", units=_PRESS, lo=0.0, hi=1000.0,
          required=bare, default=0.2,
          default_reason="not used by the filled/bonded stack model (only reported)"),
    # ---- bare contact -------------------------------------------------------
    Field("sigma_a_um", "RMS roughness of surface A (profilometry Rq)?",
          "Roughness and slope set the contact-spot geometry; conductance goes as slope / "
          "roughness. Ra is not RMS: Rq ~ 1.25 Ra for a Gaussian surface.",
          unit="um", units=_LEN, lo=0.0, hi=1000.0, applies=bare),
    Field("sigma_b_um", "RMS roughness of surface B?", "As for surface A.",
          unit="um", units=_LEN, lo=0.0, hi=1000.0, applies=bare),
    Field("slope_a", "Mean absolute asperity slope of surface A (profilometry Δa, "
          "dimensionless)? Leave blank if not measured.",
          "Conductance is linear in the slope. If missing it is estimated from roughness "
          "(Antonetti's correlation) and flagged: a weak correlation, off by 2x for some "
          "finishes.",
          unit=None, lo=0.0, hi=2.0, applies=bare, required=no,
          default_reason="estimated from roughness and flagged"),
    Field("slope_b", "Mean absolute asperity slope of surface B? Leave blank if not "
          "measured.", "As for surface A.",
          unit=None, lo=0.0, hi=2.0, applies=bare, required=no,
          default_reason="estimated from roughness and flagged"),
    Field("medium", "What fills the gaps: vacuum, or a gas?",
          "In gas, heat also crosses the gaps between the spots; at 1 atm air that path "
          "often carries a large share of the total.",
          kind="choice", choices=lambda: ("vacuum",) + tuple(available_gases()),
          applies=bare, default="air", default_reason="most electronics assemblies run in air"),
    Field("gas_pressure_kPa", "Gas pressure?",
          "Below ~10 kPa the gas becomes rarefied in micron gaps and its conductance drops.",
          unit="kPa", units=_PRESS, lo=0.0, hi=1e5,
          applies=lambda a: bare(a) and a.get("medium", "air") != "vacuum",
          default=101.325, default_reason="1 atm"),
    Field("clamping", "How is the joint held: uniform load (press, centred spring clip, "
          "weight) or bolted/screwed?",
          "Bolted faces carry most of the load near the fasteners, so the mean pressure "
          "overstates the contact; that is the single largest error source for heat sinks "
          "and lids.",
          kind="choice", choices=("uniform", "bolted", "unknown"), applies=bare,
          default="unknown", default_reason="not stated; flat contact will be assumed and flagged"),
    Field("flatness_um", "Combined out-of-flatness of the two faces (sum of each face's "
          "maximum deviation)? Leave blank if unknown.",
          "A crowned or warped face touches only in the middle; that macro constriction "
          "can dominate. Ignoring it over-predicted measured joints ~7x on average.",
          unit="um", units=_LEN, lo=0.0, hi=5000.0, applies=bare, required=no,
          default_reason="faces treated as conforming flats (flagged)"),
    Field("joint_area_mm2", "Nominal joint area (footprint)?",
          "Needed with the flatness: the macro constriction depends on the joint size.",
          unit="mm2", units=("mm2", "mm^2", "cm2", "cm^2", "in2", "in^2", "m2"), lo=0.0,
          applies=lambda a: bare(a) and a.get("flatness_um") is not None),
    # ---- TIM ----------------------------------------------------------------
    Field("blt_um", "Bond-line thickness of the TIM (area average, not the spacer size)?",
          "R_TIM = BLT / k + boundary resistances, so BLT enters linearly.",
          unit="um", units=_LEN, lo=0.0, hi=5000.0, applies=tim),
    Field("k_tim", "Bulk thermal conductivity of the TIM (from the slope of an ASTM D5470 "
          "R-vs-BLT fit, not the datasheet headline if that is a total)?",
          "Sets the bulk part BLT / k.", unit="W/mK", units=("W/mK", "W/m/K"), lo=0.0,
          hi=2000.0, applies=tim),
    Field("R_c_mm2K_W", "Boundary resistance R_c1 + R_c2 (the D5470 intercept)? Leave blank "
          "if unknown.",
          "Often 4-30 mm^2K/W for greases (NREL 2017); without it the result is a LOWER "
          "bound on R (flagged).", unit="mm2K/W", units=_RES, lo=-1e-9, applies=tim,
          required=no, default_reason="left out; R is then a lower bound (flagged)"),
    Field("tim_class", "TIM class (grease, pad, pcm, gel, adhesive, solder) for a sanity "
          "range? Leave blank to skip.",
          "The total is checked against the typical range for its class.",
          kind="choice", choices=("grease", "pad", "pcm", "gel", "adhesive", "solder"),
          applies=tim, required=no, default_reason="range check skipped"),
    # ---- bond ---------------------------------------------------------------
    Field("layers", "Layers from side A to side B, as 'name thickness_um k_W/mK "
          "[void_fraction] [material]', separated by ';'  (e.g. 'Cu3Sn 2 70; SAC305 30 58 0.04')",
          "A bonded joint is a series stack; each layer adds t / (k (1 - voids)), and each "
          "solid-solid face adds an atomic (Kapitza) resistance.",
          kind="text", applies=bond),
    # ---- advanced -----------------------------------------------------------
    Field("z_trunc", "Truncation level of the asperity heights (units of sigma), or "
          "'reference'.",
          "Light-load correction (Milanez et al. 2004). Fitted to light-load data with "
          "your own hardness coefficients.",
          kind="text", required=no, advanced=True, applies=bare),
    Field("h_K_MW", "Measured interface conductance h_K (per real contact area)?",
          "Replaces the DMM placeholder for the atomic interface term.",
          unit="MW/m2K", units=("MW/m2K", "GW/m2K", "W/m2K"), lo=0.0, required=no,
          advanced=True),
]

BY_NAME = {f.name: f for f in FIELDS}


def parse_layers(text: str) -> list[tuple]:
    """'Cu3Sn 2 70; SAC305 30 58 0.04 ; Al 0.5 237 0 Al' -> bond_fill tuples."""
    out = []
    mats = set(available_materials())
    for chunk in [c.strip() for c in text.split(";") if c.strip()]:
        parts = chunk.split()
        if len(parts) < 3:
            raise FieldError(f"layer {chunk!r}: need at least 'name thickness_um k'")
        name = parts[0]
        try:
            t, k = float(parts[1]), float(parts[2])
            void = float(parts[3]) if len(parts) > 3 and parts[3] not in mats else 0.0
        except ValueError as e:
            raise FieldError(f"layer {chunk!r}: thickness, k and voids must be numbers") from e
        mat = next((p for p in parts[3:] if p in mats), None)
        if t <= 0 or k <= 0 or not 0 <= void < 1:
            raise FieldError(f"layer {chunk!r}: thickness and k > 0, voids in [0, 1)")
        out.append((name, t, k, void, mat))
    if not out:
        raise FieldError("give at least one layer")
    return out
