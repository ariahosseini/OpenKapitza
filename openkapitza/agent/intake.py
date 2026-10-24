"""Intake: collect answers field by field, apply defaults, build the joint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fields import BY_NAME, FIELDS, Answers, Field, FieldError, parse_layers

SKIP = object()          # explicit "not measured" for an optional field


@dataclass
class Intake:
    answers: Answers = field(default_factory=dict)
    defaulted: dict[str, str] = field(default_factory=dict)   # name -> reason
    skipped: set[str] = field(default_factory=set)

    # ---- state -------------------------------------------------------------
    def applicable(self) -> list[Field]:
        return [f for f in FIELDS if f.applies(self.answers)]

    def pending(self) -> list[Field]:
        """Fields still to ask, in order (advanced ones are never asked)."""
        return [f for f in self.applicable()
                if not f.advanced and f.name not in self.answers
                and f.name not in self.skipped and f.name not in self.defaulted]

    def next_field(self) -> Field | None:
        p = self.pending()
        return p[0] if p else None

    @property
    def complete(self) -> bool:
        return self.next_field() is None

    def missing_required(self) -> list[Field]:
        return [f for f in self.applicable() if f.required(self.answers)
                and f.name not in self.answers and f.name not in self.defaulted]

    # ---- answering ---------------------------------------------------------
    def set(self, name: str, text: str) -> Any:
        """Parse and store an answer. Raises FieldError with a user-facing message."""
        f = BY_NAME.get(name)
        if f is None:
            raise FieldError(f"unknown field {name!r}")
        if not f.applies(self.answers) and not f.advanced:
            raise FieldError(f"{name} does not apply to this joint")
        if f.name == "layers":
            parse_layers(text)                 # validate now, store the text
            value = text.strip()
        elif f.name == "z_trunc":
            value = text.strip()
            if value != "reference":
                value = Field("z_trunc", "", "", unit=None, lo=0.0, hi=10.0).parse(value)
        else:
            value = f.parse(text)
        self.answers[name] = value
        self.defaulted.pop(name, None)
        self.skipped.discard(name)
        self._drop_inapplicable()
        return value

    def accept_default(self, name: str) -> Any:
        """Empty answer: take the default, or skip an optional field."""
        f = BY_NAME[name]
        d = f.default_for(self.answers)
        if d is not None:
            self.answers[name] = d
            self.defaulted[name] = f.default_reason
            self._drop_inapplicable()
            return d
        if not f.required(self.answers):
            self.skipped.add(name)
            self.defaulted[name] = f.default_reason or "not given"
            return None
        raise FieldError("this one is needed; there is no safe default")

    def unset(self, name: str) -> None:
        self.answers.pop(name, None)
        self.defaulted.pop(name, None)
        self.skipped.discard(name)

    def _drop_inapplicable(self) -> None:
        """A changed answer (say bare -> tim) can make earlier answers moot."""
        live = {f.name for f in self.applicable()}
        for name in list(self.answers):
            if name not in live:
                self.answers.pop(name)
                self.defaulted.pop(name, None)

    # ---- output ------------------------------------------------------------
    def build_kwargs(self) -> dict:
        """Keyword arguments for openkapitza.build_joint."""
        from ..builder import bond_fill, tim_fill
        missing = self.missing_required()
        if missing:
            raise FieldError("still needed: " + ", ".join(f.name for f in missing))
        a = self.answers
        kw: dict[str, Any] = dict(
            material_a=a["material_a"], material_b=a["material_b"],
            temperature_K=a["temperature_K"], pressure_MPa=a["pressure_MPa"],
            label=f"{a['material_a']}/{a['material_b']}",
        )
        for k in ("Hv_a_GPa", "Hv_b_GPa", "c1_a_GPa", "c2_a", "c1_b_GPa", "c2_b",
                  "max_pressure_MPa"):
            if a.get(k) is not None:
                kw[k] = a[k]
        jt = a.get("joint_type", "bare")
        if jt == "bare":
            kw.update(sigma_a_um=a["sigma_a_um"], sigma_b_um=a["sigma_b_um"],
                      slope_a=a.get("slope_a"), slope_b=a.get("slope_b"),
                      clamping=a.get("clamping"))
            if a.get("medium", "air") != "vacuum":
                kw.update(gas=a.get("medium", "air"),
                          gas_pressure_kPa=a.get("gas_pressure_kPa", 101.325))
            if a.get("flatness_um") is not None:
                kw.update(flatness_um=a["flatness_um"], joint_area_mm2=a["joint_area_mm2"])
            if a.get("z_trunc") is not None:
                kw["z_trunc"] = a["z_trunc"]
        else:
            # The stack model does not use roughness for the bulk; a nominal value keeps
            # the thin-TIM check meaningful only when the user gave one.
            kw.update(sigma_a_um=a.get("sigma_a_um", 1.0), sigma_b_um=a.get("sigma_b_um", 1.0))
            if jt == "tim":
                kw["fill"] = tim_fill(a["blt_um"], a["k_tim"], tim_class=a.get("tim_class"),
                                      R_c_mm2K_W=a.get("R_c_mm2K_W"))
            else:
                kw["fill"] = bond_fill(parse_layers(a["layers"]))
        return kw

    def summary_lines(self) -> list[str]:
        out = []
        for f in self.applicable():
            if f.name in self.answers:
                v = self.answers[f.name]
                tag = "  (default: " + self.defaulted[f.name] + ")" if f.name in self.defaulted else ""
                unit = f" {f.unit}" if f.unit and isinstance(v, float) else ""
                out.append(f"  {f.name:16} = {v:g}{unit}{tag}" if isinstance(v, float)
                           else f"  {f.name:16} = {v}{tag}")
            elif f.name in self.skipped:
                out.append(f"  {f.name:16} = —  ({self.defaulted.get(f.name, 'not given')})")
        return out
