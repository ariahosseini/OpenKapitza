"""Files in and out of the agent: joint descriptions, results as JSON, sessions.

A joint description is YAML or JSON with the agent's field names (see
`python -m openkapitza.agent --fields`). Values may carry units as strings:

    material_a: Al6061
    material_b: Cu
    temperature_K: 45 C
    pressure_MPa: 500 kPa
    sigma_a_um: 0.8
    sigma_b_um: 0.4
    clamping: bolted
    flatness_um: 25
    joint_area_mm2: 40 x 40 mm
    md: runs/sial_ttm          # optional: finished MD to ingest (path or list)
    label: heat sink to lid    # optional

A file may also hold several joints under `joints:` (a list), for batch runs.
Unlisted fields take their defaults, exactly as pressing Enter would.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from .fields import BY_NAME, FieldError
from .intake import Intake

SESSION_VERSION = 1
META_KEYS = ("md", "label", "sweep")


def load_descriptions(path: Path | str) -> list[dict]:
    """One or more joint descriptions from a YAML/JSON file."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)                      # JSON is valid YAML
    if isinstance(data, dict) and "joints" in data:
        data = data["joints"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise FieldError(f"{path}: expected a mapping of fields, or a list under 'joints:'")
    return data


def intake_from_dict(desc: dict) -> Intake:
    """Fill an Intake from a description; unknown keys are an error, not ignored."""
    unknown = [k for k in desc if k not in BY_NAME and k not in META_KEYS]
    if unknown:
        raise FieldError(f"unknown field(s): {', '.join(unknown)}")
    it = Intake()
    # Set in schema order so that branching fields (joint_type, medium, flatness)
    # are known before the fields that depend on them.
    for f in [f for f in BY_NAME.values() if f.name in desc]:
        value = desc[f.name]
        if value is None:
            continue
        try:
            it.set(f.name, str(value))
        except FieldError as e:
            raise FieldError(f"{f.name}: {e}") from e
    while not it.complete:
        it.accept_default(it.next_field().name)
    missing = it.missing_required()
    if missing:
        raise FieldError("missing required field(s): " + ", ".join(f.name for f in missing))
    return it


# ---- results as plain data --------------------------------------------------------
def _num(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return float(x)


def result_to_dict(result, plan, intake: Intake | None = None, md_used=None) -> dict:
    """Everything a front end or a spreadsheet needs, JSON-serialisable."""
    net = result.primary
    mc = (result.uncertainty or {}).get("h_joint")
    out: dict[str, Any] = {
        "joint": result.inputs.label,
        "h_joint_W_m2K": _num(net.h_joint.si),
        "R_total_m2K_W": _num(net.R_total.si),
        "band_W_m2K": ({"p05": _num(mc.p05), "p50": _num(getattr(mc, "p50", None)),
                        "p95": _num(mc.p95)} if mc else None),
        "branch": net.branch,
        "contributions": [{"part": c.label, "resistance_m2K_W": _num(c.resistance),
                           "fraction": _num(c.fraction)} for c in net.contributions],
        "kapitza": ({"h_K_W_m2K": _num(result.kapitza.h_K.si),
                     "uncertainty": _num(result.kapitza.h_K.uncertainty),
                     "source": result.kapitza.source} if result.kapitza else None),
        "flags": [{"code": str(f.code), "severity": f.severity.name, "message": f.message,
                   "remedy": f.remedy} for f in result.flags.flags],
        "plan": {"md": plan.md, "reason": plan.md_reason, "route": plan.route_status,
                 "potential": plan.route_potential, "cluster_jobs": list(plan.cluster_jobs),
                 "next_measurements": plan.next_measurements},
    }
    if result.accommodation_band:
        lo, hi = result.accommodation_band["h_joint"]
        out["accommodation_band_W_m2K"] = [_num(lo), _num(hi)]
    if result.macro is not None:
        out["macro"] = {"a_L_over_b_L": _num(result.macro.B),
                        "R_macro_m2K_W": _num(result.macro.R_L_area.si),
                        "conforming": result.macro.conforming}
    if intake is not None:
        out["inputs"] = dict(intake.answers)
        out["defaults_used"] = dict(intake.defaulted)
    if md_used is not None:
        out["md_used"] = md_used.to_dict()
    return out


# ---- sessions ------------------------------------------------------------------
def save_session(session, path: Path | str) -> Path:
    p = Path(path)
    data = {"version": SESSION_VERSION, "answers": session.intake.answers,
            "defaulted": session.intake.defaulted, "skipped": sorted(session.intake.skipped),
            "md": [r.to_dict() for r in session.md.values()]}
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def load_session(session, path: Path | str) -> None:
    from .ingest import MDResult
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != SESSION_VERSION:
        raise FieldError(f"{path}: unsupported session version {data.get('version')}")
    it = Intake()
    it.answers = dict(data["answers"])
    it.defaulted = dict(data["defaulted"])
    it.skipped = set(data["skipped"])
    session.intake = it
    session.md = {}
    for d in data.get("md", []):
        r = MDResult.from_dict(d)
        session.md[r.pair] = r
    session.result = session.plan = None
    session.history = []
