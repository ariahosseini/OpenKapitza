"""One-parameter sweeps: how the joint responds to pressure, flatness, BLT...

    sweep pressure_MPa=0.1:2:8          8 points, linear
    sweep pressure_MPa=0.1:2:8:log      log-spaced
    sweep flatness_um=0,5,10,25,50      explicit values

Each point is a full continuum run (without the Monte Carlo, for speed) on a
copy of the current inputs, so every option and flag applies as in a single
run. The table reports h, R, the part that carries most of the solid-path
resistance, and the worst flag at that point.
"""

from __future__ import annotations

import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from ..schema.flags import Severity
from .fields import BY_NAME, FieldError


@dataclass
class SweepPoint:
    value: float
    h: float
    R: float
    limiting: str
    worst: str
    warnings: frozenset = frozenset()   # WARNING-or-worse codes at this point


def parse_spec(expr: str) -> tuple[str, list[float]]:
    if "=" not in expr:
        raise FieldError("use sweep name=start:stop:n[:log] or name=v1,v2,...")
    name, rng = (s.strip() for s in expr.split("=", 1))
    f = BY_NAME.get(name)
    if f is None or f.kind != "float":
        raise FieldError(f"{name!r} is not a numeric field")
    if "," in rng:
        vals = [f.parse(v) for v in rng.split(",") if v.strip()]
    else:
        parts = rng.split(":")
        if len(parts) not in (3, 4):
            raise FieldError("range is start:stop:n[:log]")
        a, b = f.parse(parts[0]), f.parse(parts[1])
        try:
            n = int(parts[2])
        except ValueError as e:
            raise FieldError("n must be an integer") from e
        if n < 2 or n > 200:
            raise FieldError("n must be between 2 and 200")
        log = len(parts) == 4 and parts[3].strip().lower() == "log"
        if log and (a <= 0 or b <= 0):
            raise FieldError("a log sweep needs positive end points")
        vals = ([a * (b / a) ** (i / (n - 1)) for i in range(n)] if log
                else [a + (b - a) * i / (n - 1) for i in range(n)])
    if not vals:
        raise FieldError("no values to sweep")
    return name, vals


def run_sweep(session, name: str, values: list[float]) -> list[SweepPoint]:
    out = []
    for v in values:
        trial = copy.deepcopy(session.intake)
        trial.set(name, repr(v))
        missing = trial.missing_required()
        if missing:
            need = ", ".join(f.name for f in missing)
            raise FieldError(f"sweeping {name} also needs {need}; give it first "
                             f"(set {missing[0].name}=...)")
        r = session._compute(trial, with_uncertainty=False)
        net = r.primary
        parts = [c for c in net.contributions if c.label != "solid path (series total)"
                 and not c.label.startswith("gas gap")]
        lim = max(parts, key=lambda c: c.fraction).label if parts else "-"
        worst = max((f for f in r.flags.flags), key=lambda f: f.severity, default=None)
        w = (f"{worst.severity.name}: {worst.code}" if worst and worst.severity >= Severity.WARNING
             else "-")
        warn = frozenset(str(f.code) for f in r.flags.flags if f.severity >= Severity.WARNING)
        out.append(SweepPoint(v, net.h_joint.si, net.R_total.si, lim, w, warn))
    common = frozenset.intersection(*[p.warnings for p in out]) if out else frozenset()
    for p in out:                         # keep only warnings that change along the sweep
        p.worst = ", ".join(sorted(p.warnings - common)) or "-"
    return out


def table(name: str, unit: str | None, pts: list[SweepPoint]) -> str:
    head = f"{name + (' [' + unit + ']' if unit else ''):>20}  {'h [W/m²K]':>12}  {'R [mm²K/W]':>11}  limiting part"
    rows = [head, "-" * len(head)]
    common = frozenset.intersection(*[p.warnings for p in pts]) if pts else frozenset()
    for p in pts:
        rows.append(f"{p.value:20.4g}  {p.h:12.4g}  {p.R * 1e6:11.4g}  {p.limiting}"
                    + (f"   (+{p.worst})" if p.worst != "-" else ""))
    if common:
        rows.append(f"at every point: {', '.join(sorted(common))}")
    return "\n".join(rows)


def write_csv(path: Path | str, name: str, pts: list[SweepPoint]) -> Path:
    p = Path(path)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([name, "h_W_m2K", "R_m2K_W", "limiting_part", "warnings"])
        for q in pts:
            w.writerow([q.value, q.h, q.R, q.limiting, ";".join(sorted(q.warnings))])
    return p


def plot(path: Path | str, name: str, unit: str | None, pts: list[SweepPoint]) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise FieldError("plotting needs matplotlib (pip install matplotlib); the CSV has "
                         "the same numbers") from e
    xs = [p.value for p in pts]
    fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=150)
    scale, hunit = (1e3, "kW/m²K") if max(p.h for p in pts) >= 1e4 else (1.0, "W/m²K")
    ax.plot(xs, [p.h / scale for p in pts], marker="o", color="#2a6fbb")
    span = max(xs) / min(xs) if min(xs) > 0 else 1
    if span > 20:
        ax.set_xscale("log")
    hs = [p.h for p in pts]
    if min(hs) > 0 and max(hs) / min(hs) > 20:
        ax.set_yscale("log")
    ax.set_xlabel(f"{name}" + (f" [{unit}]" if unit else ""))
    ax.set_ylabel(f"joint conductance h [{hunit}]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)
