"""Bring finished MD results back into a joint calculation.

Three things can be ingested, all identified by what is in the folder:

  a job folder        spec.json + NEMD outputs -> one measured h_K (or, for an
                      EMD job, an equilibrium cross-check)
  a campaign root     manifest.json with "rungs" (plan_campaign) -> every rung
                      measured, each convergence axis judged; the quoted h_K is
                      the most refined measured rung, with the verdicts attached
  a prepared root     manifest.json with "jobs" (prepare_cluster) -> every
                      finished job measured and grouped by material pair

The result is an MDResult per material pair: h_K with its error, where it came
from, whether convergence was established, and the measurement's own flags.
`to_kapitza` turns it into the KapitzaEstimate the pipeline accepts, after
checking it matches the joint (materials, temperature, spot pressure).

Nothing here re-derives physics: measurement is md.measure.measure_kapitza and
md.campaign.collect_campaign, the same code the tests and the cluster runbook
use.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ..md.campaign import SPEC_FILE, collect_campaign, read_spec, spec_from_record
from ..md.dmm import KapitzaEstimate
from ..md.measure import measure_kapitza
from ..schema.flags import Code, FlagSet, Severity
from ..schema.quantities import Confidence, Provenance, Quantity

# Element label in an MD spec -> database materials it can stand for. An
# element run standing in for an alloy is accepted with a caution.
ELEMENT_MATERIALS = {
    "Si": ("Si",), "Ge": ("Ge",), "Cu": ("Cu",), "Al": ("Al", "Al6061"),
    "SiO2": ("SiO2",), "Ni": ("Ni200",),
}
ALLOY_PROXY = {"Al6061": "Al", "Ni200": "Ni"}
TEMPERATURE_TOLERANCE_K = 25.0
RELATIVE_ERROR_WARN = 0.30


@dataclass
class MDResult:
    pair: frozenset                  # element labels as in the spec
    h_K: float                       # W/m^2K, per real contact area
    uncertainty: float
    temperature: float               # K
    spot_pressure_GPa: float         # 0 when unloaded
    potential: str
    source: str                      # where it came from (folder, rung)
    converged: bool | None           # None: single run, convergence not tested
    electrons: bool                  # True for a two-temperature (TTM) run
    flags: FlagSet = field(default_factory=FlagSet)
    notes: list[str] = field(default_factory=list)
    emd: dict | None = None          # cross-check, if an EMD job was found

    @property
    def relative_uncertainty(self) -> float:
        return self.uncertainty / self.h_K if self.h_K else math.inf

    def summary(self) -> str:
        conv = {True: "converged", False: "NOT converged", None: "single run, convergence untested"}
        s = (f"{'/'.join(sorted(self.pair))}: h_K = {self.h_K / 1e6:.4g} ± "
             f"{self.uncertainty / 1e6:.3g} MW/m²K ({self.relative_uncertainty:.0%}), "
             f"{self.temperature:.0f} K, {self.potential}, {conv[self.converged]}"
             f"{', with electrons (TTM)' if self.electrons else ', phonons only'}")
        if self.emd:
            s += (f"; EMD cross-check h = {self.emd['h'] / 1e6:.3g} ± {self.emd['h_err'] / 1e6:.3g}"
                  f" MW/m²K")
        return s

    # ---- persistence (session save/load) ---------------------------------
    def to_dict(self) -> dict:
        return {"pair": sorted(self.pair), "h_K": self.h_K, "uncertainty": self.uncertainty,
                "temperature": self.temperature, "spot_pressure_GPa": self.spot_pressure_GPa,
                "potential": self.potential, "source": self.source, "converged": self.converged,
                "electrons": self.electrons, "notes": self.notes, "emd": self.emd,
                "flags": [{"code": str(f.code), "severity": f.severity.name, "message": f.message}
                          for f in self.flags.flags]}

    @classmethod
    def from_dict(cls, d: dict) -> "MDResult":
        fl = FlagSet()
        for f in d.get("flags", []):
            fl.add(f["code"], Severity[f["severity"]], f["message"])
        return cls(frozenset(d["pair"]), d["h_K"], d["uncertainty"], d["temperature"],
                   d["spot_pressure_GPa"], d["potential"], d["source"], d["converged"],
                   d["electrons"], fl, list(d.get("notes", [])), d.get("emd"))


class IngestError(ValueError):
    """The folder holds nothing that can be ingested; the message says why."""


# ---- reading -------------------------------------------------------------------
def ingest(path: Path | str) -> dict[frozenset, MDResult]:
    """Measure whatever finished MD output is under `path`, grouped by pair."""
    p = Path(path)
    if not p.exists():
        raise IngestError(f"{p} does not exist")
    manifest = p / "manifest.json"
    if manifest.is_file():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        if "rungs" in m:
            r = _campaign(p, m)
            return {r.pair: r}
        if "jobs" in m:
            return _prepared(p, m)
    if (p / SPEC_FILE).is_file():
        spec, kind, extra = read_spec(p)
        if kind == "emd":
            raise IngestError("an EMD job is a cross-check: ingest it together with its NEMD "
                              "job (their common parent folder)")
        r = _single(p, spec)
        return {r.pair: r}
    raise IngestError(f"no spec.json or manifest.json in {p}: nothing to measure (jobs "
                      "written by prepare_cluster / plan_campaign carry one)")


def _has_output(wd: Path) -> bool:
    return (wd / "temperature_profile.dat").is_file() and (wd / "reservoir_energy.dat").is_file()


def _single(wd: Path, spec, *, converged: bool | None = None, label: str | None = None) -> MDResult:
    if not _has_output(wd):
        raise IngestError(f"{wd.name}: no output yet (temperature_profile.dat / "
                          "reservoir_energy.dat missing)")
    flags = FlagSet()
    m = measure_kapitza(wd, spec, flags)
    r = MDResult(pair=frozenset({spec.element_a, spec.element_b}), h_K=m.h_K,
                 uncertainty=m.uncertainty, temperature=spec.temperature,
                 spot_pressure_GPa=spec.spot_pressure_GPa or 0.0,
                 potential=f"{spec.pair_style} {spec.pair_file}", source=label or str(wd),
                 converged=converged, electrons=spec.ttm is not None, flags=flags)
    if converged is None:
        flags.add(Code.MD_SINGLE_RUN, Severity.CAUTION,
                  f"{r.source}: one cell size and run length; finite-size convergence is not "
                  "established.",
                  remedy="Run the convergence campaign for this pair (cross-section, length, "
                         "duration, reservoir) before treating h_K as converged.")
    if r.relative_uncertainty > RELATIVE_ERROR_WARN:
        flags.add(Code.MD_LARGE_ERROR, Severity.WARNING,
                  f"{r.source}: h_K error is {r.relative_uncertainty:.0%} of the value.",
                  remedy="Longer production or a larger cross-section; the joint result "
                         "carries this error into its band.")
    return r


def _campaign(root: Path, manifest: dict) -> MDResult:
    res = collect_campaign(root)
    if not res.measurements:
        raise IngestError(f"campaign {root.name}: no rung has finished "
                          f"({len(res.failed)} pending or failed)")
    rungs = {e["name"]: e for e in manifest["rungs"]}
    # Most refined measured rung: the largest cell (cross-section x total length),
    # then the longest production. Its own error, plus the verdicts, travel with it.
    def size(name):
        g = rungs[name]["spec"]["geometry"]
        return (g["cross_section"] ** 2 * (g["bulk_a_length"] + g["bulk_b_length"]),
                rungs[name]["spec"]["production_steps"])
    best = max(res.measurements, key=size)
    spec = spec_from_record(rungs[best]["spec"])
    m = res.measurements[best]
    # No verdict at all (e.g. one finished rung): convergence untested, not failed.
    converged = all(v.converged for v in res.verdicts.values()) if res.verdicts else None
    r = MDResult(pair=frozenset({spec.element_a, spec.element_b}), h_K=m.h_K,
                 uncertainty=m.uncertainty, temperature=spec.temperature,
                 spot_pressure_GPa=spec.spot_pressure_GPa or 0.0,
                 potential=f"{spec.pair_style} {spec.pair_file}",
                 source=f"{root} (rung {best})", converged=converged,
                 electrons=spec.ttm is not None, flags=res.flags)
    for axis, v in res.verdicts.items():
        r.notes.append(f"{axis}: {'converged' if v.converged else 'not converged'} "
                       f"(last change {v.last_change:.1%})")
    if res.failed:
        r.notes.append(f"{len(res.failed)} rung(s) without a result")
    if converged is None:
        r.flags.add(Code.MD_SINGLE_RUN, Severity.CAUTION,
                    f"Campaign {root.name}: no axis has two finished rungs yet, so convergence "
                    "is untested.", remedy="Let more rungs finish, then ingest again.")
    elif not converged:
        r.flags.add(Code.MD_NOT_CONVERGED, Severity.WARNING,
                    f"Campaign {root.name}: not every axis converged; quoting the most refined "
                    f"rung ({best}).",
                    remedy="See the per-axis notes; extend the axis that is still moving.")
    return r


def _prepared(root: Path, manifest: dict) -> dict[frozenset, MDResult]:
    out: dict[frozenset, MDResult] = {}
    skipped = []
    emds = {}
    for e in manifest["jobs"]:
        name = e["name"]
        if name.startswith("campaign/"):
            continue
        wd = root / name
        if not (wd / SPEC_FILE).is_file() or not _has_output(wd) and not (wd / "emd_temperatures.dat").is_file():
            skipped.append(name)
            continue
        spec, kind, _ = read_spec(wd)
        if kind == "emd":
            emds[frozenset({spec.element_a, spec.element_b})] = wd
            continue
        try:
            r = _single(wd, spec, label=f"{root}/{name}")
        except (IngestError, ValueError, OSError) as exc:
            skipped.append(f"{name} ({exc})")
            continue
        prev = out.get(r.pair)
        # Prefer a run with electrons (it contains the e-ph resistance), then
        # the smaller relative error.
        if prev is None or (r.electrons, -r.relative_uncertainty) > (prev.electrons,
                                                                     -prev.relative_uncertainty):
            if prev is not None:
                r.notes.append(f"also measured: {prev.summary()}")
            out[r.pair] = r
        else:
            prev.notes.append(f"also measured: {r.summary()}")
    camp = root / "campaign"
    if (camp / "manifest.json").is_file():
        try:
            r = _campaign(camp, json.loads((camp / "manifest.json").read_text()))
            out[r.pair] = r
        except IngestError as exc:
            skipped.append(f"campaign ({exc})")
    for pair, wd in emds.items():
        if pair in out:
            out[pair].emd = _emd_cross_check(wd, out[pair])
    if not out:
        raise IngestError(f"{root}: no finished NEMD job ({len(skipped)} without output)")
    for r in out.values():
        if skipped:
            r.notes.append(f"{len(skipped)} job(s) in {root.name} had no usable output")
    return out


def _emd_cross_check(wd: Path, nemd: MDResult) -> dict | None:
    from ..md.emd_kapitza import check, h_from_series, read_emd
    try:
        t, dT, na, nb, area = read_emd(wd)
        res = h_from_series(t, dT, na, nb, area, model="biexp")
    except (OSError, ValueError) as exc:
        nemd.notes.append(f"EMD job {wd.name} unreadable: {exc}")
        return None
    check(res, nemd.flags, h_nemd=nemd.h_K)
    return {"h": res.h, "h_err": res.h_err, "ratio_nemd_over_emd": res.ratio_nemd,
            "h_fast": res.h_fast, "slow_weight": res.slow_weight}


# ---- applying ------------------------------------------------------------------
def matches(r: MDResult, mat_a: str, mat_b: str) -> bool:
    els = sorted(r.pair)
    for x, y in ((mat_a, mat_b), (mat_b, mat_a)):
        if len(els) == 2 and x in ELEMENT_MATERIALS.get(els[0], ()) and \
                y in ELEMENT_MATERIALS.get(els[1], ()):
            return True
    return False


def to_kapitza(r: MDResult, mat_a: str, mat_b: str, temperature_K: float,
               spot_pressure_GPa: float | None, flags: FlagSet) -> KapitzaEstimate:
    """KapitzaEstimate for this joint, with the mismatch checks as flags."""
    if not matches(r, mat_a, mat_b):
        raise IngestError(f"MD pair {'/'.join(sorted(r.pair))} does not match the joint "
                          f"{mat_a}/{mat_b}")
    for m in (mat_a, mat_b):
        if m in ALLOY_PROXY:
            flags.add(Code.MD_MISMATCH, Severity.CAUTION,
                      f"MD was run on pure {ALLOY_PROXY[m]}; the joint is {m}.",
                      remedy="Alloying elements and precipitates scatter phonons at the "
                             "interface too; treat the MD h_K as the pure-metal value.")
    if abs(r.temperature - temperature_K) > TEMPERATURE_TOLERANCE_K:
        flags.add(Code.MD_MISMATCH, Severity.CAUTION,
                  f"MD at {r.temperature:.0f} K, joint at {temperature_K:.0f} K.",
                  remedy="h_K depends on temperature (DMM: roughly with the heat capacity); "
                         "run MD at the joint temperature.")
    if spot_pressure_GPa and spot_pressure_GPa > 0.5 and r.spot_pressure_GPa < 0.1:
        flags.add(Code.MD_MISMATCH, Severity.INFO,
                  f"MD interface unloaded; inside a contact spot the pressure is "
                  f"~{spot_pressure_GPa:.2g} GPa.",
                  remedy="The spot-pressure jobs (P1-9/P1-10) measure the loaded interface.")
    for f in r.flags.flags:                      # the measurement's own caveats travel
        flags.add(f.code, f.severity, f"[MD] {f.message}", remedy=f.remedy)
    conf = Confidence.MEDIUM if (r.converged and r.relative_uncertainty < 0.15) else Confidence.LOW
    q = Quantity(si=r.h_K, unit="MW/m2K", uncertainty=r.uncertainty,
                 provenance=Provenance.MD_COMPUTED, confidence=conf, name="h_K",
                 note=f"NEMD, {r.potential}, {r.source}; per unit REAL contact area")
    return KapitzaEstimate(h_K=q, transmission=Quantity(si=float("nan"), unit="-",
                                                        provenance=Provenance.MD_COMPUTED,
                                                        name="transmission"),
                           source="md")
