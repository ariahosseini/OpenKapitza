"""The convergence campaign: plan it, write it out as jobs, collect it (P1-3).

Four axes, one factor at a time about a base cell: cross-section, bulk length,
production duration and applied temperature split (flux). The first three ask
"did h_K stop moving"; the flux axis asks whether the response is linear at
all (`check_flux_linearity`), which is a validity test, not a convergence one.

Three stages, so the cluster session is mechanical:

    plan_campaign(base, ...)        -> list of Rung (shared base deduplicated)
    write_campaign(root, rungs, ...) -> one directory per rung with the deck,
                                        the potential, a job script, and a
                                        manifest with the cost of every rung
    collect_campaign(root, ...)     -> measure_kapitza on every finished rung,
                                        then error-aware verdicts per axis

Errors travel with h_K into `check_series`, so a statistics-limited axis is
reported as such rather than as "not converged" (same lesson as P1-1).
"""

from __future__ import annotations

import dataclasses

import copy
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..compute.resources import estimate_cost, resources_for
from ..compute.runner import JobSpec, SlurmRunner
from ..schema.flags import FlagSet
from .convergence import ConvergenceSeries, check_flux_linearity, check_series
from .measure import measure_kapitza
from .nemd import NEMDSpec, build_input_deck

# "reservoir": thermostat-slab length. Si/Ge ITR is linear in 1/L_bath as well as
# 1/L_sys (Wei et al., Chin. Phys. B 34, 116301, 2025), so it is an axis too.
AXES = ("cross_section", "length", "duration", "flux", "reservoir")
UNITS = {"cross_section": "nm", "length": "nm", "duration": "ps", "flux": "K",
         "reservoir": "nm"}


@dataclass
class Rung:
    name: str
    spec: NEMDSpec
    axes: list[tuple[str, float]] = field(default_factory=list)   # (axis, value)


def _value(spec: NEMDSpec, axis: str) -> float:
    g = spec.geometry
    if axis == "cross_section":
        return round(spec.snapped_cross_section * 1e9, 4)
    if axis == "length":
        return round(g.bulk_a_length * 1e9, 4)
    if axis == "duration":
        return round(spec.production_steps * spec.timestep_fs * 1e-3, 4)
    if axis == "flux":
        return round(spec.delta_reservoir, 4)
    if axis == "reservoir":
        return round(g.reservoir_thickness * 1e9, 4)
    raise ValueError(axis)


def _vary(base: NEMDSpec, axis: str, value: float) -> NEMDSpec:
    s = copy.deepcopy(base)
    if axis == "cross_section":
        s.geometry.cross_section = value * 1e-9
    elif axis == "length":
        s.geometry.bulk_a_length = value * 1e-9
        s.geometry.bulk_b_length = value * 1e-9
    elif axis == "duration":
        s.production_steps = int(round(value / (s.timestep_fs * 1e-3)))
    elif axis == "flux":
        s.delta_reservoir = value
    elif axis == "reservoir":
        s.geometry.reservoir_thickness = value * 1e-9
    else:
        raise ValueError(axis)
    return s


def _key(spec: NEMDSpec) -> tuple:
    return tuple(_value(spec, a) for a in AXES)


def plan_campaign(base: NEMDSpec, *, cross_sections_nm=(), lengths_nm=(),
                  durations_ps=(), deltas_K=(), reservoirs_nm=()) -> list[Rung]:
    """One-factor-at-a-time ladders about `base`; identical cells run once."""
    rungs: dict[tuple, Rung] = {}

    def add(spec: NEMDSpec, axis: str | None):
        k = _key(spec)
        if k not in rungs:
            name = "r{:02d}_".format(len(rungs)) + "_".join(
                f"{a[0]}{_value(spec, a):g}" for a in AXES)
            rungs[k] = Rung(name, spec)
        if axis and (axis, _value(spec, axis)) not in rungs[k].axes:
            rungs[k].axes.append((axis, _value(spec, axis)))

    for axis, values in (("cross_section", cross_sections_nm), ("length", lengths_nm),
                         ("duration", durations_ps), ("flux", deltas_K),
                         ("reservoir", reservoirs_nm)):
        if values:
            add(base, axis)                 # the base belongs to every ladder
        for v in values:
            add(_vary(base, axis, v), axis)
    return list(rungs.values())


def _atoms(spec: NEMDSpec) -> int:
    from .nemd import ATOMS_PER_CELL
    na = ATOMS_PER_CELL[spec.lattice_style_a] / spec.lattice_a ** 3
    nb = ATOMS_PER_CELL[spec.lattice_style_b] / spec.lattice_b ** 3
    return spec.geometry.estimate_atoms(na, nb)


def _steps(spec: NEMDSpec) -> int:
    from .nemd import relax_steps
    return relax_steps(spec) + spec.total_steps


# Below roughly this many atoms per core, a classical many-body run stops
# scaling: communication dominates and the assumed parallel efficiency is
# fiction. Rungs are given cores in proportion to their size, capped.
ATOMS_PER_CORE = 1000


def cores_for(n_atoms: int, max_cores: int) -> int:
    """Power-of-two core count giving about ATOMS_PER_CORE atoms each."""
    want = max(1, n_atoms // ATOMS_PER_CORE)
    c = 1
    while c * 2 <= min(want, max_cores):
        c *= 2
    return c


def write_campaign(root: Path | str, rungs: list[Rung], potential: Path | str, *,
                   lmp_command: str = "lmp", cores: int = 64, cores_per_node: int = 64,
                   partition: str | None = None, account: str | None = None,
                   modules: tuple[str, ...] = (), flags: FlagSet | None = None) -> dict:
    """Write every rung as a ready-to-submit directory, plus manifest.json.

    Each directory holds nemd.in, the potential, submit.sbatch (Slurm) and
    run.sh (local). Nothing is submitted: submission is the cluster session's
    decision, made after reading the manifest's cost.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    potential = Path(potential)
    flags = flags if flags is not None else FlagSet()
    runner = SlurmRunner()
    entries = []
    total = 0.0
    for r in rungs:
        wd = root / r.name
        wd.mkdir(exist_ok=True)
        (wd / "nemd.in").write_text(build_input_deck(r.spec), encoding="utf-8")
        write_spec(wd, r.spec, "nemd")
        shutil.copy(potential, wd / potential.name)
        for extra in r.spec.extra_files:           # e.g. a MEAM parameter file
            shutil.copy(potential.parent / extra, wd / extra)
        if r.spec.ttm is not None:                 # electron grid for fix ttm/thermal
            from .nemd import ttm_properties
            (wd / "ttm_props.in").write_text(ttm_properties(r.spec), encoding="utf-8")
        n_atoms = _atoms(r.spec)
        rung_cores = cores_for(n_atoms, cores)
        est = estimate_cost(n_atoms, _steps(r.spec), r.spec.pair_style,
                            cores=rung_cores, cores_per_node=cores_per_node, flags=flags)
        res = resources_for(est, cores_per_node=cores_per_node,
                            partition=partition, account=account)
        cmd = f"srun {lmp_command} -in nemd.in -log log.lammps"
        (wd / "submit.sbatch").write_text(
            runner.script(JobSpec(name=r.name, command=cmd, workdir=wd,
                                  resources=res, modules=list(modules))),
            encoding="utf-8")
        (wd / "run.sh").write_text(
            f"#!/bin/bash\ncd \"$(dirname \"$0\")\"\n{lmp_command} -in nemd.in "
            f"-log log.lammps > stdout.log 2> stderr.log\n", encoding="utf-8")
        (wd / "run.sh").chmod(0o755)
        total += est.node_hours
        entries.append({
            "name": r.name, "axes": r.axes,
            "values": {a: _value(r.spec, a) for a in AXES},
            "atoms": est.n_atoms, "steps": est.n_timesteps, "cores": rung_cores,
            "node_hours": round(est.node_hours, 3),
            "walltime_hours": round(res.walltime_hours, 3),
            "spec": _spec_record(r.spec),
        })
    manifest = {"rungs": entries, "total_node_hours": round(total, 2),
                "potential": potential.name, "max_cores_per_rung": cores,
                "core_hours": round(sum(e["cores"] * e["walltime_hours"] / 2.0
                                        for e in entries), 1),
                "flags": [str(f) for f in flags]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _spec_record(spec: NEMDSpec) -> dict:
    g = spec.geometry
    return {
        "geometry": {k: getattr(g, k) for k in ("wall_thickness", "reservoir_thickness",
                                                 "bulk_a_length", "bulk_b_length",
                                                 "cross_section", "bin_width")},
        **{k: getattr(spec, k) for k in (
            "element_a", "element_b", "lattice_a", "lattice_b", "temperature",
            "delta_reservoir", "pair_style", "pair_file", "relax_structure",
            "relax_steps", "timestep_fs", "equilibrate_steps", "steady_steps",
            "production_steps", "profile_every", "profile_frames", "damping_fs",
            "seed", "mass_a", "mass_b", "spot_pressure_GPa", "common_lattice",
            "structure_mode", "data_file", "lattice_style_a", "lattice_style_b",
            "coeff_args", "relax_z", "piston_damp_ps", "spot_pressure_mode",
            "load_steps", "couple_xy")},
        "types_a": list(spec.types_a), "types_b": list(spec.types_b),
        "ttm": (None if spec.ttm is None else
                {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in dataclasses.asdict(spec.ttm).items()}),
        "supercell": list(spec.supercell) if spec.supercell else None,
        "extra_files": list(spec.extra_files),
        "pair_elements": list(spec.pair_elements) if spec.pair_elements else None,
    }


SPEC_FILE = "spec.json"


def write_spec(wd: Path | str, spec: NEMDSpec, kind: str = "nemd", **extra) -> Path:
    """Write the job's spec next to its deck, so the job can be measured later
    without the script that built it (agent.ingest reads it). kind: nemd | emd."""
    path = Path(wd) / SPEC_FILE
    path.write_text(json.dumps({"kind": kind, "spec": _spec_record(spec), **extra}, indent=2),
                    encoding="utf-8")
    return path


def read_spec(wd: Path | str) -> tuple[NEMDSpec, str, dict]:
    """(spec, kind, extra) from a job folder's spec.json."""
    rec = json.loads((Path(wd) / SPEC_FILE).read_text(encoding="utf-8"))
    kind = rec.pop("kind", "nemd")
    spec = spec_from_record(rec.pop("spec"))
    return spec, kind, rec


def spec_from_record(rec: dict) -> NEMDSpec:
    from .nemd import NEMDGeometry
    rec = dict(rec)
    g = NEMDGeometry(**rec.pop("geometry"))
    pe = rec.pop("pair_elements")
    sc = rec.pop("supercell", None)
    ef = rec.pop("extra_files", [])
    ta = tuple(rec.pop("types_a", (1,)))
    tb = tuple(rec.pop("types_b", (2,)))
    ttm = rec.pop("ttm", None)
    if ttm is not None:
        from .ttm import TTMSpec
        ttm = TTMSpec(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ttm.items()})
    return NEMDSpec(geometry=g, pair_elements=tuple(pe) if pe else None,
                    supercell=tuple(sc) if sc else None, extra_files=tuple(ef),
                    types_a=ta, types_b=tb, ttm=ttm, **rec)


@dataclass
class CampaignResult:
    measurements: dict          # rung name -> KapitzaMeasurement
    failed: dict                # rung name -> reason
    verdicts: dict              # axis -> ConvergenceVerdict
    linearity: dict | None
    flags: FlagSet


def collect_campaign(root: Path | str, *, k_expected_a: float | None = None,
                     k_expected_b: float | None = None,
                     tolerance: float = 0.05) -> CampaignResult:
    """Measure every finished rung and judge each axis with its errors."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    flags = FlagSet()
    measurements, failed = {}, {}
    for e in manifest["rungs"]:
        wd = root / e["name"]
        if not (wd / "temperature_profile.dat").is_file():
            failed[e["name"]] = "no output yet"
            continue
        try:
            measurements[e["name"]] = measure_kapitza(
                wd, spec_from_record(e["spec"]), FlagSet(),
                k_expected_a=k_expected_a, k_expected_b=k_expected_b)
        except (OSError, ValueError) as exc:
            failed[e["name"]] = str(exc)

    verdicts, linearity = {}, None
    for axis in AXES:
        pts = [(v, measurements[e["name"]]) for e in manifest["rungs"]
               for a, v in e["axes"] if a == axis and e["name"] in measurements]
        if not pts:
            continue
        if axis == "flux":
            if len(pts) >= 2:
                linearity = check_flux_linearity([m.flux for _, m in pts],
                                                 [m.extraction.delta_T.si for _, m in pts],
                                                 flags)
            continue
        series = ConvergenceSeries(axis, [v for v, _ in pts], [m.h_K for _, m in pts],
                                   UNITS[axis], [m.uncertainty for _, m in pts])
        verdicts[axis] = check_series(series, flags, tolerance=tolerance)
    return CampaignResult(measurements, failed, verdicts, linearity, flags)
