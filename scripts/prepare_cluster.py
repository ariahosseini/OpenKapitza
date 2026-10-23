"""Write every cluster job as a ready-to-submit directory, with one costed manifest.

Cluster day should start with reading a cost table and submitting, not with
setting up. This writes, under <root>:

  campaign/        P1-3 Si/Ge convergence campaign (plan_campaign preset)
  sige_spot_*/     P1-9/P1-10 Si/Ge at 2 GPa spot pressure, stress and strain modes
  sige_emd/        P1-8 equilibrium cross-check (short slabs, >= 500 tau)
  sial_pp/, sial_ttm/   P1-6/P1-14 Si/Al phonon-only and two-temperature pair
  sicu_pp/         P1-6 Si/Cu phonon-only
  sio2_strong/, sio2_weak/   P2-3/P2-6 Si/a-SiO2 on the bundled cells
  manifest.json    every job: atoms, steps, cores, node-hours, walltime
  commands.txt     jobs that are commands rather than decks (P2-5 quench)

Nothing is submitted. Every NEMD deck passes validate_deck before it is
written; a problem stops the script.

    python3 scripts/prepare_cluster.py <root> --potentials <lammps>/potentials
        [--lmp lmp] [--cores 64] [--cores-per-node 64] [--partition P]
        [--account A] [--module M ...]
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from openkapitza.compute.resources import estimate_cost, resources_for  # noqa: E402
from openkapitza.compute.runner import JobSpec, SlurmRunner  # noqa: E402
from openkapitza.md.campaign import _atoms, _steps, cores_for, plan_campaign, write_campaign  # noqa: E402
from openkapitza.md.emd_kapitza import EMDSpec, build_emd_deck  # noqa: E402
from openkapitza.md.nemd import (  # noqa: E402
    NEMDGeometry, NEMDSpec, build_input_deck, ttm_properties, validate_deck,
)
from openkapitza.md.ttm import TTMSpec  # noqa: E402
from openkapitza.schema import FlagSet  # noqa: E402

MEAM = dict(pair_style="meam", pair_file="library.meam",
            extra_files=("AlSiMgCuFe.meam",))
MEAM_COEFF = "library.meam AlS SiS MgS CuS FeS AlSiMgCuFe.meam SiS {b}S"


def _geometry(cross_nm, a_nm, b_nm, reservoir_nm=2.0):
    return NEMDGeometry(wall_thickness=1.0e-9, reservoir_thickness=reservoir_nm * 1e-9,
                        bulk_a_length=a_nm * 1e-9, bulk_b_length=b_nm * 1e-9,
                        cross_section=cross_nm * 1e-9, bin_width=0.5e-9)


def jobs() -> list[tuple[str, NEMDSpec, str]]:
    """(name, spec, kind) for every non-campaign NEMD-family job."""
    from plan_campaign import PRESETS, base_spec
    sige = base_spec(PRESETS["sige"])
    out = []
    for mode in ("stress", "strain"):
        s = copy.deepcopy(sige)
        s.spot_pressure_GPa, s.spot_pressure_mode = 2.0, mode
        out.append((f"sige_spot_{mode}", s, "nemd"))
    emd = copy.deepcopy(sige)
    emd.geometry = _geometry(4.3, 2.0, 2.0, reservoir_nm=0.5)
    out.append(("sige_emd", emd, "emd"))
    common = dict(temperature=300.0, relax_steps=50_000, equilibrate_steps=100_000,
                  steady_steps=500_000, production_steps=1_000_000, profile_every=100,
                  profile_frames=8, lattice_style_a="diamond", lattice_style_b="fcc",
                  mass_a=28.0855, **MEAM)
    for name, b, lat_b, mass_b, b_nm, ttm in (
            ("sial_pp", "Al", 4.0495e-10, 26.9815, 30.0, None),
            ("sial_ttm", "Al", 4.0495e-10, 26.9815, 30.0, TTMSpec("Al")),
            ("sicu_pp", "Cu", 3.6149e-10, 63.546, 30.0, None)):
        out.append((name, NEMDSpec(
            geometry=_geometry(4.3, 16.0, b_nm), element_a="Si", element_b=b,
            lattice_a=5.4309e-10, lattice_b=lat_b, mass_b=mass_b,
            coeff_args=MEAM_COEFF.format(b=b), ttm=ttm, **common), "nemd"))
    return out


def _write(root: Path, name: str, deck: str, files: list[tuple[Path, str]], n_atoms: int,
           steps: int, pair_style: str, a, flags: FlagSet, extra: dict | None = None) -> dict:
    wd = root / name
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "nemd.in").write_text(deck, encoding="utf-8")
    for src, dst in files:
        shutil.copy(src, wd / dst)
    for k, v in (extra or {}).items():
        (wd / k).write_text(v, encoding="utf-8")
    cores = cores_for(n_atoms, a.cores)
    est = estimate_cost(n_atoms, steps, pair_style, cores=cores,
                        cores_per_node=a.cores_per_node, flags=flags)
    res = resources_for(est, cores_per_node=a.cores_per_node, partition=a.partition,
                        account=a.account)
    cmd = f"srun {a.lmp} -in nemd.in -log log.lammps"
    (wd / "submit.sbatch").write_text(SlurmRunner().script(JobSpec(
        name=name, command=cmd, workdir=wd, resources=res, modules=list(a.module))))
    (wd / "run.sh").write_text(f"#!/bin/bash\ncd \"$(dirname \"$0\")\"\n{a.lmp} -in nemd.in "
                               f"-log log.lammps > stdout.log 2> stderr.log\n")
    (wd / "run.sh").chmod(0o755)
    return {"name": name, "atoms": n_atoms, "steps": steps, "cores": cores,
            "node_hours": round(est.node_hours, 3),
            "walltime_hours": round(res.walltime_hours, 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--potentials", required=True, type=Path)
    ap.add_argument("--lmp", default="lmp")
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--cores-per-node", type=int, default=64)
    ap.add_argument("--partition")
    ap.add_argument("--account")
    ap.add_argument("--module", action="append", default=[])
    a = ap.parse_args(argv)
    root, pot = a.root, a.potentials
    root.mkdir(parents=True, exist_ok=True)
    flags = FlagSet()
    entries = []

    # P1-3 campaign
    from plan_campaign import PRESETS, base_spec
    p = PRESETS["sige"]
    rungs = plan_campaign(base_spec(p), cross_sections_nm=p["cross_sections_nm"],
                          lengths_nm=p["lengths_nm"], durations_ps=p["durations_ps"],
                          deltas_K=p["deltas_K"], reservoirs_nm=p["reservoirs_nm"])
    for r in rungs:
        problems = validate_deck(build_input_deck(r.spec))
        if problems:
            raise SystemExit(f"campaign rung {r.name}: {problems}")
    cm = write_campaign(root / "campaign", rungs, pot / "SiCGe.tersoff", lmp_command=a.lmp,
                        cores=a.cores, cores_per_node=a.cores_per_node, partition=a.partition,
                        account=a.account, modules=tuple(a.module), flags=flags)
    for e in cm["rungs"]:
        entries.append({k: e[k] for k in ("name", "atoms", "steps", "cores", "node_hours",
                                          "walltime_hours")} | {"name": "campaign/" + e["name"],
                                                                "issue": "P1-3"})

    issues = {"sige_spot_stress": "P1-9/P1-10", "sige_spot_strain": "P1-9/P1-10",
              "sige_emd": "P1-8", "sial_pp": "P1-6/P1-14", "sial_ttm": "P1-14", "sicu_pp": "P1-6"}
    for name, spec, kind in jobs():
        files = [(pot / spec.pair_file, spec.pair_file)] + [(pot / f, f) for f in spec.extra_files]
        extra = {}
        if kind == "emd":
            emd = EMDSpec(settle_ps=50.0, production_ps=15_000.0)   # >= 500 tau at tau ~ 25 ps
            deck = build_emd_deck(spec, emd)
            steps = (spec.relax_steps + spec.equilibrate_steps
                     + int((emd.settle_ps + emd.production_ps) * 1000 / spec.timestep_fs))
        else:
            deck = build_input_deck(spec)
            problems = validate_deck(deck)
            if problems:
                raise SystemExit(f"{name}: {problems}")
            steps = _steps(spec)
            if spec.ttm is not None:
                extra["ttm_props.in"] = ttm_properties(spec)
        e = _write(root, name, deck, files, _atoms(spec), steps, spec.pair_style, a, flags, extra)
        e["issue"] = issues[name]
        entries.append(e)

    # P2-3 / P2-6: the bundled Si/a-SiO2 cells and decks
    data = REPO / "data" / "structures"
    from build_si_sio2 import geometry as sio2_geometry, nemd_spec as sio2_spec
    for cls in ("strong", "weak"):
        spec = sio2_spec(sio2_geometry(4.0, 5))
        deck = (data / f"Si100_aSiO2_{cls}_4nm.nemd.in").read_text()
        n_atoms = int(next(ln.split()[0] for ln in
                           (data / f"Si100_aSiO2_{cls}_4nm.data").read_text().splitlines()
                           if ln.strip().endswith("atoms")))
        e = _write(root, f"sio2_{cls}", deck,
                   [(data / f"Si100_aSiO2_{cls}_4nm.data", "interface.data"),
                    (pot / "SiO.tersoff", "SiO.tersoff")],
                   n_atoms, _steps(spec), "tersoff", a, flags)
        e["issue"] = "P2-3/P2-6"
        entries.append(e)

    (root / "commands.txt").write_text(
        "# P2-5 / P2-2: a-SiO2 at 1e12 K/s, then both bonding classes on that glass.\n"
        "# The quench checkpoints per segment; resubmit the same command to resume.\n"
        f"python3 scripts/build_si_sio2.py --lmp {a.lmp} --potentials {pot} --out sio2_1e12 "
        "--quench-rate 1e12 --bonding strong --bulk-nm 8\n"
        f"python3 scripts/build_si_sio2.py --lmp {a.lmp} --potentials {pot} --out sio2_1e12_weak "
        "--glass sio2_1e12/quench/glass.data --bonding weak --bulk-nm 8\n")
    total = sum(e["node_hours"] for e in entries)
    core_h = sum(e["cores"] * e["walltime_hours"] / 2.0 for e in entries)
    (root / "manifest.json").write_text(json.dumps(
        {"jobs": entries, "total_node_hours": round(total, 2), "core_hours": round(core_h, 1),
         "flags": [str(f) for f in flags]}, indent=2))
    print(f"{'job':48} {'issue':12} {'atoms':>9} {'cores':>5} {'node-h':>8} {'wall-h':>7}")
    for e in entries:
        print(f"{e['name']:48} {e['issue']:12} {e['atoms']:9,} {e['cores']:5d} "
              f"{e['node_hours']:8.2f} {e['walltime_hours']:7.2f}")
    print(f"\n{len(entries)} jobs, {total:.1f} node-hours, ~{core_h:.0f} core-hours "
          f"(walltimes include a 2x safety factor). Nothing submitted. Also: commands.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
