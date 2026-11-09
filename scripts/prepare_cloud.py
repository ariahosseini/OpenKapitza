"""Write the rented-machine job set into one folder, with a costed manifest (KB 42).

    python3 scripts/prepare_cloud.py ~/ok_cloud --potentials ~/lammps-openkapitza/potentials
        [--np 48] [--slots 4] [--core-rate 1.0] [--only NAME ...] [--smoke]

Everything still to run on CPU, regenerated with the current code (stage
checkpoints, interface tally where it applies), in the order a 4-slot machine
should take them (longest first, so the short jobs fill in at the end):

  heavy/sial_pp, heavy/sial_ttm, heavy/sicu_pp   P1-6/P1-14 MEAM Si/Al, Si/Cu
  heavy/sige_incoherent                          P1-16 strain-free 25:24 cell
  heavy/r07_rerun                                P1-3 64 nm per side, steady 400 ps
  followup/arb_hex_q2.4, followup/arb_ehex_q2.4  KB 41 drive gap: imposed flux
                                                 raised to the Langevin runs' 2.4 GW/m^2
  followup/arb_langevin_dT27                     ... and Langevin with a 27 K drive,
                                                 so its jump matches the imposed runs' 24 K
  followup/sige_damp1.0_s2                       KB 41 damping 1.0 ps, second seed
  disorder/mix2, mix4, mix8, step4_p2, step8_p1  S-9 intermixed and stepped Si/Ge

Every deck passes validate_deck before it is written. Then, on the machine:

    python3 -m openkapitza.agent
    > jobs ~/ok_cloud
    > submit all local lmp=~/lammps-openkapitza/build/lmp np=48 slots=4
    > watch

--smoke writes the same decks cut to a few hundred steps per stage, for a
check that every one runs to the end on the machine's LAMMPS (a few minutes).
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from openkapitza.compute.resources import estimate_cost   # noqa: E402
from openkapitza.md.campaign import _atoms, _steps, plan_campaign, write_spec  # noqa: E402
from openkapitza.md.nemd import build_input_deck, ttm_properties, validate_deck  # noqa: E402
from openkapitza.schema import FlagSet                     # noqa: E402

Q_LANGEVIN = 2.4e9      # W/m^2, the weekend Langevin runs' flux (KB 41)
SMOKE_STEPS = dict(relax_steps=200, equilibrate_steps=300, steady_steps=300,
                   production_steps=400, load_steps=None)


def _heavy():
    """(name, spec, issue, note) for the MEAM pairs, the incoherent cell and r07."""
    from plan_campaign import PRESETS, base_spec
    from prepare_cluster import jobs
    want = {"sial_pp": "P1-6/P1-14", "sial_ttm": "P1-14", "sicu_pp": "P1-6",
            "sige_incoherent": "P1-16"}
    out = [(n, s, want[n], "") for n, s, kind in jobs() if n in want]
    p = PRESETS["sige"]
    rungs = plan_campaign(base_spec(p), cross_sections_nm=p["cross_sections_nm"],
                          lengths_nm=p["lengths_nm"], durations_ps=p["durations_ps"],
                          deltas_K=p["deltas_K"], reservoirs_nm=p["reservoirs_nm"])
    r07 = next(r for r in rungs if r.name.startswith("r07_"))
    s = copy.deepcopy(r07.spec)
    s.steady_steps = 400_000          # KB 24: 100 ps was too short for a 128 nm column
    out.append(("r07_rerun", s, "P1-3", f"{r07.name} with steady 400 ps (KB 24)"))
    return out


def _followups():
    """name -> prepare_weekend case tuple."""
    from prepare_weekend import bonded_base, cases, nb_base
    from openkapitza.structure.nonbonded import NonbondedSpec
    lj1 = NonbondedSpec(lj_scale=1.0)

    def imposed_q(mode):
        def f(s):
            s.flux_mode, s.imposed_flux, s.presplit_K = mode, Q_LANGEVIN, 50.0
        return f

    def langevin_small(s):
        cases()["arb_langevin_d0.1"][2](s)
        s.delta_reservoir = 27.0
        s.presplit_K = 20.0

    def damp_seed(s):
        s.damping_fs, s.seed = 1000.0, 54321

    return {
        "arb_hex_q2.4": (nb_base, lj1, imposed_q("heat"), 40, "HEX at the Langevin flux"),
        "arb_ehex_q2.4": (nb_base, lj1, imposed_q("ehex"), 40, "eHEX at the Langevin flux"),
        "arb_langevin_dT27": (nb_base, lj1, langevin_small, 50,
                              "Langevin 0.1 ps, 27 K drive (jump ~ imposed runs')"),
        "sige_damp1.0_s2": (bonded_base, None, damp_seed, 480, "r00, damping 1.0 ps, seed 2"),
    }


def _shorten(spec):
    for k, v in SMOKE_STEPS.items():
        if hasattr(spec, k):
            setattr(spec, k, v)
    spec.relax_scale_with_length = False
    spec.profile_every, spec.profile_frames = 10, 4
    return spec


def _shorten_case(case):
    factory, nb, modify, h, note = case

    def mod(s):
        if modify:
            modify(s)
        _shorten(s)
    return (factory, nb, mod, h, note)


def write_heavy(wd: Path, spec, pots: Path) -> dict:
    deck = build_input_deck(spec)
    problems = validate_deck(deck)
    if problems:
        raise SystemExit(f"{wd.name}: {problems}")
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    (wd / "nemd.in").write_text(deck)
    write_spec(wd, spec, "nemd")
    for f in (spec.pair_file, *spec.extra_files):
        shutil.copy(pots / f, wd / f)
    if spec.ttm is not None:
        (wd / "ttm_props.in").write_text(ttm_properties(spec))
    return {"atoms": int(_atoms(spec)), "steps": int(_steps(spec)),
            "pair_style": spec.pair_style}


def schedule(entries: list[dict], slots: int) -> float:
    """Wall hours of a greedy run, in list order, on `slots` side-by-side slots."""
    free = [0.0] * slots
    for e in entries:
        i = free.index(min(free))
        e["start_h"] = round(free[i], 2)
        free[i] += e["wall_hours"]
    return max(free)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--potentials", type=Path, required=True)
    ap.add_argument("--np", type=int, default=48, help="MPI ranks per job")
    ap.add_argument("--slots", type=int, default=4, help="jobs side by side")
    ap.add_argument("--core-rate", type=float, default=1.0,
                    help="speed of one rank relative to the cost model's core "
                         "(measure it with the smoke run; 1.0 = as estimated)")
    ap.add_argument("--only", nargs="*", help="job names (e.g. heavy/sial_pp)")
    ap.add_argument("--smoke", action="store_true", help="a few hundred steps per stage")
    ap.add_argument("--force", action="store_true",
                    help="rewrite a folder whose jobs were already submitted (loses them)")
    a = ap.parse_args(argv)
    root, pots = a.root.expanduser(), a.potentials.expanduser()
    if (root / "jobs.json").is_file() and not a.force:
        raise SystemExit(f"{root}: jobs were already submitted from here (jobs.json); "
                         "rewriting would delete their output. Use another folder, or --force.")
    root.mkdir(parents=True, exist_ok=True)
    from prepare_disorder import CASES as DISORDER, spec_for, write_case as write_disorder
    from prepare_weekend import write_case as write_weekend
    flags = FlagSet()
    entries = []

    def add(name, e, issue, note=""):
        est = estimate_cost(e["atoms"], e["steps"], e.get("pair_style", "tersoff"),
                            cores=a.np, cores_per_node=a.np * a.slots, flags=flags)
        e.update(name=name, issue=issue, cores=a.np, note=note or e.get("note", ""),
                 wall_hours=round(est.wall_hours / a.core_rate, 2))
        entries.append(e)
        print(f"{name:28} {issue:11} {e['atoms']:8,} atoms {e['steps'] / 1e6:5.2f} Msteps "
              f"~{e['wall_hours']:5.1f} h  {e['note']}")

    def wanted(name):
        return not a.only or name in a.only

    for name, spec, issue, note in _heavy():
        if wanted(f"heavy/{name}"):
            spec = _shorten(copy.deepcopy(spec)) if a.smoke else spec
            add(f"heavy/{name}", write_heavy(root / "heavy" / name, spec, pots), issue, note)
    for name, case in _followups().items():
        if wanted(f"followup/{name}"):
            case = _shorten_case(case) if a.smoke else case
            add(f"followup/{name}", write_weekend(root / "followup" / name, name, case, pots),
                "KB 41")
    for name in DISORDER:
        if wanted(f"disorder/{name}"):
            if a.smoke:
                import prepare_disorder as pd
                orig = pd.spec_for
                pd.spec_for = lambda n: _shorten(orig(n))
                try:
                    e = write_disorder(root / "disorder" / name, name, pots)
                finally:
                    pd.spec_for = orig
            else:
                e = write_disorder(root / "disorder" / name, name, pots)
            add(f"disorder/{name}", e, "S-9")
    for e in entries:                                   # every deck, checked once more
        problems = validate_deck((root / e["name"] / "nemd.in").read_text())
        if problems:
            raise SystemExit(f"{e['name']}: {problems}")
    entries.sort(key=lambda e: -e["wall_hours"])        # longest first
    wall = schedule(entries, a.slots)
    core_h = sum(e["wall_hours"] * a.np for e in entries)
    (root / "manifest.json").write_text(json.dumps(
        {"jobs": entries, "np": a.np, "slots": a.slots, "smoke": a.smoke,
         "wall_hours": round(wall, 1), "core_hours": round(core_h),
         "flags": [str(f) for f in flags]}, indent=2))
    print(f"\n{len(entries)} jobs; on {a.slots} slots x {a.np} ranks about {wall:.1f} h "
          f"wall, {core_h:.0f} core-hours (estimates; --core-rate rescales them).")
    print(f"Then: jobs {root}  /  submit all local lmp=<lmp> np={a.np} slots={a.slots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
