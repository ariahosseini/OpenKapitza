"""Write the Phase 1 convergence campaign as ready-to-submit job directories.

Nothing is submitted. Read the printed cost and manifest.json, then submit
on the cluster (each rung has submit.sbatch) and collect with:

    python3 scripts/plan_campaign.py collect <dir>

Usage:
    python3 scripts/plan_campaign.py write <dir> --potential <file> [--preset sige]
        [--cores 64] [--partition P] [--account A] [--lmp lmp] [--module M ...]
    python3 scripts/plan_campaign.py collect <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openkapitza.md.campaign import collect_campaign, plan_campaign, write_campaign  # noqa: E402
from openkapitza.md.nemd import NEMDGeometry, NEMDSpec  # noqa: E402

# Si/Ge, SiCGe.tersoff. Base cell sits mid-ladder on every axis, so each
# ladder brackets it. Steady-state time: take from find_equilibration_time.py
# on the cluster's first rung; 100 ps is a placeholder sized from the 400 ps
# local run and must be confirmed.
PRESETS = {
    "sige": dict(
        base=dict(cross_nm=4.3, length_nm=16.0, production_ps=500.0, delta_K=40.0,
                  steady_ps=100.0),
        cross_sections_nm=(2.2, 3.3, 5.4, 6.5),
        lengths_nm=(8.0, 32.0, 64.0),
        durations_ps=(250.0, 1000.0),
        deltas_K=(20.0, 60.0),
        reservoirs_nm=(1.0, 4.0),           # base 2.0; Wei et al. 2025
        k_expected=(148.0, 60.0),
    ),
}


def base_spec(p: dict) -> NEMDSpec:
    b = p["base"]
    g = NEMDGeometry(wall_thickness=1.0e-9, reservoir_thickness=2.0e-9,
                     bulk_a_length=b["length_nm"] * 1e-9, bulk_b_length=b["length_nm"] * 1e-9,
                     cross_section=b["cross_nm"] * 1e-9, bin_width=0.5e-9)
    return NEMDSpec(
        geometry=g, element_a="Si", element_b="Ge", lattice_a=5.431e-10,
        lattice_b=5.658e-10, temperature=300.0, delta_reservoir=b["delta_K"],
        pair_style="tersoff", pair_file="SiCGe.tersoff", pair_elements=("Si(D)", "Ge"),
        relax_structure=True, relax_steps=50_000, equilibrate_steps=100_000,
        steady_steps=int(b["steady_ps"] * 1000), production_steps=int(b["production_ps"] * 1000),
        profile_every=100, profile_frames=8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("dir", type=Path)
    w.add_argument("--potential", required=True, type=Path)
    w.add_argument("--preset", default="sige", choices=sorted(PRESETS))
    w.add_argument("--cores", type=int, default=64)
    w.add_argument("--cores-per-node", type=int, default=64)
    w.add_argument("--partition")
    w.add_argument("--account")
    w.add_argument("--lmp", default="lmp")
    w.add_argument("--module", action="append", default=[])
    c = sub.add_parser("collect")
    c.add_argument("dir", type=Path)
    c.add_argument("--preset", default="sige", choices=sorted(PRESETS))
    a = ap.parse_args(argv)

    p = PRESETS[a.preset]
    if a.cmd == "write":
        rungs = plan_campaign(base_spec(p), cross_sections_nm=p["cross_sections_nm"],
                              lengths_nm=p["lengths_nm"], durations_ps=p["durations_ps"],
                              deltas_K=p["deltas_K"], reservoirs_nm=p["reservoirs_nm"])
        m = write_campaign(a.dir, rungs, a.potential, lmp_command=a.lmp, cores=a.cores,
                           cores_per_node=a.cores_per_node, partition=a.partition,
                           account=a.account, modules=tuple(a.module))
        print(f"{'rung':30} {'axes':34} {'atoms':>8} {'cores':>5} {'node-h':>7} {'wall-h':>7}")
        for e in m["rungs"]:
            axes = ",".join(f"{ax}" for ax, _ in e["axes"])
            print(f"{e['name']:30} {axes:34} {e['atoms']:8,} {e['cores']:5d} "
                  f"{e['node_hours']:7.2f} {e['walltime_hours']:7.2f}")
        print(f"\n{len(m['rungs'])} rungs, {m['total_node_hours']:.1f} node-hours "
              f"(walltime requests include a 2x safety factor), up to {a.cores} cores "
              f"per rung. Written to {a.dir}; nothing submitted.")
        for f in m["flags"]:
            print(f)
        return 0

    res = collect_campaign(a.dir, k_expected_a=p["k_expected"][0],
                           k_expected_b=p["k_expected"][1])
    for name, m in sorted(res.measurements.items()):
        print(f"{name:44} h_K {m.h_K / 1e6:7.1f} +/- {m.uncertainty / 1e6:5.1f} MW/m2K")
    for name, why in sorted(res.failed.items()):
        print(f"{name:44} NOT MEASURED: {why}")
    for axis, v in res.verdicts.items():
        state = ("converged" if v.converged else
                 "unresolved" if v.detail.get("unresolved") else "NOT converged")
        print(f"axis {axis:14} {state:13} last change {v.last_change * 100:.1f} %")
    if res.linearity:
        print(f"flux linearity: {'linear' if res.linearity['linear'] else 'NOT linear'}, "
              f"h_K from slope {res.linearity['h_K_from_slope'] / 1e6:.1f} MW/m2K")
    print(res.flags.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
