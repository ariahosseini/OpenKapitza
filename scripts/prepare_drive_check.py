"""Drive-free check of the non-bonded LJ x1, 1 GPa contact (S-5, KB 44).

    python3 scripts/prepare_drive_check.py ~/ok_jobs/drive_check \
        --potentials ~/lammps-openkapitza/potentials [--ns 20] [--smoke]

Langevin reservoirs read h_K = 66-68 on this cell and imposed flux (HEX,
eHEX) 37-41, whatever the drive amplitude (KB 41, 43). This runs the same
cell with no drive at all: the equilibrium temperature-fluctuation method
(md/emd_kapitza.py), at two slab lengths.

Each job is the NEMD deck up to the end of equilibration (same structure, LJ
contact, 1 GPa piston relaxation, walls then frozen), followed by NVE: the
two slab energies fluctuate and relax through the interface with time
tau = C_r / (A h). Slabs are short (2 and 4 nm per side) because tau grows
with slab length and the series must span hundreds of tau; on this weak
contact the interface resistance (1/h ~ 2e-8 m^2K/W) is ~100x a slab's
(L/k), so the decay is the interface's.

How it decides (scripts/analyze_drive_check.py): the equilibrium conductance
is a lower bound on a correctly measured NEMD conductance (NEMD/EMD >= 1),
and close to 1 for a low-transmission contact like this one. An EMD value
near 40 supports the imposed-flux level; one well above 40 (e.g. near 60)
would mean the imposed-flux runs read low.
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

from openkapitza.md.campaign import write_spec                         # noqa: E402
from openkapitza.md.emd_kapitza import EMDSpec, build_emd_deck         # noqa: E402
from openkapitza.md.nemd import NEMDGeometry, validate_deck            # noqa: E402
from openkapitza.structure.nonbonded import NonbondedSpec, build_data, make_spec  # noqa: E402

SLABS_NM = (2.0, 4.0)
SMOKE = EMDSpec(settle_ps=0.2, production_ps=0.5, sample_fs=10.0)


def spec_for(slab_nm: float, *, smoke: bool = False):
    from prepare_nonbonded import base
    b = base()
    b.spot_pressure_GPa = 1.0
    b.interface_tally = False                  # no flux to tally at equilibrium
    b.flux_mode = "langevin"                   # only the deck up to the drive is used
    g = b.geometry
    b.geometry = NEMDGeometry(wall_thickness=g.wall_thickness, reservoir_thickness=0.5e-9,
                              bulk_a_length=slab_nm * 1e-9, bulk_b_length=slab_nm * 1e-9,
                              cross_section=g.cross_section, bin_width=g.bin_width)
    if smoke:
        from prepare_cloud import _shorten
        _shorten(b)
    return make_spec(b, NonbondedSpec(lj_scale=1.0))


def write(wd: Path, slab_nm: float, emd: EMDSpec, pots: Path, smoke: bool) -> dict:
    spec = spec_for(slab_nm, smoke=smoke)
    deck = build_emd_deck(spec, emd)
    problems = validate_deck(deck)
    if problems:
        raise SystemExit(f"{wd.name}: {problems}")
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    data, summary = build_data(spec, NonbondedSpec(lj_scale=1.0))
    (wd / "interface.data").write_text(data)
    (wd / "nemd.in").write_text(deck)
    write_spec(wd, spec, "emd", nonbonded=summary,
               emd={"settle_ps": emd.settle_ps, "production_ps": emd.production_ps,
                    "sample_fs": emd.sample_fs})
    shutil.copy(pots / spec.pair_file, wd / spec.pair_file)
    atoms = int(summary.get("atoms", 0)) if isinstance(summary, dict) else 0
    steps = int((emd.settle_ps + emd.production_ps) * 1000 / spec.timestep_fs)
    return {"atoms": atoms, "steps": steps, "slab_nm": slab_nm,
            "production_ns": emd.production_ps / 1000}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--potentials", type=Path, required=True)
    ap.add_argument("--ns", type=float, default=20.0, help="production length, ns")
    ap.add_argument("--rate", type=float, default=4.5e6,
                    help="atom-steps/s of the target machine (M3 Pro, 5 ranks: 4.5e6, "
                         "measured on this cell, KB 43)")
    ap.add_argument("--smoke", action="store_true", help="a few hundred steps")
    a = ap.parse_args(argv)
    root, pots = a.root.expanduser(), a.potentials.expanduser()
    if (root / "jobs.json").is_file():
        raise SystemExit(f"{root}: jobs were already submitted from here")
    root.mkdir(parents=True, exist_ok=True)
    emd = SMOKE if a.smoke else EMDSpec(settle_ps=20.0, production_ps=a.ns * 1000,
                                        sample_fs=10.0)
    jobs = []
    for slab in SLABS_NM:
        name = f"emd_slab{slab:g}nm"
        e = write(root / name, slab, emd, pots, a.smoke)
        relax = 50_000 + 100_000                       # relax + equilibrate, as the NEMD deck
        hours = e["atoms"] * (e["steps"] + relax) / a.rate / 3600
        jobs.append({"name": name, **e, "est_hours": round(hours, 1),
                     "note": f"equilibrium fluctuation, {slab:g} nm per side"})
        print(f"{name:16} {e['atoms']:6,} atoms  {e['production_ns']:5.1f} ns  ~{hours:4.1f} h")
    (root / "manifest.json").write_text(json.dumps({"jobs": jobs, "smoke": a.smoke}, indent=2))
    print(f"\n{len(jobs)} jobs, ~{sum(j['est_hours'] for j in jobs):.0f} h in sequence. "
          f"Then: jobs {root}  /  submit all local lmp=<lmp> np=5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
