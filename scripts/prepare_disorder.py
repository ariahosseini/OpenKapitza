"""Write the disordered Si/Ge interface jobs (open issue S-9, KB 30, 33).

    python3 scripts/prepare_disorder.py ~/ok_jobs --potentials ~/lammps-openkapitza/potentials

The campaign's base cell (16 nm per side, 4.3 nm, Tersoff, Langevin 0.1 ps;
r00 = 483 +/- 20 MW/m^2K for the sharp interface) with the n (100) layers
nearest the plane intermixed, a linear Ge fraction across them. Real Si/Ge
interfaces are graded over a few layers; Green's-function work (Tian,
Esfarjani & Chen 2012; Song & Chen) finds mixing can RAISE transmission,
with a maximum near 4 mixed layers. Also two STEPPED interfaces (a square
wave in x: terraces of 4 or 8 (100) layers' height, 1 or 2 periods across
the 4.3 nm cell). Writes ROOT/disorder/<case>/ and adds them to
ROOT/manifest.json; in the agent: 'jobs ROOT', 'submit disorder local'.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from openkapitza.md.campaign import write_spec                 # noqa: E402
from openkapitza.md.nemd import build_input_deck, relax_steps   # noqa: E402

CASES = {"mix2": dict(intermix_layers=2), "mix4": dict(intermix_layers=4),
         "mix8": dict(intermix_layers=8),
         "step4_p2": dict(interface_steps=(4, 2)),      # 0.54 nm high, 1.09 nm terraces
         "step8_p1": dict(interface_steps=(8, 1))}      # 1.09 nm high, 2.17 nm terraces


def spec_for(case: str):
    from plan_campaign import PRESETS, base_spec
    s = base_spec(PRESETS["sige"])
    for k, v in CASES[case].items():
        setattr(s, k, v)
    return s


def write_case(wd: Path, name: str, pots: Path) -> dict:
    """Write one case into `wd` (replacing it); its manifest fields, without the name."""
    s = spec_for(name)
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    (wd / "nemd.in").write_text(build_input_deck(s))
    write_spec(wd, s, "nemd", note=f"Si/Ge disordered interface: {CASES[name]}")
    shutil.copy(pots / s.pair_file, wd)
    steps = relax_steps(s) + s.equilibrate_steps + s.steady_steps + s.production_steps
    return {"atoms": int(s.geometry.estimate_atoms(4.99e28, 4.99e28)), "steps": steps,
            "pair_style": s.pair_style}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--potentials", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="case names (default: all)")
    a = ap.parse_args(argv)
    root = a.root.expanduser()
    jobs = []
    for name in CASES:
        if a.only and name not in a.only:
            continue
        e = write_case(root / "disorder" / name, name, a.potentials.expanduser())
        jobs.append({"name": f"disorder/{name}", **e})
        atoms, steps = e["atoms"], e["steps"]
        print(f"{name:9} {int(atoms):7,} atoms  {steps / 1e6:.2f} Msteps")
    man = root / "manifest.json"
    data = json.loads(man.read_text()) if man.is_file() else {"jobs": []}
    names = {j["name"] for j in jobs}
    data["jobs"] = [j for j in data.get("jobs", []) if j["name"] not in names] + jobs
    man.write_text(json.dumps(data, indent=2))
    print(f"\n{len(jobs)} jobs under {root / 'disorder'}; in the agent: jobs {root}, "
          f"then submit disorder local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
