"""Write the intermixed Si/Ge interface jobs (open issue S-9, KB 30).

    python3 scripts/prepare_intermix.py ~/ok_jobs --potentials ~/lammps-openkapitza/potentials

The campaign's base cell (16 nm per side, 4.3 nm, Tersoff, Langevin 0.1 ps;
r00 = 483 +/- 20 MW/m^2K for the sharp interface) with the n (100) layers
nearest the plane intermixed, a linear Ge fraction across them. Real Si/Ge
interfaces are graded over a few layers; Green's-function work (Tian,
Esfarjani & Chen 2012; Song & Chen) finds mixing can RAISE transmission,
with a maximum near 4 mixed layers. Writes ROOT/intermix/mix<n>/ and adds
them to ROOT/manifest.json; in the agent: 'jobs ROOT', 'submit intermix local'.
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

LAYERS = (2, 4, 8)


def spec_for(n: int):
    from plan_campaign import PRESETS, base_spec
    s = base_spec(PRESETS["sige"])
    s.intermix_layers = n
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--potentials", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=list(LAYERS))
    a = ap.parse_args(argv)
    root = a.root.expanduser()
    jobs = []
    for n in a.layers:
        s = spec_for(n)
        wd = root / "intermix" / f"mix{n}"
        if wd.exists():
            shutil.rmtree(wd)
        wd.mkdir(parents=True)
        (wd / "nemd.in").write_text(build_input_deck(s))
        write_spec(wd, s, "nemd", note=f"Si/Ge, {n} intermixed (100) layers")
        shutil.copy(a.potentials.expanduser() / s.pair_file, wd)
        steps = relax_steps(s) + s.equilibrate_steps + s.steady_steps + s.production_steps
        atoms = s.geometry.estimate_atoms(4.99e28, 4.99e28)
        jobs.append({"name": f"intermix/mix{n}", "atoms": int(atoms), "steps": steps})
        print(f"mix{n:<3} {int(atoms):7,} atoms  {steps / 1e6:.2f} Msteps")
    man = root / "manifest.json"
    data = json.loads(man.read_text()) if man.is_file() else {"jobs": []}
    names = {j["name"] for j in jobs}
    data["jobs"] = [j for j in data.get("jobs", []) if j["name"] not in names] + jobs
    man.write_text(json.dumps(data, indent=2))
    print(f"\n{len(jobs)} jobs under {root / 'intermix'}; in the agent: jobs {root}, "
          f"then submit intermix local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
