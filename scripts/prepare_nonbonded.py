"""Write the non-bonded / gas-layer sweep (tracker S-5, S-6; KB 19) as jobs.

    python3 scripts/prepare_nonbonded.py ~/ok_jobs --potentials ~/lammps-openkapitza/potentials

Writes ROOT/nonbonded/<case>/ (nemd.in, interface.data, spec.json, the
potential file) and adds the cases to ROOT/manifest.json as nonbonded/<case>,
so the agent's job loop sees them ('jobs ROOT', then 'submit nonbonded local').

Every case is the same Si/Ge column (4.3 nm cross-section, 8 nm per side)
driven by an IMPOSED flux (fix heat), because the reservoir-tally flux is too
noisy at low h. The flux is chosen per case from a guess of h so the interface
jump is ~20 K, and each side is equilibrated at its expected temperature
(presplit) so the slow C*dT/2q transient is skipped. A guess off by 2x gives a
10-40 K jump, which is fine; the measured h does not use the guess.

Cases:
  bonded_ref                       the bonded interface (the anchor), LANGEVIN: imposed
                                   flux under-reads a strong interface in a short cell
                                   (KB 19, check 2), and at its flux the tally noise is small
  lj1_p1_langevin                  LJ x1 at 1 GPa with Langevin reservoirs and 2 ns of
                                   production: the method cross-check at this cell size
  lj{0.25,1,4}_p{0,0.5,1,2}        LJ-only contact, strength x load
  {Ar,N2,He}_c{1,0.5}_p1           a gas monolayer (full / half), 1 GPa
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

from openkapitza.md.campaign import write_spec                      # noqa: E402
from openkapitza.md.nemd import NEMDGeometry, build_input_deck, relax_steps   # noqa: E402
from openkapitza.structure.nonbonded import NonbondedSpec, write_job   # noqa: E402

JUMP_K = 20.0
# h guesses (MW/m^2K) from the first look (KB 19), LJ x1: by contact load
H_LJ1 = {0.0: 16, 0.5: 50, 1.0: 87, 2.0: 150}
H_GAS = {"Ar": 46, "N2": 46, "He": 30}
H_BONDED = 480


def base():
    from plan_campaign import PRESETS, base_spec
    s = base_spec(PRESETS["sige"])
    g = s.geometry
    s.geometry = NEMDGeometry(wall_thickness=g.wall_thickness,
                              reservoir_thickness=g.reservoir_thickness,
                              bulk_a_length=8e-9, bulk_b_length=8e-9, cross_section=4.3448e-9,
                              bin_width=0.5e-9)
    for k, v in dict(relax_steps=50_000, equilibrate_steps=100_000, steady_steps=300_000,
                     production_steps=1_000_000, profile_every=100, profile_frames=8).items():
        setattr(s, k, v)
    s.flux_mode = "heat"
    return s


def cases():
    out = {"bonded_ref": (None, None, H_BONDED),
           "lj1_p1_langevin": (NonbondedSpec(lj_scale=1.0), 1.0, H_LJ1[1.0])}
    for scale in (0.25, 1.0, 4.0):
        for p in (0.0, 0.5, 1.0, 2.0):
            h = min(H_BONDED, H_LJ1[p] * scale ** 0.75)
            out[f"lj{scale:g}_p{p:g}"] = (NonbondedSpec(lj_scale=scale), p or None, h)
    for gas, h in H_GAS.items():
        for cov in (1.0, 0.5):
            out[f"{gas}_c{cov:g}_p1"] = (NonbondedSpec(gas=gas, coverage=cov), 1.0, h)
    return out


def total_steps(spec) -> int:
    return relax_steps(spec) + spec.equilibrate_steps + spec.steady_steps + spec.production_steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="job root holding manifest.json (e.g. ~/ok_jobs)")
    ap.add_argument("--potentials", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="case names to write (default: all)")
    a = ap.parse_args(argv)
    root = a.root.expanduser()
    out = root / "nonbonded"
    jobs = []
    for name, (nb, p, h_guess) in cases().items():
        if a.only and name not in a.only:
            continue
        spec = copy.deepcopy(base())
        spec.spot_pressure_GPa = p
        if name in ("bonded_ref", "lj1_p1_langevin"):
            spec.flux_mode, spec.imposed_flux = "langevin", None
            if name == "lj1_p1_langevin":
                spec.production_steps = 2_000_000
                spec.presplit_K = 30.0      # most of the 40 K drive drops at the plane
        else:
            spec.imposed_flux = h_guess * 1e6 * JUMP_K
            spec.presplit_K = JUMP_K
        wd = out / name
        if wd.exists():
            shutil.rmtree(wd)
        wd.mkdir(parents=True)
        if nb is None:
            (wd / "nemd.in").write_text(build_input_deck(spec))
            write_spec(wd, spec, "nemd", h_guess_MW=h_guess)
            shutil.copy(a.potentials.expanduser() / spec.pair_file, wd)
            atoms = spec.geometry.estimate_atoms(4.99e28, 4.99e28)
        else:
            info = write_job(wd, spec, nb, a.potentials.expanduser())
            atoms = info["atoms"]
        jobs.append({"name": f"nonbonded/{name}", "atoms": int(atoms), "steps": total_steps(spec),
                     "flux_mode": spec.flux_mode,
                     "imposed_flux_W_m2": spec.imposed_flux, "h_guess_MW": round(h_guess, 1)})
        drive = (f"q = {spec.imposed_flux:.3g} W/m^2" if spec.flux_mode == "heat"
                 else "Langevin 320/280 K")
        print(f"{name:16} {atoms:7,} atoms  {drive:22} (h guess {h_guess:.0f} MW/m^2K)")
    man = root / "manifest.json"
    data = json.loads(man.read_text()) if man.is_file() else {"jobs": []}
    names = {j["name"] for j in jobs}
    data["jobs"] = [j for j in data.get("jobs", []) if j["name"] not in names] + jobs
    man.write_text(json.dumps(data, indent=2))
    print(f"\n{len(jobs)} jobs under {out}; added to {man}. In the agent: jobs {root}, "
          f"then submit nonbonded local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
