"""Write the post-rebuild job set (KB 26, 28; tracker H-4, P1-16, P1-17, S-6).

    python3 scripts/prepare_weekend.py ~/ok_jobs --potentials ~/lammps-openkapitza/potentials

Needs a LAMMPS with RIGID (fix ehex) and TALLY (compute heat/flux/virial/tally).
Writes ROOT/weekend/<case>/ and adds the cases to ROOT/manifest.json as
weekend/<case>; in the agent: 'jobs ROOT', then 'submit weekend local'.
Cases run in the order listed: the method arbitration first.

A. Method arbitration, LJ x1 at 1 GPa (the 20-case sweep's cell). Every case
   also records the direct interfacial heat flow (interface_flux.dat), an
   independent flux that does not depend on how the gradient is driven.
     arb_langevin_d0.1   Langevin reservoirs, damping 0.1 ps (our default)
     arb_langevin_d0.5   Langevin reservoirs, damping 0.5 ps (P1-17)
     arb_ehex            imposed flux through fix ehex (drift-corrected)
     arb_hex             the same flux through fix heat (the sweep's drive)
B. Strain and damping on the bonded Si/Ge base cell (16 nm, campaign r00):
     sige_ge_rescaled    Ge potential rescaled to a_Si: no misfit strain (P1-16)
     sige_damp0.5        Langevin damping 0.5 ps (P1-17; r00 is 0.1 ps)
     sige_damp1.0        Langevin damping 1.0 ps
C. The He anomaly (KB 26), eHEX with the tally, 1 GPa, full monolayer:
     ar_ehex, he_ehex    the two gases again under eHEX (control and subject)
     he_mass40           He's LJ with argon's mass: is it the light, fast atom?
     he_nodirect         He layer with the Si-Ge LJ switched off: is it the
     ar_nodirect           direct Si-Ge tail reaching across the thinner layer?
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

from openkapitza.md.campaign import write_spec                       # noqa: E402
from openkapitza.md.nemd import build_input_deck, relax_steps         # noqa: E402
from openkapitza.md.rescale_tersoff import write_rescaled            # noqa: E402
from openkapitza.structure.nonbonded import NonbondedSpec, write_job  # noqa: E402

JUMP_K = 20.0
RESCALED = "SiGe_Ge-rescaled.tersoff"


def nb_base():
    from prepare_nonbonded import base
    s = base()
    s.interface_tally = True
    return s


def bonded_base():
    from plan_campaign import PRESETS, base_spec
    return base_spec(PRESETS["sige"])


def imposed(h_guess_MW: float, mode: str = "ehex"):
    def f(s):
        s.flux_mode, s.imposed_flux, s.presplit_K = mode, h_guess_MW * 1e6 * JUMP_K, JUMP_K
    return f


def langevin(damping_fs: float):
    def f(s):
        s.flux_mode, s.imposed_flux = "langevin", None
        s.damping_fs = damping_fs
        s.production_steps = 2_000_000
        s.presplit_K = 30.0          # most of the 40 K drive drops at the plane
    return f


def damping(fs: float):
    def f(s):
        s.damping_fs = fs
    return f


def cases():
    """name -> (base factory, NonbondedSpec | None, spec modifier, h guess MW/m^2K, note)."""
    lj1 = NonbondedSpec(lj_scale=1.0)
    out = {
        "arb_langevin_d0.1": (nb_base, lj1, langevin(100.0), 50, "Langevin 0.1 ps + tally"),
        "arb_langevin_d0.5": (nb_base, lj1, langevin(500.0), 50, "Langevin 0.5 ps + tally"),
        "arb_ehex": (nb_base, lj1, imposed(50, "ehex"), 50, "eHEX + tally"),
        "arb_hex": (nb_base, lj1, imposed(50, "heat"), 50, "HEX + tally"),
        "sige_ge_rescaled": (bonded_base, None, None, 480, "Ge rescaled to a_Si"),
        "sige_damp0.5": (bonded_base, None, damping(500.0), 480, "r00, damping 0.5 ps"),
        "sige_damp1.0": (bonded_base, None, damping(1000.0), 480, "r00, damping 1.0 ps"),
        "ar_ehex": (nb_base, NonbondedSpec(gas="Ar"), imposed(32), 32, "Ar layer, eHEX"),
        "he_ehex": (nb_base, NonbondedSpec(gas="He"), imposed(68), 68, "He layer, eHEX"),
        "he_mass40": (nb_base, NonbondedSpec(gas="He", gas_mass=39.948), imposed(40), 40,
                      "He LJ, Ar mass"),
        "he_nodirect": (nb_base, NonbondedSpec(gas="He", lj_scale=0.0), imposed(50), 50,
                        "He layer, no Si-Ge LJ"),
        "ar_nodirect": (nb_base, NonbondedSpec(gas="Ar", lj_scale=0.0), imposed(25), 25,
                        "Ar layer, no Si-Ge LJ"),
    }
    return out


def total_steps(spec) -> int:
    return relax_steps(spec) + spec.equilibrate_steps + spec.steady_steps + spec.production_steps


def write_case(wd: Path, name: str, case: tuple, pots: Path) -> dict:
    """Write one case into `wd` (replacing it); its manifest fields, without the name.

    `case` is a cases() value. Also used by prepare_cloud.py for follow-ups."""
    factory, nb, modify, h_guess, note = case
    spec = copy.deepcopy(factory())
    if nb is not None:
        spec.spot_pressure_GPa = 1.0
    if modify:
        modify(spec)
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    if nb is None:
        if name.endswith("ge_rescaled"):
            s = spec.lattice_a / spec.lattice_b
            write_rescaled(pots / spec.pair_file, wd / RESCALED, "Si(D)", "Ge", "Ge", s)
            spec.pair_file = RESCALED
            spec.lattice_b = spec.lattice_a
        else:
            shutil.copy(pots / spec.pair_file, wd)
        (wd / "nemd.in").write_text(build_input_deck(spec))
        write_spec(wd, spec, "nemd", h_guess_MW=h_guess, note=note)
        atoms = spec.geometry.estimate_atoms(4.99e28, 4.99e28)
    else:
        atoms = write_job(wd, spec, nb, pots)["atoms"]
    return {"atoms": int(atoms), "steps": total_steps(spec), "flux_mode": spec.flux_mode,
            "imposed_flux_W_m2": spec.imposed_flux, "interface_tally": spec.interface_tally,
            "note": note, "pair_style": spec.pair_style}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="job root holding manifest.json (e.g. ~/ok_jobs)")
    ap.add_argument("--potentials", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="case names to write (default: all)")
    a = ap.parse_args(argv)
    root = a.root.expanduser()
    pots = a.potentials.expanduser()
    out = root / "weekend"
    jobs = []
    for name, case in cases().items():
        if a.only and name not in a.only:
            continue
        e = write_case(out / name, name, case, pots)
        jobs.append({"name": f"weekend/{name}", **e})
        print(f"{name:18} {e['atoms']:7,} atoms  {e['steps'] / 1e6:4.2f} Msteps  {e['note']}")
    man = root / "manifest.json"
    data = json.loads(man.read_text()) if man.is_file() else {"jobs": []}
    names = {j["name"] for j in jobs}
    data["jobs"] = [j for j in data.get("jobs", []) if j["name"] not in names] + jobs
    man.write_text(json.dumps(data, indent=2))
    print(f"\n{len(jobs)} jobs under {out}; added to {man}. In the agent: jobs {root}, "
          f"then submit weekend local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
