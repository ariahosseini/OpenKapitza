"""Time this machine on the project's own decks; size every prepared job from it.

    python3 scripts/benchmark_local.py --lmp ~/lammps-openkapitza/build/lmp \
        --potentials ~/lammps-openkapitza/potentials [--np 4 5 8]

What it does:

1. Builds two short decks with the same code that writes the cluster jobs:
   Si/Ge Tersoff (the campaign's base geometry) and Si/Al MEAM (a shorter
   cell). Run lengths are cut to a few thousand steps; per-atom-step cost
   does not depend on run length.
2. Runs each with `mpirun -np N` for every N given, and reads LAMMPS's own
   "Loop time" lines for the MD segments only (minimisation and setup are
   excluded), giving atom-steps per second.
3. Writes the measured rates to ~/.config/openkapitza/throughput.json, which
   the cost model and the agent's local submissions read
   (openkapitza.compute.resources.local_estimate).
4. Writes every prepared job (scripts/prepare_cluster.py, in a temporary
   folder) and prints its hours on this machine at the fastest rank count,
   with a verdict: here overnight, here but long, or cloud.

Nothing is submitted. Runs natively: on macOS use the LAMMPS from
scripts/mac_setup_lammps.sh, not a Linux VM (which sees only its own cores).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OVERNIGHT_H = 10.0          # fits a night or a working day
LONG_H = 48.0               # beyond this, a laptop is the wrong place

_LOOP = re.compile(r"Loop time of ([\d.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms")


# ---- log parsing --------------------------------------------------------------------
def md_loops(log: str) -> list[dict]:
    """Every timed MD segment in a LAMMPS log; minimisations are dropped."""
    out = []
    for m in _LOOP.finditer(log):
        tail = log[m.end(): m.end() + 600]
        if "Minimization stats" in tail:
            continue
        steps, secs = int(m.group(3)), float(m.group(1))
        if steps > 0 and secs > 0:
            out.append({"seconds": secs, "procs": int(m.group(2)), "steps": steps,
                        "atoms": int(m.group(4))})
    return out


def rate(loops: list[dict]) -> dict:
    """Atom-steps per second over the MD segments (the first, warm-up one dropped
    when there are several)."""
    use = loops[1:] if len(loops) > 2 else loops
    if not use:
        raise ValueError("no timed MD segment in the log")
    atom_steps = sum(l["steps"] * l["atoms"] for l in use)
    secs = sum(l["seconds"] for l in use)
    procs = use[0]["procs"]
    return {"atom_steps_per_s": atom_steps / secs, "per_core": atom_steps / secs / procs,
            "procs": procs, "atoms": use[0]["atoms"], "steps": sum(l["steps"] for l in use),
            "seconds": secs}


# ---- decks ------------------------------------------------------------------------
def decks(potentials: Path, work: Path) -> dict[str, Path]:
    from openkapitza.md.nemd import NEMDGeometry, build_input_deck
    from plan_campaign import PRESETS, base_spec
    from prepare_cluster import jobs

    short = dict(relax_steps=200, equilibrate_steps=1000, steady_steps=1000,
                 production_steps=2000, profile_every=100, profile_frames=4)
    sige = base_spec(PRESETS["sige"])
    for k, v in short.items():
        setattr(sige, k, v)
    sial = copy.deepcopy(next(s for n, s, _ in jobs() if n == "sial_pp"))
    g = sial.geometry
    sial.geometry = NEMDGeometry(wall_thickness=g.wall_thickness,
                                 reservoir_thickness=g.reservoir_thickness,
                                 bulk_a_length=8e-9, bulk_b_length=8e-9,
                                 cross_section=g.cross_section, bin_width=g.bin_width)
    for k, v in dict(short, equilibrate_steps=500, steady_steps=500, production_steps=1000).items():
        setattr(sial, k, v)
    out = {}
    for style, spec in (("tersoff", sige), ("meam", sial)):
        d = work / style
        d.mkdir(parents=True, exist_ok=True)
        (d / "nemd.in").write_text(build_input_deck(spec), encoding="utf-8")
        for f in (spec.pair_file, *spec.extra_files):
            src = potentials / f
            if not src.is_file():
                raise SystemExit(f"{src} not found: give --potentials <lammps>/potentials")
            shutil.copy(src, d / f)
        out[style] = d
    return out


def run(lmp: str, deck: Path, np_: int, launcher: str, timeout: float) -> dict:
    wd = deck.parent / f"{deck.name}_np{np_}"
    if wd.exists():
        shutil.rmtree(wd)
    shutil.copytree(deck, wd)
    cmd = ([lmp] if launcher == "none" else [launcher, "-np", str(np_), lmp]) + \
        ["-in", "nemd.in", "-log", "log.lammps", "-screen", "none"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    log = (wd / "log.lammps").read_text(errors="replace") if (wd / "log.lammps").is_file() else ""
    if p.returncode != 0 or not log:
        err = re.findall(r"^ERROR.*$", log + p.stdout + p.stderr, re.M)
        raise RuntimeError(f"{' '.join(cmd)} failed in {wd}: "
                           f"{err[0] if err else (p.stderr or p.stdout)[-400:]}")
    r = rate(md_loops(log))
    r["wall_s"] = wall
    return r


# ---- jobs ------------------------------------------------------------------------
def prepared_jobs(potentials: Path) -> list[dict]:
    """atoms, steps and pair style of every job prepare_cluster writes."""
    import contextlib
    import io
    import prepare_cluster
    from openkapitza.md.campaign import read_spec
    with tempfile.TemporaryDirectory() as d:
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_cluster.main([d, "--potentials", str(potentials)])
        man = json.loads((Path(d) / "manifest.json").read_text())
        out = []
        for j in man["jobs"]:
            spec, kind, _ = read_spec(Path(d) / j["name"])
            style = spec.pair_style.split("/")[0].lower()
            out.append({"name": j["name"], "atoms": j["atoms"], "steps": j["steps"],
                        "style": style, "ttm": spec.ttm is not None, "kind": kind,
                        "cluster_node_hours": j["node_hours"]})
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmp", required=True)
    ap.add_argument("--potentials", required=True, type=Path)
    ap.add_argument("--np", type=int, nargs="+", default=[4, 5, 8])
    ap.add_argument("--launcher", default="mpirun", help="'none' runs serial (np ignored)")
    ap.add_argument("--styles", nargs="+", default=["tersoff", "meam"])
    ap.add_argument("--out", type=Path, help="calibration file (default "
                    "~/.config/openkapitza/throughput.json)")
    ap.add_argument("--keep", type=Path, help="keep the run folders here")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args(argv)
    from openkapitza.compute.resources import DEFAULT_CALIBRATION, local_estimate

    if platform.system() == "Linux" and os.path.exists("/proc/cpuinfo") and \
            "Apple" in Path("/proc/cpuinfo").read_text(errors="ignore"):
        print("Note: this looks like a Linux VM on a Mac; it measures the VM, not the Mac.\n")
    nps = [1] if a.launcher == "none" else a.np
    work = a.keep or Path(tempfile.mkdtemp(prefix="okbench-"))
    work.mkdir(parents=True, exist_ok=True)
    built = decks(a.potentials, work)
    results: dict[str, dict] = {}
    for style in a.styles:
        results[style] = {}
        for n in nps:
            print(f"  {style:8} np={n} ...", end="", flush=True)
            r = run(a.lmp, built[style], n, a.launcher, a.timeout)
            results[style][n] = r
            print(f" {r['atom_steps_per_s']:.3g} atom-steps/s ({r['per_core']:.3g} per core, "
                  f"{r['atoms']:,} atoms, {r['steps']:,} steps in {r['seconds']:.0f} s)")

    machine = {"host": platform.node(), "system": platform.platform(),
               "processor": _chip(), "date": time.strftime("%Y-%m-%d"),
               "lmp": a.lmp, "launcher": a.launcher, "styles": {}}
    for style, by_np in results.items():
        best = max(by_np, key=lambda n: by_np[n]["atom_steps_per_s"])
        machine["styles"][style] = {
            "best_np": best, "atom_steps_per_s": by_np[best]["atom_steps_per_s"],
            "per_core": by_np[best]["per_core"],
            "by_np": {str(n): round(v["atom_steps_per_s"]) for n, v in by_np.items()}}
    out = a.out or DEFAULT_CALIBRATION
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(machine, indent=2))

    print("\nFastest rank count per potential:")
    for style, s in machine["styles"].items():
        print(f"  {style:8} np={s['best_np']}: {s['atom_steps_per_s']:.3g} atom-steps/s")
    print(f"\nCalibration written to {out}\n")

    print(f"{'job':44} {'atoms':>8} {'steps':>10} {'pot':8} {'hours here':>10}  verdict")
    rows = []
    for j in prepared_jobs(a.potentials):
        est = local_estimate(j["atoms"], j["steps"], j["style"], machine)
        h = est["hours"] if est else float("nan")
        verdict = ("here, overnight" if h <= OVERNIGHT_H else
                   "here, but long" if h <= LONG_H else "cloud")
        note = " (TTM: MEAM rate, an underestimate)" if j["ttm"] else ""
        rows.append({**j, "hours_here": h, "np": est["np"] if est else None, "verdict": verdict})
        print(f"{j['name']:44} {j['atoms']:8,} {j['steps']:10,} {j['style']:8} {h:10.1f}  "
              f"{verdict}{note}")
    here = [r for r in rows if r["verdict"] != "cloud"]
    print(f"\n{len(here)} of {len(rows)} jobs are sensible here "
          f"({sum(r['hours_here'] for r in here):.0f} h in total); "
          f"{len(rows) - len(here)} belong in the cloud "
          f"({sum(r['hours_here'] for r in rows if r['verdict'] == 'cloud'):.0f} h here).")
    (work / "benchmark.json").write_text(json.dumps({"machine": machine, "jobs": rows},
                                                    indent=2, default=str))
    print(f"Details: {work / 'benchmark.json'}")
    return 0


def _chip() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or platform.processor()
    except (OSError, subprocess.SubprocessError):
        return platform.processor()


if __name__ == "__main__":
    raise SystemExit(main())
