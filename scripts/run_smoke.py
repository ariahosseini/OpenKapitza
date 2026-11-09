"""Run a --smoke job set to the end and check every deck (KB 42).

    python3 scripts/prepare_cloud.py ~/ok_smoke --potentials P --smoke
    python3 scripts/run_smoke.py ~/ok_smoke --lmp ~/lammps-openkapitza/build/lmp --np 48 --parallel 4

For each job in ROOT/manifest.json: runs the deck (mpirun -np N, or serial
with --np 1), then checks that LAMMPS exited 0, reached OPENKAPITZA_DONE,
printed no ERROR, lost no atoms, kept finite temperatures, wrote its stage
checkpoints, and that the agent's measurement code reads the output (the
value of a few hundred steps means nothing and is not printed). It also
reads LAMMPS's loop times and compares the speed with the cost model, giving
the --core-rate to pass to prepare_cloud.py for real wall-time estimates.

Exit status 0 only if every job passes. Report: ROOT/smoke_report.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LOOP = re.compile(r"Loop time of ([\d.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms")


def run_one(wd: Path, lmp: str, np_: int, timeout: int, unpin: bool = False) -> dict:
    mpi = ["mpirun", "-np", str(np_)] + (["--bind-to", "none"] if unpin else [])
    cmd = ([lmp] if np_ <= 1 else mpi + [lmp]) + ["-in", "nemd.in", "-log", "log.lammps"]
    t0 = time.time()
    try:
        with open(wd / "stdout.log", "w") as out, open(wd / "stderr.log", "w") as err:
            code = subprocess.run(cmd, cwd=wd, stdout=out, stderr=err, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    return {"exit": code, "seconds": round(time.time() - t0, 1)}


def check(wd: Path, spec_kind: str = "nemd") -> list[str]:
    """What is wrong with a finished smoke run (empty: nothing)."""
    problems = []
    log = wd / "log.lammps"
    if not log.is_file():
        return ["no log.lammps"]
    text = log.read_text(errors="replace")
    if "OPENKAPITZA_DONE" not in text:
        problems.append("did not reach OPENKAPITZA_DONE")
    for ln in text.splitlines():
        if ln.startswith("ERROR"):
            problems.append(ln.strip()[:200])
            break
    if "Lost atoms" in text:
        problems.append("lost atoms")
    from openkapitza.md.nemd import last_thermo_block, parse_log_thermo
    rows = last_thermo_block(parse_log_thermo(log))
    temps = [r.get("Temp") for r in rows if "Temp" in r]
    if not temps:
        problems.append("no thermo output")
    elif not all(isinstance(t, float) and math.isfinite(t) and 0 < t < 5000 for t in temps):
        problems.append(f"temperature out of range: {temps[-3:]}")
    for f in ("stage.equilibrated.bin", "stage.steady.bin"):
        if f in (wd / "nemd.in").read_text() and not (wd / f).is_file():
            problems.append(f"no {f}")
    try:
        from openkapitza.md.campaign import read_spec
        from openkapitza.md.measure import measure_kapitza
        from openkapitza.schema.flags import FlagSet
        spec, _, _ = read_spec(wd)
        measure_kapitza(wd, spec, FlagSet())
    except Exception as exc:                        # report, never hide
        problems.append(f"measurement does not read the output: {exc}")
    return problems


def speed(wd: Path) -> dict | None:
    """Atom-steps per rank-second of the longest run in the log."""
    m = LOOP.findall((wd / "log.lammps").read_text(errors="replace"))
    if not m:
        return None
    t, procs, steps, atoms = max(m, key=lambda x: int(x[2]))
    t = float(t)
    if t <= 0:
        return None
    return {"rate": int(atoms) * int(steps) / (t * int(procs)), "procs": int(procs),
            "steps": int(steps)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--lmp", required=True)
    ap.add_argument("--np", type=int, default=1, help="MPI ranks per job (1: serial)")
    ap.add_argument("--parallel", type=int, default=1, help="jobs at once")
    ap.add_argument("--timeout", type=int, default=5400, help="seconds per job")
    ap.add_argument("--check-only", action="store_true", help="check finished runs only")
    a = ap.parse_args(argv)
    root = a.root.expanduser()
    man = json.loads((root / "manifest.json").read_text())
    jobs = man["jobs"]
    if not man.get("smoke"):
        print("note: this manifest is not a --smoke set; these are full-length runs")
    runs = {}
    if not a.check_only:
        with ThreadPoolExecutor(max_workers=a.parallel) as pool:
            futs = {j["name"]: pool.submit(run_one, root / j["name"], a.lmp, a.np, a.timeout,
                                       a.parallel > 1)
                    for j in jobs}
            for name, f in futs.items():
                runs[name] = f.result()
    from openkapitza.compute.resources import THROUGHPUT
    report, ratios, failed = [], [], 0
    for j in jobs:
        wd = root / j["name"]
        probs = check(wd)
        r = runs.get(j["name"], {})
        if r.get("exit") not in (None, 0):
            probs.insert(0, f"exit {r['exit']}")
        sp = speed(wd) if (wd / "log.lammps").is_file() else None
        style = j.get("pair_style", "tersoff")
        model = THROUGHPUT.get(style)
        if sp and model and sp["steps"] >= 200:
            ratios.append(sp["rate"] / model)
        failed += bool(probs)
        report.append({"name": j["name"], "ok": not probs, "problems": probs, **r,
                       "rate": sp["rate"] if sp else None, "pair_style": style})
        print(f"{'PASS' if not probs else 'FAIL'}  {j['name']:28} "
              f"{r.get('seconds', '-'):>7}s  " + ("; ".join(probs) if probs else
                                                   (f"{sp['rate']:.3g} atom-steps/rank-s"
                                                    if sp else "")))
    core_rate = round(statistics.median(ratios), 2) if ratios else None
    (root / "smoke_report.json").write_text(json.dumps(
        {"jobs": report, "np": a.np, "core_rate": core_rate, "failed": failed}, indent=2))
    print(f"\n{len(jobs) - failed}/{len(jobs)} passed.")
    if core_rate:
        print(f"Speed against the cost model: {core_rate}x (short runs; startup included). "
              f"Pass --core-rate {core_rate} to prepare_cloud.py for wall-time estimates.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
