"""First real LAMMPS run: a deliberately small Si/Ge shakedown.

Nothing in Phase 1 has ever been executed by LAMMPS. This script exercises the
whole chain once -- build deck, run, parse profile, parse reservoir energy,
extract h_K -- on a box small enough to finish in minutes on a laptop.

IT IS A PLUMBING TEST, NOT A MEASUREMENT. The box is far too small and the run
far too short for a converged h_K. Expect the number to be wrong; what matters
is that every stage runs, the files parse, and the extracted value lands in a
plausible range rather than off by orders of magnitude. Converged numbers come
from the convergence harness on a cluster.

Usage:
    python3 scripts/smoke_lammps.py [--lmp /path/to/lmp] [--potentials DIR]
                                    [--steps N] [--keep]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkapitza.compute import JobSpec, LocalRunner
from openkapitza.md.nemd import (
    NEMDGeometry,
    NEMDSpec,
    build_input_deck,
    check_spec,
    match_species,
    parse_profile,
    parse_reservoir_energy,
    potential_elements,
)
from openkapitza.md.profile import extract_kapitza, flux_from_reservoirs
from openkapitza.schema import FlagSet

# What the corrected Debye-DMM predicts for Si/Ge at 300 K, and what published
# NEMD reports. A tiny unconverged box will not match these; they are here so
# an answer that is wildly outside them is obvious.
DMM_REFERENCE_MW = 334.0
NEMD_PUBLISHED_MW = (266.0, 368.0)


def find_lammps(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    for name in ("lmp", "lmp_serial", "lmp_mpi", "lammps"):
        found = shutil.which(name)
        if found:
            return found
    return None


def potential_dirs(potentials_dir: str | None, lmp: str | None) -> list[Path]:
    dirs = []
    if potentials_dir:
        dirs.append(Path(potentials_dir))
    if os.environ.get("LAMMPS_POTENTIALS"):
        dirs.append(Path(os.environ["LAMMPS_POTENTIALS"]))
    if lmp:
        root = Path(lmp).resolve().parent.parent
        dirs += [root / "share" / "lammps" / "potentials", root / "potentials"]
    return [d for d in dirs if d.is_dir()]


def choose_potential(dirs: list[Path]) -> tuple[Path, str, tuple[str, str], str] | None:
    """Pick the best Si/Ge-capable potential actually present.

    Returns (file, pair_style, element names as written in that file, note).

    Preference order, and why:

    1. SiCGe.tersoff -- a real Si-Ge cross-species fit, and the Tersoff family
       is what the published Si/Ge NEMD comparisons used, so the result is
       answerable against them.
    2. Si.sw as a MASS-MISMATCH model: both atom types take silicon's
       potential, type 2 takes germanium's mass. There is no real chemistry
       across the junction, only an acoustic mismatch. It is a legitimate
       model system and it exercises the pipeline, but the number is not Si/Ge.

    What this deliberately does NOT do is write a SiGe.sw file from
    parameters recalled rather than cited. An invented potential yields
    numbers that look entirely reasonable and mean nothing.
    """
    # 1. A real Si-Ge cross-species fit, if one is installed. The labels are
    #    read from the file rather than assumed: LAMMPS writes the Tersoff-D
    #    silicon entry as "Si(D)", and passing "Si" fails outright.
    for d in dirs:
        for name in ("SiCGe.tersoff", "SiGe.tersoff"):
            f = d / name
            if not f.exists():
                continue
            labels = potential_elements(f)
            si, ge = match_species(labels, "Si"), match_species(labels, "Ge")
            if si and ge:
                return (f, "tersoff", (si, ge),
                        f"real Si-Ge cross-species Tersoff fit "
                        f"(labels {si!r}, {ge!r})")
            print(f"  {name} has no Si/Ge pair in {labels}; skipping")

    # 2. Mass-mismatch model on a single-species potential.
    for d in dirs:
        for name, style in (("Si.sw", "sw"), ("Si.tersoff", "tersoff")):
            f = d / name
            if not f.exists():
                continue
            labels = potential_elements(f)
            si = match_species(labels, "Si")
            if si:
                return (f, style, (si, si),
                        f"MASS-MISMATCH MODEL (label {si!r}): silicon "
                        f"potential throughout, germanium mass on side B "
                        f"-- not real Si/Ge")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", default=None)
    ap.add_argument("--potentials", default=None)
    ap.add_argument("--steps", type=int, default=200_000,
                    help="production steps; default is a few minutes")
    ap.add_argument("--workdir", default="/tmp/openkapitza_smoke")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    lmp = find_lammps(args.lmp)
    if not lmp:
        print("No LAMMPS binary found. Pass --lmp /path/to/lmp, or put lmp on "
              "PATH.\nNothing else in this script can run without it.")
        return 2
    print(f"LAMMPS:     {lmp}")

    dirs = potential_dirs(args.potentials, lmp)
    if not dirs:
        print("No LAMMPS potentials directory found. Pass --potentials DIR or "
              "set LAMMPS_POTENTIALS.")
        return 2
    print(f"Potentials: {dirs[0]}")

    chosen = choose_potential(dirs)
    if not chosen:
        print("Neither SiCGe.tersoff nor Si.sw is present in:")
        for d in dirs:
            print(f"    {d}")
        print("At least one is needed. Si.sw ships with every LAMMPS build.")
        return 2
    pot, pair_style, pair_elements, note = chosen
    print(f"Potential:  {pot.name}  [{pair_style}]")
    print(f"            {note}")
    model_only = pair_elements[0] == pair_elements[1]

    # Small on purpose: this is plumbing, not physics.
    geom = NEMDGeometry(
        wall_thickness=1.0e-9, reservoir_thickness=1.5e-9,
        bulk_a_length=8.0e-9, bulk_b_length=8.0e-9,
        cross_section=2.2e-9, bin_width=0.4e-9,
    )
    spec = NEMDSpec(
        geometry=geom, element_a="Si", element_b="Ge",
        lattice_a=5.431e-10, lattice_b=5.658e-10,
        temperature=300.0, delta_reservoir=40.0,
        pair_style=pair_style, pair_file=pot.name,
        pair_elements=pair_elements,
        equilibrate_steps=max(5_000, args.steps // 10),
        steady_steps=max(5_000, args.steps // 10),
        production_steps=args.steps,
        profile_every=100,
    )

    flags = FlagSet()
    check_spec(spec, flags)
    print(f"\nCell: {geom.total_length * 1e9:.1f} nm long, "
          f"{spec.snapped_cross_section * 1e9:.2f} nm square, "
          f"~{geom.estimate_atoms(5.0e28, 4.42e28):,} atoms")
    print(f"Pre-flight flags ({len(flags)}): "
          f"{', '.join(sorted(flags.codes())) or 'none'}")
    print("  (a box this small is EXPECTED to raise size warnings)")

    wd = Path(args.workdir)
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "nemd.in").write_text(build_input_deck(spec), encoding="utf-8")
    shutil.copy(pot, wd / pot.name)

    runner = LocalRunner()
    print(f"\nRunning in {wd} ...")
    t0 = time.time()
    handle = runner.submit(JobSpec(name="smoke", command=f"{lmp} -in nemd.in",
                                   workdir=wd))
    while not runner.poll(handle).terminal:
        time.sleep(2.0)
    result = runner.collect(handle)
    print(f"  {result.state.value} in {time.time() - t0:.0f}s "
          f"(rc={result.returncode})")

    if not result.ok:
        print("\nLAMMPS did not complete. Last lines of stderr:")
        if result.stderr_path and result.stderr_path.exists():
            print("".join(result.stderr_path.read_text().splitlines(True)[-20:]))
        if result.stdout_path and result.stdout_path.exists():
            print("--- stdout tail ---")
            print("".join(result.stdout_path.read_text().splitlines(True)[-20:]))
        print("\nThis is the expected first outcome: the decks have never been\n"
              "executed. Deck syntax, the potential file and the fix ave/chunk\n"
              "column layout are the likely culprits.")
        return 1

    print("\nParsing output ...")
    try:
        profile = parse_profile(wd / "temperature_profile.dat", geom, flags)
        e_hot, e_cold, duration = parse_reservoir_energy(wd / "reservoir_energy.dat")
    except (OSError, ValueError) as exc:
        print(f"  parsing failed: {exc}")
        print("  Check the column layout written by fix ave/chunk against "
              "parse_profile.")
        return 1

    area = spec.snapped_area
    flux = flux_from_reservoirs(e_hot, e_cold, duration, area, flags)
    print(f"  {profile.z.size} bins, flux {flux:.3e} W/m^2 "
          f"over {duration * 1e12:.1f} ps")

    try:
        r = extract_kapitza(profile, flux, flags,
                            bulk_a=geom.bulk_a, bulk_b=geom.bulk_b,
                            k_expected_a=148.0, k_expected_b=60.0)
    except ValueError as exc:
        print(f"  extraction refused: {exc}")
        return 1

    mw = r.h_K.si / 1e6
    print(f"\n  dT      = {r.delta_T.si:.3f} +/- "
          f"{r.delta_T.uncertainty or float('nan'):.3f} K")
    print(f"  h_K     = {mw:.1f} MW/m^2K")
    print(f"  fits    R^2 = {r.fit_a.r2:.4f} / {r.fit_b.r2:.4f}")
    print(f"  naive adjacent-bin dT would have been {r.naive_delta_T:.3f} K "
          f"({r.naive_error * 100:+.1f} %)")

    if model_only:
        print("\n  NOTE: this was the mass-mismatch MODEL, not Si/Ge. The "
              "published\n        Si/Ge values below do NOT apply to it; they "
              "are shown only as a\n        scale check that the pipeline is "
              "not off by orders of magnitude.")
    print(f"\n  reference: corrected Debye-DMM {DMM_REFERENCE_MW:.0f}, "
          f"published NEMD {NEMD_PUBLISHED_MW[0]:.0f}-{NEMD_PUBLISHED_MW[1]:.0f} "
          f"MW/m^2K")
    if 50.0 <= mw <= 2000.0:
        print("  -> plausible for an unconverged box; the chain works.")
    else:
        print("  -> OUTSIDE a plausible range. Something in the chain is wrong,\n"
              "     not merely unconverged.")

    print(f"\nFlags ({len(flags)}):")
    print(flags.render())
    print("\nREMINDER: this is a plumbing check. Do not quote this number.\n"
          "Run the convergence harness on a cluster for anything real.")

    if not args.keep:
        print(f"\n(run with --keep to retain {wd})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
