"""Did relaxing the in-plane box actually clear the cell stress?

Compares two runs of the same cell that differ only in `relax_structure`, and
reports the internal pressure, the in-plane box dimensions, and the extracted
h_K for each.

WHY IT MATTERS. The cell is built on ONE shared in-plane lattice constant, so
the entire Si/Ge mismatch appears as in-plane strain -- measured at 6.4 GPa of
internal pressure against ~0 for a relaxed crystal, with -4.01 % of strain
dumped on the Ge side and 0 % on the Si side. Stress of that size shifts
phonon frequencies and therefore changes h_K directly, so an unrelaxed cell
does not measure the interface it claims to.

Relaxing x and y at zero pressure (z cannot be barostatted -- non-periodic
walls) lets the shared constant settle where net in-plane stress vanishes,
splitting the mismatch between both sides.

Usage:
    python3 scripts/compare_relaxation.py <unrelaxed_dir> <relaxed_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))          # pin to THIS tree

from openkapitza.md import (
    NEMDGeometry,
    check_flux_stationarity,
    extract_kapitza,
    flux_from_reservoirs,
    parse_profile,
    parse_reservoir_energy,
)
from openkapitza.md.nemd import check_run_health, parse_log_thermo
from openkapitza.schema import FlagSet

# Matches the cell used by scripts/smoke_lammps.py.
GEOM = NEMDGeometry(
    wall_thickness=1.0e-9, reservoir_thickness=1.5e-9,
    bulk_a_length=8.0e-9, bulk_b_length=8.0e-9,
    cross_section=2.2e-9, bin_width=0.4e-9,
)
LATTICE_A = 5.431e-10
AREA = (round(2.2e-9 / LATTICE_A) * LATTICE_A) ** 2
K_SI, K_GE = 148.0, 60.0


def _num(value):
    return value.si if hasattr(value, "si") else float(value)


def _box_in_plane(log_path: Path) -> tuple[float, float] | None:
    """Final lx, ly from any thermo block that reported them."""
    # Capture the VALUES while the Lx-bearing header is in scope. Keeping the
    # header and the row separately goes wrong the moment a later thermo block
    # has no Lx column: the row is stale but the header has moved on, and
    # header.index("Lx") then raises. (Hit exactly that on the first run.)
    header: list[str] | None = None
    found: tuple[float, float] | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Step":
            header = parts
            continue
        if header is None or "Lx" not in header or len(parts) != len(header):
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            header = None
            continue
        found = (values[header.index("Lx")], values[header.index("Ly")])
    return found


def report(label: str, directory: Path) -> None:
    print(f"\n{'=' * 70}\n{label}  ({directory.name})\n{'=' * 70}")
    log = directory / "log.lammps"
    if not log.is_file():
        print("  no log.lammps")
        return

    flags = FlagSet()
    thermo = parse_log_thermo(log)
    check_run_health(thermo, 300.0, flags)

    pressures = [r["Press"] for r in thermo if "Press" in r]
    if pressures:
        # LAMMPS metal units report pressure in bar; 1 GPa = 1e4 bar.
        print(f"  internal pressure: first {pressures[0] / 1e4:+8.3f} GPa"
              f"   last {pressures[-1] / 1e4:+8.3f} GPa"
              f"   mean {sum(pressures) / len(pressures) / 1e4:+8.3f} GPa")

    box = _box_in_plane(log)
    if box:
        lx, ly = box
        n_cells = round(2.2e-9 / LATTICE_A)
        print(f"  in-plane box: {lx:.4f} x {ly:.4f} A"
              f"  -> a_eff {lx / n_cells:.4f} A"
              f"  (Si 5.4310, Ge 5.6580)")

    profile_path = directory / "temperature_profile.dat"
    energy_path = directory / "reservoir_energy.dat"
    if profile_path.is_file() and energy_path.is_file():
        check_flux_stationarity(energy_path, flags)
        profile = parse_profile(profile_path, GEOM, flags)
        e_hot, e_cold, duration = parse_reservoir_energy(energy_path)
        flux = _num(flux_from_reservoirs(e_hot, e_cold, duration, AREA, flags))
        result = extract_kapitza(
            profile, flux, flags,
            bulk_a=GEOM.bulk_a, bulk_b=GEOM.bulk_b,
            k_expected_a=K_SI, k_expected_b=K_GE,
        )
        print(f"  flux {flux:.3e} W/m2   dT {_num(result.delta_T):7.3f} K"
              f"   h_K {_num(result.h_K) / 1e6:8.1f} MW/m2K")
    else:
        print("  (no production profile yet)")

    print(f"  flags ({len(flags.flags)}): {sorted(flags.codes())}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    report("UNRELAXED  (built on one lattice constant)", Path(argv[1]))
    report("RELAXED    (in-plane box at zero pressure)", Path(argv[2]))
    print(f"\n{'=' * 70}")
    print("Reference: a relaxed crystal sits near zero internal pressure.")
    print("Published Si/Ge h_K at 300 K: 266-368 MW/m2K; corrected Debye-DMM 334.")
    print("Neither run is a converged measurement -- read the pressure and the")
    print("stationarity flags, not the h_K.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
