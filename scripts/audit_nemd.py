"""Independent cross-check of the NEMD deck generator.

The Phase-1 analogue of `scripts/audit.py`. It does NOT import the deck
builder's helpers to check the deck builder's output -- it parses the rendered
deck text back and recomputes every number from the spec by hand, so a sign,
unit or ordering error has to appear in both places to survive.

Motivation: every Phase-1 defect found so far was invisible to the unit tests
and surfaced only when real LAMMPS read the deck -- a region written in lattice
counts where Angstroms were meant (5 nm would have been built as 9 A), a
cross-section snapped to whole cells while the flux used the unsnapped area
(4.4 % straight into h_K), a `fix ave/chunk` missing its chunk ID, an averaging
window that predated `reset_timestep 0` and produced an empty profile with exit
code 0. All of those are statically checkable. This checks them, across a grid
of configurations rather than the single fixture the tests use.

LAMMPS constraints enforced here, which error only at runtime:
  - `fix ave/chunk` needs Nfreq a multiple of Nevery, and Nevery*Nrepeat <= Nfreq.
  - The averaging window must fit inside the production run that follows
    `reset_timestep 0`, or the profile file is written empty.
  - Thermostatted regions must not overlap the windows the extractor fits,
    or the Langevin damping is folded into the bulk gradient.

Run with:  python3 scripts/audit_nemd.py
"""

from __future__ import annotations

import re
import sys
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))          # pin to THIS tree, never an installed copy

from openkapitza.md.nemd import (
    M_TO_ANG,
    NEMDGeometry,
    NEMDSpec,
    build_input_deck,
    validate_deck,
)

TOL_ANG = 1e-3          # deck writes 4 decimal places


class Failure(Exception):
    pass


def _check(cond: bool, label: str, detail: str = "") -> None:
    if not cond:
        raise Failure(f"{label}{(': ' + detail) if detail else ''}")


def _region_bounds(deck: str, name: str) -> tuple[str, str]:
    """Pull the z lo/hi tokens of a `region <name> block ...` line."""
    m = re.search(rf"^region\s+{name}\s+block\s+(.+?)\s+units box\s*$",
                  deck, re.MULTILINE)
    _check(m is not None, f"region {name} missing")
    toks = m.group(1).split()
    _check(len(toks) == 6, f"region {name} malformed", m.group(1))
    return toks[4], toks[5]


def _scalar(deck: str, pattern: str, label: str) -> float:
    m = re.search(pattern, deck, re.MULTILINE)
    _check(m is not None, f"{label} not found in deck")
    return float(m.group(1))


def _int(deck: str, pattern: str, label: str) -> int:
    m = re.search(pattern, deck, re.MULTILINE)
    _check(m is not None, f"{label} not found in deck")
    return int(m.group(1))


def audit_one(spec: NEMDSpec, label: str) -> list[str]:
    """Return a list of failure strings for one spec; empty means clean."""
    problems: list[str] = []
    g = spec.geometry

    try:
        deck = build_input_deck(spec)
    except Exception as exc:                                  # noqa: BLE001
        return [f"{label}: build_input_deck raised {exc!r}"]

    def run(fn, name):
        try:
            fn()
        except Failure as exc:
            problems.append(f"{label}: {exc}")
        except Exception as exc:                              # noqa: BLE001
            problems.append(f"{label}: {name} raised {exc!r}")

    # --- the builder's own validator must be clean -------------------------
    def _validator():
        errs = validate_deck(deck)
        _check(not errs, "validate_deck reported", "; ".join(errs))
    run(_validator, "validate_deck")

    # --- z is non-periodic, or heat short-circuits -------------------------
    def _boundary():
        _check(re.search(r"^boundary\s+p p f\s*$", deck, re.MULTILINE) is not None,
               "z must be non-periodic (boundary p p f)")
    run(_boundary, "boundary")

    # --- regions, recomputed from the geometry in Angstrom ----------------
    def _regions():
        _, wall_lo_hi = _region_bounds(deck, "wall_lo")
        hot_lo, hot_hi = _region_bounds(deck, "hot")
        cold_lo, cold_hi = _region_bounds(deck, "cold")
        wall_hi_lo, _ = _region_bounds(deck, "wall_hi")

        want_wall = g.wall_thickness * M_TO_ANG
        want_hot_end = g.z_bulk_a_start * M_TO_ANG
        want_cold_start = g.z_bulk_b_end * M_TO_ANG
        want_cold_end = g.z_cold_end * M_TO_ANG

        for got, want, nm in (
            (wall_lo_hi, want_wall, "wall_lo hi"),
            (hot_lo, want_wall, "hot lo"),
            (hot_hi, want_hot_end, "hot hi"),
            (cold_lo, want_cold_start, "cold lo"),
            (cold_hi, want_cold_end, "cold hi"),
            (wall_hi_lo, want_cold_end, "wall_hi lo"),
        ):
            _check(abs(float(got) - want) < TOL_ANG, f"{nm} mismatch",
                   f"deck {got} vs expected {want:.4f} A")

        # A region written in lattice counts instead of Angstrom is the bug
        # that would have built 5 nm as 9 A. Catch the magnitude directly.
        _check(float(hot_hi) > float(hot_lo), "hot region inverted")
        _check(float(cold_hi) > float(cold_lo), "cold region inverted")
        _check(float(cold_lo) > float(hot_hi),
               "hot and cold regions overlap or are swapped")
    run(_regions, "regions")

    # --- the extractor's fit windows must be unthermostatted --------------
    def _windows():
        a_lo, a_hi = g.bulk_a
        b_lo, b_hi = g.bulk_b
        _check(a_lo >= g.z_bulk_a_start - 1e-18,
               "bulk_a starts inside the hot Langevin region")
        _check(b_hi <= g.z_bulk_b_end + 1e-18,
               "bulk_b ends inside the cold Langevin region")
        _check(abs(a_hi - g.interface_z) < 1e-18
               and abs(b_lo - g.interface_z) < 1e-18,
               "fit windows do not meet at the interface")
        _check(a_hi > a_lo and b_hi > b_lo, "degenerate fit window")
    run(_windows, "fit windows")

    # --- area: must be the SNAPPED area in build mode ---------------------
    def _area():
        # The deck hands the area to the extractor on its DONE line, not as a
        # LAMMPS variable. That printed number is what divides the heat flux,
        # so it is the one worth auditing.
        area = _scalar(
            deck, r"OPENKAPITZA_DONE\s+area_m2\s+([0-9.eE+-]+)", "area")
        want = (spec.snapped_area if spec.structure_mode == "build" else g.area)
        rel = abs(area - want) / want
        _check(rel < 1e-6, "flux area mismatch",
               f"deck {area:.6e} vs expected {want:.6e} m^2 ({rel * 100:.3f} %)")
        if spec.structure_mode == "build":
            # And it must NOT be the unsnapped area, when they differ.
            if abs(spec.snapped_area - g.area) / g.area > 1e-9:
                _check(abs(area - g.area) / g.area > 1e-9,
                       "flux area uses the UNSNAPPED cross-section")
    run(_area, "area")

    # --- fix ave/chunk timing: a runtime error if violated ----------------
    def _ave_chunk():
        m = re.search(
            r"^fix\s+\S+\s+\S+\s+ave/chunk\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)",
            deck, re.MULTILINE)
        _check(m is not None, "fix ave/chunk line not found")
        nevery, nrepeat, nfreq, chunk_id = (
            int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4))
        _check(nevery > 0 and nrepeat > 0 and nfreq > 0,
               "ave/chunk has a non-positive count")
        _check(nfreq % nevery == 0,
               "ave/chunk Nfreq is not a multiple of Nevery",
               f"{nfreq} % {nevery}")
        _check(nevery * nrepeat <= nfreq,
               "ave/chunk Nevery*Nrepeat exceeds Nfreq",
               f"{nevery}*{nrepeat} > {nfreq}")
        # The chunk ID must name a compute chunk/atom, not be a number.
        _check(not re.fullmatch(r"[0-9.]+", chunk_id),
               "ave/chunk is missing its chunk ID", f"got {chunk_id!r}")
        _check(re.search(rf"^compute\s+{re.escape(chunk_id)}\s+\S+\s+chunk/atom",
                         deck, re.MULTILINE) is not None,
               "ave/chunk references an undefined chunk", chunk_id)
        # The window must fit in the production run after reset_timestep 0.
        _check(nfreq <= spec.production_steps,
               "averaging window outruns the production run -- empty profile",
               f"Nfreq {nfreq} > production {spec.production_steps}")
    run(_ave_chunk, "ave/chunk")

    # --- reset_timestep must precede the production run -------------------
    def _reset():
        # The deck pads commands into columns, so match flexible whitespace.
        m_reset = re.search(r"^reset_timestep\s+0\s*$", deck, re.MULTILINE)
        _check(m_reset is not None,
               "reset_timestep 0 missing before production")
        i_reset = m_reset.start()
        m = list(re.finditer(r"^run\s+(\d+)\s*$", deck, re.MULTILINE))
        _check(m, "no run commands found")
        last_run = m[-1]
        _check(i_reset < last_run.start(),
               "reset_timestep 0 comes after the production run")
        _check(int(last_run.group(1)) == spec.production_steps,
               "final run length is not production_steps",
               f"{last_run.group(1)} vs {spec.production_steps}")
    run(_reset, "reset_timestep")

    # --- velocity seeding: equipartition halves the requested T -----------
    def _tinit():
        # The mobile group is thermostatted, not "all" -- walls are frozen.
        t = _scalar(
            deck, r"velocity\s+\S+\s+create\s+([0-9.eE+-]+)", "t_init")
        _check(abs(t - 2.0 * spec.temperature) < 1e-6,
               "velocity create temperature is not 2x the setpoint",
               f"{t} vs {2.0 * spec.temperature}")
    run(_tinit, "velocity create")

    # --- reservoir setpoints straddle the mean ----------------------------
    def _setpoints():
        _check(abs((spec.t_hot + spec.t_cold) / 2.0 - spec.temperature) < 1e-9,
               "reservoir setpoints do not straddle the mean temperature")
        _check(spec.t_hot > spec.t_cold, "hot setpoint is not above cold")
    run(_setpoints, "setpoints")

    # --- the setpoints actually WRITTEN into the deck ---------------------
    def _deck_setpoints():
        """Check the deck text, not just the spec properties.

        Checking spec.t_hot/t_cold alone is vacuous -- it verifies two
        properties against each other and never reads the deck, so a template
        that swapped the two groups would pass. A mutation test caught this.
        """
        found = {}
        for m in re.finditer(
            r"^fix\s+\S+\s+(\S+)\s+langevin\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)",
            deck, re.MULTILINE,
        ):
            group, t_start, t_stop = m.group(1), float(m.group(2)), float(m.group(3))
            _check(abs(t_start - t_stop) < 1e-9,
                   "langevin ramps instead of holding a fixed setpoint",
                   f"{group}: {t_start} -> {t_stop}")
            prev = found.setdefault(group, t_start)
            _check(abs(prev - t_start) < 1e-9,
                   "same group thermostatted to two different setpoints", group)

        _check(len(found) == 2,
               "expected exactly two thermostatted groups",
               f"got {sorted(found)}")

        # Identify each group by the region it was built from, then require
        # the one on the low-z (hot) side to carry the higher setpoint.
        hot_group = cold_group = None
        for group in found:
            m = re.search(rf"^group\s+{re.escape(group)}\s+region\s+(\S+)",
                          deck, re.MULTILINE)
            _check(m is not None, f"group {group} has no region definition")
            region = m.group(1)
            if region == "hot":
                hot_group = group
            elif region == "cold":
                cold_group = group
        _check(hot_group is not None and cold_group is not None,
               "could not map thermostat groups onto the hot/cold regions",
               f"{sorted(found)}")
        _check(found[hot_group] > found[cold_group],
               "the group in the HOT region has the LOWER setpoint -- "
               "heat flows the wrong way",
               f"{hot_group}={found[hot_group]} {cold_group}={found[cold_group]}")
        _check(abs(found[hot_group] - spec.t_hot) < 1e-6
               and abs(found[cold_group] - spec.t_cold) < 1e-6,
               "deck setpoints disagree with the spec",
               f"deck {found[hot_group]}/{found[cold_group]} vs "
               f"spec {spec.t_hot}/{spec.t_cold}")
    run(_deck_setpoints, "deck setpoints")

    # --- never barostat a non-periodic dimension --------------------------
    def _barostat_dims():
        """`fix npt` on a non-periodic dimension is a hard LAMMPS error.

        z is non-periodic here by construction (walls), so any npt line
        naming z is fatal. The old spot-pressure block did exactly that and
        had never been run, because nothing set spot_pressure_GPa -- including
        every configuration in this audit's own grid until it was widened.
        """
        periodic = set()
        m = re.search(r"^boundary\s+(\S+)\s+(\S+)\s+(\S+)\s*$", deck, re.MULTILINE)
        _check(m is not None, "no boundary line")
        for dim, style in zip("xyz", m.groups()):
            if style == "p":
                periodic.add(dim)

        for line in re.findall(r"^fix\s+.*?npt.*$", deck, re.MULTILINE):
            # Strip the temp triple so its numbers are not mistaken for dims.
            tail = re.sub(r"temp\s+\S+\s+\S+\s+\S+", "", line)
            for dim in "xyz":
                if re.search(rf"(?<![a-zA-Z]){dim}\s+-?[\d.]", tail):
                    _check(dim in periodic,
                           f"npt barostats non-periodic {dim}",
                           line.strip())
            if "couple" in tail:
                coupled = re.search(r"couple\s+(\S+)", tail).group(1)
                if coupled != "none":
                    for dim in coupled:
                        _check(dim in periodic,
                               f"npt couples non-periodic {dim}", line.strip())
    run(_barostat_dims, "barostat dims")

    # --- relaxation is present exactly when asked for ---------------------
    def _relax():
        has_minimize = re.search(r"^minimize\s", deck, re.MULTILINE) is not None
        has_relax_fix = re.search(r"^fix\s+relax\s", deck, re.MULTILINE) is not None
        want = spec.relax_structure and spec.structure_mode == "build"
        _check(has_minimize == want,
               f"minimize {'missing' if want else 'present'} "
               f"with relax_structure={spec.relax_structure}")
        _check(has_relax_fix == want, "relax fix presence does not match spec")
        if want:
            _check(re.search(r"^unfix\s+relax\s*$", deck, re.MULTILINE) is not None,
                   "relax fix is never unfixed, so it runs through production")
            # The relax run must come before the gradient is imposed.
            i_relax = deck.index("fix             relax")
            i_lang = deck.index("langevin")
            _check(i_relax < i_lang,
                   "relaxation happens after the reservoirs are attached")
    run(_relax, "relaxation")

    # --- unit conversions -------------------------------------------------
    def _units():
        ts = _scalar(deck, r"^timestep\s+([0-9.eE+-]+)", "timestep")
        _check(abs(ts - spec.timestep_fs * 1e-3) < 1e-9,
               "timestep is not fs converted to ps",
               f"deck {ts} ps vs {spec.timestep_fs} fs")
        bw = _scalar(deck, r"chunk/atom\s+bin/1d\s+z\s+lower\s+([0-9.eE+-]+)",
                     "bin width")
        _check(abs(bw - g.bin_width * M_TO_ANG) < TOL_ANG,
               "bin width is not metres converted to Angstrom",
               f"deck {bw} A vs {g.bin_width * M_TO_ANG} A")
    run(_units, "units")

    return problems


def main() -> int:
    base = NEMDGeometry()
    cases: list[tuple[NEMDSpec, str]] = []

    # A grid over the knobs that moved in the defects found so far.
    for cross_nm, bulk_nm, bin_nm, lat_ang, prod, every, frames, relax in product(
        (3.0, 5.0, 7.3),            # cross-section, incl. non-multiples of a
        (6.0, 20.0, 31.5),          # bulk length each side
        (0.3, 0.5),                 # bin width
        (5.431, 5.658),             # Si, Ge lattice constants
        (100_000, 20_000),          # production steps
        (100, 500),                 # profile_every
        (4, 7),                     # profile_frames
        (True, False),              # relax_structure -- was never covered
    ):
        geom = NEMDGeometry(
            bulk_a_length=bulk_nm * 1e-9,
            bulk_b_length=bulk_nm * 1e-9,
            cross_section=cross_nm * 1e-9,
            bin_width=bin_nm * 1e-9,
        )
        spec = NEMDSpec(
            geometry=geom,
            element_a="Si", element_b="Ge",
            lattice_a=lat_ang * 1e-10, lattice_b=5.658e-10,
            mass_a=28.0855, mass_b=72.63,
            pair_style="sw", pair_file="Si.sw",
            temperature=300.0,
            production_steps=prod,
            profile_every=every,
            profile_frames=frames,
            relax_structure=relax,
        )
        label = (f"cross={cross_nm}nm bulk={bulk_nm}nm bin={bin_nm}nm "
                 f"a={lat_ang}A prod={prod} every={every} frames={frames} "
                 f"relax={relax}")
        cases.append((spec, label))

    # The spot-pressure path is a loaded piston now; audit that deck too, so
    # the configuration that once crashed LAMMPS stays in the grid.
    spot = NEMDSpec(
        geometry=base, element_a="Si", element_b="Ge",
        lattice_a=5.431e-10, lattice_b=5.658e-10,
        mass_a=28.0855, mass_b=72.63,
        pair_style="tersoff", pair_file="SiCGe.tersoff",
        temperature=300.0, spot_pressure_GPa=6.0,
    )
    cases.append((spot, "spot-pressure 6 GPa"))
    from dataclasses import replace
    cases.append((replace(spot, spot_pressure_mode="strain"),
                  "spot-pressure 6 GPa uniaxial strain"))

    print(f"auditing {len(cases)} deck configurations\n")
    all_problems_pre: list[str] = []
    all_problems: list[str] = list(all_problems_pre)
    for spec, label in cases:
        all_problems += audit_one(spec, label)

    width = 100
    print("=" * width)
    if all_problems:
        print(f"FAILURES: {len(all_problems)}")
        print("=" * width)
        for p in all_problems:
            print(f"  {p}")
        return 1
    print(f"FAILURES: none across {len(cases)} configurations")
    print("=" * width)
    print("\nChecked per configuration: validate_deck clean; z non-periodic;")
    print("six region boundaries against the geometry in Angstrom; fit windows")
    print("clear of both Langevin regions and meeting at the interface; flux")
    print("area snapped and provably not the unsnapped value; ave/chunk Nfreq")
    print("divisibility, Nevery*Nrepeat <= Nfreq, a named chunk ID that is")
    print("actually defined, and a window that fits the production run;")
    print("reset_timestep 0 before the final run; velocity create at 2x the")
    print("setpoint; reservoir setpoints straddling the mean; fs->ps and m->A.")
    print("\nNOTE: this is a STATIC audit. It cannot verify the physics -- only")
    print("that the deck says what the spec meant. A converged run on a real")
    print("engine is still required; see README 'Phase 1'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
