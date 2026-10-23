"""Direct NEMD: geometry, LAMMPS input decks, and reading the result back.

Direct NEMD is the default rather than Muller-Plathe reverse NEMD. rNEMD
imposes a flux by swapping velocities between slabs, which distorts the local
velocity distribution around the swap regions. For bulk conductivity that is
tolerable; for an interface it is not, because the quantity wanted is a
discontinuity whose neighbourhood must stay undisturbed.

The layout along the transport direction z:

    [fixed wall][hot reservoir][ bulk A ]|[ bulk B ][cold reservoir][fixed wall]

Fixed walls stop the slab drifting and make the direction non-periodic, so
heat cannot short-circuit around the box. Langevin reservoirs set the hot and
cold ends; their accumulated energy is what gives the flux, since a nominal
setpoint is not a measurement.

The geometry object also publishes the exact bulk ranges, which are handed to
the extractor rather than letting it guess where the reservoirs end -- a
guess that silently biases the jump low (see md.profile).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..schema.flags import Code, FlagSet, Severity
from .profile import TemperatureProfile

# Pairings of (potential style, pair_coeff file) by element set. Tier 1 only;
# Tier 2 MLIPs arrive in Phase 4 behind the validation gate.
# Each entry is (pair_style, file, elements-as-named-in-that-file).
#
# ONLY files that actually ship with LAMMPS belong here. An earlier version
# listed "SiGe.sw", which does not exist in the distribution -- the shakedown
# found it immediately. Verify a file is present before adding it, and never
# invent potential parameters to fill a gap: a fabricated potential produces
# numbers that look fine and mean nothing.
TIER1_POTENTIALS = {
    ("Si",): ("sw", "Si.sw", ("Si",)),
    ("Si", "Ge"): ("tersoff", "SiCGe.tersoff", ("Si", "Ge")),
    ("Si", "C"): ("tersoff", "SiC.tersoff", ("Si", "C")),
    ("Al",): ("eam/alloy", "Al.eam.alloy", ("Al",)),
    ("Cu",): ("eam/alloy", "Cu.eam.alloy", ("Cu",)),
}

# A model interface built from ONE element's potential with the second atom
# type given a different mass. There is no real chemistry across the junction,
# only an acoustic mismatch, so it is not a substitute for a fitted
# cross-species potential -- but it is a standard model system, it needs only
# Si.sw, and it exercises the whole pipeline with a genuine temperature jump.
MASS_MISMATCH = ("sw", "Si.sw", ("Si", "Si"))


def potential_elements(path: Path | str) -> list[str]:
    """Element labels exactly as written in a Tersoff or SW potential file.

    Both formats begin each entry with three element labels followed by
    numeric parameters, and both allow entries to wrap across lines. A
    continuation line therefore starts with a number, which is how the two
    are told apart here.

    This exists because labels are NOT always the plain chemical symbol.
    LAMMPS distinguishes parameterisations in the label -- the silicon entry
    in a Tersoff file may be written "Si(D)" for the Tersoff-D fit -- and
    passing "Si" to pair_coeff when the file says "Si(D)" fails with
    "Potential file is missing an entry for: Si Si Si". Reading the labels
    beats guessing them.
    """
    labels: list[str] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            token = line.split()[0]
            try:                      # a continuation line starts with a number
                float(token)
                continue
            except ValueError:
                pass
            if token not in seen:
                seen.add(token)
                labels.append(token)
    return labels


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def potential_triples(path: Path | str) -> set[tuple[str, str, str]]:
    """Every (el1, el2, el3) entry present in a Tersoff or SW file.

    Knowing the labels is not sufficient. A LAMMPS potential file may carry
    several parameterisations of the same element -- SiCGe.tersoff has
    Si(B), Si(C) and Si(D) -- while providing cross-species terms for only
    ONE of them. Choosing a silicon label that has no Si-Ge entry fails with
    "Potential file is missing an entry for: Si(B) Si(B) Ge", which is how
    this was found.

    Entries are three labels followed by numeric parameters and may wrap
    across lines, so the file is tokenised whole and any run of three
    consecutive non-numeric tokens is taken as a triple.
    """
    tokens: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tokens.extend(line.split("#", 1)[0].split())

    triples: set[tuple[str, str, str]] = set()
    i = 0
    while i < len(tokens) - 2:
        window = tokens[i:i + 3]
        if not any(_is_number(tok) for tok in window):
            triples.add((window[0], window[1], window[2]))
            i += 3
        else:
            i += 1
    return triples


def complete_pairs(
    triples: set[tuple[str, str, str]], symbol_a: str, symbol_b: str
) -> list[tuple[str, str]]:
    """Label pairs for which EVERY triple LAMMPS needs is present.

    A two-element many-body system requires all eight ordered triples over
    the two labels. Returning only complete pairs means the deck cannot be
    written against a combination the potential file does not support.
    """
    labels = {lab for triple in triples for lab in triple}

    def candidates(symbol: str) -> list[str]:
        return sorted(
            lab for lab in labels
            if lab == symbol or lab.split("(", 1)[0] == symbol
        )

    found: list[tuple[str, str]] = []
    for a in candidates(symbol_a):
        for b in candidates(symbol_b):
            required = {
                (x, y, z) for x in (a, b) for y in (a, b) for z in (a, b)
            }
            if required <= triples:
                found.append((a, b))
    return found


def locate_potential(pair_file: str, potential_dir: str | None) -> Path:
    """The readable path to `pair_file`, for verification.

    `pair_file` is what goes into the deck, so it must stay resolvable by
    LAMMPS from ITS working directory -- normally a bare filename next to the
    deck. Verification runs somewhere else entirely, so it needs the real
    location. Tries the name as given first, so an absolute or
    already-correct relative path keeps working.

    Raises
    ------
    FileNotFoundError
        If neither candidate is readable, so callers can distinguish
        "unverifiable" from "verified and wrong".
    """
    candidates = [Path(pair_file)]
    if potential_dir:
        candidates.append(Path(potential_dir) / Path(pair_file).name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find {pair_file!r}" +
        (f" in {potential_dir!r}" if potential_dir else "")
    )


def resolve_pair_elements(
    path: Path | str, symbol_a: str, symbol_b: str
) -> tuple[str, str]:
    """The label pair to write into `pair_coeff`, verified against the file.

    This is the library-side entry point for the lesson that cost three
    debugging cycles: a many-body potential file names its parameterisations
    in the LABEL, and the cross terms may exist for only one of them. For
    `SiCGe.tersoff` the labels are Si(B), Si(C), C, Si(D), Ge -- and only
    Si(D) carries the full set of eight Si/Ge triples. Si(B) and Si(C) have
    two each, so a deck written against either dies on a missing entry
    minutes into a run.

    `match_species` is NOT a substitute: it returns the first label whose
    stem matches, which for "Si" in that file is Si(B) -- the wrong one.
    Prefer this function anywhere a deck is about to be written.

    Raises
    ------
    ValueError
        If no pair of labels carries every required triple. The message lists
        what the file does contain, because the next question is always
        "then what can I use?".
    """
    triples = potential_triples(path)
    pairs = complete_pairs(triples, symbol_a, symbol_b)
    if pairs:
        # Deterministic: complete_pairs sorts its candidates.
        return pairs[0]

    labels = sorted({lab for triple in triples for lab in triple})
    raise ValueError(
        f"{Path(path).name} has no complete set of triples for "
        f"{symbol_a}/{symbol_b}. Labels present: {labels}. "
        f"A two-element many-body system needs all eight ordered triples "
        f"over the two labels; this file does not provide them for any "
        f"pairing of those elements. Pick a different potential rather than "
        f"writing a deck LAMMPS will reject."
    )


def match_species(labels: list[str], symbol: str) -> str | None:
    """Find the label in `labels` denoting the element `symbol`.

    Exact match wins; otherwise the part before any parenthesised
    parameterisation tag is compared, so "Si" finds "Si(D)".
    """
    for label in labels:
        if label == symbol:
            return label
    for label in labels:
        if label.split("(", 1)[0] == symbol:
            return label
    return None


@dataclass
class NEMDGeometry:
    """Slab layout, in metres. All positions measured from z = 0."""

    wall_thickness: float = 1.0e-9
    reservoir_thickness: float = 2.0e-9
    bulk_a_length: float = 20.0e-9
    bulk_b_length: float = 20.0e-9
    cross_section: float = 5.0e-9        # side of the square cross-section
    bin_width: float = 0.5e-9

    @property
    def area(self) -> float:
        return self.cross_section ** 2

    @property
    def z_hot_start(self) -> float:
        return self.wall_thickness

    @property
    def z_bulk_a_start(self) -> float:
        return self.wall_thickness + self.reservoir_thickness

    @property
    def interface_z(self) -> float:
        return self.z_bulk_a_start + self.bulk_a_length

    @property
    def z_bulk_b_end(self) -> float:
        return self.interface_z + self.bulk_b_length

    @property
    def z_cold_end(self) -> float:
        return self.z_bulk_b_end + self.reservoir_thickness

    @property
    def total_length(self) -> float:
        return self.z_cold_end + self.wall_thickness

    @property
    def bulk_a(self) -> tuple[float, float]:
        """Exact unthermostatted range on side A, for the extractor."""
        return (self.z_bulk_a_start, self.interface_z)

    @property
    def bulk_b(self) -> tuple[float, float]:
        return (self.interface_z, self.z_bulk_b_end)

    @property
    def n_bins(self) -> int:
        return max(1, int(round(self.total_length / self.bin_width)))

    def estimate_atoms(self, number_density_a: float, number_density_b: float) -> int:
        """Atom count, for the cost model."""
        a_len = self.wall_thickness + self.reservoir_thickness + self.bulk_a_length
        b_len = self.total_length - a_len
        return int(self.area * (a_len * number_density_a + b_len * number_density_b))


@dataclass
class NEMDSpec:
    """Everything needed to write one NEMD input deck."""

    geometry: NEMDGeometry
    element_a: str
    element_b: str
    lattice_a: float                    # m
    lattice_b: float                    # m
    temperature: float                  # K, mean
    delta_reservoir: float = 40.0       # K, hot/cold split about the mean
    pair_style: str = "tersoff"
    pair_file: str = "SiCGe.tersoff"
    pair_elements: tuple[str, ...] | None = None   # names AS IN the file
    # Where to LOOK for pair_file when verifying it. Verification only: it is
    # never written into the deck. `pair_file` has to stay whatever LAMMPS
    # will resolve from its own working directory -- usually a bare filename,
    # because the runner copies the potential in beside the deck -- but a bare
    # filename is not resolvable from wherever check_spec happens to run. Not
    # separating the two made the first real run emit a spurious
    # NEMD_POTENTIAL_UNVERIFIED for a potential that was sitting right there.
    potential_dir: str | None = None
    # Relax the in-plane box before imposing the gradient. Default ON: a cell
    # built on one lattice constant carries the whole mismatch as in-plane
    # strain (6.4 GPa for Si/Ge), which changes h_K directly.
    relax_structure: bool = True
    relax_steps: int = 50_000
    # Let the top wall move as a rigid piston along z during relaxation, so the
    # cell relaxes to sigma_zz = 0 as the in-plane barostat relaxes sigma_xx,
    # sigma_yy. Default ON (open issues P1-2, P1-7): with both walls frozen the
    # cell is clamped at its built length, and because it is built on ONE
    # lattice constant the Ge half is compressed along z with nowhere to go.
    # Measured: the clamp alone reproduces the 1.5 GPa Pzz of the production
    # cell and moves the shared in-plane constant from 56 % to 67 % of the way
    # toward Ge (scripts/test_inplane_relaxation.py, 0 K).
    relax_z: bool = True
    piston_damp_ps: float = 0.5
    timestep_fs: float = 1.0
    equilibrate_steps: int = 500_000
    steady_steps: int = 500_000
    production_steps: int = 2_000_000
    profile_every: int = 100
    profile_frames: int = 8
    damping_fs: float = 100.0
    spot_pressure_GPa: float | None = None
    seed: int = 12345
    structure_mode: str = "build"       # "build" | "read_data"
    data_file: str = "interface.data"
    common_lattice: float | None = None  # m; defaults to lattice_a in build mode
    mass_a: float = 28.0855             # amu
    mass_b: float = 72.630
    extra: dict = field(default_factory=dict)

    @property
    def lattice_used(self) -> float:
        """The lattice constant the built cell actually uses.

        A coherent interface needs ONE in-plane lattice constant, so one side
        is strained. That is standard practice for a Si/Ge NEMD cell, but the
        strain is not cosmetic: it shifts phonon frequencies and therefore
        h_K. `lattice_strain` reports it and check_spec flags it.
        """
        return self.common_lattice or self.lattice_a

    @property
    def snapped_cross_section(self) -> float:
        """Cross-section rounded to a whole number of unit cells, in metres.

        x and y are periodic, so a box that is not a whole number of cells
        leaves a seam where the images meet. The flux uses THIS area, not the
        requested one, or h_K is wrong by the ratio.
        """
        a = self.lattice_used
        n = max(1, round(self.geometry.cross_section / a))
        return n * a

    @property
    def snapped_area(self) -> float:
        return self.snapped_cross_section ** 2

    @property
    def lattice_strain(self) -> tuple[float, float]:
        """Fractional in-plane strain carried by each side."""
        a = self.lattice_used
        return (a / self.lattice_a - 1.0, a / self.lattice_b - 1.0)

    @property
    def t_hot(self) -> float:
        return self.temperature + 0.5 * self.delta_reservoir

    @property
    def t_cold(self) -> float:
        return self.temperature - 0.5 * self.delta_reservoir

    @property
    def total_steps(self) -> int:
        return self.equilibrate_steps + self.steady_steps + self.production_steps


# LAMMPS `metal` units: distance in Angstrom, time in ps, energy in eV.
M_TO_ANG = 1e10
FS_TO_PS = 1e-3
GPA_TO_BAR = 1e4


DECK = """\
# OpenKapitza direct NEMD -- {element_a}/{element_b} interface
# Generated; do not edit by hand. Units: metal (Angstrom, ps, eV, bar).
#
# Layout along z:
#   [wall][hot Langevin][ bulk {element_a} ]|[ bulk {element_b} ][cold Langevin][wall]
# z is NON-PERIODIC so heat cannot short-circuit around the box.

units           metal
dimension       3
boundary        p p f
atom_style      atomic
timestep        {timestep_ps:.6f}

{structure_block}
pair_style      {pair_style}
pair_coeff      * * {pair_file} {pair_a} {pair_b}

# ---- regions, in Angstrom -------------------------------------------------
region          wall_lo  block INF INF INF INF INF {z_wall_lo:.4f} units box
region          hot      block INF INF INF INF {z_wall_lo:.4f} {z_hot_end:.4f} units box
region          cold     block INF INF INF INF {z_cold_start:.4f} {z_cold_end:.4f} units box
region          wall_hi  block INF INF INF INF {z_cold_end:.4f} INF units box

group           gwall_lo region wall_lo
group           gwall_hi region wall_hi
group           ghot     region hot
group           gcold    region cold
group           gwalls   union gwall_lo gwall_hi
group           gmobile  subtract all gwalls

velocity        gwalls set 0.0 0.0 0.0 units box
fix             freeze gwalls setforce 0.0 0.0 0.0
{relax_block}
# ---- equilibration --------------------------------------------------------
velocity        gmobile create {t_init:.2f} {seed} mom yes rot yes dist gaussian
{pressure_block}
fix             equil gmobile nvt temp {temperature:.2f} {temperature:.2f} {damp_ps:.4f}
thermo          10000
thermo_style    custom step temp pe etotal press
run             {equilibrate_steps}
unfix           equil

# ---- impose the gradient --------------------------------------------------
# Langevin reservoirs on the ends only; the bulk runs NVE so the profile
# between them is physical and not thermostatted into a straight line.
fix             nve gmobile nve
fix             hot  ghot  langevin {t_hot:.2f} {t_hot:.2f} {damp_ps:.4f} {seed_hot} tally yes
fix             cold gcold langevin {t_cold:.2f} {t_cold:.2f} {damp_ps:.4f} {seed_cold} tally yes

# Reach steady state before any averaging starts.
run             {steady_steps}

# ---- production -----------------------------------------------------------
# Restart the clock. fix ave/chunk writes on multiples of Nfreq and averages
# backwards over Nevery*Nrepeat steps, so on a non-zero starting timestep the
# window reaches back before the fix existed and NO frame is written at all --
# the file ends up holding nothing but its header. Resetting makes the
# averaging windows align exactly with this run.
reset_timestep  0

# Recreate the reservoirs so their energy accumulators cover the production
# window only. The flux comes from these, never from a nominal setpoint.
unfix           hot
unfix           cold
fix             hot  ghot  langevin {t_hot:.2f} {t_hot:.2f} {damp_ps:.4f} {seed_hot} tally yes
fix             cold gcold langevin {t_cold:.2f} {t_cold:.2f} {damp_ps:.4f} {seed_cold} tally yes

# Bin along z and average the temperature in each bin. The chunk ID
# ("layers") MUST follow Nfreq and precede the values -- omitting it makes
# LAMMPS read the first value as the chunk ID and fail with "No values from
# computes, fixes, or variables used in fix ave/chunk command".
compute         layers gmobile chunk/atom bin/1d z lower {bin_width:.4f} units box
fix             profile gmobile ave/chunk {profile_every} {n_profile_samples} \
{profile_window} layers temp file temperature_profile.dat

variable        ehot  equal f_hot
variable        ecold equal f_cold
variable        tnow  equal time
fix             energy all print {profile_every} "${{tnow}} ${{ehot}} ${{ecold}}" \
file reservoir_energy.dat screen no title "# time[ps] E_hot[eV] E_cold[eV]"

# Pxx/Pyy/Pzz, not just the scalar Press: `Press` is their mean, which
# averages away the anisotropy that matters. In-plane relaxation drives
# Pxx and Pyy to ~0 while Pzz keeps a residual, because z is non-periodic
# and cannot be barostatted -- and z is the transport direction, so Pzz is
# the component h_K is most sensitive to. Without these columns in the
# PRODUCTION block the stress check has nothing to test where it counts.
thermo_style    custom step temp press pxx pyy pzz f_hot f_cold
run             {production_steps}

print           "OPENKAPITZA_DONE area_m2 {area:.6e} interface_z_ang {interface_ang:.4f}"
"""

READ_DATA_BLOCK = """\
read_data       {data_file}
"""

# Self-contained diamond-cubic cell. Used so a crystalline run needs no
# externally supplied structure file; Phase 2's structure module replaces this
# for anything with a real lattice mismatch, an amorphous side, or defects.
BUILD_BLOCK = """\
# Built in place on a single diamond lattice -- ONE in-plane constant, so the
# B side carries {strain_b:+.2%} strain and the A side {strain_a:+.2%}. That
# strain shifts phonon frequencies and therefore h_K; it is a deliberate
# simplification of a coherent interface, not a free choice.
lattice         diamond {lattice_ang:.6f}
# x and y are PERIODIC, so they must be whole numbers of unit cells or the
# images do not match across the boundary. z is non-periodic (walls), so a
# partial cell at the end is harmless. The box extends {headroom_ang:.2f} A
# above the built cell so a piston-relaxed top wall cannot leave a fixed box
# (atoms outside an `f` boundary are lost); that space starts empty.
region          simbox block 0.0 {cross_ang:.4f} 0.0 {cross_ang:.4f} \
                0.0 {box_z_ang:.4f} units box
create_box      2 simbox
region          side_a block INF INF INF INF INF {interface_ang:.4f} units box
region          side_b block INF INF INF INF {interface_ang:.4f} {nz_ang:.4f} units box
create_atoms    1 region side_a
create_atoms    2 region side_b
mass            1 {mass_a:.4f}
mass            2 {mass_b:.4f}
"""

RELAX_BLOCK = """\

# ---- relaxation -----------------------------------------------------------
# The cell is built on ONE in-plane lattice constant, so the whole lattice
# mismatch shows up as in-plane strain: 6.4 GPa of internal pressure in the
# Si/Ge case, against ~0 for a relaxed crystal. That stress shifts phonon
# frequencies and therefore changes h_K directly, so an unrelaxed cell does
# not measure the interface it claims to.
#
# x and y are PERIODIC and can be barostatted. z cannot -- it is non-periodic
# (walls), and `fix npt` on a non-periodic dimension is a hard LAMMPS error.
# That is fine, because the mismatch strain is in-plane by construction.
# Relaxing x and y at zero pressure lets the shared in-plane constant settle
# where the net in-plane stress vanishes, splitting the mismatch between the
# two sides instead of putting all of it on the B side.
#
# `couple xy` keeps the cross-section square, which the periodic images
# require and which the cubic symmetry of both materials justifies.
minimize        1.0e-6 1.0e-8 10000 100000
# minimize advances the step counter by its iteration count. Reset it, or any
# fix ave/time window below never lands on an output step and reads as zero.
reset_timestep  0
velocity        gmobile create {t_init:.2f} {seed} mom yes rot yes dist gaussian
fix             relax gmobile npt temp {temperature:.2f} {temperature:.2f} {damp_ps:.4f} \
                couple xy x 0.0 0.0 {pdamp_ps:.4f} y 0.0 0.0 {pdamp_ps:.4f}
# Sample often enough to SEE the box converge. Inheriting a coarse interval
# from an earlier stage left two samples across 30 000 steps, which cannot
# distinguish a converged box from one still drifting -- and the relaxed
# in-plane constant is the number everything downstream inherits.
thermo          {relax_thermo}
thermo_style    custom step temp press pxx pyy pzz lx ly
{piston_on}run             {relax_steps}
unfix           relax
{piston_off}thermo_style    custom step temp press pe ke etotal
"""

# The top wall as a rigid piston along z, so the cell relaxes to sigma_zz = 0.
# setforce zeroes x and y and leaves z alone (NULL); aveforce then gives every
# wall atom the group's MEAN z-force, so the wall translates as one body and
# stops when the net force on it vanishes. Viscous damping kills the
# oscillation (time constant {damp_ps} ps). Defined in this order on purpose:
# setforce runs first, aveforce averages what it leaves.
PISTON_ON = """\
unfix           freeze
fix             freeze_lo gwall_lo setforce 0.0 0.0 0.0
fix             freeze_hi gwall_hi setforce 0.0 0.0 NULL
fix             piston gwall_hi aveforce NULL NULL 0.0
fix             piston_nve gwall_hi nve
fix             piston_damp gwall_hi viscous {gamma:.6e}
variable        zwall equal bound(gwall_hi,zmin)
# The cell breathes along z (period ~10 ps for a 20 nm Si/Ge slab), so the
# wall is frozen at its MEAN position over the second half, not wherever the
# oscillation leaves it -- the latter left up to +/-0.3 GPa in the first test.
fix             zav all ave/time 10 {nrep} {relax_steps} v_zwall
thermo_style    custom step temp press pxx pyy pzz lx ly v_zwall
"""

PISTON_OFF = """\
displace_atoms  gwall_hi move 0.0 0.0 $(f_zav-v_zwall) units box
unfix           zav
unfix           piston_damp
unfix           piston_nve
unfix           piston
unfix           freeze_hi
velocity        gwall_hi set 0.0 0.0 0.0 units box
fix             freeze_hi gwall_hi setforce 0.0 0.0 0.0
group           gside_a type 1
group           gside_b type 2
print           "OPENKAPITZA_RELAXED wall_hi_zmin $(bound(gwall_hi,zmin):%.4f) top_a $(bound(gside_a,zmax):%.4f) bottom_b $(bound(gside_b,zmin):%.4f)"
"""

# The spot-pressure template that used to live here is deliberately gone, not
# commented out, so it cannot be re-enabled by deleting a guard. It emitted
#     fix ... npt temp ... z <P> <P> <Pdamp>
# which LAMMPS rejects outright ("Cannot use fix npt on a non-periodic z
# dimension"), and before that a literal backslash it read as an npt keyword.
# Correct approach when someone builds it: leave the bottom wall fixed, make
# the top wall mobile in z, and apply a force per wall atom summing to
# sigma*A, so the normal stress is imposed rather than barostatted. Until
# then build_input_deck raises NotImplementedError for spot_pressure_GPa.


# amu * A/ps^2 in eV/A: the mass-acceleration unit of LAMMPS `metal` units.
AMU_A_PS2_IN_EV_A = 1.0364269e-4


def headroom(spec: NEMDSpec) -> float:
    """Empty space above the built cell, in metres, for the piston to move into.

    The largest plausible expansion is the B side relaxing from the shared
    lattice constant to its own: |a_b/a_used - 1| of its length. Twice that,
    and never less than 0.5 nm.
    """
    if not spec.relax_z:
        return 0.0
    g = spec.geometry
    strain = abs(spec.lattice_b / spec.lattice_used - 1.0) + abs(
        spec.lattice_a / spec.lattice_used - 1.0)
    return max(0.5e-9, 2.0 * strain * (g.bulk_a_length + g.bulk_b_length))


def _relax_steps(spec: NEMDSpec) -> int:
    """Relaxation length rounded up to a multiple of 20, so the averaging
    windows (every 10 steps, over the second half) tile it exactly."""
    return max(20, -(-spec.relax_steps // 20) * 20)


def _piston_gamma(spec: NEMDSpec) -> float:
    """Per-atom viscous coefficient giving a damping time of piston_damp_ps."""
    mass = spec.mass_b          # the top wall sits on side B
    return mass * AMU_A_PS2_IN_EV_A / spec.piston_damp_ps


def build_input_deck(spec: NEMDSpec) -> str:
    """Render the LAMMPS input deck for this run."""
    g = spec.geometry
    a = M_TO_ANG

    relax_block = ""
    if spec.relax_structure and spec.structure_mode == "build":
        relax_block = RELAX_BLOCK.format(
            temperature=spec.temperature,
            t_init=2.0 * spec.temperature,
            seed=spec.seed,
            damp_ps=spec.damping_fs * FS_TO_PS,
            pdamp_ps=10 * spec.damping_fs * FS_TO_PS,
            relax_steps=_relax_steps(spec),
            # At least 20 samples, so a drifting box is visible.
            relax_thermo=max(1, _relax_steps(spec) // 20),
            piston_on=(PISTON_ON.format(damp_ps=spec.piston_damp_ps,
                                        gamma=_piston_gamma(spec),
                                        relax_steps=_relax_steps(spec),
                                        nrep=_relax_steps(spec) // 20)
                       if spec.relax_z else ""),
            piston_off=PISTON_OFF if spec.relax_z else "",
        )

    pressure_block = ""
    if spec.spot_pressure_GPa:
        # Verified against LAMMPS: this path emits
        #   fix ... npt ... z <P> <P> <Pdamp>
        # and dies with "Cannot use fix npt on a non-periodic z dimension".
        # It also used to emit a literal backslash, which LAMMPS read as an
        # npt keyword. Neither had ever been run: spot_pressure_GPa defaults
        # to None and no fixture set it. Refusing beats emitting a deck that
        # crashes minutes into a job.
        raise NotImplementedError(
            "spot_pressure_GPa is not implemented. Normal load on a slab with "
            "non-periodic z cannot be applied with `fix npt`, which requires "
            "the barostatted dimension to be periodic; it needs a force on "
            "the wall group instead (sigma = F/A, top wall mobile in z, bottom "
            "wall fixed), which is not built yet. Leave it None and relax the "
            "in-plane box with relax_structure=True. See README 'Spot pressure'."
        )
    if spec.structure_mode == "build":
        a_ang = spec.lattice_used * M_TO_ANG
        sa, sb = spec.lattice_strain
        structure_block = BUILD_BLOCK.format(
            lattice_ang=a_ang,
            cross_ang=spec.snapped_cross_section * M_TO_ANG,
            nz_ang=g.total_length * M_TO_ANG,
            box_z_ang=(g.total_length + headroom(spec)) * M_TO_ANG,
            headroom_ang=headroom(spec) * M_TO_ANG,
            interface_ang=g.interface_z * M_TO_ANG,
            mass_a=spec.mass_a, mass_b=spec.mass_b,
            strain_a=sa, strain_b=sb,
        )
    elif spec.structure_mode == "read_data":
        structure_block = READ_DATA_BLOCK.format(data_file=spec.data_file)
    else:
        raise ValueError(f"unknown structure_mode {spec.structure_mode!r}")

    # Resolve the pair_coeff labels against the potential file whenever it is
    # readable. Falling straight back to the bare chemical symbols is what
    # produced `pair_coeff * * SiCGe.tersoff Si Ge` -- a deck LAMMPS rejects,
    # because that file's labels are Si(D) and Ge. The interrogation used to
    # live only in scripts/smoke_lammps.py, so anyone driving the library
    # directly got the invalid deck with nothing to warn them.
    pair_labels = spec.pair_elements
    if pair_labels is None:
        try:
            pair_labels = resolve_pair_elements(
                locate_potential(spec.pair_file, spec.potential_dir),
                spec.element_a, spec.element_b,
            )
        except (OSError, ValueError):
            # Unreadable (generating decks away from the potentials) or no
            # complete pair. Fall back to the bare symbols so deck generation
            # still works offline; check_spec raises the flag that says this
            # was not verified, and validate_deck cannot know either way.
            pair_labels = (spec.element_a, spec.element_b)

    # Emit several frames rather than one average over everything: the
    # extractor uses the last, and the earlier ones make drift visible if the
    # run had not actually reached steady state.
    window = max(
        spec.profile_every,
        (spec.production_steps // spec.profile_frames // spec.profile_every)
        * spec.profile_every,
    )
    n_samples = max(1, window // spec.profile_every)

    return DECK.format(
        structure_block=structure_block,
        element_a=spec.element_a,
        element_b=spec.element_b,
        pair_style=spec.pair_style,
        pair_file=spec.pair_file,
        pair_a=pair_labels[0],
        pair_b=pair_labels[1],
        timestep_ps=spec.timestep_fs * FS_TO_PS,
        damp_ps=spec.damping_fs * FS_TO_PS,
        z_wall_lo=g.wall_thickness * a,
        z_hot_end=g.z_bulk_a_start * a,
        z_cold_start=g.z_bulk_b_end * a,
        z_cold_end=g.z_cold_end * a,
        t_init=2.0 * spec.temperature,       # equipartition halves it
        temperature=spec.temperature,
        t_hot=spec.t_hot,
        t_cold=spec.t_cold,
        seed=spec.seed,
        seed_hot=spec.seed + 1,
        seed_cold=spec.seed + 2,
        equilibrate_steps=spec.equilibrate_steps,
        steady_steps=spec.steady_steps,
        production_steps=spec.production_steps,
        profile_every=spec.profile_every,
        n_profile_samples=n_samples,
        profile_window=window,
        bin_width=g.bin_width * a,
        area=(spec.snapped_area if spec.structure_mode == "build" else g.area),
        interface_ang=g.interface_z * a,
        pressure_block=pressure_block,
        relax_block=relax_block,
    )


def validate_deck(deck: str) -> list[str]:
    """Structural checks on a generated deck, returned as a list of problems.

    These are argument-order and consistency rules that LAMMPS enforces but
    only discovers at the offending line -- which, for a fix set up after
    equilibration, is minutes into a run. Checking the text costs
    microseconds.

    The rules here are the ones this package has actually got wrong:

    - `fix ave/chunk` takes the chunk ID after Nfreq and before the values.
      Omitting it makes LAMMPS read the first value as the chunk ID and fail
      with "No values from computes, fixes, or variables used".
    - Nfreq must be a multiple of Nevery, and Nevery*Nrepeat must not exceed
      Nfreq, or the averaging window does not fit.
    - z must be non-periodic, or heat short-circuits around the box and the
      measured conductance is meaningless rather than merely wrong.
    """
    problems: list[str] = []

    # Join backslash continuations so each command is one logical line.
    joined = deck.replace("\\\n", " ")
    lines = [ln.split("#", 1)[0].strip() for ln in joined.splitlines()]
    lines = [ln for ln in lines if ln]

    chunk_ids = {
        parts[1]
        for parts in (ln.split() for ln in lines)
        if len(parts) > 3 and parts[0] == "compute" and "chunk/atom" in parts
    }

    for ln in lines:
        parts = ln.split()
        if parts[0] != "fix" or "ave/chunk" not in parts:
            continue
        i = parts.index("ave/chunk")
        tail = parts[i + 1:]
        if len(tail) < 4:
            problems.append(f"ave/chunk has too few arguments: {ln}")
            continue
        try:
            nevery, nrepeat, nfreq = (int(x) for x in tail[:3])
        except ValueError:
            problems.append(f"ave/chunk Nevery/Nrepeat/Nfreq not integers: {ln}")
            continue
        chunk_id = tail[3]
        if chunk_id not in chunk_ids:
            problems.append(
                f"ave/chunk chunk ID {chunk_id!r} is not a compute chunk/atom "
                f"(known: {sorted(chunk_ids) or 'none'}); the chunk ID must "
                f"come straight after Nfreq"
            )
        if len(tail) < 5 or tail[4] in ("file", "ave", "norm", "title1"):
            problems.append(f"ave/chunk names no value to average: {ln}")
        if nevery <= 0 or nfreq % nevery:
            problems.append(
                f"ave/chunk Nfreq ({nfreq}) must be a multiple of Nevery "
                f"({nevery})"
            )
        elif nevery * nrepeat > nfreq:
            problems.append(
                f"ave/chunk Nevery*Nrepeat ({nevery * nrepeat}) exceeds Nfreq "
                f"({nfreq}); the averaging window does not fit"
            )

    # An ave/chunk whose averaging window predates the fix writes nothing at
    # all -- the file is created with a header and left empty. Guard it by
    # requiring the clock to be reset before the fix is defined.
    chunk_at = next(
        (i for i, ln in enumerate(lines)
         if ln.startswith("fix") and "ave/chunk" in ln),
        None,
    )
    if chunk_at is not None:
        reset_before = any(
            ln.split()[:2] == ["reset_timestep", "0"] for ln in lines[:chunk_at]
        )
        if not reset_before:
            problems.append(
                "no 'reset_timestep 0' before the ave/chunk fix; on a non-zero "
                "starting timestep the averaging window reaches back before "
                "the fix existed and no frame is ever written"
            )

    for ln in lines:
        if ln.startswith("boundary"):
            parts = ln.split()
            if len(parts) >= 4 and parts[3] != "f":
                problems.append(
                    f"z boundary is {parts[3]!r}, not 'f'; a periodic "
                    f"transport direction lets heat bypass the interface"
                )

    if not any(ln.startswith("pair_coeff") for ln in lines):
        problems.append("no pair_coeff line")

    return problems


def check_spec(spec: NEMDSpec, flags: FlagSet) -> None:
    """Pre-flight checks that catch the usual ways a NEMD run is wasted."""
    g = spec.geometry

    bins_per_side = min(g.bulk_a_length, g.bulk_b_length) / g.bin_width
    if bins_per_side < 20:
        flags.add(
            Code.NEMD_FEW_BINS,
            Severity.WARNING,
            f"Only {bins_per_side:.0f} bins span the shorter bulk slab.",
            remedy=(
                "After excluding the near-interface region there will be too "
                "few left to fit. Lengthen the slabs or narrow the bins."
            ),
            bins_per_side=bins_per_side,
        )

    gradient_K_per_m = spec.delta_reservoir / (g.bulk_a_length + g.bulk_b_length)
    if gradient_K_per_m > 2e9:
        flags.add(
            Code.NEMD_FLUX_NONLINEAR,
            Severity.WARNING,
            f"The imposed gradient is {gradient_K_per_m:.2e} K/m, which is "
            f"very steep.",
            remedy=(
                "Steep gradients leave linear response and make h_K look "
                "flux dependent. Reduce the reservoir split or lengthen the "
                "slabs, and confirm with the flux-linearity convergence test."
            ),
            gradient=gradient_K_per_m,
        )

    if spec.delta_reservoir > 0.4 * spec.temperature:
        flags.add(
            Code.NEMD_FLUX_NONLINEAR,
            Severity.WARNING,
            f"Reservoir split of {spec.delta_reservoir:.0f} K is large against "
            f"a mean temperature of {spec.temperature:.0f} K.",
            remedy=(
                "Material properties vary appreciably across the slab, so the "
                "profile will not be linear and the extracted h_K is not a "
                "single-temperature quantity."
            ),
        )

    if spec.structure_mode == "build":
        sa, sb = spec.lattice_strain
        worst = max(abs(sa), abs(sb))
        if worst > 0.01:
            flags.add(
                Code.NEMD_LATTICE_STRAIN,
                Severity.WARNING,
                f"The built cell uses one lattice constant, so the sides carry "
                f"{sa:+.2%} and {sb:+.2%} in-plane strain.",
                remedy=(
                    "Strain of this size shifts phonon frequencies and changes "
                    "h_K. It is the usual simplification for a coherent "
                    "interface, but report it, and use the Phase-2 structure "
                    "module with a strain-minimising supercell when the "
                    "mismatch matters."
                ),
                strain=worst,
            )

    # --- the potential file must actually support the requested pair -------
    # LAMMPS reports a missing triple only when it reaches the pair_coeff
    # line, which for a deck set up after equilibration is minutes into a
    # run. Checking here costs microseconds.
    try:
        triples = potential_triples(
            locate_potential(spec.pair_file, spec.potential_dir)
        )
    except OSError:
        flags.add(
            Code.NEMD_POTENTIAL_UNVERIFIED,
            Severity.WARNING,
            f"Could not read the potential file {spec.pair_file!r}, so the "
            f"pair_coeff labels were not verified against it.",
            remedy=(
                "This is expected when generating decks away from the LAMMPS "
                "potentials directory. Before submitting, re-run this check "
                "where the file exists -- a many-body file names its "
                "parameterisations in the label (Si(D), not Si) and may carry "
                "cross terms for only one of them."
            ),
            pair_file=str(spec.pair_file),
        )
    else:
        pairs = complete_pairs(triples, spec.element_a, spec.element_b)
        labels = sorted({lab for triple in triples for lab in triple})
        if not pairs:
            flags.add(
                Code.NEMD_POTENTIAL_INCOMPLETE,
                Severity.CRITICAL,
                f"{Path(spec.pair_file).name} has no complete set of triples "
                f"for {spec.element_a}/{spec.element_b}. Labels present: "
                f"{labels}.",
                remedy=(
                    "A two-element many-body system needs all eight ordered "
                    "triples over the two labels. LAMMPS will reject this "
                    "deck. Choose a potential that parameterises the pair; do "
                    "not hand-write a cross term."
                ),
                labels=labels,
            )
        elif spec.pair_elements is not None and tuple(spec.pair_elements) not in {
            tuple(p) for p in pairs
        }:
            flags.add(
                Code.NEMD_POTENTIAL_INCOMPLETE,
                Severity.CRITICAL,
                f"pair_elements {tuple(spec.pair_elements)} is not a complete "
                f"pairing in {Path(spec.pair_file).name}. Complete pairings: "
                f"{[tuple(p) for p in pairs]}.",
                remedy=(
                    "The requested labels exist but do not carry every "
                    "required triple -- this is the Si(B)/Si(D) trap, where "
                    "the first label matching 'Si' is not the one with the "
                    "cross terms. Use one of the complete pairings listed."
                ),
                requested=list(spec.pair_elements),
                complete=[list(p) for p in pairs],
            )

    if min(g.bulk_a_length, g.bulk_b_length) < 10e-9:
        flags.add(
            Code.NEMD_SIZE_LIMITED,
            Severity.CAUTION,
            f"Bulk slabs are {min(g.bulk_a_length, g.bulk_b_length) * 1e9:.1f} nm, "
            f"short against the phonon mean free paths in a crystalline solid.",
            remedy=(
                "The apparent conductivity will be size-limited and h_K may "
                "depend on slab length. Run the length convergence test and "
                "extrapolate rather than quoting one box."
            ),
        )


def read_chunk_frames(path: Path | str) -> list[tuple[int, np.ndarray]]:
    """Every frame of a `fix ave/chunk` file: (timestep, rows of coord, count, value).

    Coordinates are left in the file's units (Angstrom).
    """
    frames: list[tuple[int, list[tuple[float, float, float]]]] = []
    with open(Path(path), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 3:           # frame header: timestep nchunks total
                frames.append((int(float(parts[0])), []))
                continue
            if len(parts) >= 4 and frames:
                frames[-1][1].append((float(parts[1]), float(parts[2]), float(parts[3])))
    return [(step, np.asarray(rows)) for step, rows in frames if rows]


def _profile_from_rows(rows: np.ndarray, geometry: NEMDGeometry,
                       flags: FlagSet) -> TemperatureProfile:
    coord = rows[:, 0] / M_TO_ANG                     # Angstrom -> m
    counts = rows[:, 1]
    temps = rows[:, 2]
    keep = counts > 0
    dropped = int((~keep).sum())
    if dropped:
        flags.add(
            Code.NEMD_FEW_BINS,
            Severity.CAUTION,
            f"{dropped} of {counts.size} bins were empty and have been dropped.",
            remedy=(
                "Empty bins usually mean the binning extends past the slab, or "
                "the structure has a gap. Check the geometry against the data "
                "file."
            ),
            dropped=dropped,
        )
    return TemperatureProfile(
        z=coord[keep],
        T=temps[keep],
        interface_z=geometry.interface_z,
        area=geometry.area,
        counts=counts[keep],
    )


def parse_profile_frames(
    path: Path | str, geometry: NEMDGeometry, flags: FlagSet
) -> list[tuple[int, TemperatureProfile]]:
    """One TemperatureProfile per frame, with the frame's end timestep."""
    frames = read_chunk_frames(path)
    if not frames:
        raise ValueError(f"no chunk data found in {path}")
    return [(step, _profile_from_rows(rows, geometry, flags)) for step, rows in frames]


def parse_profile(
    path: Path | str, geometry: NEMDGeometry, flags: FlagSet, *,
    frames: str = "all",
) -> TemperatureProfile:
    """Read a LAMMPS `fix ave/chunk` file into a TemperatureProfile.

    The format is three comment lines, then per-frame blocks of
    `chunk coord ncount value`.

    `frames="all"` (default) averages every frame, weighted by atom count.
    The deck uses `fix ave/chunk` WITHOUT `ave running`, so each frame is an
    INDEPENDENT average over its own slice of the production window, and
    together they cover exactly the window the flux is measured over.

    This used to keep only the last frame, on the stated belief that earlier
    frames were running averages carrying the transient. They are not: with
    the default `ave one` each frame is its own window. Keeping the last of
    four discarded 75 % of the profile statistics and measured the jump over
    a quarter of the window the flux came from. Whether the production window
    holds a transient is `check_flux_stationarity`'s question, asked of the
    same window. `frames="last"` keeps the old behaviour for comparison.
    """
    raw = read_chunk_frames(path)
    if not raw:
        raise ValueError(f"no chunk data found in {path}")
    if frames == "last":
        return _profile_from_rows(raw[-1][1], geometry, flags)
    if frames != "all":
        raise ValueError("frames must be 'all' or 'last'")
    shapes = {rows.shape for _, rows in raw}
    if len(shapes) != 1:
        raise ValueError(f"frames have different bin counts: {sorted(shapes)}")
    stack = np.stack([rows for _, rows in raw])        # frames x bins x 3
    counts = stack[:, :, 1]
    total = counts.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        temp = np.where(total > 0, (stack[:, :, 2] * counts).sum(axis=0) / total, 0.0)
    rows = np.column_stack([stack[:, :, 0].mean(axis=0), total / len(raw), temp])
    return _profile_from_rows(rows, geometry, flags)


def parse_relaxed_positions(path: Path | str) -> dict[str, float] | None:
    """The piston relaxation's printed positions (Angstrom), or None.

    Keys: wall_hi_zmin, top_a, bottom_b. Present only when relax_z ran.
    """
    import re
    pat = re.compile(r"^OPENKAPITZA_RELAXED wall_hi_zmin (\S+) top_a (\S+) bottom_b (\S+)")
    try:
        with open(Path(path), "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = pat.match(line.strip())
                if m:
                    return {"wall_hi_zmin": float(m.group(1)), "top_a": float(m.group(2)),
                            "bottom_b": float(m.group(3))}
    except OSError:
        return None
    return None


def parse_log_thermo(path: Path | str) -> list[dict[str, float]]:
    """Thermo rows from a LAMMPS log, as dicts keyed by column name.

    A thermo block is a header line beginning with "Step" followed by numeric
    rows, ending at "Loop time". Several blocks appear in one log, one per
    `run`; all are returned in order.
    """
    rows: list[dict[str, float]] = []
    headers: list[str] | None = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Step":
                headers = parts
                continue
            if headers is None:
                continue
            if len(parts) == len(headers) and all(_is_number(p) for p in parts):
                rows.append({h: float(v) for h, v in zip(headers, parts)})
            else:
                headers = None
    return rows


def last_thermo_block(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """The final thermo block only, split on a non-increasing Step column.

    `parse_log_thermo` flattens every `run` in the log into one list, and
    averaging across all of them mixes stages that describe different
    physics. With relaxation enabled the log now holds minimisation rows at
    6.5 GPa and 540 K alongside production rows at 1.5 GPa and 300 K, so a
    mean over everything described no state the run was ever in -- it
    reported 3.64 GPa for a cell sitting at 1.51 GPa.

    `reset_timestep 0` before production makes the boundary unambiguous.
    """
    if not rows:
        return []
    start = 0
    for i in range(1, len(rows)):
        if rows[i].get("Step", 0) <= rows[i - 1].get("Step", 0):
            start = i
    return rows[start:]


def check_run_health(
    rows: list[dict[str, float]],
    target_temperature: float,
    flags: FlagSet,
    *,
    temperature_tolerance: float = 0.08,
    max_pressure_GPa: float = 1.0,
    use_last_block: bool = True,
) -> None:
    """Did the run actually do what the deck asked?

    Two things the first successful shakedown showed are worth checking
    automatically rather than reading off a log by eye.

    A mean temperature well away from the setpoint means equilibration did
    not finish; the system is still converting kinetic to potential energy
    and nothing measured on top of it is a steady state.

    A large internal pressure means the cell is strained. Building both
    sides on one lattice constant put 6.4 GPa into a Si/Ge cell -- the same
    4 % mismatch the strain flag describes, now with a magnitude. Pressure
    of that order shifts phonon frequencies enough to change h_K outright,
    so it is reported as a number rather than a caution.
    """
    if not rows:
        flags.add(
            Code.NEMD_UNEQUILIBRATED,
            Severity.CAUTION,
            "No thermo output could be read from the log, so the run could "
            "not be checked for equilibration or cell stress.",
            remedy="Confirm thermo_style and that the log was captured.",
        )
        return

    # Restrict to the production block: averaging across stages mixes
    # minimisation, relaxation and production, which describe different
    # states. See last_thermo_block.
    if use_last_block:
        rows = last_thermo_block(rows)

    temps = [r["Temp"] for r in rows if "Temp" in r]
    if temps:
        mean_T = sum(temps) / len(temps)
        drift = abs(mean_T - target_temperature) / target_temperature
        if drift > temperature_tolerance:
            flags.add(
                Code.NEMD_UNEQUILIBRATED,
                Severity.WARNING,
                f"Mean temperature over the run was {mean_T:.0f} K against a "
                f"{target_temperature:.0f} K setpoint ({drift * 100:.0f} % off).",
                remedy=(
                    "Equilibration has not finished: the system is still "
                    "relaxing, so the profile measured on top of it is not a "
                    "steady state. Lengthen the equilibration stage, and "
                    "check for stored strain energy draining into the "
                    "kinetic degrees of freedom."
                ),
                mean_T=mean_T,
            )

    # Check the stress COMPONENTS, not just the scalar pressure.
    #
    # `Press` is (Pxx+Pyy+Pzz)/3, so it averages away exactly the anisotropy
    # that matters here. Measured on a relaxed Si/Ge cell: Pxx = Pyy = 0.07
    # GPa but Pzz = 1.51 GPa, giving a mean of 0.55 GPa -- comfortably under a
    # 1 GPa threshold, no flag, while 1.5 GPa of uniaxial stress sits along
    # the TRANSPORT direction. In-plane relaxation cannot fix that, because z
    # is non-periodic and cannot be barostatted; the residual is real and it
    # is the component h_K is most sensitive to.
    #
    # So flag on the worst component when components are reported, and name
    # it. Fall back to the scalar when the thermo style did not include them.
    component_means: dict[str, float] = {}
    for key in ("Pxx", "Pyy", "Pzz"):
        values = [r[key] for r in rows if key in r]
        if values:
            component_means[key] = (sum(values) / len(values)) / 1e4

    if component_means:
        worst_key = max(component_means, key=lambda k: abs(component_means[k]))
        worst = component_means[worst_key]
        if abs(worst) > max_pressure_GPa:
            detail = ", ".join(f"{k} {v:+.2f}" for k, v in component_means.items())
            along_transport = (
                " This is the transport direction, so h_K is directly "
                "affected." if worst_key == "Pzz" else ""
            )
            flags.add(
                Code.NEMD_CELL_STRESS,
                Severity.WARNING,
                f"Residual stress of {worst:+.2f} GPa along {worst_key[1:]} "
                f"({detail} GPa); a relaxed crystal sits near zero."
                f"{along_transport}",
                remedy=(
                    "In-plane relaxation cannot remove a z component: z is "
                    "non-periodic (walls) and cannot be barostatted. Build "
                    "each side with its own out-of-plane constant set by its "
                    "Poisson ratio -- the pseudomorphic strained-layer "
                    "structure -- rather than putting both sides on one "
                    "lattice constant. That is the Phase-2 structure module."
                ),
                **{f"stress_{k}": v for k, v in component_means.items()},
            )

    presses = [r["Press"] for r in rows if "Press" in r]
    if presses and not component_means:
        mean_P_GPa = (sum(presses) / len(presses)) / 1e4   # bar -> GPa
        if abs(mean_P_GPa) > max_pressure_GPa:
            flags.add(
                Code.NEMD_CELL_STRESS,
                Severity.WARNING,
                f"Mean internal pressure was {mean_P_GPa:.2f} GPa; a relaxed "
                f"crystal at this temperature sits near zero.",
                remedy=(
                    "The cell is strained, almost always from building both "
                    "sides on one lattice constant. Stress of this magnitude "
                    "shifts phonon frequencies and changes h_K directly, so "
                    "the result is not the unstrained interface. Use a "
                    "strain-minimising supercell, or relax the cell before "
                    "imposing the gradient."
                ),
                pressure_GPa=mean_P_GPa,
            )


def parse_reservoir_series(
    path: Path | str,
) -> tuple[list[float], list[float], list[float]]:
    """The reservoir accumulators as time series: (t s, E_hot J, E_cold J).

    `parse_reservoir_energy` collapses this to last-minus-first, which is the
    right thing for the flux but throws away the information needed to tell
    whether the run was at steady state while that flux was measured.
    """
    EV_TO_J = 1.602176634e-19
    PS_TO_S = 1e-12
    t: list[float] = []
    hot: list[float] = []
    cold: list[float] = []
    with open(Path(path), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                values = [float(p) for p in parts[:3]]
            except ValueError:
                continue
            t.append(values[0] * PS_TO_S)
            hot.append(values[1] * EV_TO_J)
            cold.append(values[2] * EV_TO_J)
    return t, hot, cold


def check_flux_stationarity(
    path: Path | str,
    flags: FlagSet,
    *,
    tolerance: float = 0.15,
    n_windows: int = 10,
) -> dict[str, float]:
    """Is the reservoir power actually constant across the averaging window?

    WHY THIS EXISTS. A 20 ps and a 60 ps run of the same cell returned
    h_K = 17453 and 1348 MW/m^2K -- a factor of 13 -- while raising the
    IDENTICAL set of eleven flags between them. `NEMD_UNEQUILIBRATED` existed
    but never fired, because it compares mean temperature against the
    setpoint, and temperature equilibrating is NOT the same as heat flow
    reaching steady state.

    HOW (see `md.stationarity`). A trend test on the window-mean reservoir
    power whose standard error comes from the data's own scatter, so the
    verdict distinguishes three cases the first version conflated:
    drifting (WARNING, NEMD_UNEQUILIBRATED), stationary to within
    `tolerance` (no flag), and unresolved -- the data cannot show drift below
    the tolerance (CAUTION, NEMD_STATIONARITY_UNRESOLVED). The first version
    compared two half-window slopes against a fixed 15 %, and on a real
    400 ps run that ratio's own noise was 36 %, so it reported every run as
    not steady (open issue P1-1).

    Returns the measured relative drift per reservoir.
    """
    from .stationarity import Verdict, analyse_drift

    t, hot, cold = parse_reservoir_series(path)
    if len(t) < 4 * 3:
        flags.add(
            Code.NEMD_UNEQUILIBRATED,
            Severity.WARNING,
            f"Only {len(t)} reservoir-energy samples, too few to test whether "
            f"the heat flow was stationary.",
            remedy=(
                "Write the reservoir accumulators more often, or lengthen the "
                "production run. Without this test an unconverged run can "
                "return a plausible-looking h_K that is wrong by an order of "
                "magnitude."
            ),
            n_samples=len(t),
        )
        return {}

    results = {}
    for name, series in (("hot", hot), ("cold", cold)):
        try:
            results[name] = analyse_drift(t, series, tolerance=tolerance,
                                          n_windows=n_windows)
        except ValueError:
            continue
    drifts = {k: abs(r.drift) for k, r in results.items()}

    def _fmt(names):
        return ", ".join(
            f"{k} {results[k].drift * 100:+.0f} +/- {results[k].drift_se * 100:.0f} %"
            for k in sorted(names))

    drifting = [k for k, r in results.items() if r.verdict is Verdict.DRIFTING]
    unresolved = [k for k, r in results.items() if r.verdict is Verdict.UNRESOLVED]
    if drifting:
        flags.add(
            Code.NEMD_UNEQUILIBRATED,
            Severity.WARNING,
            f"Reservoir power is still drifting across the averaging window "
            f"({_fmt(drifting)}, significant against its own noise); the run "
            f"was not at steady state while the flux was measured.",
            remedy=(
                "h_K from a non-stationary run is not a measurement -- it can "
                "be wrong by an order of magnitude while every other "
                "diagnostic looks unremarkable. Lengthen the steady-state "
                "stage before production, then re-run."
            ),
            **{f"drift_{k}": results[k].drift for k in drifting},
            **{f"drift_se_{k}": results[k].drift_se for k in drifting},
        )
    elif unresolved:
        worst = max(results[k].resolution for k in unresolved)
        flags.add(
            Code.NEMD_STATIONARITY_UNRESOLVED,
            Severity.CAUTION,
            f"No significant drift in reservoir power ({_fmt(unresolved)}), but "
            f"the data can only resolve drift down to {worst * 100:.0f} %, above "
            f"the {tolerance * 100:.0f} % tolerance. Steady state is plausible, "
            f"not shown.",
            remedy=(
                "This is a statistics limit, not a transient: lengthen the "
                "PRODUCTION window or widen the cross-section to cut the "
                "power noise. A longer equilibration will not help."
            ),
            resolution=worst,
        )
    return drifts


def parse_reservoir_energy(path: Path | str) -> tuple[float, float, float]:
    """Read the reservoir accumulators: (E_hot J, E_cold J, duration s).

    LAMMPS `metal` units give energy in eV and time in ps. The values are
    cumulative, so the usable quantity is the change across the production
    window, taken as last minus first.
    """
    EV_TO_J = 1.602176634e-19
    PS_TO_S = 1e-12
    rows = []
    with open(Path(path), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                rows.append(tuple(float(p) for p in parts[:3]))
    if len(rows) < 2:
        raise ValueError(f"need at least two samples in {path}, got {len(rows)}")

    t0, h0, c0 = rows[0]
    t1, h1, c1 = rows[-1]
    return (
        abs(h1 - h0) * EV_TO_J,
        abs(c1 - c0) * EV_TO_J,
        (t1 - t0) * PS_TO_S,
    )
