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
    timestep_fs: float = 1.0
    equilibrate_steps: int = 500_000
    steady_steps: int = 500_000
    production_steps: int = 2_000_000
    profile_every: int = 100
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
# Reset the energy accumulators so the measured flux covers the production
# window only. The flux comes from these, never from a nominal setpoint.
unfix           hot
unfix           cold
fix             hot  ghot  langevin {t_hot:.2f} {t_hot:.2f} {damp_ps:.4f} {seed_hot} tally yes
fix             cold gcold langevin {t_cold:.2f} {t_cold:.2f} {damp_ps:.4f} {seed_cold} tally yes

compute         ke all ke/atom
variable        kT atom c_ke/(1.5*8.617333262e-5)

compute         layers gmobile chunk/atom bin/1d z lower {bin_width:.4f} units box
fix             profile gmobile ave/chunk {profile_every} {n_profile_samples} \
{production_steps} v_kT file temperature_profile.dat

variable        ehot  equal f_hot
variable        ecold equal f_cold
variable        tnow  equal time
fix             energy all print {profile_every} "${{tnow}} ${{ehot}} ${{ecold}}" \
file reservoir_energy.dat screen no title "# time[ps] E_hot[eV] E_cold[eV]"

thermo_style    custom step temp f_hot f_cold
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
# partial cell at the end is harmless and the box matches the geometry exactly.
region          simbox block 0.0 {cross_ang:.4f} 0.0 {cross_ang:.4f} \
                0.0 {nz_ang:.4f} units box
create_box      2 simbox
region          side_a block INF INF INF INF INF {interface_ang:.4f} units box
region          side_b block INF INF INF INF {interface_ang:.4f} INF units box
create_atoms    1 region side_a
create_atoms    2 region side_b
mass            1 {mass_a:.4f}
mass            2 {mass_b:.4f}
"""

PRESSURE_BLOCK = """\
# Spot pressure: the local stress inside a real contact spot is of order the
# contact microhardness (GPa), NOT the apparent contact pressure (MPa).
fix             press gmobile npt temp {temperature:.2f} {temperature:.2f} {damp_ps:.4f} \\
                z {pressure_bar:.2f} {pressure_bar:.2f} {pdamp_ps:.4f}
run             {press_steps}
unfix           press
"""


def build_input_deck(spec: NEMDSpec) -> str:
    """Render the LAMMPS input deck for this run."""
    g = spec.geometry
    a = M_TO_ANG

    pressure_block = ""
    if spec.spot_pressure_GPa:
        pressure_block = PRESSURE_BLOCK.format(
            temperature=spec.temperature,
            damp_ps=spec.damping_fs * FS_TO_PS,
            pdamp_ps=10 * spec.damping_fs * FS_TO_PS,
            pressure_bar=spec.spot_pressure_GPa * GPA_TO_BAR,
            press_steps=max(10_000, spec.equilibrate_steps // 5),
        )

    if spec.structure_mode == "build":
        a_ang = spec.lattice_used * M_TO_ANG
        sa, sb = spec.lattice_strain
        structure_block = BUILD_BLOCK.format(
            lattice_ang=a_ang,
            cross_ang=spec.snapped_cross_section * M_TO_ANG,
            nz_ang=g.total_length * M_TO_ANG,
            interface_ang=g.interface_z * M_TO_ANG,
            mass_a=spec.mass_a, mass_b=spec.mass_b,
            strain_a=sa, strain_b=sb,
        )
    elif spec.structure_mode == "read_data":
        structure_block = READ_DATA_BLOCK.format(data_file=spec.data_file)
    else:
        raise ValueError(f"unknown structure_mode {spec.structure_mode!r}")

    n_samples = max(1, spec.production_steps // spec.profile_every)

    return DECK.format(
        structure_block=structure_block,
        element_a=spec.element_a,
        element_b=spec.element_b,
        pair_style=spec.pair_style,
        pair_file=spec.pair_file,
        pair_a=(spec.pair_elements or (spec.element_a, spec.element_b))[0],
        pair_b=(spec.pair_elements or (spec.element_a, spec.element_b))[1],
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
        bin_width=g.bin_width * a,
        area=(spec.snapped_area if spec.structure_mode == "build" else g.area),
        interface_ang=g.interface_z * a,
        pressure_block=pressure_block,
    )


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


def parse_profile(
    path: Path | str, geometry: NEMDGeometry, flags: FlagSet
) -> TemperatureProfile:
    """Read a LAMMPS `fix ave/chunk` file into a TemperatureProfile.

    The format is three comment lines, then per-frame blocks of
    `chunk coord ncount value`. Only the LAST frame is used: earlier frames
    are the running averages LAMMPS emits along the way, and mixing them in
    would blend the transient into the steady state.
    """
    path = Path(path)
    rows: list[tuple[float, float, float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 3:           # frame header: timestep nchunks total
                rows = []
                continue
            if len(parts) >= 4:
                rows.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if not rows:
        raise ValueError(f"no chunk data found in {path}")

    coord = np.array([r[0] for r in rows]) / M_TO_ANG   # Angstrom -> m
    counts = np.array([r[1] for r in rows])
    temps = np.array([r[2] for r in rows])

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
