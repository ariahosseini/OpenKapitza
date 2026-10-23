"""Crystalline interfaces in any orientation: lattice matching and slabs (P2-1).

Section 3 of the agent instruction: "cut surfaces, stack them and find a
supercell that minimises lattice-mismatch strain; report the residual strain
and the supercell size." The NEMD deck already builds (100) cube-on-cube
diamond/fcc pairs itself; this module covers everything else -- other
orientations, other structures (zincblende, wurtzite GaN, 4H-SiC), and
multi-species sides -- and writes a data file for the read-in NEMD path.

LATTICE MATCHING (Zur & McGill, J. Appl. Phys. 55, 378, 1984 -- the method
pymatgen's ZSLGenerator implements). Enumerate small supercells of each
surface lattice and keep pairs whose vectors agree within a strain
tolerance. Restricted here to RECTANGULAR supercells, so the NEMD box stays
orthogonal; every surface used for electronics has them ((100), (110),
(111) and hexagonal (0001) all do). The smallest-area match within the
tolerance wins, then the smallest strain.

Lengths in Angstrom throughout.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .analysis import Cell
from .crystals import MASS

MASS = {**MASS, "C": 12.011, "Ga": 69.723, "N": 14.007}


# --------------------------------------------------------------------------
# Crystals
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Crystal:
    name: str
    lattice: np.ndarray                    # 3x3, rows are cell vectors
    basis: tuple[tuple[str, tuple[float, float, float]], ...]   # species, fractional

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.lattice)))

    @property
    def species(self) -> tuple[str, ...]:
        seen: list[str] = []
        for s, _ in self.basis:
            if s not in seen:
                seen.append(s)
        return tuple(seen)

    def density(self) -> float:
        m = sum(MASS[s] for s, _ in self.basis)
        return m * 1.66053906660e-24 / (self.volume * 1e-24)

    def nearest_neighbour(self) -> float:
        pos = self.cartesian()
        best = math.inf
        for t in itertools.product((-1, 0, 1), repeat=3):
            shift = np.asarray(t) @ self.lattice
            d = np.linalg.norm(pos[:, None, :] + shift - pos[None, :, :], axis=-1)
            d = d[d > 1e-6]
            if d.size:
                best = min(best, float(d.min()))
        return best

    def cartesian(self) -> np.ndarray:
        return np.array([f for _, f in self.basis]) @ self.lattice

    def centring(self) -> np.ndarray:
        """Fractional translations, besides the cell vectors, that map the
        crystal onto itself (fcc and diamond: the three a/2<110> vectors).

        Found from the basis, not hard-coded: a vector taking every atom onto
        an atom of the same species is a lattice translation. Without these
        the in-plane search on a conventional cubic cell misses the primitive
        surface cells -- the 45-degree (100) cell and every short (111) one.
        """
        frac = np.array([f for _, f in self.basis]) % 1.0
        spec = [s for s, _ in self.basis]
        out = [np.zeros(3)]
        for j in range(1, len(frac)):
            if spec[j] != spec[0]:
                continue
            t = (frac[j] - frac[0]) % 1.0
            moved = (frac + t) % 1.0
            ok = all(any(spec[k] == spec[i] and np.allclose(
                         ((moved[i] - frac[k] + 0.5) % 1.0) - 0.5, 0, atol=1e-6)
                         for k in range(len(frac))) for i in range(len(frac)))
            if ok and not any(np.allclose(t, o) for o in out):
                out.append(t)
        return np.array(out)


def cubic(a: float) -> np.ndarray:
    return np.eye(3) * a


def hexagonal(a: float, c: float) -> np.ndarray:
    return np.array([[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, c]])


_FCC = ((0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0))


def diamond(name: str, a: float, el: str = "Si") -> Crystal:
    b = [(el, p) for p in _FCC] + [(el, tuple(x + .25 for x in p)) for p in _FCC]
    return Crystal(name, cubic(a), tuple(b))


def zincblende(name: str, a: float, cation: str, anion: str) -> Crystal:
    b = [(cation, p) for p in _FCC] + [(anion, tuple(x + .25 for x in p)) for p in _FCC]
    return Crystal(name, cubic(a), tuple(b))


def fcc(name: str, a: float, el: str) -> Crystal:
    return Crystal(name, cubic(a), tuple((el, p) for p in _FCC))


def wurtzite(name: str, a: float, c: float, cation: str, anion: str,
             u: float = 0.375) -> Crystal:
    b = ((cation, (1 / 3, 2 / 3, 0.0)), (cation, (2 / 3, 1 / 3, 0.5)),
         (anion, (1 / 3, 2 / 3, u)), (anion, (2 / 3, 1 / 3, 0.5 + u)))
    return Crystal(name, hexagonal(a, c), b)


def sic_4h(name: str, a: float, c: float, u: float = 0.1875) -> Crystal:
    """4H-SiC, stacking ABCB; Si at z = 0, 1/4, 1/2, 3/4, C displaced by u."""
    sites = {"A": (0.0, 0.0), "B": (1 / 3, 2 / 3), "C": (2 / 3, 1 / 3)}
    b = []
    for k, s in enumerate("ABCB"):
        x, y = sites[s]
        b.append(("Si", (x, y, k / 4)))
        b.append(("C", (x, y, k / 4 + u)))
    return Crystal(name, hexagonal(a, c), tuple(b))


# Room-temperature experimental constants (standard handbook values). For MD,
# build on the POTENTIAL's own constants (knowledge base: a relaxed cell
# follows the potential, whatever it was built from).
LIBRARY = {
    "Si": lambda: diamond("Si", 5.431, "Si"),
    "Ge": lambda: diamond("Ge", 5.658, "Ge"),
    "Al": lambda: fcc("Al", 4.0495, "Al"),
    "Cu": lambda: fcc("Cu", 3.6149, "Cu"),
    "3C-SiC": lambda: zincblende("3C-SiC", 4.3596, "Si", "C"),
    "GaN": lambda: wurtzite("GaN", 3.189, 5.185, "Ga", "N", 0.377),
    "4H-SiC": lambda: sic_4h("4H-SiC", 3.073, 10.053),
}


# --------------------------------------------------------------------------
# Surfaces and rectangular supercells
# --------------------------------------------------------------------------

def surface_normal(crystal: Crystal, hkl: tuple[int, int, int]) -> np.ndarray:
    """Unit normal of the (hkl) plane (Miller indices on the given cell)."""
    recip = np.linalg.inv(crystal.lattice).T          # rows: reciprocal vectors / 2pi
    n = np.asarray(hkl, dtype=float) @ recip
    return n / np.linalg.norm(n)


def in_plane_vectors(crystal: Crystal, hkl: tuple[int, int, int], *,
                     l_max: float = 40.0, n_max: int = 12) -> np.ndarray:
    """Lattice vectors lying in the (hkl) plane, shorter than l_max."""
    n = surface_normal(crystal, hkl)
    idx = np.array(list(itertools.product(range(-n_max, n_max + 1), repeat=3)), dtype=float)
    vecs = np.vstack([(idx + c) @ crystal.lattice for c in crystal.centring()])
    ok = (np.abs(vecs @ n) < 1e-6) & (np.linalg.norm(vecs, axis=1) > 1e-6) & \
         (np.linalg.norm(vecs, axis=1) <= l_max)
    return vecs[ok]


@dataclass(frozen=True)
class RectCell:
    v1: np.ndarray            # 3D lattice vectors, orthogonal, in the plane
    v2: np.ndarray

    @property
    def lengths(self) -> tuple[float, float]:
        return float(np.linalg.norm(self.v1)), float(np.linalg.norm(self.v2))

    @property
    def area(self) -> float:
        return self.lengths[0] * self.lengths[1]


def rectangular_cells(crystal: Crystal, hkl: tuple[int, int, int], *,
                      l_max: float = 40.0) -> list[RectCell]:
    """Rectangular supercells of the (hkl) surface lattice, smallest first.

    Each pair of orthogonal in-plane lattice vectors spans one. Only the
    shortest representative of each (length1, length2) is kept.
    """
    vecs = in_plane_vectors(crystal, hkl, l_max=l_max)
    n = surface_normal(crystal, hkl)
    norms = np.linalg.norm(vecs, axis=1)
    order = np.argsort(norms, kind="stable")
    vecs, norms = vecs[order], norms[order]
    gram = vecs @ vecs.T
    hand = np.cross(vecs[:, None, :], vecs[None, :, :]) @ n
    ii, jj = np.nonzero((np.abs(gram) <= 1e-6 * np.outer(norms, norms)) & (hand > 0))
    cells: dict[tuple[float, float], RectCell] = {}
    for i, j in zip(ii, jj):                 # row-major over the sorted list: shortest first
        key = (round(norms[i], 4), round(norms[j], 4))
        if key not in cells:
            cells[key] = RectCell(vecs[i], vecs[j])
    return sorted(cells.values(), key=lambda c: (c.area, c.lengths))


@dataclass(frozen=True)
class Match:
    cell_a: RectCell
    cell_b: RectCell
    strain_b: tuple[float, float]      # B strained onto A: L_A/L_B - 1 per axis
    width: tuple[float, float]         # common cell, mismatch split evenly

    @property
    def max_strain(self) -> float:
        return max(abs(self.strain_b[0]), abs(self.strain_b[1]))

    def strains(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """In-plane strain of each side in the common (split) cell."""
        la, lb = self.cell_a.lengths, self.cell_b.lengths
        return ((self.width[0] / la[0] - 1, self.width[1] / la[1] - 1),
                (self.width[0] / lb[0] - 1, self.width[1] / lb[1] - 1))


def match(a: Crystal, hkl_a: tuple[int, int, int], b: Crystal, hkl_b: tuple[int, int, int],
          *, max_strain: float = 0.02, l_max: float = 40.0) -> Match:
    """Smallest-area rectangular supercell pair with mismatch <= max_strain."""
    cells_a = rectangular_cells(a, hkl_a, l_max=l_max)
    cells_b = rectangular_cells(b, hkl_b, l_max=l_max)
    la = np.array([c.lengths for c in cells_a])                 # (Na, 2)
    lb = np.array([c.lengths for c in cells_b])                 # (Nb, 2)
    strain = la[:, None, :] / lb[None, :, :] - 1.0              # (Na, Nb, 2)
    worst = np.abs(strain).max(axis=2)
    ia, ib = np.nonzero(worst <= max_strain)
    if len(ia) == 0:
        raise ValueError(f"no rectangular match within {max_strain:.1%} up to {l_max} A; "
                         f"raise max_strain or l_max")
    area = la[ia, 0] * la[ia, 1]
    k = np.lexsort((worst[ia, ib], area))[0]                    # smallest area, then strain
    i, j = int(ia[k]), int(ib[k])
    s = (float(strain[i, j, 0]), float(strain[i, j, 1]))
    width = (0.5 * (la[i, 0] + lb[j, 0]), 0.5 * (la[i, 1] + lb[j, 1]))
    return Match(cells_a[i], cells_b[j], s, width)


# --------------------------------------------------------------------------
# Slabs
# --------------------------------------------------------------------------

@dataclass
class Slab:
    pos: np.ndarray            # (N, 3) in the cell frame: x along v1, y along v2, z normal
    species: list[str]
    lengths: tuple[float, float]
    thickness: float
    meta: dict = field(default_factory=dict)

    def bottom_species(self, tol: float = 0.3) -> set[str]:
        return {s for s, z in zip(self.species, self.pos[:, 2]) if z < self.pos[:, 2].min() + tol}

    def top_species(self, tol: float = 0.3) -> set[str]:
        return {s for s, z in zip(self.species, self.pos[:, 2]) if z > self.pos[:, 2].max() - tol}


def build_slab(crystal: Crystal, cell: RectCell, thickness: float, *,
               z_offset: float = 0.0) -> Slab:
    """Atoms of `crystal` in the rectangular cell, from its lowest plane up.

    `z_offset` shifts the cut through the crystal (to choose a termination);
    the slab is then translated so its lowest atom sits at z = 0.
    """
    x = cell.v1 / np.linalg.norm(cell.v1)
    y = cell.v2 / np.linalg.norm(cell.v2)
    z = np.cross(x, y)
    rot = np.vstack([x, y, z])                          # frame rows
    L1, L2 = cell.lengths
    reach = L1 + L2 + thickness + 2 * float(np.linalg.norm(crystal.lattice, axis=1).max())
    nmax = int(math.ceil(reach / float(np.linalg.norm(crystal.lattice, axis=1).min()))) + 1
    idx = np.array(list(itertools.product(range(-nmax, nmax + 1), repeat=3)))
    pts = idx @ crystal.lattice
    frac = np.array([f for _, f in crystal.basis])
    names = [s for s, _ in crystal.basis]
    pos, spec = [], []
    for k, f in enumerate(frac):
        r = (pts + f @ crystal.lattice) @ rot.T
        r[:, 2] -= z_offset
        keep = (r[:, 2] >= -1e-6) & (r[:, 2] < thickness - 1e-6)
        r = r[keep]
        r[:, 0] %= L1
        r[:, 1] %= L2
        pos.append(r)
        spec += [names[k]] * len(r)
    pos = np.vstack(pos)
    # Periodic images collapse onto the same wrapped position: keep one.
    key = np.round(np.c_[pos[:, 0] % L1, pos[:, 1] % L2, pos[:, 2]], 4)
    key[np.isclose(key[:, 0], L1, atol=1e-4), 0] = 0.0
    key[np.isclose(key[:, 1], L2, atol=1e-4), 1] = 0.0
    _, first = np.unique(key, axis=0, return_index=True)
    first = np.sort(first)
    pos, spec = pos[first], [spec[i] for i in first]
    z_shift = float(pos[:, 2].min())
    pos[:, 2] -= z_shift
    return Slab(pos, spec, (L1, L2), thickness,
                meta={"crystal": crystal.name, "z_shift": z_shift + z_offset})


def slab_cell(slab: Slab, types: dict[str, int], z_box: float) -> Cell:
    """As an analysis Cell (periodic in x, y only)."""
    t = np.array([types[s] for s in slab.species])
    masses = {types[s]: MASS[s] for s in set(slab.species)}
    return Cell(slab.pos.copy(), t, np.zeros(3), np.array([*slab.lengths, z_box]),
                masses, periodic=(True, True, False))


# --------------------------------------------------------------------------
# Bicrystal
# --------------------------------------------------------------------------

INTERFACE_CONTACT = 0.95        # closest A-B distance / mean nearest-neighbour distance


@dataclass
class Bicrystal:
    cell: Cell
    types_a: tuple[int, ...]
    types_b: tuple[int, ...]
    type_species: dict[int, str]
    match: Match
    gap: float
    interface_z: float
    report: dict = field(default_factory=dict)


def join_bicrystal(a: Crystal, hkl_a, b: Crystal, hkl_b, *, length_a: float,
                   length_b: float, z_start: float = 0.0, headroom: float = 10.0,
                   max_strain: float = 0.02, repeat: tuple[int, int] = (1, 1),
                   z_offset_a: float = 0.0, z_offset_b: float = 0.0,
                   shift: tuple[float, float] = (0.0, 0.0)) -> Bicrystal:
    """A slab of `a` below a slab of `b`, laterally strained to a common cell.

    `shift` (A) translates B in-plane relative to A: the interface registry,
    which lattice matching alone does not fix (`scan_registry` chooses it).

    `repeat` tiles the matched cell laterally (the match is the smallest
    commensurate cell, often too narrow for NEMD). The gap is set from
    lateral registry so the closest A-B pair is INTERFACE_CONTACT times the
    mean nearest-neighbour distance.
    """
    m = match(a, hkl_a, b, hkl_b, max_strain=max_strain)
    sa = build_slab(a, m.cell_a, length_a, z_offset=z_offset_a)
    sb = build_slab(b, m.cell_b, length_b, z_offset=z_offset_b)
    W = (m.width[0] * repeat[0], m.width[1] * repeat[1])

    def place(slab: Slab, own: tuple[float, float]) -> np.ndarray:
        p = slab.pos.copy()
        p[:, 0] *= m.width[0] / own[0]
        p[:, 1] *= m.width[1] / own[1]
        tiles = [p + np.array([i * m.width[0], j * m.width[1], 0.0])
                 for i in range(repeat[0]) for j in range(repeat[1])]
        return np.vstack(tiles)

    pa = place(sa, m.cell_a.lengths)
    pb = place(sb, m.cell_b.lengths)
    pb[:, 0] = (pb[:, 0] + shift[0]) % W[0]
    pb[:, 1] = (pb[:, 1] + shift[1]) % W[1]
    spa = sa.species * (repeat[0] * repeat[1])
    spb = sb.species * (repeat[0] * repeat[1])
    top_a = pa[:, 2].max()
    ta = pa[np.isclose(pa[:, 2], top_a, atol=0.3)]
    bb = pb[np.isclose(pb[:, 2], pb[:, 2].min(), atol=0.3)]

    def wrap(d, L):
        return (d + L / 2) % L - L / 2

    d2 = (wrap(ta[:, None, 0] - bb[None, :, 0], W[0]) ** 2
          + wrap(ta[:, None, 1] - bb[None, :, 1], W[1]) ** 2)       # (n_top_a, n_bottom_b)
    dpar2 = float(d2.min())
    # Mean over B's interface atoms of the nearest in-plane A neighbour: the
    # registry score. The minimum alone is zero for every shift once the
    # matched cell holds many atoms per plane.
    mean_clearance = float(np.sqrt(d2.min(axis=0)).mean())
    r0 = INTERFACE_CONTACT * 0.5 * (a.nearest_neighbour() + b.nearest_neighbour())
    dz_b = bb[:, 2].mean() - pb[:, 2].min()
    gap = math.sqrt(max(r0 * r0 - dpar2, (0.5 * r0) ** 2))
    pa[:, 2] += z_start
    pb[:, 2] += z_start + top_a + gap - pb[:, 2].min() - dz_b

    species_a = list(a.species)
    species_b = list(b.species)
    tmap_a = {s: k + 1 for k, s in enumerate(species_a)}
    tmap_b = {s: len(species_a) + k + 1 for k, s in enumerate(species_b)}
    types = np.r_[[tmap_a[s] for s in spa], [tmap_b[s] for s in spb]]
    masses = {**{t: MASS[s] for s, t in tmap_a.items()}, **{t: MASS[s] for s, t in tmap_b.items()}}
    pos = np.vstack([pa, pb])
    top = pos[:, 2].max()
    cell = Cell(pos, types, np.zeros(3), np.array([W[0], W[1], top + headroom]), masses,
                periodic=(True, True, False))
    (ea, eb) = m.strains()
    report = {
        "cell_a_lengths": m.cell_a.lengths, "cell_b_lengths": m.cell_b.lengths,
        "width": W, "strain_a": ea, "strain_b": eb,
        "atoms": len(types), "termination_a_top": sorted(sa.top_species()),
        "termination_b_bottom": sorted(sb.bottom_species()),
        "closest_ab": math.sqrt(dpar2 + gap * gap), "r0": r0,
        "shift": tuple(float(x) for x in shift), "lateral_clearance": math.sqrt(dpar2),
        "mean_clearance": mean_clearance,
    }
    return Bicrystal(cell, tuple(tmap_a.values()), tuple(tmap_b.values()),
                     {**{t: s for s, t in tmap_a.items()}, **{t: s for s, t in tmap_b.items()}},
                     m, gap, z_start + top_a + 0.5 * gap, report)


def bicrystal_for_nemd(a: Crystal, hkl_a, b: Crystal, hkl_b, geometry, *,
                       max_strain: float = 0.02, min_width: float = 20.0,
                       headroom: float = 10.0, z_offset_a: float = 0.0,
                       z_offset_b: float = 0.0,
                       shift: tuple[float, float] = (0.0, 0.0)) -> Bicrystal:
    """A bicrystal laid out on an NEMDGeometry, for the read-in NEMD deck.

    Side A fills z in [0, interface_z), side B [interface_z, total_length):
    the deck's walls and reservoirs are z-regions of the geometry, so the
    atoms must sit where the geometry says. The matched cell is tiled until
    each lateral length is at least `min_width` (A).
    """
    m = match(a, hkl_a, b, hkl_b, max_strain=max_strain)
    repeat = tuple(max(1, math.ceil(min_width / w)) for w in m.width)
    iz = geometry.interface_z * 1e10
    total = geometry.total_length * 1e10
    kw = dict(headroom=headroom, max_strain=max_strain, repeat=repeat,
              z_offset_a=z_offset_a, z_offset_b=z_offset_b, shift=shift)
    bc = join_bicrystal(a, hkl_a, b, hkl_b, length_a=iz, length_b=total - iz, **kw)
    # B starts a gap above A's top plane, not at iz: trim it so it ends at total.
    b_start = bc.cell.pos[np.isin(bc.cell.types, bc.types_b), 2].min()
    bc = join_bicrystal(a, hkl_a, b, hkl_b, length_a=iz, length_b=total - b_start, **kw)
    bc.report["repeat"] = repeat
    bc.report["geometry_interface_z"] = iz
    bc.report["interface_offset"] = bc.interface_z - iz    # A: built plane vs geometry
    return bc


# --------------------------------------------------------------------------
# Registry: the in-plane shift of B relative to A
# --------------------------------------------------------------------------

@dataclass
class RegistryPoint:
    shift: tuple[float, float]
    gap: float
    lateral_clearance: float     # closest in-plane distance between facing planes, A
    mean_clearance: float        # mean over B's interface atoms of the nearest A, A
    score: float                 # lower is better


@dataclass
class RegistryScan:
    points: list[RegistryPoint]
    best: RegistryPoint
    method: str
    spread: float                # max - min score, same units as score


def scan_registry(a: Crystal, hkl_a, b: Crystal, hkl_b, *, n: int = 6,
                  score="hollow", max_strain: float = 0.02,
                  z_offset_a: float = 0.0, z_offset_b: float = 0.0,
                  thickness_a: float | None = None,
                  thickness_b: float | None = None) -> RegistryScan:
    """Score an n x n grid of in-plane shifts over one matched cell.

    Lattice matching (Zur-McGill, ZSL) fixes the cells, not where B sits on
    A (Tang & Butler, arXiv:2608.19933). `score`:
      "hollow"  -- geometric, no potential: the MEAN lateral clearance of B's
                   interface atoms from A's top plane (score = -clearance).
                   A DIAGNOSTIC of whether registry can matter (its spread is
                   ~0 for large incommensurate-like cells: Si(111)/Al(111)),
                   NOT a selector: on Si(100)/Al(100) with MEAM it
                   anti-correlates with the relaxed energy (r = -0.38).
      callable  -- f(Bicrystal) -> energy (lower is better), e.g.
                   `lammps_static_energy(...)`: the real criterion.
    Thin slabs (a few periods) are enough: only the facing planes differ.
    """
    m = match(a, hkl_a, b, hkl_b, max_strain=max_strain)
    ta = thickness_a or 2.0 * max(np.linalg.norm(a.lattice, axis=1))
    tb = thickness_b or 2.0 * max(np.linalg.norm(b.lattice, axis=1))
    method = score if isinstance(score, str) else getattr(score, "__name__", "energy")
    if isinstance(score, str) and score != "hollow":
        raise ValueError("score must be 'hollow' or a callable")
    points = []
    for i in range(n):
        for j in range(n):
            sh = (m.width[0] * i / n, m.width[1] * j / n)
            bc = join_bicrystal(a, hkl_a, b, hkl_b, length_a=ta, length_b=tb,
                                max_strain=max_strain, z_offset_a=z_offset_a,
                                z_offset_b=z_offset_b, shift=sh)
            clr = bc.report["lateral_clearance"]
            mcl = bc.report["mean_clearance"]
            val = -mcl if score == "hollow" else float(score(bc))
            points.append(RegistryPoint((float(sh[0]), float(sh[1])), bc.gap, clr, mcl, val))
    best = min(points, key=lambda p: p.score)
    vals = [p.score for p in points]
    return RegistryScan(points, best, method, max(vals) - min(vals))


def lammps_static_energy(lmp: str, pair_style: str, pair_coeff: str, *,
                         potential_dir=None, files: tuple[str, ...] = (),
                         workdir=None, minimize: bool = False):
    """Energy scorer for `scan_registry`: LAMMPS `run 0` (or a short minimize).

    Returns f(Bicrystal) -> total energy per area, eV/A^2. Unrelaxed by
    default: a screen of registries at the registry-set gap; `minimize=True`
    relaxes positions (box fixed) for a fairer, slower comparison.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path
    from .analysis import write_lammps_data

    def energy(bc: Bicrystal) -> float:
        wd = Path(workdir or tempfile.mkdtemp(prefix="registry_"))
        wd.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(Path(potential_dir) / f, wd / f)
        write_lammps_data(bc.cell, wd / "cell.data")
        # Shrink-wrapped z: free surfaces on both slabs, and atoms that start
        # on the data file's box face are not lost when minimisation moves them.
        body = ("units metal\nboundary p p m\natom_style atomic\nread_data cell.data\n"
                f"pair_style {pair_style}\npair_coeff * * {pair_coeff}\n"
                + ("minimize 1e-8 1e-10 2000 20000\n" if minimize else "run 0\n")
                + 'print "OPENKAPITZA_PE $(pe:%.8f) $(lx*ly:%.6f)"\n')
        (wd / "in.e").write_text(body)
        res = subprocess.run([lmp, "-in", "in.e", "-log", "none"], cwd=wd,
                             capture_output=True, text=True)
        for ln in res.stdout.splitlines():
            if ln.startswith("OPENKAPITZA_PE"):
                _, pe, area = ln.split()
                return float(pe) / float(area)
        raise RuntimeError(f"LAMMPS energy failed in {wd}: {res.stdout[-400:]}{res.stderr[-400:]}")
    energy.__name__ = "lammps_static_energy" + ("_minimized" if minimize else "")
    return energy


@dataclass
class Termination:
    z_offset: float              # pass to build_slab / join_bicrystal (A)
    bottom: tuple[str, ...]      # species on the slab's bottom face
    top: tuple[str, ...]         # species on its top face, for this thickness
    bottom_coordination_loss: int = 0   # bonds cut per bottom-face atom (covalent)


def terminations(crystal: Crystal, hkl, *, thickness: float, max_strain_cell=None,
                 tol: float = 0.05) -> list[Termination]:
    """Every distinct cut of `crystal` normal to (hkl), with the faces it gives.

    For compound (polar) surfaces the termination is a CHOICE with physical
    consequences (Ga- vs N-polar GaN, Si- vs C-face SiC, which species bonds
    across the interface). This lists them, with the bonds each cut breaks per
    bottom-face atom; the caller picks one, or scores them by energy when a
    cross potential exists. Cuts are placed just below each distinct plane of
    the middle period of a thick probe slab.
    """
    cell = max_strain_cell or rectangular_cells(crystal, hkl)[0]
    period = 3.0 * float(np.linalg.norm(crystal.lattice, axis=1).max())
    probe = build_slab(crystal, cell, 3.0 * period)
    pos, L1, L2 = probe.pos, probe.lengths[0], probe.lengths[1]
    nn = crystal.nearest_neighbour()
    # Bonds from the central cell to a 3 x 3 lateral tiling. Minimum-image
    # distances are wrong here: a primitive surface cell (Si(100): 3.84 A) is
    # smaller than twice the bond reach, so an atom and its own periodic image
    # would be merged into one neighbour.
    tiles = np.vstack([pos + np.array([i * L1, j * L2, 0.0])
                       for i in (-1, 0, 1) for j in (-1, 0, 1)])
    r = np.linalg.norm(tiles[None, :, :] - pos[:, None, :], axis=2)   # (n, 9n)
    ii, jj = np.nonzero((r > 0.1) & (r < 1.15 * nn))
    zi_all, zj_all = pos[ii, 2], tiles[jj, 2]
    zs = np.unique(np.round(pos[:, 2], 3))
    mid = zs[(zs > period) & (zs < 2.0 * period)]
    out: dict[tuple, Termination] = {}
    for zp in mid:
        zc = float(zp) - tol
        z_offset = zc + probe.meta["z_shift"]
        sl = build_slab(crystal, cell, thickness, z_offset=z_offset)
        # each bond is seen from both ends: count from the atom ABOVE the cut
        crossing = int(np.sum((zi_all > zc) & (zj_all < zc)))
        above = pos[:, 2] > zc
        first = above & (pos[:, 2] < pos[above, 2].min() + 0.3)
        lost = round(crossing / max(1, int(first.sum())))
        # Same species but different bonds cut is a different surface
        # (Si(111) shuffle: 1, glide: 3), so the bond count is part of the key.
        key = (tuple(sorted(sl.bottom_species())), tuple(sorted(sl.top_species())), lost)
        if key not in out:
            out[key] = Termination(z_offset, key[0], key[1], lost)
    return list(out.values())
