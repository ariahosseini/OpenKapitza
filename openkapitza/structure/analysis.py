"""Structural analysis for built and quenched cells (Phase 2).

Pure Python on numpy, so it is testable without LAMMPS: read a LAMMPS data
file or a `write_data` / custom dump, then density, partial radial
distribution functions, coordination numbers and bond-angle distributions
under periodic boundaries in an orthogonal box.

The agent instruction (section 3) makes validation mandatory for every built
structure: density, RDF first and second peaks, bond angles, coordination.
`md.structure.amorphous` turns these numbers into a pass/flag verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AMU_G = 1.66053906660e-24          # g per amu
A3_CM3 = 1e-24                     # cm^3 per A^3


@dataclass
class Cell:
    """Atoms in an orthogonal box. Lengths in Angstrom."""

    pos: np.ndarray                # (N, 3)
    types: np.ndarray              # (N,) integer type ids
    lo: np.ndarray                 # (3,)
    hi: np.ndarray                 # (3,)
    masses: dict[int, float] = field(default_factory=dict)   # type -> amu
    periodic: tuple[bool, bool, bool] = (True, True, True)

    @property
    def lengths(self) -> np.ndarray:
        return self.hi - self.lo

    @property
    def volume(self) -> float:
        return float(np.prod(self.lengths))

    def density(self) -> float:
        """g/cm^3 over the whole box."""
        m = sum(self.masses[int(t)] for t in self.types)
        return m * AMU_G / (self.volume * A3_CM3)

    def select(self, mask: np.ndarray) -> "Cell":
        return Cell(self.pos[mask], self.types[mask], self.lo.copy(), self.hi.copy(),
                    dict(self.masses), self.periodic)


def read_lammps_data(path: Path | str) -> Cell:
    """Atomic-style LAMMPS data file (as written by `write_data`)."""
    lines = Path(path).read_text().splitlines()
    lo, hi = np.zeros(3), np.zeros(3)
    masses: dict[int, float] = {}
    atoms: list[tuple[int, int, float, float, float]] = []
    section = None
    for raw in lines:
        line = raw.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[2:4] in (["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]):
            k = "xyz".index(parts[2][0])
            lo[k], hi[k] = float(parts[0]), float(parts[1])
            continue
        if parts[0] in ("Masses", "Atoms", "Velocities", "Bonds", "Pair"):
            section = parts[0]
            continue
        if section == "Masses" and len(parts) >= 2:
            masses[int(parts[0])] = float(parts[1])
        elif section == "Atoms" and len(parts) >= 5:
            atoms.append((int(parts[0]), int(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4])))
    if not atoms:
        raise ValueError(f"no atoms in {path}")
    atoms.sort()
    arr = np.array([a[2:] for a in atoms])
    types = np.array([a[1] for a in atoms])
    return Cell(arr, types, lo, hi, masses)


def write_lammps_data(cell: Cell, path: Path | str, *, comment: str = "OpenKapitza") -> None:
    """Atomic-style data file LAMMPS can `read_data`."""
    n_types = max(int(cell.types.max()), max(cell.masses) if cell.masses else 0)
    out = [f"# {comment}", "", f"{len(cell.types)} atoms", f"{n_types} atom types", ""]
    for k, ax in enumerate("xyz"):
        out.append(f"{cell.lo[k]:.6f} {cell.hi[k]:.6f} {ax}lo {ax}hi")
    out += ["", "Masses", ""]
    out += [f"{t} {cell.masses[t]:.4f}" for t in sorted(cell.masses)]
    out += ["", "Atoms # atomic", ""]
    out += [f"{i + 1} {int(t)} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
            for i, (t, p) in enumerate(zip(cell.types, cell.pos))]
    Path(path).write_text("\n".join(out) + "\n")


# --------------------------------------------------------------------------
# Neighbours
# --------------------------------------------------------------------------

def _min_image(d: np.ndarray, cell: Cell) -> np.ndarray:
    L = cell.lengths
    for k in range(3):
        if cell.periodic[k]:
            d[..., k] -= L[k] * np.round(d[..., k] / L[k])
    return d


def pair_distances(cell: Cell, idx_a: np.ndarray, idx_b: np.ndarray,
                   r_max: float, *, chunk: int = 512):
    """(i, j, r) for i in idx_a, j in idx_b, 0 < r < r_max, minimum image.

    Brute force in chunks: fine for the few-thousand-atom cells used here.
    """
    if any(cell.periodic[k] and 2 * r_max > cell.lengths[k] for k in range(3)):
        raise ValueError("r_max exceeds half the box; minimum image breaks down")
    pb = cell.pos[idx_b]
    I, J, R = [], [], []
    for s in range(0, len(idx_a), chunk):
        ia = idx_a[s:s + chunk]
        d = _min_image(pb[None, :, :] - cell.pos[ia][:, None, :], cell)
        r = np.sqrt((d * d).sum(-1))
        m = (r < r_max) & (r > 1e-8)
        ii, jj = np.nonzero(m)
        I.append(ia[ii]); J.append(idx_b[jj]); R.append(r[ii, jj])
    return np.concatenate(I), np.concatenate(J), np.concatenate(R)


# --------------------------------------------------------------------------
# RDF, coordination, angles
# --------------------------------------------------------------------------

@dataclass
class RDF:
    r: np.ndarray
    g: np.ndarray

    def first_peak(self, r_lo: float = 0.0, r_hi: float = math.inf) -> float:
        m = (self.r >= r_lo) & (self.r <= r_hi)
        return float(self.r[m][np.argmax(self.g[m])])


def partial_rdf(cell: Cell, type_a: int, type_b: int, *, r_max: float = 6.0,
                dr: float = 0.02) -> RDF:
    """g_ab(r), normalised so g -> 1 for an ideal gas of the same density."""
    ia = np.nonzero(cell.types == type_a)[0]
    ib = np.nonzero(cell.types == type_b)[0]
    _, _, r = pair_distances(cell, ia, ib, r_max)
    edges = np.arange(0.0, r_max + dr, dr)
    hist, _ = np.histogram(r, bins=edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 * math.pi * centres ** 2 * dr
    rho_b = len(ib) / cell.volume
    g = hist / (len(ia) * rho_b * shell)
    return RDF(centres, g)


def coordination(cell: Cell, centre_type: int, neighbour_type: int,
                 cutoff: float) -> np.ndarray:
    """Number of neighbour_type atoms within cutoff of each centre_type atom."""
    ic = np.nonzero(cell.types == centre_type)[0]
    inb = np.nonzero(cell.types == neighbour_type)[0]
    i, _, _ = pair_distances(cell, ic, inb, cutoff)
    counts = np.zeros(len(cell.types), dtype=int)
    np.add.at(counts, i, 1)
    return counts[ic]


def bond_angles(cell: Cell, centre_type: int, end_type: int, cutoff: float) -> np.ndarray:
    """All end-centre-end angles (degrees) for bonds shorter than cutoff."""
    ic = np.nonzero(cell.types == centre_type)[0]
    ie = np.nonzero(cell.types == end_type)[0]
    i, j, _ = pair_distances(cell, ic, ie, cutoff)
    angles = []
    order = np.argsort(i, kind="stable")
    i, j = i[order], j[order]
    starts = np.flatnonzero(np.r_[True, i[1:] != i[:-1]])
    ends = np.r_[starts[1:], len(i)]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        c = i[s]
        v = _min_image(cell.pos[j[s:e]] - cell.pos[c], cell)
        v /= np.linalg.norm(v, axis=1)[:, None]
        cos = np.clip(v @ v.T, -1.0, 1.0)
        iu = np.triu_indices(e - s, 1)
        angles.append(np.degrees(np.arccos(cos[iu])))
    return np.concatenate(angles) if angles else np.zeros(0)


def angle_peak(angles: np.ndarray, *, bin_deg: float = 2.0) -> float:
    """Mode of an angle distribution (bin centre)."""
    if angles.size == 0:
        return float("nan")
    edges = np.arange(0.0, 180.0 + bin_deg, bin_deg)
    hist, _ = np.histogram(angles, bins=edges)
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))
