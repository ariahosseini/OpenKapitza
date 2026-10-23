"""Ideal crystal cells, built in Python so they double as known answers.

beta-cristobalite (idealised Fd-3m, a = 7.16 A): Si on the diamond sites, O at
the midpoint of every Si-Si bond, so Si-O = sqrt(3)/8 a = 1.550 A, Si-O-Si =
180 deg, O-Si-O = 109.47 deg, coordination 4 and 2, density 2.17 g/cm^3. It
is the usual melt-quench starting point for SiO2 because its density is
already close to the glass's (2.20).
"""

from __future__ import annotations

import itertools

import numpy as np

from .analysis import Cell

MASS = {"Si": 28.0855, "O": 15.9994, "Ge": 72.63, "Al": 26.9815, "Cu": 63.546}

DIAMOND_A = np.array([(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)])
DIAMOND_B = DIAMOND_A + 0.25


def _tile(basis: np.ndarray, n: tuple[int, int, int], a: tuple[float, float, float]):
    shifts = np.array(list(itertools.product(range(n[0]), range(n[1]), range(n[2]))))
    frac = (basis[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    return frac * np.asarray(a)


def diamond(a: float, n: tuple[int, int, int], *, type_id: int = 1,
            mass: float = MASS["Si"], az: float | None = None) -> Cell:
    """Diamond-cubic cell, n cells per direction; `az` stretches z only."""
    cell_a = (a, a, az or a)
    pos = _tile(np.vstack([DIAMOND_A, DIAMOND_B]), n, cell_a)
    hi = np.array([n[0] * a, n[1] * a, n[2] * (az or a)])
    return Cell(pos, np.full(len(pos), type_id), np.zeros(3), hi, {type_id: mass})


def beta_cristobalite(n: tuple[int, int, int], a: float = 7.16, *,
                      si_type: int = 1, o_type: int = 2) -> Cell:
    """Idealised beta-cristobalite: 8 Si + 16 O per cubic cell."""
    dirs = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]) / 8.0
    o_basis = (DIAMOND_A[:, None, :] + dirs[None, :, :]).reshape(-1, 3) % 1.0
    si = _tile(np.vstack([DIAMOND_A, DIAMOND_B]), n, (a, a, a))
    o = _tile(o_basis, n, (a, a, a))
    pos = np.vstack([si, o])
    types = np.r_[np.full(len(si), si_type), np.full(len(o), o_type)]
    hi = np.array(n, dtype=float) * a
    return Cell(pos, types, np.zeros(3), hi, {si_type: MASS["Si"], o_type: MASS["O"]})
