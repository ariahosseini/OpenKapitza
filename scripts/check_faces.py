"""How far the contact faces of a non-bonded cell have reconstructed (KB 45).

    python3 scripts/check_faces.py JOB_DIR [JOB_DIR ...]      # reads relaxed.data

A bulk-terminated (100) face has in-layer neighbours a/sqrt(2) apart (3.8-4.0 A);
a dimer pairs two top-layer atoms at ~2.3-2.6 A. The fraction of top-layer
atoms with a neighbour closer than 2.8 A is the dimerised fraction; a fully
(2x1)-reconstructed face reads ~1. The face against the frozen wall is the
bulk-terminated reference (~0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DIMER_CUT = 2.8      # A
LAYER = 1.0          # A, half-thickness of the top layer: wide enough for buckled dimers
                     # (~0.7 A tilt), short of the second layer (a/4 = 1.36 A below)


def read_data(path: Path):
    lines = Path(path).read_text().splitlines()
    box = {}
    for ln in lines[:30]:
        p = ln.split()
        if len(p) >= 4 and p[2] in ("xlo", "ylo", "zlo"):
            box[p[2][0]] = float(p[1]) - float(p[0])
    i = next(k for k, ln in enumerate(lines) if ln.strip().startswith("Atoms"))
    rows = []
    for ln in lines[i + 2:]:
        p = ln.split()
        if len(p) < 5:
            break
        rows.append((int(p[1]), float(p[2]), float(p[3]), float(p[4])))
    a = np.array(rows)
    return a[:, 0].astype(int), a[:, 1:], box["x"], box["y"]


def dimer_fraction(types, xyz, lx, ly, typ: int, top: bool) -> float:
    s = xyz[types == typ]
    z0 = s[:, 2].max() if top else s[:, 2].min()
    layer = s[np.abs(s[:, 2] - z0) < LAYER]
    d = layer[:, None, :2] - layer[None, :, :2]
    d[..., 0] -= lx * np.round(d[..., 0] / lx)
    d[..., 1] -= ly * np.round(d[..., 1] / ly)
    dz = layer[:, None, 2] - layer[None, :, 2]
    r = np.sqrt((d ** 2).sum(-1) + dz ** 2)
    np.fill_diagonal(r, np.inf)
    return float((r.min(axis=1) < DIMER_CUT).mean())


def faces(path: Path) -> dict:
    """Dimerised fraction of the A contact face (top of type 1), the B contact
    face (bottom of type 2) and the A wall-side face (reference)."""
    t, xyz, lx, ly = read_data(path)
    return {"a_contact": dimer_fraction(t, xyz, lx, ly, 1, True),
            "b_contact": dimer_fraction(t, xyz, lx, ly, 2, False),
            "a_wall_side": dimer_fraction(t, xyz, lx, ly, 1, False)}


def main(argv=None) -> int:
    for d in (argv if argv is not None else sys.argv[1:]):
        f = faces(Path(d) / "relaxed.data")
        print(f"{Path(d).name:20} dimerised: A face {f['a_contact']:.0%}, B face "
              f"{f['b_contact']:.0%} (wall side {f['a_wall_side']:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
