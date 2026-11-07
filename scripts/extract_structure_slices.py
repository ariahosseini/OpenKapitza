"""Cut thin slabs out of relaxed MD cells for the README's atom pictures.

    python3 scripts/extract_structure_slices.py ~/ok_jobs

Reads <root>/<job>/relaxed.data (LAMMPS write_data, atomic style) for the
jobs in SLICES and writes docs/figures/data/structures/<name>.npz: the atoms
whose y lies in a slab of the given thickness, as (type, x, y, z) float32
arrays plus the box and the type -> element map. A slab one lattice constant
thick viewed along y shows the diamond lattice as its familiar projection;
the whole cell would be a grey wall. The full data files (2-3 MB each) stay
out of the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "figures" / "data" / "structures"

# name: (job folder, slab thickness A, type -> element)
SLICES = {
    "sige_bonded": ("nonbonded/bonded_ref", 5.4, {1: "Si", 2: "Ge"}),
    "sige_vdw_1GPa": ("nonbonded/lj1_p1", 5.4, {1: "Si", 2: "Ge"}),
    "sige_ar_layer": ("nonbonded/Ar_c1_p1", 5.4, {1: "Si", 2: "Ge", 3: "Ar"}),
    "sige_n2_half": ("nonbonded/N2_c0.5_p1", 5.4, {1: "Si", 2: "Ge", 3: "N2"}),
    "sige_he_layer": ("nonbonded/He_c1_p1", 5.4, {1: "Si", 2: "Ge", 3: "He"}),
    "si_asio2": ("sio2_strong", 5.4, {1: "Si", 2: "Si(ox)", 3: "O"}),
}


# Steady temperature profiles shown under the cell (all frames averaged).
PROFILES = {"bonded": "nonbonded/bonded_ref", "vdw_1GPa": "nonbonded/lj1_p1",
            "ar_layer": "nonbonded/Ar_c1_p1"}


def mean_profile(path: Path) -> tuple[list[float], list[float]]:
    """Bin centre (A) and temperature (K), averaged over every frame; empty
    bins (walls, headroom) dropped."""
    frames, cur, n = [], None, 0
    for ln in path.read_text().splitlines():
        p = ln.split()
        if not p or ln.startswith("#"):
            continue
        if len(p) == 3:
            cur, n = [], int(p[1])
            frames.append(cur)
            continue
        cur.append((float(p[1]), float(p[2]), float(p[3])))
    a = np.array([f for f in frames if len(f) == len(frames[0])])
    z, cnt, T = a[0, :, 0], a[:, :, 1].mean(0), a[:, :, 2].mean(0)
    keep = cnt > 0.5 * np.median(cnt[cnt > 0])
    return z[keep].round(2).tolist(), T[keep].round(3).tolist()


def read_data(path: Path) -> tuple[np.ndarray, dict]:
    lines = path.read_text().splitlines()
    box = {}
    for ln in lines[:40]:
        p = ln.split()
        if len(p) >= 4 and p[2] in ("xlo", "ylo", "zlo"):
            box[p[2][0]] = (float(p[0]), float(p[1]))
    i = next(k for k, ln in enumerate(lines) if ln.startswith("Atoms"))
    rows = []
    for ln in lines[i + 2:]:
        p = ln.split()
        if not p:
            if rows:
                break
            continue
        if not p[0].isdigit():
            break
        rows.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return np.asarray(rows), box


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (job, t, elements) in SLICES.items():
        f = root / job / "relaxed.data"
        if not f.is_file():
            print(f"{name:16} skipped: no {f}")
            continue
        a, box = read_data(f)
        # Slab faces halfway between (001) atomic planes (spacing a/4), so thermal
        # motion does not drop atoms in and out of the picture.
        ys = np.sort(a[a[:, 0] == 1, 2])
        y0 = ys[: max(1, len(ys) // 50)].mean() + t / 8
        keep = (a[:, 2] >= y0) & (a[:, 2] < y0 + t)
        s = a[keep].astype(np.float32)
        np.savez_compressed(OUT / f"{name}.npz", atoms=s,
                            box=np.array([box[k] for k in "xyz"], dtype=np.float32),
                            slab=np.array([y0, y0 + t], dtype=np.float32),
                            types=np.array(sorted(elements)),
                            elements=np.array([elements[k] for k in sorted(elements)]))
        print(f"{name:16} {keep.sum():6} of {len(a)} atoms")
    import json
    prof = {}
    for name, job in PROFILES.items():
        f = root / job / "temperature_profile.dat"
        if f.is_file():
            z, T = mean_profile(f)
            sys.path.insert(0, str(REPO))
            from openkapitza.md.campaign import read_spec
            from openkapitza.md.measure import measure_kapitza
            from openkapitza.schema.flags import FlagSet
            spec, _, _ = read_spec(root / job)
            m = measure_kapitza(root / job, spec, FlagSet())
            prof[name] = {"z_ang": z, "T_K": T, "flux_mode": spec.flux_mode,
                          "q_W_m2": m.flux, "h_MW": m.h_K / 1e6}
    if prof:
        (OUT.parent / "cell_profiles.json").write_text(json.dumps(prof, indent=1))
        print("profiles:", ", ".join(prof))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
