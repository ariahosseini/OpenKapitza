"""Non-bonded interfaces: two solids held together by van der Waals forces only,
optionally with a layer of gas atoms trapped between them.

Real contact spots are rarely the clean, covalently bonded interface of the
Si/Ge campaign. Two surfaces pressed together in air touch through physisorbed
gas or contamination, or bond only weakly. This module builds those cells for
the same NEMD deck, so they compare like for like with the bonded case:

- The column is the bonded Si/Ge build (one diamond lattice, Ge strained onto
  Si in-plane), so atoms, walls, reservoirs and bins sit exactly where
  `NEMDGeometry` puts them.
- Side B is lifted to open a gap of `plane_gap_ang` between the facing atomic
  planes (the bonded spacing is a/4 = 1.36 A). The piston relaxation then
  finds the gap the forces want, under the chosen contact pressure.
- Within each side the solid keeps its own potential (Tersoff, one instance
  per side, so no Si-Ge bond can form). Across the interface atoms interact
  only through Lennard-Jones, scaled by `lj_scale` so the bonding strength can
  be swept.
- `gas` puts a layer of gas atoms (GAS_LJ in md/accommodation.py) in the gap
  on a square grid, with `coverage` the fraction of sites filled. Gas-solid
  and gas-gas interactions are Lennard-Jones with Lorentz-Berthelot mixing.

Solid LJ parameters are UFF (Rappe et al., JACS 114, 10024, 1992): Si
x = 4.295 A, D = 0.402 kcal/mol; Ge x = 4.280 A, D = 0.379 kcal/mol
(sigma = x / 2^(1/6)). UFF describes the non-bonded tail, which is what is
wanted here; it is not fitted to interfacial heat transport.

Atom types: 1 = side A, 2 = side B, 3 = gas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..md.accommodation import GAS_LJ

KCAL_EV = 0.0433641
UFF = {   # element: (sigma A, eps eV)
    "Si": (4.295 / 2 ** (1 / 6), 0.402 * KCAL_EV),
    "Ge": (4.280 / 2 ** (1 / 6), 0.379 * KCAL_EV),
}
MASS = {"Si": 28.0855, "Ge": 72.630}
DIAMOND = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                    [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])


@dataclass
class NonbondedSpec:
    lj_scale: float = 1.0            # multiplies the A-B Lennard-Jones well
    plane_gap_ang: float = 3.4       # facing-plane separation as built (no gas)
    gas: str | None = None           # "Ar", "N2", "He", "Ne": a layer in the gap
    coverage: float = 1.0            # fraction of gas sites filled
    gas_spacing_ang: float | None = None   # grid pitch; default 2^(1/6) sigma_gas
    cutoff_ang: float = 12.0
    seed: int = 7

    @property
    def label(self) -> str:
        if self.gas:
            return f"{self.gas}{self.coverage:g}"
        return f"lj{self.lj_scale:g}"


def lj_pairs(el_a: str, el_b: str, nb: NonbondedSpec) -> dict[tuple[int, int], tuple[float, float]]:
    """(eps eV, sigma A) for every cross pair LJ handles."""
    sa, ea = UFF[el_a]
    sb, eb = UFF[el_b]
    out = {(1, 2): (nb.lj_scale * math.sqrt(ea * eb), 0.5 * (sa + sb))}
    if nb.gas:
        g = GAS_LJ[nb.gas]
        out[(1, 3)] = (math.sqrt(ea * g["eps"]), 0.5 * (sa + g["sigma"]))
        out[(2, 3)] = (math.sqrt(eb * g["eps"]), 0.5 * (sb + g["sigma"]))
        out[(3, 3)] = (g["eps"], g["sigma"])
    return out


def pair_block(spec, nb: NonbondedSpec) -> str:
    """pair_style hybrid: one Tersoff per side, LJ across and for the gas."""
    nt = 3 if nb.gas else 2
    pe = spec.pair_elements
    if pe is None:
        from ..md.nemd import locate_potential, resolve_pair_elements
        try:
            pe = resolve_pair_elements(locate_potential(spec.pair_file, spec.potential_dir),
                                       spec.element_a, spec.element_b)
        except (OSError, ValueError):
            pe = (spec.element_a, spec.element_b)
    la, lb = pe[0], pe[1]
    null = " NULL" if nt == 3 else ""
    lines = [f"pair_style      hybrid {spec.pair_style} {spec.pair_style} lj/cut {nb.cutoff_ang:g}",
             f"pair_coeff      * * {spec.pair_style} 1 {spec.pair_file} {la} NULL{null}",
             f"pair_coeff      * * {spec.pair_style} 2 {spec.pair_file} NULL {lb}{null}"]
    for (i, j), (eps, sig) in lj_pairs(spec.element_a, spec.element_b, nb).items():
        lines.append(f"pair_coeff      {i} {j} lj/cut {eps:.6f} {sig:.4f}")
    lines.append("pair_modify     shift yes")
    return "\n".join(lines) + "\n"


def gas_gap(el_a: str, el_b: str, nb: NonbondedSpec) -> tuple[float, float]:
    """(A-plane to gas, gas to B-plane) distances, at the LJ minimum."""
    p = lj_pairs(el_a, el_b, nb)
    return 2 ** (1 / 6) * p[(1, 3)][1] * 0.9, 2 ** (1 / 6) * p[(2, 3)][1] * 0.9


def build_data(spec, nb: NonbondedSpec, headroom_ang: float = 5.0) -> tuple[str, dict]:
    """LAMMPS data file text for the cell, and a summary."""
    from ..md.nemd import M_TO_ANG
    g = spec.geometry
    a = spec.lattice_used * M_TO_ANG
    n = max(1, round(spec.snapped_cross_section * M_TO_ANG / a))
    L = n * a
    total = g.total_length * M_TO_ANG
    iz = g.interface_z * M_TO_ANG
    nz = int(math.ceil(total / a)) + 1
    cells = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(nz)], float)
    pos = (cells[:, None, :] + DIAMOND[None, :, :]).reshape(-1, 3) * a
    pos = pos[pos[:, 2] < total - 1e-6]
    is_a = pos[:, 2] < iz - 1e-6
    top_a = pos[is_a, 2].max()
    bot_b = pos[~is_a, 2].min()
    if nb.gas:
        da, db = gas_gap(spec.element_a, spec.element_b, nb)
        gap = da + db
    else:
        gap = nb.plane_gap_ang
    lift = gap - (bot_b - top_a)
    pos[~is_a, 2] += lift
    types = np.where(is_a, 1, 2)
    gas_n = 0
    if nb.gas:
        pitch = nb.gas_spacing_ang or 2 ** (1 / 6) * GAS_LJ[nb.gas]["sigma"]
        m = max(1, round(L / pitch))
        grid = (np.array([[i, j] for i in range(m) for j in range(m)], float) + 0.25) * (L / m)
        rng = np.random.default_rng(nb.seed)
        keep = rng.random(len(grid)) < nb.coverage if nb.coverage < 1 else np.ones(len(grid), bool)
        grid = grid[keep]
        gz = np.full((len(grid), 1), top_a + da)
        pos = np.vstack([pos, np.hstack([grid, gz])])
        types = np.concatenate([types, np.full(len(grid), 3)])
        gas_n = len(grid)
    zhi = total + lift + headroom_ang
    nt = 3 if nb.gas else 2
    masses = [MASS[spec.element_a], MASS[spec.element_b]] + ([GAS_LJ[nb.gas]["mass"]] if nb.gas else [])
    out = [f"OpenKapitza non-bonded {spec.element_a}/{spec.element_b} ({nb.label}); "
           f"plane gap {gap:.3f} A", "",
           f"{len(pos)} atoms", f"{nt} atom types", "",
           f"0.0 {L:.6f} xlo xhi", f"0.0 {L:.6f} ylo yhi", f"0.0 {zhi:.6f} zlo zhi", "",
           "Masses", ""] + [f"{i + 1} {m}" for i, m in enumerate(masses)] + ["", "Atoms # atomic", ""]
    out += [f"{i + 1} {t} {x:.6f} {y:.6f} {z:.6f}" for i, (t, (x, y, z)) in enumerate(zip(types, pos))]
    summary = {"atoms": int(len(pos)), "gas_atoms": gas_n, "plane_gap_ang": gap,
               "lift_ang": lift, "box_z_ang": zhi, "label": nb.label}
    return "\n".join(out) + "\n", summary


def make_spec(base, nb: NonbondedSpec):
    """The NEMD spec for this cell: read_data, hybrid pair section."""
    s = replace(base, structure_mode="read_data", data_file="interface.data",
                pair_block=pair_block(base, nb), types_a=(1,), types_b=(2,))
    s.extra = dict(base.extra, nonbonded={"lj_scale": nb.lj_scale, "gas": nb.gas,
                                          "coverage": nb.coverage,
                                          "plane_gap_ang": nb.plane_gap_ang})
    return s


def write_job(wd: Path | str, base, nb: NonbondedSpec, potentials: Path | str | None = None) -> dict:
    """nemd.in, interface.data, spec.json (and the potential file) in `wd`."""
    import shutil
    from ..md.campaign import write_spec
    from ..md.nemd import build_input_deck
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    spec = make_spec(base, nb)
    data, summary = build_data(spec, nb)
    (wd / "interface.data").write_text(data)
    (wd / "nemd.in").write_text(build_input_deck(spec))
    write_spec(wd, spec, "nemd", nonbonded=summary)
    if potentials:
        shutil.copy(Path(potentials) / spec.pair_file, wd / spec.pair_file)
    return summary
