"""A Tersoff potential with one element's length scale changed (P1-16, KB 28).

Our Si/Ge cell puts Ge on the Si lattice, 4 % compressed. Compressed Ge moves
its phonon spectrum toward Si's and raises h_K, which is the likely reason our
values sit above the literature range. Feng et al. remove the strain the other
way: Ge's potential is rescaled so that its own lattice constant is a_Si.

Rescaling is exact for a Tersoff form: every length (R, S, i.e. LAMMPS's R and
D) is multiplied by s and every inverse length (lambda1, lambda2, lambda3) is
divided by s; energies (A, B) and the angular terms are unchanged, so the
equilibrium lattice constant becomes s * a and the cohesive energy is kept.
Frequencies rise by 1/s (force constants go as E/L^2): with s = a_Si/a_Ge
that is +4 %, against roughly +12 % for Ge compressed by 4 % (Grueneisen ~1).
The two cells therefore bracket the strain effect rather than remove all of it.

The Si-Ge cross terms are re-mixed with Tersoff's 1989 rules from the rescaled
pure values: lambda_ij = (lambda_i + lambda_j)/2, R_ij = sqrt(R_i R_j),
S_ij = sqrt(S_i S_j); A_ij and B_ij (with chi) are energies and do not change.
With s = 1 the rules reproduce SiCGe.tersoff exactly (tested).

LAMMPS entry layout (metal units):
    e1 e2 e3  m gamma lambda3 c d costheta0  n beta lambda2 B  R D  lambda1 A
R and D of an (i, j, k) entry are the i-k cutoff; the two-body terms of (i, j, j)
belong to the i-j pair.
"""

from __future__ import annotations

import math
from pathlib import Path

FIELDS = ("m", "gamma", "lambda3", "c", "d", "costheta0", "n", "beta", "lambda2", "B",
          "R", "D", "lambda1", "A")


def read_entries(path: Path | str) -> dict[tuple[str, str, str], dict[str, float]]:
    """Every (e1, e2, e3) entry of a LAMMPS Tersoff file (continuation lines joined)."""
    words: list[str] = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.split("#", 1)[0]
        words += ln.split()
    out = {}
    for i in range(0, len(words) - 16, 17):
        e = tuple(words[i:i + 3])
        out[e] = dict(zip(FIELDS, (float(x) for x in words[i + 3:i + 17])))
    return out


def _cut(p: dict) -> tuple[float, float]:
    """(inner, outer) cutoff of an entry: Tersoff's R and S."""
    return p["R"] - p["D"], p["R"] + p["D"]


def rescale(entries: dict, el_a: str, el_b: str, scaled: str, s: float) -> dict:
    """The entries among {el_a, el_b}, with `scaled`'s lengths times s and the
    cross terms re-mixed."""
    if scaled not in (el_a, el_b):
        raise ValueError(f"{scaled!r} is neither {el_a!r} nor {el_b!r}")
    els = (el_a, el_b)
    pure = {e: dict(entries[(e, e, e)]) for e in els}
    p = pure[scaled]
    p["lambda1"] /= s
    p["lambda2"] /= s
    p["lambda3"] /= s
    r, q = _cut(p)
    p["R"], p["D"] = s * (r + q) / 2, s * (q - r) / 2

    def pair(i: str, j: str) -> dict:
        """Two-body lengths of the i-j pair."""
        if i == j:
            return {k: pure[i][k] for k in ("lambda1", "lambda2", "R", "D")}
        (ri, si), (rj, sj) = _cut(pure[i]), _cut(pure[j])
        r, q = math.sqrt(ri * rj), math.sqrt(si * sj)
        return {"lambda1": (pure[i]["lambda1"] + pure[j]["lambda1"]) / 2,
                "lambda2": (pure[i]["lambda2"] + pure[j]["lambda2"]) / 2,
                "R": (r + q) / 2, "D": (q - r) / 2}

    out = {}
    for key, old in entries.items():
        i, j, k = key
        if not set(key) <= set(els):
            continue
        e = dict(old)
        if i == j == k:
            e.update(pure[i])
        else:
            ik = pair(i, k)
            e["R"], e["D"] = ik["R"], ik["D"]
            if j == k:
                ij = pair(i, j)
                e["lambda1"], e["lambda2"] = ij["lambda1"], ij["lambda2"]
            e["lambda3"] = pure[i]["lambda3"]
        out[key] = e
    return out


def write_entries(path: Path | str, entries: dict, header: str = "") -> None:
    lines = [f"# {ln}" if ln else "#" for ln in header.splitlines()]
    lines.append("# DATE: generated UNITS: metal CONTRIBUTOR: OpenKapitza md/rescale_tersoff.py")
    for (i, j, k), p in entries.items():
        vals = " ".join(f"{p[f]:.10g}" for f in FIELDS)
        lines.append(f"{i:7} {j:7} {k:7} {vals}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_rescaled(src: Path | str, dest: Path | str, el_a: str, el_b: str, scaled: str,
                   s: float) -> Path:
    """Write the rescaled pair of elements from `src` to `dest`; returns dest."""
    ent = rescale(read_entries(src), el_a, el_b, scaled, s)
    write_entries(dest, ent, header=(
        f"{Path(src).name} with {scaled}'s length scale x {s:.6f} (lambdas / {s:.6f}),\n"
        f"cross terms re-mixed with Tersoff (1989) rules. Strain-free {el_a}/{el_b}\n"
        f"cell for open issue P1-16 (Feng et al. approach); see md/rescale_tersoff.py."))
    return Path(dest)
