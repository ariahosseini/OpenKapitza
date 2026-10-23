"""Electron-phonon resistance at a metal/non-metal interface (open issue P1-14).

Classical NEMD has no electrons. At a metal/semiconductor interface heat in
the metal is carried mostly by electrons, which must first hand it to the
metal's phonons before it can cross into the semiconductor. That hand-off is
a resistance in SERIES with the phonon-phonon interface resistance NEMD
measures, so a phonon-only h_K over-estimates the real conductance.

Estimate (Majumdar & Reddy, Appl. Phys. Lett. 84, 4768, 2004):

    h_ep = sqrt(g * k_p)          1/h = 1/h_pp + 1/h_ep

with g the metal's electron-phonon coupling factor and k_p its LATTICE
thermal conductivity -- which is exactly what work package 5c's MD bulk k
returns for a metal (`md.conductivity`, labelled lattice-only there).

Check on Al/Si (N. Yang, T. Luo, ..., G. Chen, arXiv:1401.5550, 2NN MEAM of
Lee et al.): phonon-only NEMD 0.47 GW/m^2K extrapolated; their series model
uses G_pp = 0.46, g = 2.45e17 W/m^3K and their own Green-Kubo k_p = 17.0 W/mK,
giving h_ep = 2.0 and a total of 0.37 GW/m^2K. This formula reproduces it from
those inputs (0.375). An earlier version of this note back-solved k_p = 12.3
to hit 0.37 from 0.47 and attributed the paper to "Liang et al."; both wrong.

k_p is the input that matters most (h_ep ~ sqrt(k_p)) and the least certain:
for Al, first principles give 10.7 W/mK from phonon-phonon scattering alone
and 8.1 with phonon-electron scattering (Tong, Li, Ruan, Bao, arXiv:1907.01364);
classical MD has no phonon-electron scattering, so MD k_p sits above the real
lattice value (the MEAM above: 17.0).

This is an estimate, flagged as one. Doing it properly is a two-temperature
model coupled into the MD (LAMMPS `fix ttm`), which is not built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..schema.flags import Code, FlagSet, Severity

# Electron-phonon coupling factor g near room temperature, W m^-3 K^-1.
# Sourced values only; add a metal only with a citation.
COUPLING: dict[str, dict] = {
    "Al": {"g": 2.45e17,
           "citation": "Lin, Zhigilei & Celli, PRB 77, 075133 (2008), ab initio, "
                       "low T_e; used by Yang et al., arXiv:1401.5550"},
    "Cu": {"g": 8.31e16,
           "citation": "Zhang et al. (2022), arXiv:2202.03089 (ab initio, T_e = 315 K)"},
}


@dataclass(frozen=True)
class SeriesEstimate:
    h_pp: float        # phonon-only interface conductance, W/m^2K (NEMD)
    h_ep: float        # electron-phonon conductance, W/m^2K
    h_total: float     # series combination
    reduction: float   # 1 - h_total/h_pp


def ep_conductance(g: float, k_phonon: float) -> float:
    """sqrt(g * k_p), W/m^2K."""
    if g <= 0 or k_phonon <= 0:
        raise ValueError("g and k_phonon must be positive")
    return math.sqrt(g * k_phonon)


def series_with_ep(h_pp: float, metal: str, k_phonon: float, flags: FlagSet,
                   *, g: float | None = None) -> SeriesEstimate:
    """Combine a phonon-only h_K with the metal's electron-phonon conductance."""
    if g is None:
        if metal not in COUPLING:
            flags.add(
                Code.NEMD_PHONON_ONLY_METAL, Severity.WARNING,
                f"{metal}: no sourced electron-phonon coupling factor; h_K is the "
                f"phonon-only value and over-estimates the real conductance.",
                remedy="Supply g (W/m^3K) with a citation, or use a two-temperature model.",
            )
            return SeriesEstimate(h_pp, math.inf, h_pp, 0.0)
        g = COUPLING[metal]["g"]
    h_ep = ep_conductance(g, k_phonon)
    h = 1.0 / (1.0 / h_pp + 1.0 / h_ep)
    est = SeriesEstimate(h_pp, h_ep, h, 1.0 - h / h_pp)
    flags.add(
        Code.NEMD_PHONON_ONLY_METAL, Severity.CAUTION,
        f"{metal}: electron-phonon resistance added in series as an ESTIMATE "
        f"(sqrt(g k_p), g = {g:.3g} W/m^3K, k_p = {k_phonon:.3g} W/mK): "
        f"h_K {h_pp / 1e6:.0f} -> {h / 1e6:.0f} MW/m^2K ({est.reduction * 100:.0f} % lower).",
        remedy=("Majumdar & Reddy (2004) series model, not a coupled calculation. "
                "k_p is the metal's LATTICE conductivity (MD bulk k for a metal)."),
        reduction=est.reduction,
    )
    return est
