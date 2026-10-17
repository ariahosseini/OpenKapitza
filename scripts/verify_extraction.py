"""Validate the NEMD extractor against synthetic profiles with a known answer.

Builds temperature profiles where the true jump is set by construction, then
contaminates them the way a real run is contaminated -- thermostatted
reservoirs, a non-equilibrium bend near the interface, and statistical noise
-- and checks that the extractor recovers the planted value.

The same profiles are run through the adjacent-bin shortcut to quantify how
far wrong that approach goes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from openkapitza.md.profile import (
    TemperatureProfile,
    extract_kapitza,
    naive_adjacent_bin_delta_T,
)
from openkapitza.schema import FlagSet


def synth(
    *, delta_T=8.0, k_a=148.0, k_b=60.0, flux=2.0e9, L=40e-9, n_bins=120,
    noise=0.0, bend_amp=0.0, bend_len=3e-9, reservoir_frac=0.10, seed=0,
):
    """A profile with a planted jump, optionally contaminated.

    Side A occupies [0, L), side B occupies [L, 2L), interface at z = L.
    Heat flows A -> B, so A is hotter and both sides slope downward.
    """
    rng = np.random.default_rng(seed)
    z = np.linspace(0, 2 * L, n_bins, endpoint=False) + L / n_bins
    interface = L
    T = np.empty_like(z)

    a = z < interface
    b = ~a
    # Side A: linear, reaching T_a_int at the interface
    T_a_int = 310.0
    T[a] = T_a_int + (flux / k_a) * (interface - z[a])
    # Side B: starts delta_T lower and keeps falling
    T_b_int = T_a_int - delta_T
    T[b] = T_b_int - (flux / k_b) * (z[b] - interface)

    # Non-equilibrium bend near the interface: the profile curves toward the
    # interface over a few nm, which is exactly what biases a naive fit.
    if bend_amp:
        d = np.abs(z - interface)
        sign = np.where(a, 1.0, -1.0)
        T += sign * bend_amp * np.exp(-d / bend_len)

    # Thermostatted reservoirs: flat, pinned, not following the bulk gradient
    nres = int(reservoir_frac * n_bins)
    if nres:
        T[:nres] = T[nres] + 1.5
        T[-nres:] = T[-nres - 1] - 1.5

    if noise:
        T = T + rng.normal(0.0, noise, T.shape)

    return TemperatureProfile(z=z, T=T, interface_z=interface, area=(5e-9) ** 2)


def run(label, **kw):
    truth = kw.get("delta_T", 8.0)
    flux = kw.get("flux", 2.0e9)
    L = kw.get("L", 40e-9)
    n_bins = kw.get("n_bins", 120)
    p = synth(**kw)
    f = FlagSet()
    # A real run declares its geometry; the driver knows where the
    # thermostatted regions end, so the fit windows are never guessed.
    res = int(kw.get("reservoir_frac", 0.10) * n_bins) / n_bins * 2 * L
    bulk_a, bulk_b = (res, L), (L, 2 * L - res)
    try:
        r = extract_kapitza(p, flux, f, bulk_a=bulk_a, bulk_b=bulk_b,
                            k_expected_a=kw.get("k_a", 148.0),
                            k_expected_b=kw.get("k_b", 60.0))
    except ValueError as e:
        print(f"{label:38s} EXTRACTION REFUSED: {e}")
        return
    naive = naive_adjacent_bin_delta_T(p)
    err = (r.delta_T.si - truth) / truth * 100
    nerr = (naive - truth) / truth * 100
    codes = ",".join(sorted(f.codes())) or "-"
    print(f"{label:38s} dT={r.delta_T.si:6.3f} ({err:+6.2f}%)   "
          f"naive={naive:6.3f} ({nerr:+7.2f}%)   "
          f"h_K={r.h_K.si/1e6:7.1f} MW/m2K   {codes}")


print("=" * 118)
print("SYNTHETIC RECOVERY -- planted jump dT = 8.000 K unless stated")
print("=" * 118)
run("clean")
run("+ noise 0.15 K", noise=0.15)
run("+ noise 0.4 K", noise=0.4, seed=3)
run("+ interface bend 1.5 K", bend_amp=1.5)
run("+ bend + noise", bend_amp=1.5, noise=0.2, seed=5)
run("+ bend 3 K, wide 6nm", bend_amp=3.0, bend_len=6e-9, noise=0.2, seed=7)

print()
print("=" * 118)
print("FLUX DEPENDENCE -- the naive error grows with flux, the fit does not")
print("=" * 118)
for q in (5e8, 1e9, 2e9, 4e9):
    run(f"flux {q:.0e} W/m2", flux=q, bend_amp=1.0, noise=0.15)

print()
print("=" * 118)
print("ASYMMETRIC CONDUCTIVITIES and SMALL JUMPS")
print("=" * 118)
run("k_a/k_b = 148/2 (Si/SiO2-like)", k_a=148.0, k_b=2.0, flux=2e8, bend_amp=0.5)
run("small jump dT = 1.0 K", delta_T=1.0, noise=0.1, bend_amp=0.3)
run("large jump dT = 30 K", delta_T=30.0, noise=0.3, bend_amp=2.0)
