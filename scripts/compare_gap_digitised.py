"""Compare the gap model against the digitised Song & Yovanovich figures.

Reproduces the table in `tests/benchmarks/SOURCES.md`, "Digitised gap data".

THIS IS NOT A VALIDATION. The input CSV comes from figure digitisation with a
documented bias — upper bounds recover within 1-3.5 %, but lower gas pressures
come out 25-45 % high because the extractor loses the dense overlapping
low-pressure cluster, which is exactly the rarefied regime of interest. The
output is a diagnostic: it is good for exposing a TREND, not for quoting an
agreement figure. See SOURCES.md before citing anything printed here.

Method
------
`h_g` is recovered as `h_j(digitised) - h_c(published vacuum)`. Using the
source's own vacuum conductance rather than ours removes the contact path
without needing `k_s` at the test temperature, so this compares the gap model
alone. The subtraction is well-conditioned only where `h_g` is comparable to
or larger than `h_c`, i.e. at the HIGH pressure end; at low gas pressure it is
a small difference of two larger numbers, compounding the extraction bias.

Hardness coefficients are forced to the thesis values c1 = 6.3 GPa,
c2 = -0.26 for both materials, matching `build_benchmark_csv.py`, so the
contact-side inputs are the same ones the sources used.

Run with:  python3 scripts/compare_gap_digitised.py
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))          # pin to THIS tree, never an installed copy

from openkapitza.contact import contact_hardness, mean_plane_separation
from openkapitza.gap import gas_state, gap_conductance, gap_conductance_integral
from openkapitza.schema import (
    FlagSet,
    Quantity,
    default_accommodation,
    load_gas,
    load_material,
)

CSV_PATH = REPO / "tests" / "benchmarks" / "gap_conductance_digitised.csv"
TORR = 133.322368

# Thesis hardness coefficients, applied to both materials (see module docstring).
C1_GPA, C2 = 6.3, -0.26

M_OVER_Y_BANDS = (
    (0.0, 0.1, "< 0.1  (continuum)"),
    (0.1, 0.5, "0.1 - 0.5"),
    (0.5, 2.0, "0.5 - 2"),
    (2.0, math.inf, "> 2    (rarefied)"),
)


def _geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean([math.log(v) for v in values]))


def _log_log_slope(xs: list[float], ys: list[float]) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = statistics.fmean(lx), statistics.fmean(ly)
    denom = sum((a - mx) ** 2 for a in lx)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / denom


def evaluate(row: dict[str, str]) -> dict[str, float] | None:
    """Return model and measured h_g for one digitised point, or None if unusable."""
    material = row["material"]
    temperature = Quantity.parse(float(row["temperature_K"]), "K")

    mat_a, mat_b = load_material(material), load_material(material)
    for mat in (mat_a, mat_b):
        mat.c1 = Quantity.parse(C1_GPA, "GPa")
        mat.c2 = Quantity.parse(C2, "-")

    sigma = Quantity.parse(float(row["sigma_eff_um"]), "um")
    slope = Quantity.parse(float(row["slope_eff"]), "-")
    pressure = Quantity.parse(float(row["pressure_MPa"]), "MPa")

    flags = FlagSet()
    hardness = contact_hardness(mat_a, mat_b, pressure, sigma, slope, flags)
    Y = mean_plane_separation(sigma, hardness.P_over_Hc, flags)

    p_gas_torr = float(row["gas_pressure_torr"])
    state = gas_state(load_gas(row["gas"], p_gas_torr * TORR), temperature)
    acc = default_accommodation(row["gas"], material, FlagSet())

    lumped = gap_conductance(state, Y, acc, acc, FlagSet())
    integral = gap_conductance_integral(state, Y, sigma, acc, acc, FlagSet())

    measured = float(row["h_joint_W_m2K"]) - float(row["h_c_vacuum_W_m2K"])
    if measured <= 0.0:
        return None                     # digitisation put h_j below the vacuum line

    return {
        "p_gas_torr": p_gas_torr,
        "lumped": lumped.h_g.si / measured,
        "integral": integral.h_g.si / measured,
        "M_over_Y": lumped.M.si / Y.si,
        "knudsen": state.mfp.si / Y.si,
        "Y_over_sigma": Y.si / sigma.si,
    }


def main() -> int:
    if not CSV_PATH.exists():
        print(f"missing {CSV_PATH}; run scripts/digitize_gap_figures.py first")
        return 1

    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    recs = [r for r in (evaluate(row) for row in rows) if r is not None]
    dropped = len(rows) - len(recs)

    print("=" * 74)
    print("DIGITISED gap comparison - DIAGNOSTIC ONLY, NOT A VALIDATION")
    print("Lower gas pressures in the source CSV are biased 25-45 % high.")
    print("=" * 74)
    print(f"{len(recs)} usable points of {len(rows)}"
          + (f" ({dropped} dropped: h_j below vacuum line)" if dropped else ""))
    print(f"gas pressure {min(r['p_gas_torr'] for r in recs):.1f}"
          f" - {max(r['p_gas_torr'] for r in recs):.1f} torr")
    print(f"Knudsen      {min(r['knudsen'] for r in recs):.4f}"
          f" - {max(r['knudsen'] for r in recs):.3f}")
    print(f"Y/sigma      {min(r['Y_over_sigma'] for r in recs):.2f}"
          f" - {max(r['Y_over_sigma'] for r in recs):.2f}")

    print("\ngeometric mean of model / measured h_g")
    print(f"  {'band':<22} {'n':>4} {'lumped':>9} {'integral':>9}")
    for lo, hi, label in M_OVER_Y_BANDS:
        sel = [r for r in recs if lo <= r["M_over_Y"] < hi]
        if not sel:
            continue
        print(f"  {label:<22} {len(sel):4d}"
              f" {_geometric_mean([r['lumped'] for r in sel]):9.3f}"
              f" {_geometric_mean([r['integral'] for r in sel]):9.3f}")
    print(f"  {'ALL':<22} {len(recs):4d}"
          f" {_geometric_mean([r['lumped'] for r in recs]):9.3f}"
          f" {_geometric_mean([r['integral'] for r in recs]):9.3f}")

    print("\nspread and trend")
    pressures = [r["p_gas_torr"] for r in recs]
    for name in ("lumped", "integral"):
        vals = [r[name] for r in recs]
        spread = math.exp(statistics.pstdev([math.log(v) for v in vals]))
        within = sum(1 for v in vals if 0.5 <= v <= 2.0) / len(vals) * 100.0
        print(f"  {name:<9} spread x{spread:.3f}  within 2x {within:5.1f} %"
              f"  log-log slope vs P_g {_log_log_slope(pressures, vals):+.4f}")

    print("\nThe trend, not the aggregate, is the finding: agreement decays")
    print("monotonically toward the continuum limit, where the rarefaction term")
    print("vanishes and h_g -> k_g/Y. That implicates the uniform-gap assumption,")
    print("not the accommodation coefficients. The lumped form remains the")
    print("default; see tests/benchmarks/SOURCES.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
