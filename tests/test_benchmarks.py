"""Validation against measured joint conductance.

Skips cleanly while tests/benchmarks/joint_conductance.csv holds no rows. See
that directory's README for what to populate it with and why Si/Ge is not a
valid Phase-0 benchmark.

The suite reports the full distribution of model/measurement ratios rather
than asserting a tight tolerance on each point. Contact conductance
correlations are not precision instruments: agreement within a factor of two
across a pressure sweep is the realistic standard, and the shape of the
pressure dependence matters more than any single point.
"""

from __future__ import annotations

import csv
import math
import statistics
import unittest
from pathlib import Path

from openkapitza import build_joint, run_continuum

DATA = Path(__file__).parent / "benchmarks" / "joint_conductance.csv"

# Realistic acceptance for an empirical contact correlation against experiment.
FACTOR_TOLERANCE = 2.0
# Fraction of points that must land inside that band for the suite to pass.
REQUIRED_WITHIN = 0.80


def load_cases() -> list[dict]:
    if not DATA.exists():
        return []
    with open(DATA, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("case_id")]


def _f(row: dict, key: str) -> float | None:
    v = (row.get(key) or "").strip()
    return float(v) if v else None


class TestBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        if not cls.cases:
            raise unittest.SkipTest(
                f"no benchmark rows in {DATA}; see tests/benchmarks/README.md"
            )

    def test_against_measurements(self):
        ratios: list[float] = []
        failures: list[str] = []

        for row in self.cases:
            gas = (row.get("gas") or "").strip() or None
            joint = build_joint(
                row["material_a"], row["material_b"],
                sigma_a_um=_f(row, "sigma_a_um"),
                sigma_b_um=_f(row, "sigma_b_um"),
                slope_a=_f(row, "slope_a"),
                slope_b=_f(row, "slope_b"),
                pressure_MPa=_f(row, "pressure_MPa"),
                temperature_K=_f(row, "temperature_K"),
                gas=gas,
                gas_pressure_kPa=_f(row, "gas_pressure_kPa"),
                label=row["case_id"],
            )
            result = run_continuum(joint, with_uncertainty=False)
            predicted = result.primary.h_joint.si
            measured = _f(row, "h_measured_W_m2K")
            ratio = predicted / measured
            ratios.append(ratio)

            if not (1 / FACTOR_TOLERANCE <= ratio <= FACTOR_TOLERANCE):
                failures.append(
                    f"  {row['case_id']:24s} predicted {predicted:10.4g}  "
                    f"measured {measured:10.4g}  ratio {ratio:6.2f}  "
                    f"[{result.primary.branch}]"
                )

        log_ratios = [math.log(r) for r in ratios]
        geo_mean = math.exp(statistics.fmean(log_ratios))
        spread = math.exp(statistics.pstdev(log_ratios)) if len(ratios) > 1 else 1.0
        within = sum(
            1 for r in ratios if 1 / FACTOR_TOLERANCE <= r <= FACTOR_TOLERANCE
        ) / len(ratios)

        summary = (
            f"\n{len(ratios)} cases | geometric mean ratio {geo_mean:.3f} "
            f"(1.0 = perfect) | log-spread factor {spread:.2f} | "
            f"{within * 100:.0f} % within a factor of {FACTOR_TOLERANCE:g}"
        )
        if failures:
            summary += "\noutside the band:\n" + "\n".join(failures)
        print(summary)

        self.assertGreaterEqual(
            within, REQUIRED_WITHIN,
            f"only {within * 100:.0f} % of cases within a factor of "
            f"{FACTOR_TOLERANCE:g}{summary}",
        )
        # A systematic bias matters more than scatter: it points at the
        # hardness or slope inputs rather than at the correlation.
        self.assertLess(
            abs(math.log(geo_mean)), math.log(1.5),
            f"systematic bias: geometric mean ratio {geo_mean:.3f}. A "
            f"consistent over- or under-prediction usually means H_c or the "
            f"asperity slope is wrong, not the correlation.{summary}",
        )

    def test_pressure_dependence_shape(self):
        """Within one specimen pair, h should rise with pressure as measured."""
        by_specimen: dict[str, list[tuple[float, float, float]]] = {}
        for row in self.cases:
            key = (
                f"{row['material_a']}|{row['material_b']}|"
                f"{row['sigma_a_um']}|{(row.get('gas') or 'vac')}"
            )
            joint = build_joint(
                row["material_a"], row["material_b"],
                sigma_a_um=_f(row, "sigma_a_um"),
                sigma_b_um=_f(row, "sigma_b_um"),
                slope_a=_f(row, "slope_a"),
                slope_b=_f(row, "slope_b"),
                pressure_MPa=_f(row, "pressure_MPa"),
                temperature_K=_f(row, "temperature_K"),
                gas=(row.get("gas") or "").strip() or None,
                gas_pressure_kPa=_f(row, "gas_pressure_kPa"),
            )
            pred = run_continuum(joint, with_uncertainty=False).primary.h_joint.si
            by_specimen.setdefault(key, []).append(
                (_f(row, "pressure_MPa"), pred, _f(row, "h_measured_W_m2K"))
            )

        checked = 0
        for key, pts in by_specimen.items():
            if len(pts) < 3:
                continue
            pts.sort()
            p = [math.log(x[0]) for x in pts]
            pm = [math.log(x[1]) for x in pts]
            mm = [math.log(x[2]) for x in pts]
            slope_pred = _slope(p, pm)
            slope_meas = _slope(p, mm)
            checked += 1
            self.assertAlmostEqual(
                slope_pred, slope_meas, delta=0.25,
                msg=(
                    f"{key}: predicted d(ln h)/d(ln P) = {slope_pred:.3f} vs "
                    f"measured {slope_meas:.3f}. The exponent is a stronger "
                    f"test of the model than the magnitude, which absorbs "
                    f"hardness error."
                ),
            )
        if checked == 0:
            self.skipTest("no specimen group has 3+ pressure points")


def _slope(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = statistics.fmean(x), statistics.fmean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = sum((a - mx) ** 2 for a in x)
    return num / den


if __name__ == "__main__":
    unittest.main(verbosity=2)
