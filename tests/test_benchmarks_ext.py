"""Extended benchmark: the other conforming, vacuum data sets in Bahrami's
thesis Appendix A.2 (Antonetti Ni 200, Milanez SS 304, McWaid SS 304, Nho
Ni 200 / Al 6061-T6 / Al-Ni). Built by scripts/build_benchmark_ext_csv.py.

Rows whose R_exp is printed too coarsely (rounding > 10 %) are excluded.
Result on the 158 usable points: pred/meas geometric mean 0.950, RMS 13.3 %,
all within 2x; Al 6061-containing pairs (72 points) 0.951, 11.1 %.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_benchmarks import _f, predict  # noqa: E402

from openkapitza.contact.truncated import (  # noqa: E402
    p_over_hc_gaussian, p_over_hc_truncated, tg_conductance,
)

DATA = Path(__file__).parent / "benchmarks" / "joint_conductance_ext.csv"
MAX_ROUNDING = 0.10


def gm(v):
    return math.exp(statistics.fmean(math.log(x) for x in v))


def rms(v):
    return math.sqrt(statistics.fmean((x - 1) ** 2 for x in v))


class Extended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(DATA, newline="", encoding="utf-8") as fh:
            cls.all = list(csv.DictReader(fh))
        cls.rows = [r for r in cls.all if float(r["rounding_rel"]) <= MAX_ROUNDING]
        cls.ratio = {r["case_id"]: predict(r) / _f(r, "h_measured_W_m2K") for r in cls.rows}

    def test_counts(self):
        self.assertEqual(len(self.all), 174)
        self.assertEqual(len(self.rows), 158)

    def test_conversion_matches_bahramis_own_model_column(self):
        # Same CMY, same inputs: our h vs his R_j (same-material sets) within 3 %.
        # Confirms the transcription and A = pi b_L^2 for every set.
        by = {}
        for r in self.rows:
            if r["material_a"] == r["material_b"] and r["dataset"] != "M_T1":
                by.setdefault(r["dataset"], []).append(
                    predict(r) / _f(r, "h_model_bahrami_W_m2K"))
        for ds, v in by.items():
            self.assertAlmostEqual(gm(v), 1.0, delta=0.03, msg=ds)

    def test_overall_agreement(self):
        v = list(self.ratio.values())
        self.assertAlmostEqual(gm(v), 0.95, delta=0.03)
        self.assertLess(rms(v), 0.15)
        self.assertTrue(all(0.5 <= x <= 2.0 for x in v))

    def test_aluminium_pairs(self):
        """Al 6061-T6 and Al 6061-T6 / Ni 200 (Nho): the package-metal check P0-2
        asked for -- but in VACUUM, so it tests the contact path, not air."""
        v = [self.ratio[r["case_id"]] for r in self.rows
             if "Al6061" in (r["material_a"], r["material_b"])]
        self.assertGreater(len(v), 60)
        self.assertAlmostEqual(gm(v), 0.95, delta=0.05)
        self.assertLess(rms(v), 0.13)

    def test_light_load_trend_repeats_on_independent_data(self):
        light, normal = [], []
        for r in self.rows:
            P, som = _f(r, "pressure_MPa") * 1e6, _f(r, "sigma_eff_um") / _f(r, "slope_eff")
            y = 1.53 * p_over_hc_gaussian(P, som, _f(r, "c1_GPa") * 1e9, _f(r, "c2")) ** -0.097
            (light if y > 3.2 else normal).append(1 / self.ratio[r["case_id"]])
        self.assertGreater(len(light), 30)
        self.assertGreater(gm(light), 1.10)                     # 1.15
        self.assertLess(abs(gm(normal) - 1.0), 0.05)            # 1.03


class MilanezT1(unittest.TestCase):
    """Thesis Table A.63 is the S1 specimen of Milanez, Yovanovich & Mantelli
    (their Table 2: RMS 25.3 % Gaussian -> 16.1 % TG at z_trunc = 4.1). With
    their own c1/c2 (10.67 GPa, -0.37) the code gives 20.8 % -> 15.5 %.
    With the thesis c1/c2 (6.27, -0.23) plain CMY is already 14.7 % and TG
    over-corrects to 22.4 %: z_trunc was fitted with THEIR hardness, so the
    reference values belong with it."""

    def _rms(self, c1, c2):
        with open(DATA, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["dataset"] == "M_T1"]
        s, m, ks = 0.72e-6, 0.041, 18.8
        g, t = [], []
        for r in rows:
            P, hm = _f(r, "pressure_MPa") * 1e6, _f(r, "h_measured_W_m2K")
            xg = p_over_hc_gaussian(P, s / m * 1e6, c1, c2)
            xt = p_over_hc_truncated(P, s / m * 1e6, c1, c2, 4.1)
            g.append(1.25 * ks * m / s * xg ** 0.95 / hm)
            t.append(tg_conductance(ks, s, m, xt, 4.1) / hm)
        return rms(g), rms(t)

    def test_reproduces_their_improvement_with_their_hardness(self):
        g, t = self._rms(10.67e9, -0.37)
        self.assertAlmostEqual(t, 0.161, delta=0.02)
        self.assertLess(t, g - 0.04)

    def test_reference_z_does_not_transfer_to_other_hardness_fits(self):
        g, t = self._rms(6.27e9, -0.23)
        self.assertGreater(t, g)


if __name__ == "__main__":
    unittest.main()
