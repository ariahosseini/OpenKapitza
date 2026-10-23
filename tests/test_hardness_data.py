"""Hardness data checked against Bahrami, Yovanovich & Culham, AIAA 2004-0989 (P0-6)."""

from __future__ import annotations

import math
import statistics
import sys
import unittest
from pathlib import Path

from openkapitza import build_joint, run_continuum
from openkapitza.contact import hot_hardness_factor
from openkapitza.schema import FlagSet, load_material
from openkapitza.schema.flags import Code


class Table3(unittest.TestCase):
    def test_vickers_coefficients(self):
        for key, c1, c2 in (("SS304", 6.271, -0.229), ("Ni200", 6.304, -0.264)):
            m = load_material(key, FlagSet())
            self.assertAlmostEqual(m.c1.si / 1e9, c1)
            self.assertAlmostEqual(m.c2.si, c2)


class HotHardness(unittest.TestCase):
    def test_fit_reproduces_tables_11_and_12(self):
        # (material, T C, c1(T)/c1(T_rm) from the tables; max diff of their fit 2.9 %)
        for key, t, ratio in (("Ni200", 185.9, 0.79), ("Ni200", 95.6, 0.92),
                              ("SS304", 197.2, 0.75), ("SS304", 150.8, 0.80)):
            fac = hot_hardness_factor(load_material(key, FlagSet()), t + 273.15)
            self.assertAlmostEqual(fac, ratio, delta=0.035, msg=f"{key} {t}")

    def test_not_applied_but_reported(self):
        j = lambda T: build_joint("SS304", "SS304", sigma_a_um=1, sigma_b_um=1, slope_a=0.1,
                                  slope_b=0.1, pressure_MPa=1.0, temperature_K=T)
        cold = run_continuum(j(300.0), with_uncertainty=False)
        hot = run_continuum(j(410.0), with_uncertainty=False)
        self.assertAlmostEqual(cold.hardness.H_c.si, hot.hardness.H_c.si)
        self.assertNotIn(Code.HARDNESS_TEMPERATURE, cold.flags.codes())
        self.assertIn(Code.HARDNESS_TEMPERATURE, hot.flags.codes())
        self.assertIsNone(hot_hardness_factor(load_material("Cu", FlagSet()), 400.0))

    def test_why_it_is_not_applied(self):
        """Softened c1 moves the Hegazy benchmark from 1.03 to about 1.19 (pred/meas)."""
        sys.path.insert(0, str(Path(__file__).parent))
        from test_benchmarks import _f, load_cases, predict
        rows = load_cases()
        if not rows:
            self.skipTest("no benchmark data")
        base, hot = [], []
        for r in rows:
            mat = load_material(r["material_a"], FlagSet())
            fac = hot_hardness_factor(mat, _f(r, "temperature_K"))
            hp, hm = predict(r), _f(r, "h_measured_W_m2K")
            base.append(hp / hm)
            hot.append(hp * fac ** (-0.95 / (1 + 0.071 * _f(r, "c2"))) / hm)
        gm = lambda v: math.exp(statistics.fmean(map(math.log, v)))
        self.assertLess(abs(gm(base) - 1), 0.05)
        self.assertGreater(gm(hot), 1.15)


if __name__ == "__main__":
    unittest.main()
