"""Quantum heat-capacity bracket on a classical-MD h_K (S-11, KB 31)."""

from __future__ import annotations

import unittest
from dataclasses import replace

from openkapitza import build_joint, run_continuum
from openkapitza.md.dmm import user_kapitza
from openkapitza.md.quantum import debye_fraction, quantum_band


class Fractions(unittest.TestCase):
    def test_limits_and_monotonic(self):
        self.assertAlmostEqual(debye_fraction(1e5, 300.0), 1.0, places=4)
        # Debye C_V/3Nk at T = theta is 0.952 (textbook value)
        self.assertAlmostEqual(debye_fraction(300.0, 300.0), 0.952, places=3)
        self.assertLess(debye_fraction(100.0, 645.0), debye_fraction(300.0, 645.0))

    def test_band_orders_the_cutoffs(self):
        lo, hi = quantum_band(300.0, 645.0, 374.0)
        self.assertLess(lo, hi)
        self.assertAlmostEqual(lo, debye_fraction(300.0, 645.0))
        self.assertEqual(quantum_band(300.0, 374.0, 645.0), (lo, hi))


class InPipeline(unittest.TestCase):
    def run_with(self, source, T=300.0):
        k = replace(user_kapitza(485e6), source=source)
        r = run_continuum(build_joint("Si", "Ge", sigma_a_um=0.3, sigma_b_um=0.3,
                                      slope_a=0.05, slope_b=0.05, pressure_MPa=1.0,
                                      temperature_K=T), kapitza=k)
        return {str(f.code): f for f in r.flags.flags}

    def test_md_value_gets_the_bracket(self):
        f = self.run_with("md")["QUANTUM_EFFECTS"]
        lo, hi = f.context["quantum_band"]
        self.assertTrue(0.75 < lo < hi < 0.95)
        self.assertIn("MD value in use, 485", f.message)
        self.assertIn("not a correction", f.message)

    def test_measured_value_is_left_alone(self):
        self.assertNotIn("QUANTUM_EFFECTS", self.run_with("user"))

    def test_hot_md_run_is_silent(self):
        self.assertNotIn("QUANTUM_EFFECTS", self.run_with("md", T=900.0))


if __name__ == "__main__":
    unittest.main()
