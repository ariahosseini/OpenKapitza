"""Ballistic contact spots (P0-11), slope instrument (P0-12), doped Si (P0-13)."""

from __future__ import annotations

import math
import unittest

from openkapitza import build_joint, run_continuum
from openkapitza.contact.ballistic import (
    C_BALLISTIC, Carrier, carrier_of, debye_heat_capacity, mean_spot_radius, side_factor,
)
from openkapitza.schema.db import load_material


def codes(r):
    return {str(f.code): f for f in r.flags.flags}


class SpotRadius(unittest.TestCase):
    def test_yovanovich_ratio(self):
        # a/(sigma/m) = 0.32, 0.38, 0.48 at P/H_c = 1e-4, 1e-3, 1e-2 (KB 23)
        for x, want in ((1e-4, 0.32), (1e-3, 0.38), (1e-2, 0.48)):
            self.assertAlmostEqual(mean_spot_radius(1.0, 1.0, x), want, delta=0.01)


class Carriers(unittest.TestCase):
    def test_debye_heat_capacity_limits(self):
        n = 5e28
        self.assertAlmostEqual(debye_heat_capacity(n, 100.0, 3000.0) / (3 * n * 1.380649e-23),
                               1.0, places=2)
        self.assertLess(debye_heat_capacity(n, 645.0, 300.0), 3 * n * 1.380649e-23)

    def test_silicon_gray_path_and_long_share(self):
        c = carrier_of(load_material("Si"), 300.0)
        self.assertEqual(c.kind, "phonon")
        self.assertTrue(30e-9 < c.mfp_m < 60e-9, c.mfp_m)     # the classic ~40 nm
        self.assertEqual(c.long_fraction, 0.40)
        self.assertAlmostEqual(c.mfp_long_m, 1e-6)

    def test_metals_use_the_electron_path(self):
        c = carrier_of(load_material("Cu"), 300.0)
        self.assertEqual((c.kind, round(c.mfp_m * 1e9, 1)), ("electron", 39.9))

    def test_side_factor(self):
        c = Carrier("phonon", 100e-9, 0.5, 1e-6)
        a = 1e-6
        self.assertAlmostEqual(side_factor(c, a, high=False), 1 + C_BALLISTIC * 0.1)
        hi = 1 / (0.5 / (1 + C_BALLISTIC * 0.1) + 0.5 / (1 + C_BALLISTIC))
        self.assertAlmostEqual(side_factor(c, a, high=True), hi)


class InPipeline(unittest.TestCase):
    def test_polished_silicon_is_flagged_with_a_band(self):
        r = run_continuum(build_joint("Si", "Si", sigma_a_um=0.02, sigma_b_um=0.02,
                                      slope_a=0.02, slope_b=0.02, pressure_MPa=1.0,
                                      temperature_K=300.0))
        f = codes(r)["CONTACT_BALLISTIC"]
        self.assertEqual(int(f.severity), 1)
        self.assertGreater(f.context["F_high"], 1.3)
        self.assertGreater(f.context["F_low"], 1.05)

    def test_lapped_metal_is_not(self):
        r = run_continuum(build_joint("Cu", "Cu", sigma_a_um=1.0, sigma_b_um=1.0,
                                      slope_a=0.1, slope_b=0.1, pressure_MPa=1.0,
                                      temperature_K=300.0))
        self.assertNotIn("CONTACT_BALLISTIC", codes(r))

    def test_band_brackets_below_the_diffusive_value(self):
        r = run_continuum(build_joint("Si", "Si", sigma_a_um=0.02, sigma_b_um=0.02,
                                      slope_a=0.02, slope_b=0.02, pressure_MPa=1.0,
                                      temperature_K=300.0))
        f = codes(r)["CONTACT_BALLISTIC"]
        self.assertLess(f.context["F_low"], f.context["F_high"])
        self.assertTrue(0.1 < f.context["spot_radius_nm"] / 1000 < 1.0)   # a ~ 0.3 um


class SlopeInstrument(unittest.TestCase):
    def run_with(self, inst):
        return codes(run_continuum(build_joint(
            "Al", "Al", sigma_a_um=0.5, sigma_b_um=0.5, slope_a=0.05, slope_b=0.05,
            pressure_MPa=1.0, temperature_K=300.0, slope_instrument=inst)))

    def test_afm_and_optical_warn(self):
        for inst in ("AFM", "optical"):
            f = self.run_with(inst)["SLOPE_INSTRUMENT"]
            self.assertEqual(int(f.severity), 2)

    def test_stylus_and_unknown_are_silent(self):
        for inst in ("stylus", "unknown", None):
            self.assertNotIn("SLOPE_INSTRUMENT", self.run_with(inst))


class DopedSilicon(unittest.TestCase):
    def test_flag_above_onset_only(self):
        kw = dict(sigma_a_um=0.3, sigma_b_um=0.5, slope_a=0.05, slope_b=0.08,
                  pressure_MPa=1.0, temperature_K=300.0)
        r = run_continuum(build_joint("Si", "Al", doping_a_cm3=1e20, **kw))
        self.assertEqual(int(codes(r)["DOPED_SI_CONDUCTIVITY"].severity), 1)
        r = run_continuum(build_joint("Si", "Al", doping_a_cm3=1e17, **kw))
        self.assertNotIn("DOPED_SI_CONDUCTIVITY", codes(r))
        r = run_continuum(build_joint("Al", "Al", doping_a_cm3=1e20, **kw))
        self.assertNotIn("DOPED_SI_CONDUCTIVITY", codes(r))


if __name__ == "__main__":
    unittest.main()
