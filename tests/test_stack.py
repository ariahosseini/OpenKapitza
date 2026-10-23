"""Filled and bonded joints as a series stack (P0-1)."""

from __future__ import annotations

import unittest

from openkapitza import bond_fill, build_joint, render, run_continuum, tim_fill
from openkapitza.network.stack import (
    H_K_PESSIMISTIC, FillSpec, Layer, TIM_CLASS_RANGE, stack_resistance,
)
from openkapitza.schema import FlagSet, Provenance, Quantity
from openkapitza.schema.flags import Code, Severity


def _q(v, u):
    return Quantity.parse(v, u, provenance=Provenance.USER_MEASURED)


def _joint(fill, sigma=1.0):
    return build_joint("Al", "Cu", sigma_a_um=sigma, sigma_b_um=sigma, pressure_MPa=0.2,
                       temperature_K=330.0, fill=fill)


class Tim(unittest.TestCase):
    def test_bulk_plus_boundaries(self):
        f = FlagSet()
        st = stack_resistance(tim_fill(100, 4.0, R_c_mm2K_W=15.0), None, f)
        self.assertAlmostEqual(st.R_total.si, 100e-6 / 4.0 + 15e-6)
        self.assertTrue(st.complete)
        self.assertNotIn(Code.STACK_INCOMPLETE, f.codes())

    def test_measured_total_overrides_the_build_up(self):
        st = stack_resistance(tim_fill(100, 4.0, R_measured_mm2K_W=40.0), None, FlagSet())
        self.assertAlmostEqual(st.R_total.si, 40e-6)

    def test_missing_boundaries_make_an_upper_bound_and_skip_the_class_check(self):
        f = FlagSet()
        st = stack_resistance(tim_fill(50, 3.0, tim_class="grease"), None, f)
        self.assertFalse(st.complete)
        self.assertIn(Code.STACK_INCOMPLETE, f.codes())
        self.assertNotIn(Code.STACK_CLASS_RANGE, f.codes())   # partial R vs total range: no

    def test_class_range_flags_an_implausible_total(self):
        f = FlagSet()
        stack_resistance(tim_fill(2000, 3.0, tim_class="grease", R_c_mm2K_W=10), None, f)
        self.assertIn(Code.STACK_CLASS_RANGE, f.codes())
        f = FlagSet()
        stack_resistance(tim_fill(100, 3.0, tim_class="grease", R_c_mm2K_W=10), None, f)
        self.assertNotIn(Code.STACK_CLASS_RANGE, f.codes())   # 43 mm^2K/W: inside 20-100

    def test_class_ranges_are_blazej_in_si(self):
        self.assertEqual(TIM_CLASS_RANGE["pad"], (1.0e-4, 3.0e-4))   # 1-3 K cm^2/W

    def test_tim_thinner_than_the_roughness_is_warned(self):
        f = FlagSet()
        stack_resistance(tim_fill(2.0, 3.0, R_c_mm2K_W=5), _q(1.5, "um"), f)
        self.assertIn(Code.STACK_THIN_TIM, f.codes())


class Bond(unittest.TestCase):
    def test_layers_voids_and_faces_add_in_series(self):
        f = FlagSet()
        fill = bond_fill([("Cu3Sn", 2, 70), ("SAC", 30, 60, 0.04)], h_K_a_MW=300, h_K_b_MW=500)
        st = stack_resistance(fill, None, f)
        expect = 2e-6 / 70 + 30e-6 / (60 * 0.96) + 1 / 300e6 + 1 / 500e6
        self.assertAlmostEqual(st.R_total.si, expect)
        self.assertAlmostEqual(st.R_kapitza, 1 / 300e6 + 1 / 500e6)
        void = next(x for x in f.flags if x.code == Code.STACK_VOIDS)
        self.assertEqual(void.severity, Severity.CAUTION)          # 4 % < 5 %
        self.assertIn("OVER-states", void.remedy)

    def test_heavy_voiding_is_a_warning(self):
        f = FlagSet()
        stack_resistance(bond_fill([("SAC", 30, 60, 0.10)], h_K_a_MW=300, h_K_b_MW=300), None, f)
        self.assertEqual(next(x for x in f.flags if x.code == Code.STACK_VOIDS).severity,
                         Severity.WARNING)

    def test_missing_kapitza_graded_by_how_much_it_could_matter(self):
        f = FlagSet()
        st = stack_resistance(bond_fill([("SAC", 30, 60)]), None, f)
        flag = next(x for x in f.flags if x.code == Code.STACK_INCOMPLETE)
        self.assertEqual(flag.severity, Severity.INFO)               # thick solder
        self.assertTrue(st.complete)
        f = FlagSet()
        st = stack_resistance(bond_fill([("Au film", 0.2, 300)]), None, f)
        flag = next(x for x in f.flags if x.code == Code.STACK_INCOMPLETE)
        self.assertEqual(flag.severity, Severity.WARNING)            # thin film: h_K matters
        self.assertFalse(st.complete)
        share = (2 / H_K_PESSIMISTIC) / (0.2e-6 / 300 + 2 / H_K_PESSIMISTIC)
        self.assertAlmostEqual(flag.context["kapitza_share_pessimistic"], share)

    def test_dmm_used_when_both_neighbours_are_in_the_database(self):
        # Si | Al film | Cu: both faces have a DMM estimate from materials.yaml.
        r = run_continuum(build_joint("Si", "Cu", sigma_a_um=0.05, sigma_b_um=0.5,
                                      pressure_MPa=0.1, temperature_K=300.0,
                                      fill=bond_fill([("Al", 0.5, 237, 0.0, "Al")])),
                          with_uncertainty=False)
        labels = [t.label for t in r.stack.terms]
        self.assertIn("interface a (Kapitza)", labels)
        self.assertIn("interface b (Kapitza)", labels)
        self.assertGreater(r.stack.R_kapitza, 0)


class Pipeline(unittest.TestCase):
    def test_dispatch_report_and_uncertainty(self):
        r = run_continuum(_joint(tim_fill(100, 4.0, tim_class="grease", R_c_mm2K_W=15)),
                          n_samples=200)
        self.assertIsNotNone(r.stack)
        self.assertIsNone(r.gap)                                     # no gas path
        self.assertAlmostEqual(r.primary.h_joint.si, 1 / (25e-6 + 15e-6))
        self.assertEqual(r.primary.xi.si, 1.0)
        mc = r.uncertainty["h_joint"]
        self.assertLess(mc.p05, r.primary.h_joint.si); self.assertGreater(mc.p95, r.primary.h_joint.si)
        text = render(r)
        self.assertIn("filled joint (TIM)", text)
        self.assertIn("tim stack", text)
        self.assertNotIn("H_c (contact hardness)", text)

    def test_bare_joints_unchanged(self):
        r = run_continuum(build_joint("Al", "Cu", sigma_a_um=1.0, sigma_b_um=1.0,
                                      pressure_MPa=1.0, temperature_K=300.0, slope_a=0.1,
                                      slope_b=0.1), with_uncertainty=False)
        self.assertIsNone(r.stack)
        self.assertIn(r.primary.branch, ("plastic", "elastic"))

    def test_bad_inputs_refused(self):
        with self.assertRaises(ValueError):
            stack_resistance(FillSpec("glue", []), None, FlagSet())
        with self.assertRaises(ValueError):
            stack_resistance(bond_fill([("SAC", 30, 60, 1.0)]), None, FlagSet())


if __name__ == "__main__":
    unittest.main()
