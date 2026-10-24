"""Out-of-flatness (macro) constriction and the clamping check, against
Bahrami's 2004 thesis Appendix A.1 (205 crowned-contact points, vacuum)."""

from __future__ import annotations

import csv
import math
import statistics
import unittest
from pathlib import Path

from openkapitza import build_joint, run_continuum
from openkapitza.contact.macro import macro_contact
from openkapitza.schema.flags import Code, Severity

DATA = Path(__file__).parent / "benchmarks" / "nonconforming.csv"


def _f(r, k):
    return float(r[k])


def rows():
    with open(DATA, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cmy_R(r):
    """This package's plastic CMY at the apparent pressure, whole-joint K/W."""
    s, m = _f(r, "sigma_eff_um") * 1e-6, _f(r, "slope_eff")
    c1, c2, ks = _f(r, "c1_GPa") * 1e9, _f(r, "c2"), _f(r, "k_s_W_mK")
    A = math.pi * (_f(r, "b_L_mm") * 1e-3) ** 2
    x = (_f(r, "force_N") / A / (c1 * (1.62 * s / m * 1e6) ** c2)) ** (1 / (1 + 0.071 * c2))
    return 1 / (1.25 * ks * m / s * x ** 0.95 * A)


def macro(r):
    return macro_contact(_f(r, "force_N"), _f(r, "E_eff_GPa") * 1e9, _f(r, "sigma_eff_um") * 1e-6,
                         _f(r, "b_L_mm") ** 2 * 1e-6 / (2 * _f(r, "rho_m")),   # delta from rho
                         _f(r, "b_L_mm") * 1e-3, _f(r, "k_s_W_mK"))


gm = lambda v: math.exp(statistics.fmean(map(math.log, v)))
rms = lambda v: math.sqrt(statistics.fmean([(x - 1) ** 2 for x in v]))


class Thesis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = rows()
        cls.m = [macro(r) for r in cls.rows]

    def test_count(self):
        self.assertEqual(len(self.rows), 205)

    def test_reproduces_the_thesis_model_columns(self):
        # Printed to 2-3 significant figures; compare within that rounding.
        for r, (rho, a_H, alpha, tau, F_c, a_L, B, R_L) in zip(self.rows, self.m):
            self.assertAlmostEqual(B, _f(r, "thesis_B"), delta=0.006, msg=r["case_id"])
            self.assertAlmostEqual(tau / _f(r, "thesis_tau"), 1.0, delta=0.002, msg=r["case_id"])
            self.assertAlmostEqual(alpha, _f(r, "thesis_alpha"),
                                   delta=0.005 + 0.05 * alpha, msg=r["case_id"])   # printed 2 dp; sigma itself rounded
            self.assertAlmostEqual(R_L, _f(r, "thesis_R_L_K_W"),
                                   delta=max(0.006, 0.02 * R_L), msg=r["case_id"])

    def test_cmy_plus_macro_predicts_measured_resistance(self):
        v = [(cmy_R(r) + m[7]) / _f(r, "R_exp_K_W") for r, m in zip(self.rows, self.m)]
        self.assertAlmostEqual(gm(v), 1.03, delta=0.04)
        self.assertLess(rms(v), 0.17)           # 15.3 %; thesis approximate model 14.8 %
        self.assertTrue(all(0.5 < x < 2.0 for x in v))

    def test_ignoring_flatness_overpredicts_conductance(self):
        v = [cmy_R(r) / _f(r, "R_exp_K_W") for r in self.rows]
        self.assertLess(gm(v), 0.2)             # 0.14: h over-predicted ~7x
        self.assertLess(min(v), 0.01)           # worst ~125x

    def test_flat_joints_are_the_ones_cmy_gets_right(self):
        flat = [cmy_R(r) / _f(r, "R_exp_K_W") for r, m in zip(self.rows, self.m) if m[6] >= 0.9]
        self.assertGreater(len(flat), 5)
        self.assertAlmostEqual(gm(flat), 1.0, delta=0.25)


class Geometry(unittest.TestCase):
    def test_full_contact_above_the_critical_load(self):
        # delta/sigma ~ 1: flat at any load (b_L^2 <= 2.25 sigma rho  => F_c = 0)
        rho, a_H, alpha, tau, F_c, a_L, B, R_L = macro_contact(
            1000.0, 100e9, 1e-6, 1e-6, 12.5e-3, 20.0)
        self.assertEqual(F_c, 0.0)
        self.assertEqual(B, 1.0)
        self.assertEqual(R_L, 0.0)

    def test_more_crown_less_contact(self):
        Bs = [macro_contact(2000.0, 100e9, 1e-6, d, 12.5e-3, 20.0)[6] for d in (5e-6, 20e-6, 80e-6)]
        self.assertTrue(Bs[0] > Bs[1] > Bs[2])


class Pipeline(unittest.TestCase):
    def _run(self, **kw):
        base = dict(sigma_a_um=0.4, sigma_b_um=0.4, slope_a=0.07, slope_b=0.07,
                    pressure_MPa=1.0, temperature_K=300.0)
        base.update(kw)
        return run_continuum(build_joint("Al6061", "Al6061", **base), with_uncertainty=False)

    def test_no_flatness_is_stated_as_an_assumption(self):
        r = self._run()
        self.assertIsNone(r.macro)
        f = next(x for x in r.flags.flags if x.code == Code.FLATNESS_ASSUMED)
        self.assertEqual(f.severity, Severity.INFO)

    def test_bolted_without_flatness_warns(self):
        r = self._run(clamping="bolted")
        self.assertIn(Code.CLAMP_BOLTED, r.flags.codes())
        f = next(x for x in r.flags.flags if x.code == Code.FLATNESS_ASSUMED)
        self.assertEqual(f.severity, Severity.CAUTION)

    def test_crowned_joint_lowers_h_and_warns(self):
        flat = self._run()
        crowned = self._run(flatness_um=20.0, joint_area_mm2=63.5 ** 2, clamping="uniform")
        self.assertLess(crowned.primary.h_joint.si, 0.2 * flat.primary.h_joint.si)
        self.assertLess(crowned.macro.B, 0.8)
        f = next(x for x in crowned.flags.flags if x.code == Code.MACRO_CONSTRICTION)
        self.assertEqual(f.severity, Severity.WARNING)
        labels = [c.label for c in crowned.primary.contributions]
        self.assertIn("macro constriction (flatness)", labels)

    def test_nearly_flat_joint_barely_changes(self):
        flat = self._run()
        near = self._run(flatness_um=0.5, joint_radius_mm=12.5)
        self.assertGreater(near.primary.h_joint.si, 0.95 * flat.primary.h_joint.si)
        self.assertTrue(near.macro.conforming)

    def test_flatness_uncertainty_propagates(self):
        j = build_joint("Al6061", "Al6061", sigma_a_um=0.4, sigma_b_um=0.4, slope_a=0.07,
                        slope_b=0.07, pressure_MPa=1.0, temperature_K=300.0, flatness_um=20.0,
                        joint_radius_mm=30.0, sigma_rel_unc=None, slope_rel_unc=None,
                        pressure_rel_unc=None, hardness_rel_unc=None)
        mc = run_continuum(j, n_samples=200).uncertainty["h_joint"]
        self.assertGreater(mc.p95 / mc.p05, 1.2)

    def test_bad_inputs(self):
        kw = dict(sigma_a_um=1, sigma_b_um=1, pressure_MPa=1, temperature_K=300)
        with self.assertRaises(ValueError):
            build_joint("Cu", "Cu", flatness_um=10, **kw)                  # no joint size
        with self.assertRaises(ValueError):
            build_joint("Cu", "Cu", flatness_um=10, joint_radius_mm=5, joint_area_mm2=80, **kw)
        with self.assertRaises(ValueError):
            build_joint("Cu", "Cu", clamping="glued", **kw)


if __name__ == "__main__":
    unittest.main()


class Filled(unittest.TestCase):
    def test_tim_thinner_than_the_crown_warns(self):
        from openkapitza import tim_fill
        kw = dict(sigma_a_um=0.5, sigma_b_um=0.5, pressure_MPa=0.2, temperature_K=330.0,
                  joint_area_mm2=900.0)
        thin = run_continuum(build_joint("Al", "Cu", fill=tim_fill(25, 3.7, R_c_mm2K_W=4.2),
                                         flatness_um=40.0, **kw), with_uncertainty=False)
        f = next(x for x in thin.flags.flags if x.code == Code.MACRO_CONSTRICTION)
        self.assertEqual(f.severity, Severity.WARNING)
        thick = run_continuum(build_joint("Al", "Cu", fill=tim_fill(100, 3.7, R_c_mm2K_W=4.2),
                                          flatness_um=40.0, **kw), with_uncertainty=False)
        f = next(x for x in thick.flags.flags if x.code == Code.MACRO_CONSTRICTION)
        self.assertEqual(f.severity, Severity.INFO)
