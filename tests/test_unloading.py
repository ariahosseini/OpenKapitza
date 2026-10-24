"""Unloading (hysteresis), Milanez, Yovanovich & Mantelli 2004 Eqs. 16-19
(Mikic's model; z_trunc unset = fully Gaussian)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from openkapitza import build_joint, run_continuum
from openkapitza.contact.truncated import (
    p_over_hc_gaussian, p_over_hc_truncated, tg_conductance, tg_conductance_exact,
)
from openkapitza.contact.unloading import (
    _quad, area_integral, force_integral, pressure_ratio, spot_density, unloading_conductance,
)
from openkapitza.schema.flags import Code, Severity

S2 = math.sqrt(2.0)
# Milanez S1: combined sigma, m from their Fig. 3, their c1/c2, Table 1 maximum load
S, M, K, PMAX = 0.72e-6, 0.041, 18.8, 2.768e6
SOM = S / M * 1e6


def hmax(z):
    x = (p_over_hc_truncated(PMAX, SOM, 10.67e9, -0.37, z) if z
         else p_over_hc_gaussian(PMAX, SOM, 10.67e9, -0.37))
    return PMAX / x


class Integrals(unittest.TestCase):
    def test_start_of_unloading_is_the_first_loading_point(self):
        for lam in (1.5, 2.5, 3.5):
            self.assertAlmostEqual(area_integral(lam, 1e-9) / (0.5 * math.erfc(lam / S2)), 1.0,
                                   places=6)
            self.assertAlmostEqual(force_integral(lam, 1e-9) / (0.5 * math.erfc(lam / S2)), 1.0,
                                   places=6)
            t = 0.5 * math.erfc(4.0 / S2)
            self.assertAlmostEqual(pressure_ratio(lam, 1e-9, t),
                                   0.5 * math.erfc(lam / S2) - t, places=8)

    def test_eq18_derived_form_integrates_to_the_closed_form(self):
        # n = n_max - 1/4 Int_0^Z e^{-u^2}/erfc(u/sqrt2) [u - r/sqrt(2pi)] X dX, u = lam + X^2
        lam, Z = 2.0, 0.8
        def g(X):
            u = lam + X * X
            e = np.array([math.erfc(v / S2) for v in u])
            r = np.exp(-u * u / 2) / e
            return np.exp(-u * u) / e * (u - r / math.sqrt(2 * math.pi)) * X
        n = spot_density(lam, 0.0) - 0.25 * _quad(g, 0.0, Z)
        self.assertAlmostEqual(n / spot_density(lam, Z), 1.0, places=6)
        self.assertLess(spot_density(lam, 3.0), 1e-12)          # all contacts open

    def test_full_unloading_curve_is_physical(self):
        lam = 2.5
        prev = math.inf
        for Z in (0.0, 0.2, 0.5, 1.0, 1.5):
            r = pressure_ratio(lam, Z, 0.0)
            self.assertLess(r, prev)
            prev = r


class Conductance(unittest.TestCase):
    def test_at_maximum_load_equals_the_exact_first_loading_model(self):
        for z in (None, 4.1):
            H = hmax(z)
            r = unloading_conductance(K, S, M, PMAX, PMAX, H, z)
            self.assertAlmostEqual(r.h_c / tg_conductance_exact(K, S, M, PMAX / H, z), 1.0,
                                   places=3)

    def test_hysteresis_grows_as_the_load_comes_off(self):
        for z in (None, 4.1):
            H = hmax(z)
            prev_ratio, prev_h = 1.0, math.inf
            for P in (2.0e6, 1.0e6, 4e5, 1e5, 3e4):
                x = (p_over_hc_truncated(P, SOM, 10.67e9, -0.37, z) if z
                     else p_over_hc_gaussian(P, SOM, 10.67e9, -0.37))
                first = tg_conductance(K, S, M, x, z) if z else 1.25 * K * M / S * x ** 0.95
                h = unloading_conductance(K, S, M, P, PMAX, H, z).h_c
                self.assertGreater(h / first, prev_ratio)
                self.assertLess(h, prev_h)
                prev_ratio, prev_h = h / first, h
            self.assertTrue(2.0 < prev_ratio < 4.0)              # 2.6-3.1 at 30 kPa

    def test_truncated_model_has_an_edge(self):
        x = p_over_hc_truncated(3e6, 1 / 0.08, 6.27e9, -0.23, 3.0)
        self.assertFalse(unloading_conductance(18.0, 1e-6, 0.08, 1e5, 3e6, 3e6 / x, 3.0).at_edge)
        self.assertTrue(unloading_conductance(18.0, 1e-6, 0.08, 1e3, 3e6, 3e6 / x, 3.0).at_edge)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            unloading_conductance(K, S, M, 2 * PMAX, PMAX, hmax(None))


class Pipeline(unittest.TestCase):
    def _run(self, **kw):
        base = dict(sigma_a_um=1, sigma_b_um=1, slope_a=0.08, slope_b=0.08, pressure_MPa=1.0,
                    temperature_K=300.0, regime_override="plastic")
        base.update(kw)
        return run_continuum(build_joint("SS304", "SS304", **base), n_samples=100)

    def test_relaxed_joint_conducts_better_and_says_so(self):
        first, relaxed = self._run(), self._run(max_pressure_MPa=5.0)
        self.assertGreater(relaxed.primary.h_joint.si, 1.5 * first.primary.h_joint.si)
        f = next(x for x in relaxed.flags.flags if x.code == Code.UNLOADING)
        self.assertGreater(f.context["ratio"], 1.5)
        self.assertNotIn(Code.CMY_LIGHT_LOAD, relaxed.flags.codes())
        self.assertGreater(relaxed.Y.si, 0)
        self.assertGreater(relaxed.uncertainty["h_joint"].p95, relaxed.uncertainty["h_joint"].p05)

    def test_max_not_above_working_pressure_is_first_loading(self):
        a, b = self._run(), self._run(max_pressure_MPa=1.0)
        self.assertAlmostEqual(a.primary.h_joint.si, b.primary.h_joint.si)
        self.assertEqual(next(x for x in b.flags.flags if x.code == Code.UNLOADING).severity,
                         Severity.INFO)

    def test_truncated_unloading_at_light_load_is_cautioned(self):
        r = self._run(pressure_MPa=0.1, max_pressure_MPa=3.0, slope_a=0.098, slope_b=0.022,
                      sigma_a_um=3.9, sigma_b_um=0.12, z_trunc=3.5)
        sev = [x.severity for x in r.flags.flags if x.code == Code.UNLOADING]
        self.assertIn(Severity.CAUTION, sev)


class HardnessOverrides(unittest.TestCase):
    def test_measured_c1_c2_remove_the_fallback(self):
        kw = dict(sigma_a_um=0.5, sigma_b_um=0.5, slope_a=0.05, slope_b=0.05, pressure_MPa=1.0,
                  temperature_K=300.0)
        base = run_continuum(build_joint("Cu", "Cu", **kw), with_uncertainty=False)
        self.assertIn(Code.HARDNESS_FALLBACK, base.flags.codes())
        mine = run_continuum(build_joint("Cu", "Cu", c1_a_GPa=0.9, c2_a=-0.05, c1_b_GPa=0.9,
                                         c2_b=-0.05, **kw), with_uncertainty=False)
        self.assertNotIn(Code.HARDNESS_FALLBACK, mine.flags.codes())
        self.assertEqual(mine.hardness.method, "song_yovanovich")
        hv = run_continuum(build_joint("Cu", "Cu", Hv_a_GPa=0.8, Hv_b_GPa=0.8, **kw),
                           with_uncertainty=False)
        self.assertAlmostEqual(hv.hardness.H_c.si, 0.8e9)

    def test_bad_overrides(self):
        kw = dict(sigma_a_um=1, sigma_b_um=1, pressure_MPa=1, temperature_K=300)
        for bad in (dict(c1_a_GPa=1.0), dict(c1_a_GPa=1.0, c2_a=0.5), dict(Hv_a_GPa=-1)):
            with self.assertRaises(ValueError):
                build_joint("Cu", "Cu", **kw, **bad)

    def test_agent_fields(self):
        from openkapitza.agent import Session
        s = Session(n_samples=50)
        s.start()
        for a in ["Cu", "", "Cu", "", "300", "bare", "1", "0.5", "0.5", "0.05", "0.05",
                  "vacuum", "uniform", "", "set c1_a_GPa=900 MPa", "set c2_a=-0.05",
                  "set c1_b_GPa=0.9", "set c2_b=-0.05", "set max_pressure_MPa=4", "run"]:
            out = s.handle(a)
        self.assertNotIn(Code.HARDNESS_FALLBACK, s.result.flags.codes())
        self.assertIn(Code.UNLOADING, s.result.flags.codes())


if __name__ == "__main__":
    unittest.main()
