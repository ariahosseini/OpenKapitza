"""Truncated-Gaussian light-load model (P0-7), checked against Milanez,
Yovanovich & Mantelli, J. Thermophys. Heat Transfer 18(1) 2004."""

from __future__ import annotations

import math
import statistics
import sys
import unittest
from pathlib import Path

from openkapitza import build_joint, render, run_continuum
from openkapitza.contact.truncated import (
    p_over_hc_gaussian, p_over_hc_truncated, reference_truncation, tg_conductance,
    tg_conductance_exact, tg_factor, truncation_function,
)
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code

# Their Table 1 with the combined sigma, m printed on Figs 3-8.
# (sigma um, m, c1 GPa, c2, z_trunc, P_min kPa, P_max kPa, H_c at P_min, at P_max)
TABLE1 = {
    "S1": (0.72, 0.041, 10.67, -0.37, 4.1, 15.8, 2768, 6.2, 3.8),
    "S2": (1.29, 0.049, 10.67, -0.37, 3.8, 15.8, 2720, 6.6, 3.3),
    "S3": (3.07, 0.070, 10.67, -0.37, 3.5, 15.8, 3450, 6.5, 2.7),
    "N1": (1.74, 0.076, 4.33, -0.079, 4.2, 16.2, 1970, 3.7, 3.4),
    "N2": (3.0, 0.11, 4.33, -0.079, 4.2, 16.2, 1780, 3.6, 3.4),
    "N3": (4.2, 0.088, 4.33, -0.079, 3.9, 16.2, 2080, 3.6, 3.2),
}


class Paper(unittest.TestCase):
    def test_hardness_reproduces_table1(self):
        for name, (s, m, c1, c2, z, p0, p1, h0, h1) in TABLE1.items():
            som = s / m
            for p, h in ((p0, h0), (p1, h1)):
                hc = p * 1e3 / p_over_hc_truncated(p * 1e3, som, c1 * 1e9, c2, z) / 1e9
                self.assertAlmostEqual(hc, h, delta=0.03 * h, msg=f"{name} at {p} kPa")

    def test_truncation_raises_hardness_only_at_light_load(self):
        s, m, c1, c2, z = TABLE1["S3"][:5]
        light = p_over_hc_gaussian(15.8e3, s / m, c1 * 1e9, c2) / p_over_hc_truncated(
            15.8e3, s / m, c1 * 1e9, c2, z)
        heavy = p_over_hc_gaussian(3.45e6, s / m, c1 * 1e9, c2) / p_over_hc_truncated(
            3.45e6, s / m, c1 * 1e9, c2, z)
        self.assertGreater(light, 2.0)
        self.assertLess(heavy, 1.05)

    def test_fig2_sweep(self):
        # "for a variation of z_trunc from 4.1 to 3.3, the model varies from
        #  approximately 8 to 20 W/m^2K" (S3, 15.8 kPa, k ~ 18.2 W/mK)
        s, m, c1, c2 = 3.07e-6, 0.070, 10.67e9, -0.37
        h = {z: tg_conductance(18.2, s, m, p_over_hc_truncated(15.8e3, s / m * 1e6, c1, c2, z), z)
             for z in (4.1, 3.3)}
        self.assertAlmostEqual(h[4.1], 8.0, delta=1.0)
        self.assertAlmostEqual(h[3.3], 20.0, delta=3.0)

    def test_limits(self):
        # f -> infinity is CMY; at f = 40 the paper's "< 1 %" is 1.1 %
        self.assertAlmostEqual(tg_factor(1e-2, 6.0), 1.0, places=6)
        x40 = 40 / truncation_function(1.0, 4.0)
        self.assertAlmostEqual(tg_factor(x40, 4.0), 1.011, places=3)
        self.assertGreater(tg_factor(1e-6, 3.5), 5.0)

    def test_correlation_against_exact_model(self):
        # Paper: within 4 % over 1e-6..1e-2, z 3..4.5. True for P/H_c >= 1e-5;
        # at 1e-6 the correlation is up to 7 % LOW (documented in the module).
        worst_hi, worst_lo = 0.0, 0.0
        for e in [x / 4 for x in range(-24, -7)]:
            for z in (3.0, 3.5, 4.0, 4.5):
                d = tg_conductance(1, 1, 1, 10 ** e, z) / tg_conductance_exact(1, 1, 1, 10 ** e, z) - 1
                if e >= -5:
                    worst_hi = max(worst_hi, abs(d))
                worst_lo = max(worst_lo, abs(d))
        self.assertLess(worst_hi, 0.04)
        self.assertLess(worst_lo, 0.075)
        # untruncated exact model vs CMY's 1.25, 0.95 fit
        self.assertAlmostEqual(tg_conductance_exact(1, 1, 1, 1e-4, None) / (1.25 * 1e-4 ** 0.95),
                               1.0, delta=0.02)


class Lookup(unittest.TestCase):
    def test_table_values_and_interpolation(self):
        f = FlagSet()
        self.assertEqual(reference_truncation("SS304", 26.2, f).z_trunc.si, 3.8)
        self.assertAlmostEqual(reference_truncation("Ni200", 37.5, f).z_trunc.si, 4.05)
        self.assertEqual([x.severity.name for x in f.flags], ["INFO", "INFO"])
        r = reference_truncation("SS304", 8.0, f)
        self.assertEqual(r.z_trunc.si, 4.1)
        self.assertFalse(r.interpolated)
        self.assertIn(Code.TG_TRUNCATION_SOURCE, f.codes())
        self.assertIsNone(reference_truncation("Cu", 20.0, FlagSet()))


class Pipeline(unittest.TestCase):
    def _run(self, z, **kw):
        j = build_joint("SS304", "SS304", sigma_a_um=3.9, sigma_b_um=0.12, slope_a=0.098,
                        slope_b=0.022, pressure_MPa=0.0158, temperature_K=300.0, z_trunc=z,
                        hardness_rel_unc=None, **kw)
        return run_continuum(j, with_uncertainty=False)

    def test_light_load_flag_replaced_and_conductance_raised(self):
        cmy, tg = self._run(None), self._run("reference")
        self.assertIn(Code.CMY_LIGHT_LOAD, cmy.flags.codes())
        self.assertNotIn(Code.CMY_LIGHT_LOAD, tg.flags.codes())
        self.assertEqual(tg.branches["plastic"].h_joint.si > 3 * cmy.branches["plastic"].h_joint.si, True)
        self.assertEqual(tg.hardness.method, "truncated_gaussian")
        self.assertLess(tg.Y.si, cmy.Y.si)                 # truncation closes the gap
        self.assertIn("z_trunc (height cut-off)", render(tg))

    def test_numeric_z_out_of_range_is_warned(self):
        r = self._run(2.5)
        self.assertIn(Code.TG_TRUNCATION_RANGE, r.flags.codes())

    def test_uncertainty_on_z_widens_the_band(self):
        j = lambda u: build_joint("SS304", "SS304", sigma_a_um=3.9, sigma_b_um=0.12,
                                  slope_a=0.098, slope_b=0.022, pressure_MPa=0.0158,
                                  temperature_K=300.0, z_trunc=3.5, z_trunc_unc=u,
                                  sigma_rel_unc=None, slope_rel_unc=None, pressure_rel_unc=None,
                                  hardness_rel_unc=None, regime_override="plastic")
        fixed = run_continuum(j(None), n_samples=200).uncertainty["h_joint"]
        spread = run_continuum(j(0.2), n_samples=200).uncertainty["h_joint"]
        self.assertGreater(spread.p95 / spread.p05, 1.5)
        self.assertLess(fixed.p95 / fixed.p05, 1.2)

    def test_reference_needs_slopes(self):
        with self.assertRaises(ValueError):
            build_joint("SS304", "SS304", sigma_a_um=1, sigma_b_um=1, pressure_MPa=0.1,
                        temperature_K=300, z_trunc="reference")


class Hegazy(unittest.TestCase):
    """Independent data: our 207-point benchmark (Hegazy 1985). With the
    published bead-blasted z_trunc (held at the table ends outside 17.7-48 um)
    the light-load under-prediction shrinks: measured/predicted 1.05 -> 1.00 at
    Y/sigma 3.2-3.4 and 1.23 -> 1.14 above 3.4, overall RMS unchanged."""

    def test_light_load_bias_reduced(self):
        sys.path.insert(0, str(Path(__file__).parent))
        from test_benchmarks import _f, load_cases, predict
        rows = load_cases()
        if not rows:
            self.skipTest("no benchmark data")
        light_cmy, light_tg = [], []
        for r in rows:
            s, m = _f(r, "sigma_eff_um") * 1e-6, _f(r, "slope_eff")
            P, c1, c2 = _f(r, "pressure_MPa") * 1e6, _f(r, "c1_GPa") * 1e9, _f(r, "c2")
            som = s / m * 1e6
            if 1.53 * p_over_hc_gaussian(P, som, c1, c2) ** -0.097 < 3.4:
                continue
            z = reference_truncation(r["material_a"], som, FlagSet()).z_trunc.si
            ht = tg_conductance(_f(r, "k_s_W_mK"), s, m, p_over_hc_truncated(P, som, c1, c2, z), z)
            hm = _f(r, "h_measured_W_m2K")
            light_cmy.append(hm / predict(r)); light_tg.append(hm / ht)
        gm = lambda v: math.exp(statistics.fmean(map(math.log, v)))
        self.assertGreater(len(light_cmy), 20)
        self.assertGreater(gm(light_cmy), 1.2)
        self.assertLess(gm(light_tg), 1.16)


if __name__ == "__main__":
    unittest.main()
