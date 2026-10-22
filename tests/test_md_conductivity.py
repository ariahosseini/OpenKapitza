"""Bulk lattice thermal conductivity by periodic Langevin NEMD (work package 5c).

Pure-Python tests pin the flux bookkeeping (two halves, both reservoirs), the
unit conversion, the profile fit with reservoir margins, the 1/k vs 1/L
extrapolation and its flags, and the section-11 handling of metals. One engine
test runs a tiny cell when a LAMMPS binary is supplied:

    OPENKAPITZA_LMP=/path/to/lmp OPENKAPITZA_POTENTIALS=/path/to/potentials \\
        python3 -m unittest tests.test_md_conductivity
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from openkapitza.md.conductivity import (
    EV_PS_A_TO_W_MK, KappaPoint, KappaResult, KappaSpec, build_kappa_deck,
    extrapolate, kappa_from_run, parse_box, parse_production_thermo, read_profile,
    run_length, to_quantity,
)
from openkapitza.md.elastic import CubicSpec
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code, Severity
from openkapitza.schema.quantities import Confidence, Provenance

SI = CubicSpec("Si", "diamond", 5.431, 28.0855, "sw", "Si.sw", "Si", "Si")


def _synthetic_run(k_true=100.0, lz=200.0, lx=20.0, rate=2.0, n_bins=40):
    """Exact sawtooth profile for a given k, flux through two halves."""
    area = lx * lx
    flux = rate / (2 * area)                      # eV/ps/A^2
    grad = flux / k_true * EV_PS_A_TO_W_MK        # K/A
    z = (np.arange(n_bins) + 0.5) / n_bins
    t_hot, zc = 320.0, z * lz
    temp = np.where(z < 0.5, t_hot - grad * zc, t_hot - grad * (lz - zc))
    steps = np.arange(0, 100001, 1000.0)
    t_ps = steps * 1e-3
    thermo = np.column_stack([steps, np.full_like(steps, 300.0), np.full_like(steps, 320.0),
                              np.full_like(steps, 280.0), -rate * t_ps, rate * t_ps])
    return thermo, z, temp, np.full(n_bins, 50.0), (lx, lx, lz)


class Bookkeeping(unittest.TestCase):
    def test_unit_conversion(self):
        # 1 eV/ps/A^2 over 1 K/A is 1.602e-19 / 1e-12 / 1e-10 W/mK.
        self.assertAlmostEqual(EV_PS_A_TO_W_MK, 1602.176634, places=6)

    def test_recovers_k_from_exact_profile(self):
        thermo, z, temp, counts, box = _synthetic_run(k_true=137.0)
        p = kappa_from_run(thermo, z, temp, counts, box, 1.0, FlagSet())
        self.assertAlmostEqual(p.k, 137.0, places=6)
        self.assertAlmostEqual(p.k_halves[0], p.k_halves[1], places=6)
        self.assertLess(p.stderr, 1e-6)

    def test_flux_is_split_between_halves(self):
        # Forgetting the factor 2 doubles k; this pins it.
        thermo, z, temp, counts, box = _synthetic_run(k_true=50.0, rate=4.0)
        p = kappa_from_run(thermo, z, temp, counts, box, 1.0, FlagSet())
        self.assertAlmostEqual(p.k, 50.0, places=6)

    def test_imbalanced_reservoirs_flagged(self):
        thermo, z, temp, counts, box = _synthetic_run()
        thermo[:, 5] *= 1.2
        flags = FlagSet()
        kappa_from_run(thermo, z, temp, counts, box, 1.0, flags)
        self.assertTrue(any("steady state" in f.message for f in flags))

    def test_same_sign_rates_rejected(self):
        thermo, z, temp, counts, box = _synthetic_run()
        thermo[:, 4] *= -1
        with self.assertRaises(ValueError):
            kappa_from_run(thermo, z, temp, counts, box, 1.0, FlagSet())


class Extrapolation(unittest.TestCase):
    def _points(self, k_inf, c, lengths, rel=0.02):
        return [KappaPoint(L, 1.0 / (1.0 / k_inf + c / L), rel / (1.0 / k_inf + c / L),
                           (0, 0), 0, 0, 0, 0) for L in lengths]

    def test_recovers_k_inf(self):
        r = extrapolate(self._points(200.0, 1.0, [100, 200, 400]), FlagSet())
        self.assertAlmostEqual(r.k_inf, 200.0, places=6)
        self.assertGreater(r.extrapolation_ratio, 1.0)

    def test_two_lengths_warned(self):
        flags = FlagSet()
        extrapolate(self._points(200.0, 1.0, [100, 400]), flags)
        sev = [f.severity for f in flags if f.code == Code.MD_K_SIZE_EXTRAPOLATION]
        self.assertEqual(sev, [Severity.WARNING])

    def test_large_ratio_warned(self):
        flags = FlagSet()
        r = extrapolate(self._points(1000.0, 5.0, [50, 100, 200]), flags)
        self.assertGreater(r.extrapolation_ratio, 2.0)
        self.assertTrue(any("longest cell" in f.message for f in flags))

    def test_negative_intercept_rejected(self):
        pts = [KappaPoint(L, k, 1.0, (0, 0), 0, 0, 0, 0)
               for L, k in [(100, 20.0), (200, 50.0), (400, 400.0)]]
        with self.assertRaises(ValueError):
            extrapolate(pts, FlagSet())


    def test_k_falling_with_length_warned(self):
        pts = [KappaPoint(L, k, 1.0, (0, 0), 0, 0, 0, 0)
               for L, k in [(100, 50.0), (200, 30.0), (400, 20.0)]]
        flags = FlagSet()
        extrapolate(pts, flags)
        self.assertTrue(any("falls with length" in f.message for f in flags))


    def test_undetermined_extrapolation_is_critical(self):
        # The real SW-Si 500 K pair (17.4 and 34.8 nm): k_inf 545 +/- 3500.
        pts = [KappaPoint(174.1, 46.53, 14.40, (0, 0), 0, 0, 0, 0),
               KappaPoint(348.3, 85.74, 36.21, (0, 0), 0, 0, 0, 0)]
        flags = FlagSet()
        extrapolate(pts, flags)
        self.assertEqual(flags.worst, Severity.CRITICAL)
        self.assertTrue(any("undetermined" in f.message for f in flags))


class Policy(unittest.TestCase):
    def _result(self, metal):
        p = KappaPoint(400.0, 90.0, 3.0, (0, 0), 0, 0, 0, 0)
        return KappaResult(120.0, 10.0, [p], 0.1, 1.33,
                           meta={"is_metal": metal, "temperature": 300.0})

    def test_capped_low(self):
        q = to_quantity(self._result(False))
        self.assertEqual(q.provenance, Provenance.MD_COMPUTED)
        self.assertEqual(q.confidence, Confidence.LOW)
        self.assertEqual(q.name, "thermal_conductivity")

    def test_metal_labelled_lattice_only(self):
        q = to_quantity(self._result(True))
        self.assertEqual(q.name, "thermal_conductivity_lattice")
        self.assertIn("LATTICE PART ONLY", q.note)


class Parsing(unittest.TestCase):
    def test_deck_follows_kappa_template(self):
        deck = build_kappa_deck(KappaSpec(SI, length_cells=10))
        for needle in ("boundary        p p p", "compute         Thot  all temp/region hot",
                       "fix_modify      hot  temp Thot", "tally yes", "KAPPA_BOX",
                       "KAPPA_PRODUCTION_BEGIN", "reset_timestep  0", "change_box"):
            self.assertIn(needle, deck)

    def test_parse_thermo_and_box(self):
        text = ("KAPPA_BOX 21.7 21.7 108.7 2560\nKAPPA_PRODUCTION_BEGIN\n"
                "   Step Temp c_Thot c_Tcold f_hot f_cold\n"
                "0 300 320 280 0 0\n1000 301 320 280 -1.8 1.8\n2000 300 320 280 -3.7 3.6\n"
                "Loop time of 1 on 1 procs\nKAPPA_PRODUCTION_END\n")
        self.assertEqual(parse_box(text), (21.7, 21.7, 108.7, 2560))
        rows = parse_production_thermo(text)
        self.assertEqual(rows.shape, (3, 6))

    def test_read_last_profile_frame(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "profile.dat")
            p.write_text("# header\n# header\n# header\n"
                         "1000 2 100\n1 0.25 50 310\n2 0.75 50 290\n"
                         "2000 2 100\n1 0.25 50 312\n2 0.75 50 288\n")
            z, t, n = read_profile(p)
        self.assertEqual(t.tolist(), [312.0, 288.0])


LMP = os.environ.get("OPENKAPITZA_LMP")
POT = os.environ.get("OPENKAPITZA_POTENTIALS")


@unittest.skipUnless(LMP and POT, "set OPENKAPITZA_LMP and OPENKAPITZA_POTENTIALS")
class Engine(unittest.TestCase):
    def test_tiny_cell_runs_and_balances(self):
        spec = KappaSpec(SI, length_cells=12, cross_cells=3, npt_ps=2, nvt_ps=2,
                         steady_ps=10, production_ps=10, bins=24)
        p = run_length(Path(LMP), spec, Path(POT), FlagSet())
        self.assertGreater(p.k, 0.0)
        self.assertLess(p.meta["imbalance"], 0.2)


if __name__ == "__main__":
    unittest.main()
