"""Electron-phonon series estimate for metal/non-metal interfaces (P1-14)."""

from __future__ import annotations

import math
import unittest

from openkapitza.md.electron_phonon import (
    COUPLING, COUPLING_RANGE, LATTICE_K, ep_conductance, series_band, series_with_ep,
)
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code, Severity


class SeriesModel(unittest.TestCase):
    def test_reproduces_yang_al_si_series_value_from_their_inputs(self):
        # Yang et al., arXiv:1401.5550: G_pp 0.46 GW/m^2K, k_p 17.0 W/mK (their
        # Green-Kubo), g 2.45e17 -> G_ep 2.0, series 0.37. Their inputs, not a fit.
        flags = FlagSet()
        est = series_with_ep(0.46e9, "Al", 17.0, flags)
        self.assertAlmostEqual(est.h_ep / 1e9, 2.0, delta=0.05)
        self.assertAlmostEqual(est.h_total / 1e9, 0.37, delta=0.01)
        self.assertIn(Code.NEMD_PHONON_ONLY_METAL, flags.codes())

    def test_formula(self):
        self.assertAlmostEqual(ep_conductance(4e17, 9.0), math.sqrt(3.6e18))

    def test_unsourced_metal_is_flagged_not_guessed(self):
        flags = FlagSet()
        est = series_with_ep(5e8, "Unobtainium", 10.0, flags)
        self.assertEqual(est.h_total, 5e8)
        self.assertEqual(flags.worst, Severity.WARNING)

    def test_every_coupling_value_has_a_citation(self):
        for metal, rec in COUPLING.items():
            self.assertTrue(rec.get("citation"), metal)


class Band(unittest.TestCase):
    def test_band_is_ordered_and_brackets_the_central_value(self):
        for metal in ("Al", "Cu"):
            b = series_band(0.4e9, metal, FlagSet())
            lo, mid, hi = b.h_total
            self.assertLess(lo, mid + 1e-6); self.assertLess(mid, hi + 1e-6)
            self.assertTrue(all(h < 0.4e9 for h in b.h_total))   # series only lowers

    def test_cu_band_spans_the_coupling_spread(self):
        b = series_band(0.4e9, "Cu", FlagSet())
        g_lo, g_hi = COUPLING_RANGE["Cu"]
        self.assertAlmostEqual(b.h_ep[0], math.sqrt(g_lo * LATTICE_K["Cu"]["low"]))
        self.assertAlmostEqual(b.h_ep[2], math.sqrt(g_hi * LATTICE_K["Cu"]["high"]))
        self.assertGreater(b.h_ep[2] / b.h_ep[0], 2.0)            # g spans ~5x

    def test_default_lattice_k_is_first_principles_not_md(self):
        """The real lattice value (with p-e scattering) is below the p-p-only bound."""
        for metal, rec in LATTICE_K.items():
            self.assertLessEqual(rec["low"], rec["value"])
            self.assertLessEqual(rec["value"], rec["high"])
            self.assertTrue(rec["citation"])
        self.assertLess(LATTICE_K["Al"]["value"], 17.0)            # MEAM MD k_p

    def test_flag_states_the_band_and_its_direction(self):
        f = FlagSet()
        series_band(0.46e9, "Al", f)
        flag = next(x for x in f.flags if x.code == Code.NEMD_PHONON_ONLY_METAL)
        self.assertEqual(flag.severity, Severity.CAUTION)
        self.assertIn("ONE sourced value", flag.message)          # Al: single g
        self.assertIn("LOWER bound", flag.remedy)
        self.assertIn("6-15 %", flag.remedy)

    def test_unsourced_metal_returns_none_with_a_warning(self):
        f = FlagSet()
        self.assertIsNone(series_band(0.4e9, "Unobtainium", f))
        self.assertEqual(f.worst, Severity.WARNING)


if __name__ == "__main__":
    unittest.main()
