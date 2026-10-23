"""Electron-phonon series estimate for metal/non-metal interfaces (P1-14)."""

from __future__ import annotations

import math
import unittest

from openkapitza.md.electron_phonon import COUPLING, ep_conductance, series_with_ep
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code, Severity


class SeriesModel(unittest.TestCase):
    def test_reproduces_liang_al_si_series_value(self):
        # Liang et al. 2014: phonon-only 0.47 GW/m^2K, series model 0.37.
        flags = FlagSet()
        est = series_with_ep(0.47e9, "Al", 12.3, flags)
        self.assertAlmostEqual(est.h_total / 1e9, 0.37, delta=0.005)
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


if __name__ == "__main__":
    unittest.main()
