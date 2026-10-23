"""Measured interface conductance (Kapitza) benchmark: Al/Si by TDTR, Hopkins et
al., PRB 82, 085307 (2010), Table I. 80 nm evaporated Al on Si(100); "a" samples
had the native oxide stripped (BOE) just before deposition. Room temperature
(the paper says room temperature; 295 K is our assumption)."""

from __future__ import annotations

import csv
import math
import statistics
import unittest
from pathlib import Path

from openkapitza.md.dmm import dmm_conductance
from openkapitza.schema import FlagSet, Provenance, Quantity, load_material

DATA = Path(__file__).parent / "benchmarks" / "kapitza_measured.csv"


def rows():
    with open(DATA, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class AlSi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = rows()
        f = FlagSet()
        T = Quantity.parse(295.0, "K", provenance=Provenance.USER_MEASURED)
        cls.dmm = dmm_conductance(load_material("Si", f), load_material("Al", f), T, f).h_K.si

    def test_table_transcribed(self):
        self.assertEqual(len(self.rows), 8)
        clean = [float(r["h_MW_m2K"]) for r in self.rows if r["native_oxide"] == "no"]
        oxide = [float(r["h_MW_m2K"]) for r in self.rows if r["native_oxide"] == "yes"]
        self.assertEqual(max(clean), 193.0)
        self.assertTrue(min(clean) > max(oxide))     # stripping the oxide always helps

    def test_native_oxide_resistance(self):
        # Paper: the native oxide adds 1.74e-9 +/- 9e-11 m^2K/W (~2.16 nm of SiO2),
        # from fitting its Eq. 4 (which also models roughness). The plain pairwise
        # difference here gives 1.91e-9: the paired samples differ in roughness.
        pairs = {}
        for r in self.rows:
            pairs.setdefault(r["case_id"].rstrip("a"), {})[r["native_oxide"]] = float(r["h_MW_m2K"]) * 1e6
        dR = statistics.fmean(1 / p["yes"] - 1 / p["no"] for p in pairs.values())
        self.assertAlmostEqual(dR, 1.91e-9, delta=0.01e-9)
        self.assertLess(abs(dR / 1.74e-9 - 1), 0.12)

    def test_debye_dmm_overpredicts_the_best_interface(self):
        """Our Debye-DMM (321 MW/m^2K) is 1.66x the cleanest, smoothest Al/Si
        interface (193) and 2.2x the as-received one (144). Their sine-dispersion
        DMM gives ~230. The DMM is an upper estimate for real, oxidised, rough
        interfaces -- which is why it is a placeholder, not a result."""
        best = max(float(r["h_MW_m2K"]) for r in self.rows) * 1e6
        ratio = self.dmm / best
        self.assertGreater(ratio, 1.4)
        self.assertLess(ratio, 2.0)
        self.assertTrue(all(self.dmm > float(r["h_MW_m2K"]) * 1e6 for r in self.rows))

    def test_roughness_trend_within_each_oxide_class(self):
        # Rougher Si (TMAH) gives lower h in both classes.
        for ox in ("yes", "no"):
            pts = [(float(r["roughness_rms_nm"]), float(r["h_MW_m2K"])) for r in self.rows
                   if r["native_oxide"] == ox]
            smooth = min(pts)[1]
            self.assertEqual(smooth, max(p[1] for p in pts))


if __name__ == "__main__":
    unittest.main()
