"""TIM stack against measured greases: DeVoto et al., NREL/CP-5400-68014
(ITherm 2017). ASTM D5470, Ni-plated polished meter blocks, 139 kPa, BLT set
by glass spheres (95th-percentile diameters 29-266 um, their Table 1). Each
grease fits R = a*BLT + b with R^2 >= 0.991; 1/a is k, b is both boundaries."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

from openkapitza import tim_fill
from openkapitza.network.stack import MM2K_W, stack_resistance
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code

DATA = Path(__file__).parent / "benchmarks" / "tim_greases.csv"
BLTS = (29.0, 52.0, 78.5, 125.5, 158.0, 196.0, 266.0)    # their Table 1, 95th pct


def rows():
    with open(DATA, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class Greases(unittest.TestCase):
    def test_table3_is_the_inverse_slope_and_intercept(self):
        for r in rows():
            self.assertAlmostEqual(1 / float(r["slope_mm2K_W_per_um"]), float(r["k_W_mK"]), delta=0.06)
            self.assertAlmostEqual(float(r["intercept_mm2K_W"]), float(r["R_contact_mm2K_W"]), delta=0.05)

    def test_stack_with_their_k_and_rc_reproduces_their_fit(self):
        # The stack form (BLT/k + R_c) IS the D5470 regression; using the rounded
        # Table 3 k and R_c reproduces the Table 2 line to 5 % over 29-266 um (grease D:
        # Table 3 rounds k = 1/1.494 = 0.669 to 0.7).
        for r in rows():
            a, b = float(r["slope_mm2K_W_per_um"]), float(r["intercept_mm2K_W"])
            for blt in BLTS:
                st = stack_resistance(tim_fill(blt, float(r["k_W_mK"]),
                                               R_c_mm2K_W=float(r["R_contact_mm2K_W"])), None, FlagSet())
                self.assertAlmostEqual(st.R_total.si / MM2K_W / (a * blt + b), 1.0, delta=0.05,
                                       msg=f"grease {r['grease']} at {blt} um")

    def test_good_greases_at_production_blt_are_not_flagged(self):
        # The high-k greases (A, C, E) at 29-78.5 um: 12-34 mm^2K/W.
        for r in rows():
            if float(r["k_W_mK"]) < 3:
                continue
            for blt in (29.0, 52.0, 78.5):
                f = FlagSet()
                stack_resistance(tim_fill(blt, float(r["k_W_mK"]), tim_class="grease",
                                          R_c_mm2K_W=float(r["R_contact_mm2K_W"])), None, f)
                self.assertNotIn(Code.STACK_CLASS_RANGE, f.codes(), msg=f"{r['grease']} {blt}")

    def test_poor_grease_at_thick_blt_is_flagged(self):
        f = FlagSet()
        stack_resistance(tim_fill(266.0, 0.7, tim_class="grease", R_c_mm2K_W=28.3), None, f)
        self.assertIn(Code.STACK_CLASS_RANGE, f.codes())          # 408 mm^2K/W


if __name__ == "__main__":
    unittest.main()
