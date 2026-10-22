"""Nanoindentation by MD, compare-only (work package 5d).

Pure-Python tests pin the cap contact-area model, the Hertz fit, pop-in
detection, the post-yield hardness, the size and section-11 flags, and that
the result is never fillable. One engine test runs a tiny Ni slab when a
LAMMPS binary is supplied (OPENKAPITZA_LMP, OPENKAPITZA_POTENTIALS).
"""

from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

import numpy as np

from openkapitza.md import fill_policy
from openkapitza.md.elastic import CubicSpec
from openkapitza.md.indentation import (
    EV_PER_A3_TO_GPA, EV_PER_A_TO_NN, IndentSpec, analyse_curve,
    build_indent_deck, cap_contact_radius, compare_with_microhardness,
    hertz_modulus, parse_indent_output, run_indentation, to_quantity,
)
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code
from openkapitza.schema.quantities import Provenance

NI = CubicSpec("Ni", "fcc", 3.52, 58.6934, "eam", "Ni_u3.eam", None, "Ni")


def _hertz_rows(e_star_gpa=200.0, radius=20.0, depths=None, popin_at=None, drop=0.3):
    depths = np.arange(0.2, 6.01, 0.2) if depths is None else depths
    e = e_star_gpa / EV_PER_A3_TO_GPA
    fz = (4 / 3) * e * math.sqrt(radius) * depths ** 1.5          # eV/A
    if popin_at is not None:
        i = int(np.argmin(abs(depths - popin_at)))
        fz[i + 1:] = fz[i] * (1 - drop) + 0.5 * (fz[i + 1:] - fz[i + 1])
    rows = np.column_stack([depths, 0 * depths, 0 * depths, fz, 0 * depths])
    return rows


class Geometry(unittest.TestCase):
    def test_cap_radius(self):
        a = cap_contact_radius(10.0, np.array([0.0, 1.0, 10.0, 12.0]))
        np.testing.assert_allclose(a, [0.0, math.sqrt(19.0), 10.0, 10.0])

    def test_hertz_fit_exact(self):
        rows = _hertz_rows(e_star_gpa=231.0)
        e = hertz_modulus(rows[:, 0], rows[:, 3], 20.0) * EV_PER_A3_TO_GPA
        self.assertAlmostEqual(e, 231.0, places=6)


class Curve(unittest.TestCase):
    def test_no_popin_flagged_and_no_hardness(self):
        flags = FlagSet()
        r = analyse_curve(_hertz_rows(), 20.0, 200.0, flags)
        self.assertIsNone(r.popin_depth)
        self.assertTrue(math.isnan(r.hardness))
        self.assertTrue(any("never yielded" in f.message for f in flags))
        self.assertAlmostEqual(r.reduced_modulus, 200.0, places=6)

    def test_popin_found_and_elastic_fit_stops_there(self):
        flags = FlagSet()
        r = analyse_curve(_hertz_rows(popin_at=3.0), 20.0, 200.0, flags)
        self.assertAlmostEqual(r.popin_depth, 3.0, places=6)
        self.assertAlmostEqual(r.reduced_modulus, 200.0, places=6)
        self.assertTrue(math.isfinite(r.hardness))
        self.assertAlmostEqual(r.popin_load,
                               (4 / 3) * 200 / EV_PER_A3_TO_GPA * math.sqrt(20) * 3 ** 1.5
                               * EV_PER_A_TO_NN, places=6)

    def test_narrow_box_flagged(self):
        flags = FlagSet()
        r = analyse_curve(_hertz_rows(), 20.0, 30.0, flags)
        self.assertLess(r.lateral_ratio, 3.0)
        self.assertTrue(any("contact diameters" in f.message for f in flags))


class Policy(unittest.TestCase):
    def _result(self):
        return analyse_curve(_hertz_rows(popin_at=3.0), 20.0, 200.0, FlagSet())

    def test_never_fillable(self):
        self.assertFalse(fill_policy.POLICIES["nanohardness"].fillable)
        q = to_quantity(self._result())
        self.assertEqual(q.provenance, Provenance.MD_COMPUTED)
        self.assertIn("COMPARE ONLY", q.note)

    def test_always_says_not_microhardness(self):
        flags = FlagSet()
        compare_with_microhardness(self._result(), flags)
        self.assertIn(Code.MD_INDENT_NOT_MICROHARDNESS, flags.codes())

    def test_reversed_size_effect_flagged(self):
        r = self._result()
        flags = FlagSet()
        compare_with_microhardness(r, flags, microhardness_GPa=r.hardness + 1.0)
        self.assertIn(Code.MD_INDENT_SIZE_EFFECT_REVERSED, flags.codes())
        flags = FlagSet()
        compare_with_microhardness(r, flags, microhardness_GPa=0.5 * r.hardness)
        self.assertNotIn(Code.MD_INDENT_SIZE_EFFECT_REVERSED, flags.codes())


class Parsing(unittest.TestCase):
    def test_deck(self):
        deck = build_indent_deck(IndentSpec(NI, max_depth=1.0, step=0.5))
        for needle in ("boundary        p p s", "indent 10.0 sphere v_cx v_cy v_zc",
                       "fix_modify      ind energy yes", "variable        k loop 2",
                       "jump            SELF step", "INDENT_SURFACE"):
            self.assertIn(needle, deck)

    def test_parse(self):
        text = ("INDENT_SURFACE 42.2 56.3 56.3 3400\n"
                "INDENT 0.250000 0.0 0.0 0.29 42.1\nINDENT 0.500000 0.0 0.0 1.46 42.0\n")
        (zs, lx, ly, n), rows = parse_indent_output(text)
        self.assertEqual((zs, n), (42.2, 3400))
        self.assertEqual(rows.shape, (2, 5))


LMP = os.environ.get("OPENKAPITZA_LMP")
POT = os.environ.get("OPENKAPITZA_POTENTIALS")


@unittest.skipUnless(LMP and POT, "set OPENKAPITZA_LMP and OPENKAPITZA_POTENTIALS")
class Engine(unittest.TestCase):
    def test_tiny_slab_hertzian_start(self):
        if not (Path(POT) / NI.potential).is_file():
            self.skipTest("Ni_u3.eam not in OPENKAPITZA_POTENTIALS")
        spec = IndentSpec(NI, radius=15.0, lateral_cells=10, depth_cells=8,
                          max_depth=1.5, step=0.25)
        r = run_indentation(Path(LMP), spec, Path(POT), FlagSet())
        self.assertTrue(np.all(np.diff(r.load) > 0))
        self.assertGreater(r.reduced_modulus, 100.0)
        self.assertLess(r.reduced_modulus, 400.0)


if __name__ == "__main__":
    unittest.main()
