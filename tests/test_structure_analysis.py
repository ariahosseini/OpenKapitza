"""Structure analysis against ideal crystals with known answers (Phase 2)."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from openkapitza.structure.analysis import (
    angle_peak, bond_angles, coordination, partial_rdf, read_lammps_data,
    write_lammps_data,
)
from openkapitza.structure.crystals import beta_cristobalite, diamond


class Cristobalite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = beta_cristobalite((3, 3, 3))

    def test_density(self):
        self.assertAlmostEqual(self.c.density(), 2.17, delta=0.01)

    def test_si_o_bond_and_coordination(self):
        rdf = partial_rdf(self.c, 1, 2, r_max=4.0)
        self.assertAlmostEqual(rdf.first_peak(1.0, 2.0), math.sqrt(3) / 8 * 7.16, delta=0.02)
        self.assertTrue(np.all(coordination(self.c, 1, 2, 2.0) == 4))
        self.assertTrue(np.all(coordination(self.c, 2, 1, 2.0) == 2))

    def test_angles(self):
        self.assertAlmostEqual(angle_peak(bond_angles(self.c, 1, 2, 2.0)), 109.0, delta=1.5)
        self.assertAlmostEqual(angle_peak(bond_angles(self.c, 2, 1, 2.0)), 179.0, delta=1.5)

    def test_si_si_distance(self):
        rdf = partial_rdf(self.c, 1, 1, r_max=4.0)
        self.assertAlmostEqual(rdf.first_peak(2.0, 4.0), math.sqrt(3) / 4 * 7.16, delta=0.02)


class Diamond(unittest.TestCase):
    def test_silicon_bond_angle_and_coordination(self):
        c = diamond(5.431, (3, 3, 3))
        self.assertTrue(np.all(coordination(c, 1, 1, 2.6) == 4))
        self.assertAlmostEqual(angle_peak(bond_angles(c, 1, 1, 2.6)), 109.0, delta=1.5)
        self.assertAlmostEqual(c.density(), 2.329, delta=0.005)


class DataFiles(unittest.TestCase):
    def test_round_trip(self):
        c = beta_cristobalite((2, 2, 2))
        with tempfile.TemporaryDirectory() as d:
            write_lammps_data(c, Path(d, "c.data"))
            back = read_lammps_data(Path(d, "c.data"))
        np.testing.assert_allclose(back.pos, c.pos, atol=1e-5)
        self.assertEqual(back.masses, {1: 28.0855, 2: 15.9994})

    def test_r_max_beyond_half_box_is_refused(self):
        c = diamond(5.431, (1, 1, 1))
        with self.assertRaises(ValueError):
            partial_rdf(c, 1, 1, r_max=4.0)


if __name__ == "__main__":
    unittest.main()
