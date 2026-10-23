"""Si(100) / a-SiO2 join (Phase 2): cut placement, layout, chemistry descriptors.

Uses beta-cristobalite as a stand-in "glass" (it has the right types and
bonding), so these run without LAMMPS.
"""

from __future__ import annotations

import unittest

import numpy as np

from openkapitza.md.nemd import NEMDGeometry
from openkapitza.schema import FlagSet
from openkapitza.structure.analysis import Cell, pair_distances
from openkapitza.structure.crystals import beta_cristobalite
from openkapitza.structure.interface import (
    O_T, SI_C, SI_OX, InterfaceSpec, best_cut, bonds_crossing, interface_chemistry, join,
)


def _glass(width, cells_z=3):
    c = beta_cristobalite((4, 4, cells_z))
    scale = np.array([width / c.lengths[0], width / c.lengths[1], 1.0])
    return Cell(c.pos * scale, c.types, np.zeros(3), c.lengths * scale, dict(c.masses))


class Cut(unittest.TestCase):
    def test_crossing_count_matches_geometry(self):
        # Ideal cristobalite: Si planes at k*a/4, O planes midway. A cut between
        # an Si plane and the next O plane crosses each of that Si plane's
        # upward bonds: 32 Si per plane (4x4 cells) x 2 = 64.
        g = _glass(27.155)
        a = g.lengths[2] / 3
        self.assertEqual(bonds_crossing(g, a / 16), 64)
        self.assertEqual(bonds_crossing(g, a / 4 + a / 16), 64)

    def test_best_cut_is_a_minimum(self):
        g = _glass(27.155)
        z, n = best_cut(g, step=0.2)
        for dz in (0.4, 0.8, 1.2):
            self.assertLessEqual(n, bonds_crossing(g, (z + dz) % g.lengths[2]))


class Join(unittest.TestCase):
    def setUp(self):
        self.geom = NEMDGeometry(wall_thickness=0.8e-9, reservoir_thickness=1.0e-9,
                                 bulk_a_length=3.0e-9, bulk_b_length=3.0e-9,
                                 cross_section=2.7155e-9, bin_width=0.4e-9)
        self.spec = InterfaceSpec(self.geom, anneal=False)

    def test_layout_types_and_gap(self):
        cell, rep = join(_glass(self.spec.width, 7), self.spec, FlagSet())
        si_c = cell.pos[cell.types == SI_C]
        ox = cell.pos[cell.types != SI_C]
        self.assertLess(si_c[:, 2].max(), self.geom.interface_z * 1e10)
        self.assertAlmostEqual(ox[:, 2].min() - si_c[:, 2].max(), self.spec.gap, places=6)
        self.assertLess(ox[:, 2].max(), self.geom.total_length * 1e10)
        self.assertEqual(set(np.unique(cell.types)), {SI_C, SI_OX, O_T})
        self.assertEqual(rep.glass_repeats, 1)

    def test_lateral_mismatch_refused(self):
        with self.assertRaises(ValueError):
            join(_glass(28.0), self.spec, FlagSet())

    def test_stacking_is_flagged(self):
        f = FlagSet()
        _, rep = join(_glass(self.spec.width, 2), self.spec, f)
        self.assertGreater(rep.glass_repeats, 1)
        self.assertTrue(any("stacked" in fl.message for fl in f))

    def test_chemistry_counts_cross_interface_bonds(self):
        cell, _ = join(_glass(self.spec.width, 7), self.spec, FlagSet())
        ch = interface_chemistry(cell)
        # Gap 1.6 A < Si-O cutoff 2.0 A, so some crystal Si see oxide O.
        self.assertGreaterEqual(ch.si_o_bonds_per_nm2, 0.0)
        self.assertAlmostEqual(sum(ch.suboxide.values()), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
