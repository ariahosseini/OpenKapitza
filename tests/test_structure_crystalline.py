"""Crystalline interfaces in any orientation: matching and slabs (P2-1).

Known answers: centring translations of fcc/diamond, primitive surface cells,
hand-derivable matches, exact atom counts per bulk period, fourfold
coordination inside covalent slabs, and no overlaps at a joined interface.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from openkapitza.structure.analysis import pair_distances
from openkapitza.structure.crystalline import (
    LIBRARY, bicrystal_for_nemd, build_slab, join_bicrystal, match, rectangular_cells, slab_cell,
)

L = {k: f() for k, f in LIBRARY.items()}


class Lattices(unittest.TestCase):
    def test_centring(self):
        self.assertEqual(len(L["Si"].centring()), 4)       # fcc-based: 0 + three a/2<110>
        self.assertEqual(len(L["Al"].centring()), 4)
        self.assertEqual(len(L["GaN"].centring()), 1)      # hexagonal: none extra

    def test_primitive_surface_cells(self):
        a = 5.431
        c100 = rectangular_cells(L["Si"], (1, 0, 0), l_max=12)[0].lengths
        np.testing.assert_allclose(c100, (a / math.sqrt(2),) * 2, atol=1e-6)
        c111 = rectangular_cells(L["Si"], (1, 1, 1), l_max=12)[0].lengths
        np.testing.assert_allclose(sorted(c111), sorted((a / math.sqrt(2), a * math.sqrt(1.5))), atol=1e-6)
        cgan = rectangular_cells(L["GaN"], (0, 0, 1), l_max=12)[0].lengths
        np.testing.assert_allclose(sorted(cgan), sorted((3.189, 3.189 * math.sqrt(3))), atol=1e-6)


class Matching(unittest.TestCase):
    def test_si_ge_is_one_to_one_at_four_percent(self):
        m = match(L["Si"], (1, 0, 0), L["Ge"], (1, 0, 0), max_strain=0.05)
        self.assertAlmostEqual(m.max_strain, 1 - 5.431 / 5.658, places=6)   # B (Ge) compressed onto Si
        self.assertAlmostEqual(m.cell_a.lengths[0], 5.431 / math.sqrt(2), places=6)

    def test_si_al_beats_cube_on_cube(self):
        m = match(L["Si"], (1, 0, 0), L["Al"], (1, 0, 0), max_strain=0.01)
        self.assertLess(m.max_strain, 0.01)
        self.assertLess(m.cell_a.area, (3 * 5.431) ** 2 / 2)   # far smaller than 3:4 cubes

    def test_tolerance_is_honoured_and_failure_is_explicit(self):
        m = match(L["4H-SiC"], (0, 0, 1), L["GaN"], (0, 0, 1), max_strain=0.01)
        self.assertLessEqual(m.max_strain, 0.01)
        with self.assertRaises(ValueError):
            match(L["4H-SiC"], (0, 0, 1), L["GaN"], (0, 0, 1), max_strain=1e-5, l_max=12)

    def test_split_strain_is_opposite_on_the_two_sides(self):
        m = match(L["Si"], (1, 1, 1), L["Cu"], (1, 1, 1), max_strain=0.02)
        (ea, eb) = m.strains()
        self.assertLess(ea[0] * eb[0], 0)


class Slabs(unittest.TestCase):
    def _count(self, name, hkl, thickness):
        c = rectangular_cells(L[name], hkl)[0]
        s = build_slab(L[name], c, thickness)
        expect = len(L[name].basis) * c.area * thickness / L[name].volume
        return len(s.species), expect, s

    def test_exact_atom_counts_per_bulk_period(self):
        for name, hkl, t in (("Si", (1, 0, 0), 3 * 5.431), ("GaN", (0, 0, 1), 2 * 5.185),
                             ("4H-SiC", (0, 0, 1), 10.053), ("Al", (1, 1, 1), 3 * 4.0495 * math.sqrt(3))):
            n, expect, _ = self._count(name, hkl, t)
            self.assertAlmostEqual(n, expect, delta=1e-6, msg=name)

    def test_covalent_slab_interior_is_fourfold_and_overlap_free(self):
        for name, hkl, nn in (("Si", (1, 1, 1), 2.35), ("4H-SiC", (0, 0, 1), 1.89)):
            c = rectangular_cells(L[name], hkl)[0]
            s = build_slab(L[name], c, 25.0)
            # widen laterally so the neighbour search is valid
            reps = [np.array([i * c.lengths[0], j * c.lengths[1], 0.0]) for i in range(3) for j in range(3)]
            s.pos = np.vstack([s.pos + r for r in reps]); s.species = s.species * 9
            s.lengths = (3 * c.lengths[0], 3 * c.lengths[1])
            types = {sp: k + 1 for k, sp in enumerate(L[name].species)}
            cell = slab_cell(s, types, 60.0)
            inner = (cell.pos[:, 2] > 5) & (cell.pos[:, 2] < 18)
            allidx = np.arange(len(cell.types))
            i, _, r = pair_distances(cell, allidx[inner], allidx, nn * 1.15)
            counts = np.bincount(i, minlength=len(cell.types))[allidx[inner]]
            self.assertTrue(np.all(counts == 4), name)
            self.assertGreater(r.min(), 0.9 * nn)


class Bicrystal(unittest.TestCase):
    def test_si111_al111_join(self):
        b = join_bicrystal(L["Si"], (1, 1, 1), L["Al"], (1, 1, 1), length_a=20.0, length_b=20.0,
                           max_strain=0.01, repeat=(2, 1))
        self.assertEqual(b.types_a, (1,)); self.assertEqual(b.types_b, (2,))
        self.assertAlmostEqual(b.report["closest_ab"], b.report["r0"], places=6)
        c = b.cell
        ia = np.nonzero(c.types == 1)[0]; ib = np.nonzero(c.types == 2)[0]
        _, _, r = pair_distances(c, ia, ib, 3.0)
        self.assertGreater(r.min(), 0.99 * b.report["r0"])
        self.assertLess(c.pos[ia, 2].max(), b.interface_z)
        self.assertGreater(c.pos[ib, 2].min(), b.interface_z)

    def test_multispecies_sides_get_their_own_types(self):
        b = join_bicrystal(L["4H-SiC"], (0, 0, 1), L["GaN"], (0, 0, 1), length_a=15.0,
                           length_b=15.0, max_strain=0.01)
        self.assertEqual(b.types_a, (1, 2)); self.assertEqual(b.types_b, (3, 4))
        self.assertEqual(b.type_species, {1: "Si", 2: "C", 3: "Ga", 4: "N"})


class Registry(unittest.TestCase):
    def test_shift_translates_b_only(self):
        from openkapitza.structure.crystalline import join_bicrystal
        kw = dict(length_a=10.0, length_b=10.0, max_strain=0.01)
        b0 = join_bicrystal(L["Si"], (1, 0, 0), L["Al"], (1, 0, 0), **kw)
        b1 = join_bicrystal(L["Si"], (1, 0, 0), L["Al"], (1, 0, 0), shift=(1.0, 0.5), **kw)
        ia = b0.cell.types == 1
        np.testing.assert_allclose(b0.cell.pos[ia], b1.cell.pos[ia])
        W = b0.cell.hi[:2]
        d = (b1.cell.pos[~ia][:, :2] - b0.cell.pos[~ia][:, :2]) % W
        np.testing.assert_allclose(d, np.tile([1.0, 0.5], (d.shape[0], 1)), atol=1e-9)

    def test_scan_picks_the_minimum_of_any_scorer(self):
        from openkapitza.structure.crystalline import scan_registry
        target = (4.294, 2.147)
        def fake(bc):            # energy minimum at a known shift
            s = bc.report["shift"]; return (s[0] - target[0]) ** 2 + (s[1] - target[1]) ** 2
        r = scan_registry(L["Si"], (1, 0, 0), L["Al"], (1, 0, 0), n=4, max_strain=0.01, score=fake)
        self.assertEqual(len(r.points), 16)
        np.testing.assert_allclose(r.best.shift, target, atol=1e-2)

    def test_registry_averages_out_on_a_large_matched_cell(self):
        from openkapitza.structure.crystalline import scan_registry
        big = scan_registry(L["Si"], (1, 1, 1), L["Al"], (1, 1, 1), n=4, max_strain=0.01)
        small = scan_registry(L["Si"], (1, 0, 0), L["Al"], (1, 0, 0), n=4, max_strain=0.01)
        self.assertLess(big.spread, 1e-3)
        self.assertGreater(small.spread, 0.1)


class NEMDLayout(unittest.TestCase):
    def test_sides_sit_where_the_geometry_puts_them(self):
        from openkapitza.md.nemd import NEMDGeometry
        g = NEMDGeometry(wall_thickness=0.8e-9, reservoir_thickness=1.0e-9,
                         bulk_a_length=2e-9, bulk_b_length=2e-9, bin_width=0.4e-9)
        b = bicrystal_for_nemd(L["Si"], (1, 1, 1), L["Cu"], (1, 1, 1), g,
                               max_strain=0.02, min_width=20.0)
        iz, total = g.interface_z * 1e10, g.total_length * 1e10
        za = b.cell.pos[b.cell.types == 1, 2]; zb = b.cell.pos[b.cell.types == 2, 2]
        self.assertGreaterEqual(za.min(), 0.0); self.assertLess(za.max(), iz)
        self.assertGreater(zb.min(), iz - 1.0)            # within half a gap of the plane
        self.assertLess(zb.max(), total + 1.0)
        self.assertLess(abs(b.report["interface_offset"]), 1.5)
        self.assertTrue(all(w >= 20.0 for w in b.report["width"]))
        self.assertGreater(b.cell.hi[2], zb.max())     # headroom for the piston


class BuildScript(unittest.TestCase):
    def test_couples_the_barostat_only_on_isotropic_planes(self):
        import importlib.util, sys
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "scripts" / "build_bicrystal.py"
        spec = importlib.util.spec_from_file_location("build_bicrystal", path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.assertTrue(mod._isotropic(L["Si"], (1, 0, 0)))
        self.assertTrue(mod._isotropic(L["Si"], (1, 1, 1)))
        self.assertFalse(mod._isotropic(L["Si"], (1, 1, 0)))
        self.assertTrue(mod._isotropic(L["GaN"], (0, 0, 1)))
        self.assertFalse(mod._isotropic(L["GaN"], (1, 0, 0)))


if __name__ == "__main__":
    unittest.main()
