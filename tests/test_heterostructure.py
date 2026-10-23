"""Heterostructure build: fcc on diamond in a shared lateral supercell (P1-6).

Pins the lateral match, the plane layout (no overlap at the interface, the
gap set from lateral registry), the deck syntax (LAMMPS continues lines with
"&", never a literal backslash), MEAM pair_coeff, and that the Si/Ge
single-lattice path is untouched.
"""

from __future__ import annotations

import math
import unittest

from openkapitza.md.nemd import (
    NEMDGeometry, NEMDSpec, build_input_deck, check_spec, hetero_layout, match_lateral,
)
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code


def _sial(**kw):
    g = NEMDGeometry(wall_thickness=1.0e-9, reservoir_thickness=1.5e-9,
                     bulk_a_length=8.0e-9, bulk_b_length=8.0e-9,
                     cross_section=3.2e-9, bin_width=0.4e-9)
    base = dict(geometry=g, element_a="Si", element_b="Al", lattice_a=5.431e-10,
                lattice_b=4.050e-10, temperature=300.0, lattice_style_a="diamond",
                lattice_style_b="fcc", mass_a=28.0855, mass_b=26.9815,
                pair_style="meam", pair_file="library.meam",
                extra_files=("AlSiMgCuFe.meam",),
                coeff_args="library.meam AlS SiS MgS CuS FeS AlSiMgCuFe.meam SiS AlS")
    base.update(kw)
    return NEMDSpec(**base)


class LateralMatch(unittest.TestCase):
    def test_si_al_ratio_is_three_to_four(self):
        na, nb = match_lateral(5.431, 4.050, 32.0)
        self.assertEqual(na * 4, nb * 3)

    def test_mismatch_within_one_percent(self):
        s = _sial()
        sa, sb = s.lattice_strain
        self.assertLess(max(abs(sa), abs(sb)), 0.005)
        self.assertAlmostEqual(sa, -sb, delta=1e-4)        # split evenly

    def test_explicit_supercell_is_honoured(self):
        self.assertEqual(_sial(supercell=(3, 4)).lateral_cells, (3, 4))


class Layout(unittest.TestCase):
    def test_closest_approach_is_the_contact_distance(self):
        lay = hetero_layout(_sial())
        self.assertAlmostEqual(lay["closest_ab"], lay["r0"], places=6)
        self.assertGreater(lay["closest_ab"], 2.3)

    def test_regions_do_not_overlap(self):
        lay = hetero_layout(_sial())
        self.assertLess(lay["top_a"], lay["cut_a"])
        self.assertLessEqual(lay["cut_a"], lay["start_b"] + 1e-9)
        self.assertLess(lay["start_b"], lay["first_b"])


class Deck(unittest.TestCase):
    def test_no_literal_backslash_reaches_lammps(self):
        for line in build_input_deck(_sial()).splitlines():
            self.assertFalse(line.rstrip().endswith("\\"), line)

    def test_meam_pair_coeff(self):
        self.assertIn("pair_coeff      * * library.meam AlS SiS MgS CuS FeS "
                      "AlSiMgCuFe.meam SiS AlS", build_input_deck(_sial()))

    def test_two_lattices_two_create_atoms(self):
        deck = build_input_deck(_sial())
        self.assertEqual(deck.count("lattice         custom"), 2)
        self.assertIn("create_atoms    2 region side_b", deck)

    def test_meam_skips_triple_check_and_says_so(self):
        f = FlagSet()
        check_spec(_sial(), f)
        self.assertIn(Code.NEMD_POTENTIAL_UNVERIFIED, f.codes())
        self.assertNotIn(Code.NEMD_POTENTIAL_INCOMPLETE, f.codes())

    def test_same_structure_keeps_single_lattice_path(self):
        g = NEMDGeometry()
        s = NEMDSpec(geometry=g, element_a="Si", element_b="Ge", lattice_a=5.431e-10,
                     lattice_b=5.658e-10, temperature=300.0)
        self.assertFalse(s.heterostructure)
        self.assertNotIn("lattice         custom", build_input_deck(s))


if __name__ == "__main__":
    unittest.main()
