"""a-SiO2 melt-quench builder and its validation gate (Phase 2)."""

from __future__ import annotations

import unittest

import numpy as np

from openkapitza.md.nemd import NEMDGeometry, NEMDSpec, build_input_deck
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code
from openkapitza.structure.amorphous import (
    GLASS_DENSITY, QuenchSpec, analyse_silica, build_quench_deck, judge, start_cell,
)


class StartCell(unittest.TestCase):
    def test_cubic_start_is_at_glass_density(self):
        self.assertAlmostEqual(start_cell(QuenchSpec()).density(), GLASS_DENSITY, places=6)

    def test_fixed_lateral_keeps_xy_and_sets_z_for_density(self):
        c = start_cell(QuenchSpec(cells=(4, 4, 8), box=(27.1605, 27.1605, 60.0),
                                  fix_lateral=True))
        np.testing.assert_allclose(c.lengths[:2], 27.1605)
        self.assertAlmostEqual(c.density(), GLASS_DENSITY, places=6)


class Deck(unittest.TestCase):
    def test_rate_sets_quench_length(self):
        s = QuenchSpec(t_melt=6000, t_final=300, quench_rate=1e13, n_segments=10)
        self.assertAlmostEqual(s.quench_ps, 570.0)
        deck = build_quench_deck(s)
        self.assertEqual(deck.count(f"run             {s.steps(57.0)}"), 10)
        self.assertIn("Quench rate 1e+13 K/s", deck)

    def test_segments_tile_the_ramp(self):
        s = QuenchSpec(t_melt=6000, t_final=300, n_segments=4)
        deck = build_quench_deck(s)
        for t0, t1 in ((6000, 4575), (4575, 3150), (3150, 1725), (1725, 300)):
            self.assertIn(f"nvt temp {t0:.2f} {t1:.2f}", deck)
        self.assertEqual(deck.count("write_restart"), 5)        # melt + 4 segments

    def test_resume_continues_the_schedule(self):
        s = QuenchSpec(t_melt=6000, t_final=300, n_segments=4)
        deck = build_quench_deck(s, resume_from=2)
        self.assertIn("read_restart    seg02.restart", deck)
        self.assertNotIn("read_data", deck)
        self.assertNotIn("velocity", deck)                     # keep the restart's velocities
        self.assertNotIn("nvt temp 4575.00 3150.00", deck)     # segment 2 already done
        self.assertIn("nvt temp 3150.00 1725.00", deck)
        self.assertIn("pair_coeff", deck)                      # not stored in a restart

    def test_progress_marker_needs_its_restart_file(self):
        import tempfile
        from pathlib import Path
        from openkapitza.structure.amorphous import last_completed_segment
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(last_completed_segment(Path(d)))
            Path(d, "progress.txt").write_text("3\n")
            self.assertIsNone(last_completed_segment(Path(d)))   # restart missing
            Path(d, "seg03.restart").write_bytes(b"x")
            self.assertEqual(last_completed_segment(Path(d)), 3)

    def test_fixed_lateral_barostats_z_only(self):
        self.assertIn("z 0.0 0.0 0.5", build_quench_deck(QuenchSpec(fix_lateral=True)))
        self.assertIn("iso 0.0 0.0 0.5", build_quench_deck(QuenchSpec()))

    def test_box_set_to_mean_before_barostat_off(self):
        deck = build_quench_deck(QuenchSpec())
        self.assertLess(deck.index("fix             bav"), deck.index("change_box"))
        self.assertLess(deck.index("change_box"), deck.index("write_data"))


class Gate(unittest.TestCase):
    def test_crystal_is_not_a_glass(self):
        """Ideal cristobalite has Si-O-Si = 180 deg: the gate must reject it."""
        f = FlagSet()
        r = analyse_silica(start_cell(QuenchSpec()))
        judge(r, QuenchSpec(), f)
        self.assertFalse(r.passes)
        self.assertFalse(r.metrics["angle_SiOSi"].passes)
        self.assertIn(Code.STRUCT_VALIDATION_FAILED, f.codes())

    def test_quench_rate_is_always_stated(self):
        f = FlagSet()
        judge(analyse_silica(start_cell(QuenchSpec())), QuenchSpec(quench_rate=1e12), f)
        self.assertIn(Code.STRUCT_QUENCH_RATE, f.codes())
        self.assertTrue(any("12 orders" in fl.message for fl in f))


class NEMDReadData(unittest.TestCase):
    def test_three_type_interface_deck(self):
        g = NEMDGeometry(wall_thickness=0.8e-9, reservoir_thickness=1.0e-9,
                         bulk_a_length=4.0e-9, bulk_b_length=4.0e-9,
                         cross_section=2.71605e-9, bin_width=0.4e-9)
        s = NEMDSpec(geometry=g, element_a="Si", element_b="SiO2", lattice_a=5.4321e-10,
                     lattice_b=5.4321e-10, temperature=300.0, structure_mode="read_data",
                     data_file="interface.data", pair_style="tersoff",
                     pair_file="SiO.tersoff", coeff_args="SiO.tersoff Si Si O")
        deck = build_input_deck(s)
        self.assertIn("read_data       interface.data", deck)
        self.assertIn("pair_coeff      * * SiO.tersoff Si Si O", deck)
        self.assertIn("aveforce", deck)              # piston relaxation runs on read-in cells
        self.assertNotIn("create_atoms", deck)


if __name__ == "__main__":
    unittest.main()
