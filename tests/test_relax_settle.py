"""Piston relaxation: scaled with column length, applied as a ramp, checked (KB 16)."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plan_campaign import PRESETS, base_spec  # noqa: E402

from openkapitza.md.campaign import _steps  # noqa: E402
from openkapitza.md.measure import check_relax_settle  # noqa: E402
from openkapitza.md.nemd import (  # noqa: E402
    NEMDGeometry, build_input_deck, parse_relax_settle, relax_steps, validate_deck)
from openkapitza.schema import FlagSet  # noqa: E402
from openkapitza.schema.flags import Code  # noqa: E402


def sige(length_nm: float = 16.0, **kw):
    s = base_spec(PRESETS["sige"])
    g = s.geometry
    s.geometry = NEMDGeometry(wall_thickness=g.wall_thickness,
                              reservoir_thickness=g.reservoir_thickness,
                              bulk_a_length=length_nm * 1e-9, bulk_b_length=length_nm * 1e-9,
                              cross_section=g.cross_section, bin_width=g.bin_width)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class TestScaling(unittest.TestCase):
    def test_relaxation_scales_up_with_the_column_never_down(self):
        self.assertEqual(relax_steps(sige(8)), 50_000)
        self.assertEqual(relax_steps(sige(16)), 50_000)       # the measured reference
        self.assertEqual(relax_steps(sige(32)), 100_000)
        self.assertEqual(relax_steps(sige(64)), 200_000)
        self.assertEqual(relax_steps(sige(64, relax_scale_with_length=False)), 50_000)
        self.assertEqual(relax_steps(sige(64, relax_z=False)), 50_000)
        self.assertEqual(relax_steps(sige(64, relax_structure=False)), 0)

    def test_cost_counts_the_scaled_relaxation(self):
        s = sige(64)
        self.assertEqual(_steps(s), 200_000 + s.total_steps)


class TestDeck(unittest.TestCase):
    def test_wall_is_set_by_a_ramp_over_the_column_and_reported(self):
        d = build_input_deck(sige(64))
        self.assertIn("run             200000", d)
        self.assertIn("displace_atoms  gmobile ramp z 0.0 ${wfix_dz} z ${wfix_lo} ${wfix_hi}", d)
        self.assertIn('print           "OPENKAPITZA_WALLFIX dz ${wfix_dz} pzz_mean_bar', d)
        self.assertIn("fix             pzzav all ave/time", d)
        # the ramp is computed before anything moves: constants, then displacements
        self.assertLess(d.index("variable        wfix_hi"), d.index("displace_atoms  gmobile"))
        self.assertLess(d.index("displace_atoms  gmobile"), d.index("displace_atoms  gwall_hi"))
        self.assertEqual(validate_deck(d), [])

    def test_strain_mode_reaverages_pzz_over_the_load_stage(self):
        d = build_input_deck(sige(16, spot_pressure_GPa=2.0, spot_pressure_mode="strain"))
        load = d[d.index("uniaxial-strain load"):d.index("OPENKAPITZA_LOADED")]
        self.assertIn("unfix           pzzav", load)
        self.assertIn("fix             pzzav all ave/time", load)
        self.assertEqual(validate_deck(d), [])


OLD_LOG = """\
displace_atoms  gwall_hi move 0.0 0.0 $(f_zav-v_zwall) units box
displace_atoms  gwall_hi move 0.0 0.0 {dz} units box
Displacing atoms ...
"""
NEW_LOG = """\
print           "OPENKAPITZA_WALLFIX dz ${{wfix_dz}} pzz_mean_bar $(f_pzzav:%.2f)"
OPENKAPITZA_WALLFIX dz {dz} pzz_mean_bar {pzz}
displace_atoms  gwall_hi move 0.0 0.0 {dz} units box
"""


class TestGuard(unittest.TestCase):
    def check(self, text, spec=None):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log.lammps"
            p.write_text(text)
            f = FlagSet()
            r = check_relax_settle(p, spec or sige(16), f)
        return r, [x.code for x in f], [x.message for x in f]

    def test_old_logs_are_read_from_the_echoed_displacement(self):
        r, codes, _ = self.check(OLD_LOG.format(dz=0.7718))     # r06 of the M3 Pro campaign
        self.assertAlmostEqual(r["dz"], 0.7718)
        self.assertEqual(codes, [Code.NEMD_RELAX_UNSETTLED])
        r, codes, _ = self.check(OLD_LOG.format(dz=-0.148))     # r00: fine
        self.assertEqual(codes, [])

    def test_new_logs_judge_the_mean_pzz(self):
        _, codes, msgs = self.check(NEW_LOG.format(dz=-0.03, pzz=18824.9))
        self.assertEqual(codes, [Code.NEMD_RELAX_UNSETTLED])
        self.assertIn("+1.882 GPa", msgs[0])
        _, codes, _ = self.check(NEW_LOG.format(dz=-0.03, pzz=120.0))
        self.assertEqual(codes, [])

    def test_pzz_is_judged_about_the_applied_load(self):
        loaded = sige(16, spot_pressure_GPa=2.0, spot_pressure_mode="stress")
        _, codes, _ = self.check(NEW_LOG.format(dz=0.1, pzz=20100.0), loaded)
        self.assertEqual(codes, [])
        _, codes, msgs = self.check(NEW_LOG.format(dz=0.1, pzz=5000.0), loaded)
        self.assertEqual(codes, [Code.NEMD_RELAX_UNSETTLED])
        self.assertIn("against the applied 2", msgs[0])

    def test_no_piston_no_check(self):
        r, codes, _ = self.check(OLD_LOG.format(dz=5.0), sige(16, relax_z=False))
        self.assertIsNone(r)
        self.assertEqual(codes, [])

    def test_parser_prefers_the_new_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log.lammps"
            p.write_text(NEW_LOG.format(dz=0.25, pzz=-300.0))
            self.assertEqual(parse_relax_settle(p), {"dz": 0.25, "pzz_mean_bar": -300.0})
            self.assertIsNone(parse_relax_settle(Path(d) / "missing.lammps"))


if __name__ == "__main__":
    unittest.main()
