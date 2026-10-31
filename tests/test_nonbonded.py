"""Non-bonded and gas-layer interface cells (structure/nonbonded.py)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from openkapitza.md.campaign import read_spec
from openkapitza.md.nemd import NEMDGeometry, validate_deck
from openkapitza.structure.nonbonded import (
    UFF, NonbondedSpec, build_data, lj_pairs, make_spec, pair_block, write_job,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def base():
    from plan_campaign import PRESETS, base_spec
    s = base_spec(PRESETS["sige"])
    g = s.geometry
    s.geometry = NEMDGeometry(wall_thickness=g.wall_thickness,
                              reservoir_thickness=g.reservoir_thickness,
                              bulk_a_length=6e-9, bulk_b_length=6e-9, cross_section=2.2e-9,
                              bin_width=0.3e-9)
    s.pair_elements = ("Si(D)", "Ge")
    return s


def atoms(text: str) -> np.ndarray:
    lines = text.splitlines()
    i = lines.index("Atoms # atomic") + 2
    return np.array([[float(x) for x in ln.split()[1:5]] for ln in lines[i:] if ln.strip()])


class Pairs(unittest.TestCase):
    def test_uff_values_and_mixing(self):
        self.assertAlmostEqual(UFF["Si"][0], 3.826, places=3)
        self.assertAlmostEqual(UFF["Si"][1], 0.01743, places=4)
        p = lj_pairs("Si", "Ge", NonbondedSpec(lj_scale=4))
        self.assertAlmostEqual(p[(1, 2)][0], 4 * (UFF["Si"][1] * UFF["Ge"][1]) ** 0.5)
        self.assertEqual(set(lj_pairs("Si", "Ge", NonbondedSpec(gas="Ar"))),
                         {(1, 2), (1, 3), (2, 3), (3, 3)})

    def test_one_tersoff_per_side_so_no_cross_bond(self):
        b = pair_block(base(), NonbondedSpec())
        self.assertIn("hybrid tersoff tersoff lj/cut", b)
        self.assertIn("tersoff 1 SiCGe.tersoff Si(D) NULL\n", b)
        self.assertIn("tersoff 2 SiCGe.tersoff NULL Ge\n", b)
        g = pair_block(base(), NonbondedSpec(gas="Ar"))
        self.assertIn("Si(D) NULL NULL", g)
        self.assertIn("pair_coeff      3 3 lj/cut", g)


class Cell(unittest.TestCase):
    def test_gap_opened_between_the_facing_planes(self):
        spec = make_spec(base(), NonbondedSpec(plane_gap_ang=3.4))
        text, info = build_data(spec, NonbondedSpec(plane_gap_ang=3.4))
        a = atoms(text)
        top_a = a[a[:, 0] == 1, 3].max()
        bot_b = a[a[:, 0] == 2, 3].min()
        self.assertAlmostEqual(bot_b - top_a, 3.4, places=4)
        self.assertEqual(info["gas_atoms"], 0)
        self.assertLess(a[:, 3].max(), info["box_z_ang"])

    def test_gas_layer_sits_in_the_gap(self):
        nb = NonbondedSpec(gas="Ar")
        text, info = build_data(make_spec(base(), nb), nb)
        a = atoms(text)
        gas = a[a[:, 0] == 3]
        self.assertEqual(len(gas), info["gas_atoms"])
        self.assertGreater(len(gas), 20)
        self.assertTrue(np.all(gas[:, 3] > a[a[:, 0] == 1, 3].max()))
        self.assertTrue(np.all(gas[:, 3] < a[a[:, 0] == 2, 3].min()))
        half = NonbondedSpec(gas="Ar", coverage=0.5)
        _, info2 = build_data(make_spec(base(), half), half)
        self.assertLess(info2["gas_atoms"], info["gas_atoms"])

    def test_job_folder_is_complete_and_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            nb = NonbondedSpec(gas="Ar")
            b = base()
            b.spot_pressure_GPa = 1.0
            write_job(d, b, nb)
            deck = (Path(d) / "nemd.in").read_text()
            self.assertEqual(validate_deck(deck), [])
            self.assertIn("read_data       interface.data", deck)
            self.assertIn("pair_style      hybrid", deck)
            spec, kind, extra = read_spec(d)
            self.assertEqual(spec.pair_block, make_spec(b, nb).pair_block)
            self.assertEqual(extra["nonbonded"]["label"], "Ar1")


class ImposedFlux(unittest.TestCase):
    def test_heat_mode_deck(self):
        from openkapitza.md.nemd import EV_PER_PS_IN_W, build_input_deck, heat_power_ev_ps
        s = base()
        s.flux_mode, s.imposed_flux, s.presplit_K = "heat", 1e9, 20.0
        deck = build_input_deck(s)
        self.assertEqual(validate_deck(deck), [])
        f = heat_power_ev_ps(s, s.snapped_area)
        self.assertAlmostEqual(f * EV_PER_PS_IN_W / s.snapped_area, 1e9)
        self.assertEqual(deck.count(f"heat 1 {f:.8g}"), 2)          # steady and production
        self.assertNotIn("langevin", deck)
        self.assertIn(f"variable        ehot  equal {f:.8g}*step*0.001", deck)
        self.assertIn("nvt temp 310.00 310.00", deck)
        self.assertIn("nvt temp 290.00 290.00", deck)
        self.assertIn("unfix           equil_b", deck)
        s.imposed_flux = None
        with self.assertRaises(ValueError):
            build_input_deck(s)

    def test_resume_keeps_the_imposed_energy_formula(self):
        from openkapitza.md.checkpoint import build_resume_deck
        from openkapitza.md.nemd import build_input_deck
        s = base()
        s.flux_mode, s.imposed_flux = "heat", 1e9
        r = build_resume_deck(build_input_deck(s), 4000, s.production_steps, 0.001, 9.9, -9.9)
        self.assertNotIn("+9.9", r)
        self.assertIn("*step*0.001", r)

    def test_spec_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            s = base()
            s.flux_mode, s.imposed_flux, s.presplit_K = "heat", 2e9, 15.0
            write_job(d, s, NonbondedSpec())
            r, _, _ = read_spec(d)
            self.assertEqual((r.flux_mode, r.imposed_flux, r.presplit_K), ("heat", 2e9, 15.0))


class SweepScript(unittest.TestCase):
    def test_writes_every_case_and_registers_them(self):
        import importlib.util
        import json
        p = REPO / "scripts" / "prepare_nonbonded.py"
        spec = importlib.util.spec_from_file_location("prepare_nonbonded", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as d:
            pot = Path(d) / "pot"
            pot.mkdir()
            (pot / "SiCGe.tersoff").write_text("# placeholder\n")
            root = Path(d) / "jobs"
            root.mkdir()
            (root / "manifest.json").write_text(json.dumps({"jobs": [{"name": "campaign/r00"}]}))
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                m.main([str(root), "--potentials", str(pot), "--only", "bonded_ref", "Ar_c1_p1"])
            names = [j["name"] for j in json.loads((root / "manifest.json").read_text())["jobs"]]
            self.assertEqual(names, ["campaign/r00", "nonbonded/bonded_ref", "nonbonded/Ar_c1_p1"])
            self.assertTrue((root / "nonbonded" / "Ar_c1_p1" / "interface.data").is_file())
            self.assertEqual(len(m.cases()), 19)


if __name__ == "__main__":
    unittest.main()
