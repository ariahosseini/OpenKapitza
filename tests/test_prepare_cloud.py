"""The rented-machine job set (scripts/prepare_cloud.py, KB 42)."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

from openkapitza.md.nemd import validate_deck

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))
from test_weekend import SIGE  # noqa: E402


def load():
    sp = importlib.util.spec_from_file_location("prepare_cloud", REPO / "scripts" /
                                                "prepare_cloud.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


class CloudSet(unittest.TestCase):
    def write(self, *extra):
        self._td = tempfile.TemporaryDirectory()
        d = Path(self._td.name)
        pot = d / "pot"
        pot.mkdir()
        (pot / "SiCGe.tersoff").write_text(SIGE)
        for f in ("library.meam", "AlSiMgCuFe.meam"):
            (pot / f).write_text("# stand-in\n")
        root = d / "cloud"
        with contextlib.redirect_stdout(io.StringIO()):
            load().main([str(root), "--potentials", str(pot), *extra])
        return root, json.loads((root / "manifest.json").read_text())

    def tearDown(self):
        if hasattr(self, "_td"):
            self._td.cleanup()

    def test_full_set_is_complete_valid_and_longest_first(self):
        root, man = self.write()
        names = [j["name"] for j in man["jobs"]]
        self.assertEqual(len(names), 14)
        for want in ("heavy/sial_pp", "heavy/sial_ttm", "heavy/sicu_pp",
                     "heavy/sige_incoherent", "heavy/r07_rerun", "followup/arb_hex_q2.4",
                     "followup/arb_ehex_q2.4", "followup/arb_langevin_dT27",
                     "followup/sige_damp1.0_s2", "disorder/mix4", "disorder/step8_p1"):
            self.assertIn(want, names)
        walls = [j["wall_hours"] for j in man["jobs"]]
        self.assertEqual(walls, sorted(walls, reverse=True))
        self.assertEqual((man["np"], man["slots"]), (48, 4))
        for n in names:
            deck = (root / n / "nemd.in").read_text()
            self.assertEqual(validate_deck(deck), [], n)
            self.assertTrue((root / n / "spec.json").is_file(), n)
        self.assertTrue((root / "heavy/sial_ttm/ttm_props.in").is_file())
        self.assertTrue((root / "heavy/sicu_pp/AlSiMgCuFe.meam").is_file())
        spec = json.loads((root / "followup/arb_hex_q2.4/spec.json").read_text())
        flat = json.dumps(spec)
        self.assertIn("2400000000", flat.replace(".0", ""))
        s2 = json.loads((root / "followup/sige_damp1.0_s2/spec.json").read_text())
        self.assertIn("54321", json.dumps(s2))
        r07 = (root / "heavy/r07_rerun/nemd.in").read_text()
        self.assertIn("run             400000", r07)

    def test_smoke_decks_are_short(self):
        root, man = self.write("--smoke", "--only", "heavy/sial_ttm", "followup/arb_ehex_q2.4",
                               "disorder/mix2")
        self.assertTrue(man["smoke"])
        self.assertEqual(len(man["jobs"]), 3)
        for j in man["jobs"]:
            deck = (root / j["name"] / "nemd.in").read_text()
            runs = [int(x) for x in re.findall(r"^run\s+(\d+)", deck, re.M)]
            self.assertTrue(runs and max(runs) <= 1000, (j["name"], runs))

    def test_schedule_fills_slots(self):
        m = load()
        e = [{"wall_hours": h} for h in (10, 8, 8, 6, 2, 1, 1)]
        self.assertEqual(m.schedule(e, 4), 10)
        self.assertEqual(e[4]["start_h"], 6)


if __name__ == "__main__":
    unittest.main()


class SmokeChecks(unittest.TestCase):
    def load(self):
        sp = importlib.util.spec_from_file_location("run_smoke", REPO / "scripts" /
                                                    "run_smoke.py")
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        return m

    def test_missing_and_broken_runs_are_reported(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            self.assertEqual(m.check(wd), ["no log.lammps"])
            (wd / "nemd.in").write_text("run 10\n")
            (wd / "log.lammps").write_text("ERROR: Pair style tersoff requires atom IDs\n")
            probs = m.check(wd)
            self.assertIn("did not reach OPENKAPITZA_DONE", probs)
            self.assertTrue(any(p.startswith("ERROR") for p in probs))
            self.assertIn("no thermo output", probs)

    def test_speed_reads_the_longest_loop(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "log.lammps").write_text(
                "Loop time of 2.0 on 4 procs for 100 steps with 1000 atoms\n"
                "Loop time of 10.0 on 4 procs for 400 steps with 1000 atoms\n")
            self.assertAlmostEqual(m.speed(Path(d))["rate"], 1000 * 400 / 40.0)
