"""The local benchmark: log parsing, rates, the calibrated estimate, job sizing."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from openkapitza.compute.resources import load_calibration, local_estimate

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_local.py"
spec = importlib.util.spec_from_file_location("benchmark_local", SCRIPT)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

LOG = """\
minimize 1e-8 1e-10 1000 10000
Loop time of 14.85 on 5 procs for 142 steps with 35840 atoms

98.2% CPU use with 5 MPI tasks x no OpenMP threads

Minimization stats:
  Stopping criterion = energy tolerance
run 200
Loop time of 2.2 on 5 procs for 200 steps with 35840 atoms
run 1000
Loop time of 10.0 on 5 procs for 1000 steps with 35840 atoms
run 0
Loop time of 0.0001 on 5 procs for 0 steps with 35840 atoms
run 2000
Loop time of 20.0 on 5 procs for 2000 steps with 35840 atoms
"""

CAL = {"processor": "Apple M3 Pro", "date": "2026-10-28",
       "styles": {"tersoff": {"best_np": 5, "atom_steps_per_s": 3.584e6, "per_core": 7.168e5},
                  "meam": {"best_np": 5, "atom_steps_per_s": 5.0e5, "per_core": 1.0e5}}}


class TestParsing(unittest.TestCase):
    def test_minimisation_and_empty_runs_are_dropped(self):
        loops = B.md_loops(LOG)
        self.assertEqual([l["steps"] for l in loops], [200, 1000, 2000])
        self.assertTrue(all(l["procs"] == 5 for l in loops))

    def test_rate_drops_the_warm_up_segment(self):
        r = B.rate(B.md_loops(LOG))
        self.assertEqual(r["steps"], 3000)                     # 1000 + 2000, not the 200
        self.assertAlmostEqual(r["atom_steps_per_s"], 35840 * 3000 / 30.0)
        self.assertAlmostEqual(r["per_core"], r["atom_steps_per_s"] / 5)

    def test_no_md_is_an_error(self):
        with self.assertRaises(ValueError):
            B.rate(B.md_loops("Loop time of 1 on 1 procs for 10 steps with 5 atoms\n"
                              "Minimization stats:\n"))


class TestEstimate(unittest.TestCase):
    def test_hours_from_the_measured_total_rate(self):
        e = local_estimate(35840, 1_000_000, "tersoff", CAL)
        self.assertAlmostEqual(e["hours"], 35840 * 1e6 / 3.584e6 / 3600)
        self.assertEqual((e["np"], e["machine"]), (5, "Apple M3 Pro"))
        self.assertIsNotNone(local_estimate(1000, 1000, "meam/c", CAL))   # style variant

    def test_unmeasured_is_none(self):
        self.assertIsNone(local_estimate(1000, 1000, "sw", CAL))
        self.assertIsNone(local_estimate(1000, 1000, "tersoff", {}))

    def test_calibration_file_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.json"
            p.write_text(json.dumps(CAL))
            self.assertEqual(load_calibration(p)["processor"], "Apple M3 Pro")
            self.assertIsNone(load_calibration(Path(d) / "missing.json"))


class TestJobs(unittest.TestCase):
    def test_every_prepared_job_is_sized(self):
        with tempfile.TemporaryDirectory() as d:
            pot = Path(d)
            for f in ("SiCGe.tersoff", "library.meam", "AlSiMgCuFe.meam", "SiO.tersoff"):
                (pot / f).write_text("# placeholder\n")
            jobs = B.prepared_jobs(pot)
        names = {j["name"] for j in jobs}
        self.assertIn("sial_ttm", names)
        self.assertTrue(any(n.startswith("campaign/") for n in names))
        styles = {j["name"]: j["style"] for j in jobs}
        self.assertEqual(styles["sial_pp"], "meam")
        self.assertEqual(styles["sio2_weak"], "tersoff")
        self.assertTrue(next(j for j in jobs if j["name"] == "sial_ttm")["ttm"])
        for j in jobs:
            self.assertIsNotNone(local_estimate(j["atoms"], j["steps"], j["style"], CAL), j)

    def test_decks_are_the_project_decks_cut_short(self):
        with tempfile.TemporaryDirectory() as d:
            pot = Path(d) / "pot"
            pot.mkdir()
            for f in ("SiCGe.tersoff", "library.meam", "AlSiMgCuFe.meam"):
                (pot / f).write_text("# placeholder\n")
            built = B.decks(pot, Path(d) / "w")
            t = (built["tersoff"] / "nemd.in").read_text()
            m = (built["meam"] / "nemd.in").read_text()
        self.assertIn("pair_style      tersoff", t)
        self.assertIn("pair_style      meam", m)
        self.assertIn("run             2000", t)          # production cut to 2,000 steps


class TestJobLoopNote(unittest.TestCase):
    def test_local_submission_states_the_measured_time(self):
        from unittest import mock
        from openkapitza.agent import jobs as J
        from openkapitza.compute import resources as R
        fix = Path(__file__).resolve().parent / "data" / "md_runs" / "sial_plumbing"
        e = J.JobEntry(name="sial_pp", workdir=str(fix), atoms=6680, steps=30_000)
        with tempfile.TemporaryDirectory() as d:
            cal = Path(d) / "t.json"
            cal.write_text(json.dumps(CAL))
            with mock.patch.object(R, "DEFAULT_CALIBRATION", cal):
                note = J._local_note(e, "mpirun -np 4")
            self.assertIn("h here at np=5", note)
            self.assertIn("running np=4; fastest at np=5", note)
            with mock.patch.object(R, "DEFAULT_CALIBRATION", Path(d) / "none.json"):
                self.assertIn("benchmark_local.py", J._local_note(e, ""))


if __name__ == "__main__":
    unittest.main()
