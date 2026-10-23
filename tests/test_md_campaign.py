"""Convergence campaign: plan, write, collect (open issue P1-3).

Pins: one-factor-at-a-time ladders share the base cell (run once); every rung
directory is complete and submittable; the manifest round-trips the spec;
and collection measures each rung and feeds errors into the axis verdicts.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from openkapitza.md.campaign import (
    collect_campaign, plan_campaign, spec_from_record, write_campaign,
)
from openkapitza.md.nemd import build_input_deck

from tests.test_md_measure import _spec, _write_run


class Planning(unittest.TestCase):
    def test_base_is_shared_across_ladders(self):
        base = _spec()
        rungs = plan_campaign(base, cross_sections_nm=(3.0, 7.0), lengths_nm=(10, 40),
                              deltas_K=(20, 60))
        # base + 2 + 2 + 2 distinct cells
        self.assertEqual(len(rungs), 7)
        base_rung = [r for r in rungs if len(r.axes) == 3]
        self.assertEqual(len(base_rung), 1)
        self.assertEqual({a for a, _ in base_rung[0].axes},
                         {"cross_section", "length", "flux"})

    def test_values_land_on_the_right_axis(self):
        rungs = plan_campaign(_spec(), lengths_nm=(10, 40))
        lengths = sorted(v for r in rungs for a, v in r.axes if a == "length")
        self.assertEqual(lengths, [10.0, 20.0, 40.0])


class Writing(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.pot = self.root / "Si.sw"
        self.pot.write_text("dummy potential\n")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_every_rung_is_complete_and_costed(self):
        rungs = plan_campaign(_spec(), lengths_nm=(10, 40))
        m = write_campaign(self.root / "camp", rungs, self.pot, cores=32,
                           partition="normal", account="proj")
        self.assertEqual(len(m["rungs"]), 3)
        for e in m["rungs"]:
            wd = self.root / "camp" / e["name"]
            for f in ("nemd.in", "Si.sw", "submit.sbatch", "run.sh"):
                self.assertTrue((wd / f).is_file(), f)
            sb = (wd / "submit.sbatch").read_text()
            self.assertIn("#SBATCH --partition=normal", sb)
            self.assertIn("srun", sb)
            self.assertGreater(e["node_hours"], 0)
        self.assertAlmostEqual(m["total_node_hours"],
                               sum(e["node_hours"] for e in m["rungs"]), places=1)

    def test_manifest_spec_round_trips_to_the_same_deck(self):
        rungs = plan_campaign(_spec(), lengths_nm=(10,))
        m = write_campaign(self.root / "camp", rungs, self.pot)
        for e, r in zip(m["rungs"], rungs):
            self.assertEqual(build_input_deck(spec_from_record(e["spec"])),
                             build_input_deck(r.spec))


class Collecting(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        pot = self.root / "Si.sw"
        pot.write_text("x\n")
        self.rungs = plan_campaign(_spec(), cross_sections_nm=(3.0, 7.0))
        self.manifest = write_campaign(self.root / "camp", self.rungs, pot)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_flat_series_converges_and_missing_rungs_are_reported(self):
        names = [e["name"] for e in self.manifest["rungs"]]
        for r in self.rungs[:-1]:
            _write_run(self.root / "camp" / r.name, r.spec, dT=5.0, bin_noise=0.001)
        res = collect_campaign(self.root / "camp")
        self.assertIn(names[-1], res.failed)
        self.assertEqual(len(res.measurements), len(names) - 1)

        _write_run(self.root / "camp" / self.rungs[-1].name, self.rungs[-1].spec,
                   dT=5.0, bin_noise=0.001)
        res = collect_campaign(self.root / "camp")
        v = res.verdicts["cross_section"]
        self.assertTrue(v.converged, v)
        self.assertIsNotNone(v.detail["h_K_err"])


if __name__ == "__main__":
    unittest.main()
