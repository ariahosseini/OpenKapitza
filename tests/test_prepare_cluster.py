"""The cluster dry run writes every job, validated and costed, offline."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prepare_cluster.py"


class DryRun(unittest.TestCase):
    def test_writes_every_job_with_a_manifest(self):
        spec = importlib.util.spec_from_file_location("prepare_cluster", SCRIPT)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            pot = Path(d) / "pot"; pot.mkdir()
            for f in ("SiCGe.tersoff", "library.meam", "AlSiMgCuFe.meam", "SiO.tersoff"):
                (pot / f).write_text("# placeholder for the dry-run test\n")
            root = Path(d) / "out"
            self.assertEqual(mod.main([str(root), "--potentials", str(pot)]), 0)
            m = json.loads((root / "manifest.json").read_text())
            names = {e["name"] for e in m["jobs"]}
            for must in ("sige_emd", "sial_pp", "sial_ttm", "sicu_pp", "sio2_strong",
                         "sio2_weak", "sige_spot_strain"):
                self.assertIn(must, names)
            self.assertTrue(any(n.startswith("campaign/") and n.endswith("_r1") for n in names))
            self.assertTrue((root / "sial_ttm" / "ttm_props.in").exists())
            self.assertTrue((root / "sio2_weak" / "interface.data").exists())
            self.assertIn("1e12", (root / "commands.txt").read_text())
            # every job is self-describing, so agent.ingest can measure it later
            from openkapitza.md.campaign import read_spec
            spec, kind, _ = read_spec(root / "sial_ttm")
            self.assertEqual((kind, spec.element_b), ("nemd", "Al"))
            self.assertIsNotNone(spec.ttm)
            self.assertEqual(read_spec(root / "sige_emd")[1], "emd")
            self.assertEqual(read_spec(root / "sio2_weak")[2].get("bonding"), "weak")
            rung = next(n for n in names if n.startswith("campaign/"))
            self.assertEqual(read_spec(root / rung)[0].element_a, "Si")
            for e in m["jobs"]:
                d_ = root / e["name"]
                self.assertTrue((d_ / "submit.sbatch").exists(), e["name"])
                self.assertGreater(e["node_hours"], 0)


if __name__ == "__main__":
    unittest.main()
