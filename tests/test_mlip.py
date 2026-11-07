"""Tier-2 MLIP plumbing (md/mlip.py, P4-1, KB 36)."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

from openkapitza.md.mlip import MLIPModel, check_model, load_model, pair_block
from openkapitza.md.nemd import build_input_deck, validate_deck
from openkapitza.schema.flags import FlagSet

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def codes(flags):
    return {str(f.code): int(f.severity) for f in flags.flags}


class PairBlocks(unittest.TestCase):
    def test_each_kind(self):
        cases = {
            ("snap", ("a.snapcoeff", "a.snapparam")): ("snap", "* * a.snapcoeff a.snapparam Si Ge"),
            ("pace", ("m.yace",)): ("pace", "* * m.yace Si Ge"),
            ("mace", ("m.pt",)): ("mliap unified m.pt 0", "* * Si Ge"),
            ("deepmd", ("g.pb",)): ("deepmd g.pb", "* * Si Ge"),
            ("nequip", ("m.pth",)): ("nequip", "* * m.pth Si Ge"),
            ("allegro", ("m.pth",)): ("allegro", "* * m.pth Si Ge"),
        }
        for (kind, files), (style, coeff) in cases.items():
            b = pair_block(MLIPModel(kind, files, ("Si", "Ge")))
            self.assertEqual(b, f"pair_style      {style}\npair_coeff      {coeff}\n")

    def test_bad_models_refused(self):
        with self.assertRaises(ValueError):
            MLIPModel("gap", ("x",), ("Si",))
        with self.assertRaises(ValueError):
            MLIPModel("snap", ("only_one",), ("Si",))

    def test_deck_takes_the_block_unchanged(self):
        from plan_campaign import PRESETS, base_spec
        s = base_spec(PRESETS["sige"])
        s.pair_block = pair_block(MLIPModel("pace", ("m.yace",), ("Si", "Ge")))
        d = build_input_deck(s)
        self.assertEqual(validate_deck(d), [])
        self.assertIn("pair_style      pace\npair_coeff      * * m.yace Si Ge", d)


class Checks(unittest.TestCase):
    def test_interface_coverage_foundation_gate_and_gpu(self):
        f = FlagSet()
        m = MLIPModel("mace", ("m.pt",), ("Si", "Ge"), trained_on=("bulk Si", "bulk Ge"),
                      foundation=True)
        check_model(m, "Si", "Ge", f)
        c = codes(f)
        self.assertEqual(c["MLIP_TRAINING_COVERAGE"], 2)
        self.assertIn("MLIP_FOUNDATION", c)
        self.assertIn("MLIP_UNGATED", c)
        self.assertIn("MLIP_NEEDS_GPU", c)

    def test_a_gated_interface_model_on_a_gpu_is_clean(self):
        f = FlagSet()
        m = MLIPModel("mace", ("m.pt",), ("Si", "Ge"),
                      trained_on=("bulk Si", "Si-Ge interface (100)"),
                      gate={"phonon_dos": "pass", "elastic": "pass", "bulk_k": "pass"})
        check_model(m, "Si", "Ge", f, backend="slurm", has_gpu=True)
        self.assertEqual(codes(f), {})

    def test_wrong_elements_are_critical(self):
        f = FlagSet()
        check_model(MLIPModel("snap", ("a", "b"), ("Ge", "Ge")), "Si", "Ge", f)
        self.assertEqual(codes(f)["MLIP_TRAINING_COVERAGE"], 3)


class Examples(unittest.TestCase):
    def test_example_models_load(self):
        g = load_model(REPO / "examples" / "mlip" / "ge_snap_zuo.yaml")
        self.assertEqual((g.kind, g.elements), ("snap", ("Ge", "Ge")))
        t = load_model(REPO / "examples" / "mlip" / "sige_model_template.yaml")
        self.assertEqual(t.kind, "mace")

    def test_prepare_script_writes_a_job_and_stops_on_critical(self):
        import importlib.util
        p = REPO / "scripts" / "prepare_mlip.py"
        sp = importlib.util.spec_from_file_location("prepare_mlip", p)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            md = Path(d) / "models"
            md.mkdir()
            (md / "sige-mliap_lammps.pt").write_text("x")
            root = Path(d) / "jobs"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = mod.main([str(root), "--model",
                               str(REPO / "examples/mlip/sige_model_template.yaml"),
                               "--model-dir", str(md)])
                self.assertEqual(rc, 0)
                self.assertTrue((root / "mlip" / "sige_mace" / "sige-mliap_lammps.pt").is_file())
                rc = mod.main([str(root), "--model",
                               str(REPO / "examples/mlip/ge_snap_zuo.yaml"),
                               "--model-dir", str(md)])
                self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()


class Cost(unittest.TestCase):
    def test_mlip_styles_are_costed_as_mlip(self):
        from openkapitza.compute.resources import estimate_cost
        t = estimate_cost(35000, 750000, "tersoff", cores=5)
        for style in ("snap", "pace", "mliap", "deepmd", "nequip", "allegro"):
            e = estimate_cost(35000, 750000, style, cores=5)
            self.assertEqual(e.potential, "mlip")
            self.assertGreater(e.wall_hours, 50 * t.wall_hours)
