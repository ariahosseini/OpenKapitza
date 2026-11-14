"""Drive-free check of the non-bonded contact (scripts/*_drive_check.py, KB 44)."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from openkapitza.md.emd_kapitza import EV_OVER_KB, KB
from openkapitza.md.nemd import validate_deck

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))
from test_weekend import SIGE  # noqa: E402


def load(name):
    sp = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


class Prepare(unittest.TestCase):
    def test_two_equilibrium_decks_from_the_nemd_cell(self):
        with tempfile.TemporaryDirectory() as d:
            pot = Path(d) / "pot"
            pot.mkdir()
            (pot / "SiCGe.tersoff").write_text(SIGE)
            root = Path(d) / "dc"
            with contextlib.redirect_stdout(io.StringIO()):
                load("prepare_drive_check").main([str(root), "--potentials", str(pot)])
            man = json.loads((root / "manifest.json").read_text())
            self.assertEqual([j["name"] for j in man["jobs"]], ["emd_slab2nm", "emd_slab4nm"])
            atoms = [j["atoms"] for j in man["jobs"]]
            self.assertLess(atoms[0], atoms[1])
            for j in man["jobs"]:
                wd = root / j["name"]
                deck = (wd / "nemd.in").read_text()
                self.assertEqual(validate_deck(deck), [])
                self.assertIn("OPENKAPITZA_EMD_DONE", deck)
                self.assertIn("run             20000000", deck)        # 20 ns at 1 fs
                self.assertNotIn("impose the gradient", deck)
                self.assertIn("unfix           piston", deck)           # load off, walls frozen
                self.assertTrue((wd / "interface.data").is_file())
                spec = json.loads((wd / "spec.json").read_text())
                self.assertEqual(spec["kind"], "emd")
                self.assertEqual(spec["spec"]["spot_pressure_GPa"], 1.0)


class Analyse(unittest.TestCase):
    def fake_run(self, wd: Path, h: float, n_tau: int = 600, seed: int = 1):
        """An Ornstein-Uhlenbeck slab-energy series with the tau a given h implies."""
        n_a = n_b = 2400
        area = 18.88e-18
        c_r = 3 * KB * n_a * n_b / (n_a + n_b)
        tau = c_r / (area * h)
        dt = tau / 20
        n = int(n_tau * tau / dt)
        rng = np.random.default_rng(seed)
        x = np.empty(n)
        x[0] = 0.0
        a = math.exp(-dt / tau)
        noise = rng.normal(size=n) * math.sqrt(1 - a * a)
        for i in range(1, n):
            x[i] = a * x[i - 1] + noise[i]
        e_a = x * 3 * n_a / EV_OVER_KB
        t_ps = np.arange(n) * dt * 1e12
        wd.mkdir(parents=True)
        np.savetxt(wd / "emd_temperatures.dat",
                   np.c_[t_ps, np.full(n, 300.0), np.full(n, 300.0), e_a, np.zeros(n)],
                   header="time T_a T_b E_a E_b")
        (wd / "log.lammps").write_text(
            f"OPENKAPITZA_EMD n_a {n_a} n_b {n_b} area {area * 1e20:.6f}\n")
        (wd / "job.done").write_text("0\n")

    def test_recovers_h_and_reads_the_drives(self):
        m = load("analyze_drive_check")
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d) / "emd_slab2nm"
            self.fake_run(wd, 40e6)
            r = m.analyse(wd)
            wd2 = Path(d) / "emd_slab4nm"
            self.fake_run(wd2, 40e6, seed=4)
            self.assertEqual(m.choose(m.analyse(wd2)), "single")
            self.assertAlmostEqual(r["single"]["h"], 40.0, delta=6.0)
            self.assertEqual(m.choose(r), "single")         # one decay: no invented fast part
            two = {"single": {"h": 1}, "biexp": {"h": 2, "tau_ps": 80, "tau_fast_ps": 1.3,
                                                  "slow_weight": 0.75}}
            self.assertEqual(m.choose(two), "biexp")
            v = m.verdict(40.0, 2.0)
            self.assertIn("favoured", v["HEX"]["reads"])
            self.assertIn("allowed", v["Langevin"]["reads"])
            v = m.verdict(62.0, 2.0)                      # EMD near the Langevin level
            self.assertIn("LOW", v["HEX"]["reads"])
            self.assertIn("LOW", v["eHEX"]["reads"])


if __name__ == "__main__":
    unittest.main()
