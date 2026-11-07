"""eHEX drive, interface heat-flow tally, rescaled Tersoff, weekend job set (KB 28)."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from openkapitza.md.nemd import EV_PER_PS_IN_W, build_input_deck, validate_deck
from openkapitza.md.rescale_tersoff import FIELDS, read_entries, rescale, write_entries

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# The Si(D)/Ge entries of LAMMPS's SiCGe.tersoff (Tersoff 1989).
SIGE = """\
Si(D) Si(D) Si(D) 3.0 1.0 0.0 1.0039e5 16.217 -0.59825 0.78734 1.1000e-6 1.7322 471.18 2.85 0.15 2.4799 1830.8
Ge Ge Ge 3.0 1.0 0.0 1.0643e5 15.652 -0.43884 0.75627 9.0166e-7 1.7047 419.23 2.95 0.15 2.4451 1769.0
Si(D) Ge Ge 3.0 1.0 0.0 1.0039e5 16.217 -0.59825 0.78734 1.1000e-6 1.71845 444.7177 2.8996 0.1500 2.4625 1799.6347
Si(D) Si(D) Ge 3.0 1.0 0.0 1.0039e5 16.217 -0.59825 0.78734 0.0 0.0 0.0 2.8996 0.1500 0.0 0.0
Si(D) Ge Si(D) 3.0 1.0 0.0 1.0039e5 16.217 -0.59825 0.78734 0.0 0.0 0.0 2.85 0.15 0.0 0.0
Ge Si(D) Si(D) 3.0 1.0 0.0 1.0643e5 15.652 -0.43884 0.75627 9.0166e-7 1.71845 444.7177 2.8996 0.1500 2.4625 1799.6347
Ge Si(D) Ge 3.0 1.0 0.0 1.0643e5 15.652 -0.43884 0.75627 0.0 0.0 0.0 2.95 0.15 0.0 0.0
Ge Ge Si(D) 3.0 1.0 0.0 1.0643e5 15.652 -0.43884 0.75627 0.0 0.0 0.0 2.8996 0.1500 0.0 0.0
"""


def entries():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.tersoff"
        p.write_text(SIGE)
        return read_entries(p)


class Rescale(unittest.TestCase):
    def test_mixing_rules_reproduce_the_published_cross_terms(self):
        e = entries()
        r = rescale(e, "Si(D)", "Ge", "Ge", 1.0)
        for k in e:
            for f in FIELDS:
                self.assertAlmostEqual(r[k][f], e[k][f], delta=5e-4 * max(1, abs(e[k][f])))

    def test_lengths_scale_and_energies_do_not(self):
        e, s = entries(), 0.96
        r = rescale(e, "Si(D)", "Ge", "Ge", s)
        g, g0 = r[("Ge", "Ge", "Ge")], e[("Ge", "Ge", "Ge")]
        self.assertAlmostEqual(g["lambda1"], g0["lambda1"] / s)
        self.assertAlmostEqual(g["R"] - g["D"], s * (g0["R"] - g0["D"]))
        self.assertAlmostEqual(g["R"] + g["D"], s * (g0["R"] + g0["D"]))
        self.assertEqual((g["A"], g["B"]), (g0["A"], g0["B"]))
        self.assertEqual(r[("Si(D)", "Si(D)", "Si(D)")], e[("Si(D)", "Si(D)", "Si(D)")])
        x = r[("Si(D)", "Ge", "Ge")]
        self.assertTrue(e[("Si(D)", "Ge", "Ge")]["R"] > x["R"] > g["R"])

    def test_round_trip(self):
        r = rescale(entries(), "Si(D)", "Ge", "Ge", 0.96)
        with tempfile.TemporaryDirectory() as d:
            write_entries(Path(d) / "y", r, header="test")
            back = read_entries(Path(d) / "y")
        self.assertEqual(set(back), set(r))
        self.assertAlmostEqual(back[("Ge", "Ge", "Ge")]["R"], r[("Ge", "Ge", "Ge")]["R"])


def nb_spec(**kw):
    from prepare_weekend import nb_base
    from openkapitza.structure.nonbonded import NonbondedSpec, make_spec
    s = nb_base()
    s.spot_pressure_GPa = 1.0
    for k, v in kw.items():
        setattr(s, k, v)
    return make_spec(s, NonbondedSpec(gas="Ar"))


class Decks(unittest.TestCase):
    def test_ehex_deck(self):
        s = nb_spec(flux_mode="ehex", imposed_flux=1e9, presplit_K=20.0)
        d = build_input_deck(s)
        self.assertEqual(validate_deck(d), [])
        self.assertEqual(d.count(" ehex 1 "), 4)              # hot and cold, two stages
        self.assertNotIn(" heat 1 ", d)
        self.assertNotIn("langevin", d)

    def test_tally_block(self):
        d = build_input_deck(nb_spec(flux_mode="ehex", imposed_flux=1e9))
        self.assertIn("group           tal_a type 1", d)
        self.assertIn("compute         q_into_b  tal_nota heat/flux/virial/tally tal_a", d)
        self.assertIn("compute         q_into_bs tal_b heat/flux/virial/tally tal_notb", d)
        self.assertIn("file interface_flux.dat", d)
        prod = d.index("reset_timestep  0")
        self.assertGreater(d.index("compute         q_into_b"), prod)

    def test_tally_refused_across_a_tersoff_bond(self):
        from plan_campaign import PRESETS, base_spec
        s = base_spec(PRESETS["sige"])
        s.interface_tally = True
        with self.assertRaises(ValueError):
            build_input_deck(s)

    def test_resume_appends_and_truncates_the_tally(self):
        from openkapitza.md.checkpoint import build_resume_deck, truncate_energy
        s = nb_spec(flux_mode="ehex", imposed_flux=1e9)
        r = build_resume_deck(build_input_deck(s), 4000, s.production_steps, 0.001, 0.0, 0.0)
        self.assertIn("append interface_flux.dat", r)
        with tempfile.TemporaryDirectory() as dd:
            p = Path(dd) / "interface_flux.dat"
            p.write_text("# t\n# c\n3000 1 1\n4000 2 2\n5000 3 3\n")
            self.assertEqual(truncate_energy(p, 4000, 1.0), 2)
            self.assertNotIn("5000", p.read_text())


class TallyMeasurement(unittest.TestCase):
    def test_parse_and_convert(self):
        from openkapitza.md.measure import parse_interface_tally
        area = 2e-17
        q = 1e9 * area / EV_PER_PS_IN_W                    # eV/ps for 1 GW/m^2
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "interface_flux.dat"
            p.write_text("# header\n" + "".join(f"{100 * (i + 1)} {q * (1 + 0.01 * (-1) ** i)} "
                                                f"{q * 0.9}\n" for i in range(80)))
            t = parse_interface_tally(p, area)
        self.assertAlmostEqual(t["q_into_b"] / 1e9, 1.0, places=3)
        self.assertAlmostEqual(t["q_into_bs"] / 1e9, 0.9, places=6)
        self.assertEqual(t["rows"], 80)


class WeekendScript(unittest.TestCase):
    def test_writes_cases_in_order(self):
        p = REPO / "scripts" / "prepare_weekend.py"
        sp = importlib.util.spec_from_file_location("prepare_weekend", p)
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        self.assertEqual(list(m.cases())[:2], ["arb_langevin_d0.1", "arb_langevin_d0.5"])
        with tempfile.TemporaryDirectory() as d:
            pot = Path(d) / "pot"
            pot.mkdir()
            (pot / "SiCGe.tersoff").write_text(SIGE)
            root = Path(d) / "jobs"
            root.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                m.main([str(root), "--potentials", str(pot),
                        "--only", "arb_ehex", "sige_ge_rescaled", "he_mass40"])
            names = [j["name"] for j in json.loads((root / "manifest.json").read_text())["jobs"]]
            self.assertEqual(names, ["weekend/arb_ehex", "weekend/sige_ge_rescaled",
                                     "weekend/he_mass40"])
            wd = root / "weekend" / "sige_ge_rescaled"
            deck = (wd / "nemd.in").read_text()
            self.assertIn("SiGe_Ge-rescaled.tersoff", deck)
            self.assertTrue((wd / "SiGe_Ge-rescaled.tersoff").is_file())
            data = (root / "weekend" / "he_mass40" / "interface.data").read_text()
            self.assertIn("3 39.948", data)


if __name__ == "__main__":
    unittest.main()


class Analysis(unittest.TestCase):
    def load(self):
        p = REPO / "scripts" / "analyze_weekend.py"
        sp = importlib.util.spec_from_file_location("analyze_weekend", p)
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        return m

    def test_ratio_and_verdict(self):
        m = self.load()
        r, e = m.ratio({"h": 66.0, "err": 3.3}, {"h": 41.1, "err": 0.8})
        self.assertAlmostEqual(r, 66.0 / 41.1)
        self.assertIn("higher", m.verdict(r, e))
        self.assertIn("same", m.verdict(1.02, 0.03))

    def test_collect_groups_finished_failed_and_waiting(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "weekend"
            for name, rc in (("arb_ehex", "0"), ("arb_hex", "0"), ("he_ehex", "1"),
                             ("ar_ehex", None)):
                (w / name).mkdir(parents=True)
                if rc is not None:
                    (w / name / "job.done").write_text(rc)
            fake = {"arb_ehex": 40.0, "arb_hex": 44.0}
            m.measure = lambda wd: {"h": fake[wd.name], "err": 1.0, "flux_mode": "ehex",
                                    "damping_ps": 0.1, "warn": []}
            out = m.collect(Path(d))
        self.assertEqual(set(out["results"]), {"r00", "arb_ehex", "arb_hex"})
        self.assertIn("failed", out["missing"]["he_ehex"])
        self.assertIn("running", out["missing"]["ar_ehex"])
        hex_row = next(c for c in out["comparisons"]["A. drives (LJ x1, 1 GPa)"]
                       if c["a"] == "arb_hex")
        self.assertAlmostEqual(hex_row["ratio"], 1.1)
        self.assertIn("higher", hex_row["verdict"])


class Intermix(unittest.TestCase):
    def spec(self, n):
        from plan_campaign import PRESETS, base_spec
        s = base_spec(PRESETS["sige"])
        s.intermix_layers = n
        return s

    def test_plan_is_linear_and_straddles_the_plane(self):
        from openkapitza.md.nemd import intermix_layers_plan
        s = self.spec(4)
        plan = intermix_layers_plan(s)
        z_if = s.geometry.interface_z * 1e10
        self.assertEqual([p[1] for p in plan], [2, 2, 1, 1])
        self.assertEqual([round(p[2], 3) for p in plan], [0.125, 0.375, 0.375, 0.125])
        self.assertTrue(plan[1][0] < z_if < plan[2][0])
        with self.assertRaises(ValueError):
            intermix_layers_plan(self.spec(3))

    def test_deck_swaps_types_after_fixing_the_side_groups(self):
        d = build_input_deck(self.spec(2))
        self.assertEqual(validate_deck(d), [])
        self.assertLess(d.index("group           geo_a region side_a"),
                        d.index("set             region mix0 type/ratio"))
        self.assertIn("group           gside_a union geo_a", d)
        self.assertNotIn("group           gside_a type 1", d)
        self.assertNotIn("type/ratio", build_input_deck(self.spec(0)))
