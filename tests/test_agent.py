"""Agent layer (Phase 3): fields, intake, planner, session, terminal wrapper,
and the three scripted cases in examples/agent/."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from openkapitza.agent import FIELDS, FieldError, Intake, Session, plan
from openkapitza.agent.__main__ import main as cli
from openkapitza.agent.fields import BY_NAME, parse_layers

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "agent"


def run_script(name: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli(["--answers", str(EXAMPLES / name), "--samples", "100"])
    return buf.getvalue()


class Fields(unittest.TestCase):
    def test_units_and_aliases(self):
        self.assertAlmostEqual(BY_NAME["temperature_K"].parse("25 C"), 298.15)
        self.assertAlmostEqual(BY_NAME["pressure_MPa"].parse("100 kPa"), 0.1)
        self.assertAlmostEqual(BY_NAME["sigma_a_um"].parse("0.0008 mm"), 0.8)
        self.assertAlmostEqual(BY_NAME["joint_area_mm2"].parse("40 x 40 mm"), 1600.0)
        self.assertAlmostEqual(BY_NAME["joint_area_mm2"].parse("2 cm2"), 200.0)
        self.assertAlmostEqual(BY_NAME["R_c_mm2K_W"].parse("0.1 Kcm2/W"), 10.0)
        self.assertEqual(BY_NAME["material_a"].parse("copper"), "Cu")
        self.assertEqual(BY_NAME["joint_type"].parse("grease"), "tim")
        self.assertEqual(BY_NAME["medium"].parse("Nitrogen"), "N2")

    def test_bad_input_is_explained(self):
        for name, text in (("pressure_MPa", "-1"), ("sigma_a_um", "1 MPa"),
                           ("slope_a", "0.1 um"), ("material_a", "unobtainium"),
                           ("temperature_K", "hot")):
            with self.assertRaises(FieldError):
                BY_NAME[name].parse(text)

    def test_every_field_says_why(self):
        for f in FIELDS:
            self.assertTrue(f.question and f.why, f.name)

    def test_layers(self):
        self.assertEqual(parse_layers("Al 0.5 237 0 Al; SAC305 30 58 0.04"),
                         [("Al", 0.5, 237.0, 0.0, "Al"), ("SAC305", 30.0, 58.0, 0.04, None)])
        with self.assertRaises(FieldError):
            parse_layers("SAC305 30")


class IntakeFlow(unittest.TestCase):
    def test_bare_joint_question_order_and_branching(self):
        it = Intake()
        for name, text in (("material_a", "SS304"), ("material_b", "SS304"),
                           ("temperature_K", "300"), ("pressure_MPa", "1"),
                           ("sigma_a_um", "1"), ("sigma_b_um", "1")):
            it.set(name, text)
        pending = [f.name for f in it.pending()]
        self.assertIn("medium", pending)
        self.assertNotIn("blt_um", pending)
        it.set("medium", "vacuum")
        self.assertNotIn("gas_pressure_kPa", [f.name for f in it.pending()])
        it.set("joint_type", "tim")                       # switching drops bare-only answers
        self.assertNotIn("sigma_a_um", it.answers)
        self.assertIn("blt_um", [f.name for f in it.pending()])

    def test_required_without_default_cannot_be_skipped(self):
        it = Intake()
        with self.assertRaises(FieldError):
            it.accept_default("material_a")
        it.set("material_a", "Cu")
        self.assertEqual(it.accept_default("phase_a"), "crystalline")
        self.assertEqual(it.defaulted["phase_a"], "the database's phase for this material")


class Planner(unittest.TestCase):
    def _session(self, answers: list[str]) -> Session:
        s = Session(n_samples=50)
        s.start()
        for a in answers:
            s.handle(a)
        return s

    def test_md_recommended_with_a_route_and_jobs_written(self):
        # Si | 0.2 um Al film | Si: the two Si/Al faces carry a large share of R.
        s = self._session(["Si", "", "Si", "", "300", "bond", "",
                           "Al 0.2 237 0 Al", "run"])
        self.assertEqual(s.plan.md, "recommended")
        self.assertEqual(s.plan.route_status, "ready")
        self.assertIn("sial_ttm", s.plan.cluster_jobs)
        with tempfile.TemporaryDirectory() as d:
            pot = Path(d) / "pot"; pot.mkdir()
            for f in ("SiCGe.tersoff", "library.meam", "AlSiMgCuFe.meam", "SiO.tersoff"):
                (pot / f).write_text("# placeholder\n")
            s.potentials = str(pot)
            out = s.handle(f"prepare jobs {Path(d) / 'jobs'}")
            self.assertIn("sial_pp", out)
            self.assertIn("node-hours", out)
            self.assertTrue((Path(d) / "jobs" / "sial_ttm" / "submit.sbatch").exists())

    def test_no_route_for_alloys(self):
        s = self._session(["SS304", "", "Al6061", "", "300", "bond", "",
                           "film 0.05 400", "run"])
        self.assertIn(s.plan.md, ("optional", "recommended"))
        self.assertEqual(s.plan.route_status, "none")

    def test_classic_steel_joint_needs_no_md(self):
        s = self._session(["SS304", "", "SS304", "", "300", "bare", "1 MPa", "1", "1",
                           "0.1", "0.1", "vacuum", "uniform", "", "run"])
        self.assertEqual(s.plan.md, "not needed")
        self.assertEqual(plan(s.result).md, "not needed")

    def test_smooth_aluminium_joint_could_need_it_but_has_no_route(self):
        # Smooth, high-k contact spots make the interface share non-trivial: sweeping the
        # DMM placeholder over its band moves h by ~14 %. Alloys have no MD route.
        s = self._session(["Al6061", "", "Al6061", "", "300", "bare", "1 MPa", "0.4", "0.4",
                           "0.07", "0.07", "vacuum", "uniform", "", "run"])
        self.assertEqual(s.plan.md, "recommended")
        self.assertEqual(s.plan.route_status, "none")


class SessionCommands(unittest.TestCase):
    def test_why_back_show_set_and_errors(self):
        s = Session(n_samples=50)
        self.assertIn("first solid", s.start())
        self.assertIn("conductivities", s.handle("why"))
        s.handle("Cu")
        self.assertIn("first solid", s.handle("back"))       # re-asks side A
        self.assertNotIn("material_a", s.intake.answers)
        self.assertIn("didn't work", s.handle("nonsense"))
        s.handle("Cu")
        self.assertIn("material_a", s.handle("show"))
        self.assertIn("set", s.handle("set temperature_K=350"))
        self.assertEqual(s.intake.answers["temperature_K"], 350.0)
        self.assertIn("unknown field", s.handle("set colour=blue"))
        self.assertIn("run once first", s.handle("what if pressure_MPa=2"))
        self.assertIn("Bye", s.handle("quit"))
        self.assertTrue(s.done)


class ScriptedCases(unittest.TestCase):
    """End to end through the terminal wrapper."""

    def test_screwed_heat_sink(self):
        out = run_script("heatsink_screwed.txt")
        self.assertIn("Joint conductance:", out)
        self.assertIn("Bolted/screwed joint", out)
        self.assertIn("macro constriction (flatness)", out)
        self.assertIn("Atomistic (MD) calculation: not needed", out)
        self.assertIn("With flatness_um = 5", out)
        self.assertIn("Heat splits", out)

    def test_die_on_spreader_with_grease(self):
        out = run_script("die_spreader_grease.txt")
        self.assertIn("17.7 mm²K/W", out)              # 50/3.7 + 4.2 (NREL grease A)
        self.assertIn("TIM (bulk)", out)
        self.assertIn("With blt_um = 25", out)
        self.assertIn("Confidence: good", out)

    def test_solder_die_attach(self):
        out = run_script("die_attach_solder.txt")
        self.assertIn("SAC305 (bulk)", out)
        self.assertIn("MW/m²K", out)
        self.assertIn("voids", out)
        self.assertNotIn("TIM boundary", out)


if __name__ == "__main__":
    unittest.main()
