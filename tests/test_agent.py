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


# ---- MD ingestion, files, sweeps, sessions -----------------------------------------
import json  # noqa: E402
import shutil  # noqa: E402

from openkapitza.agent.ingest import IngestError, MDResult, ingest, matches  # noqa: E402
from openkapitza.agent.io import intake_from_dict, load_descriptions  # noqa: E402
from openkapitza.md.campaign import read_spec  # noqa: E402

RUN = Path(__file__).resolve().parent / "data" / "md_runs" / "sial_plumbing"
SI_AL = ["Si", "", "Al", "", "300", "bare", "5", "0.05", "0.2", "0.02", "0.05", "vacuum",
         "uniform", ""]


class Ingest(unittest.TestCase):
    def test_single_job_reproduces_the_recorded_measurement(self):
        (r,) = ingest(RUN).values()
        self.assertEqual(r.pair, frozenset({"Si", "Al"}))
        self.assertAlmostEqual(r.h_K / 1e6, 444.6, delta=0.5)       # tracker P1-6
        self.assertAlmostEqual(r.uncertainty / 1e6, 173.3, delta=0.5)
        self.assertIsNone(r.converged)
        codes = r.flags.codes()
        self.assertIn("MD_SINGLE_RUN", codes)
        self.assertIn("MD_LARGE_ERROR", codes)
        self.assertTrue(matches(r, "Si", "Al") and matches(r, "Al6061", "Si"))
        self.assertFalse(matches(r, "Si", "Cu"))

    def test_prepared_root_and_campaign_root(self):
        spec, _, _ = read_spec(RUN)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "prepared"
            shutil.copytree(RUN, root / "sial_pp")
            (root / "sicu_pp").mkdir()                       # pending job: skipped
            (root / "manifest.json").write_text(json.dumps(
                {"jobs": [{"name": "sial_pp"}, {"name": "sicu_pp"}]}))
            (r,) = ingest(root).values()
            self.assertAlmostEqual(r.h_K / 1e6, 444.6, delta=0.5)
            self.assertTrue(any("no usable output" in n for n in r.notes))
            camp = Path(d) / "camp"
            shutil.copytree(RUN, camp / "r0")
            from openkapitza.md.campaign import _spec_record
            (camp / "manifest.json").write_text(json.dumps(
                {"rungs": [{"name": "r0", "spec": _spec_record(spec), "axes": []}]}))
            (r,) = ingest(camp).values()
            self.assertIsNone(r.converged)                   # one rung: untested
            with self.assertRaises(IngestError):
                ingest(Path(d))                              # nothing to measure

    def test_round_trip(self):
        (r,) = ingest(RUN).values()
        r2 = MDResult.from_dict(json.loads(json.dumps(r.to_dict())))
        self.assertEqual(r2.pair, r.pair)
        self.assertEqual(r2.flags.codes(), r.flags.codes())


class SessionMD(unittest.TestCase):
    def _session(self):
        s = Session(n_samples=100)
        s.start()
        for a in SI_AL:
            s.handle(a)
        return s

    def test_ingest_after_run_replaces_the_dmm(self):
        s = self._session()
        s.handle("run")
        self.assertEqual(s.result.kapitza.source, "dmm_debye")
        out = s.handle(f"ingest {RUN}")
        self.assertIn("Re-ran the joint with the MD h_K", out)
        self.assertEqual(s.result.kapitza.source, "md")
        self.assertEqual(s.plan.md, "in use")
        self.assertAlmostEqual(s.result.kapitza.h_K.uncertainty / 1e6, 173.3, delta=0.5)
        text = s.handle("run")
        self.assertIn("in use", text)
        self.assertIn("finite-size convergence is not established", text)
        self.assertIn("Al/Si", s.handle("md"))
        s.handle("md clear")
        s.handle("run")
        self.assertEqual(s.result.kapitza.source, "dmm_debye")

    def test_user_h_K_beats_md(self):
        s = self._session()
        s.handle(f"ingest {RUN}")
        s.handle("set h_K_MW=200")
        s.handle("run")
        self.assertEqual(s.result.kapitza.source, "user")

    def test_mismatched_md_is_not_used(self):
        s = Session(n_samples=50)
        s.start()
        for a in ["SS304", "", "SS304", "", "300", "bare", "1", "1", "1", "0.1", "0.1",
                  "vacuum", "uniform", "", "run"]:
            s.handle(a)
        self.assertIn("does not match", s.handle(f"ingest {RUN}").replace(
            "None of it matches", "does not match"))
        self.assertNotEqual(s.result.kapitza.source, "md")

    def test_md_at_bond_faces(self):
        s = Session(n_samples=50)
        s.start()
        for a in ["Si", "", "Si", "", "300", "bond", "", "Al 0.2 237 0 Al"]:
            s.handle(a)
        s.handle(f"ingest {RUN}")
        s.handle("run")
        self.assertEqual(s.plan.md, "in use")
        terms = {t.label: t.resistance for t in s.result.stack.terms}
        self.assertAlmostEqual(terms["interface a (Kapitza)"], 1 / 444.55e6, delta=1e-11)


class FilesSweepsSessions(unittest.TestCase):
    def test_description_file_and_errors(self):
        (desc,) = load_descriptions(EXAMPLES / "heatsink_screwed.yaml")
        it = intake_from_dict(desc)
        self.assertAlmostEqual(it.answers["joint_area_mm2"], 1600.0)
        self.assertAlmostEqual(it.answers["temperature_K"], 318.15)
        self.assertEqual(it.defaulted.get("gas_pressure_kPa"), "1 atm")
        with self.assertRaises(FieldError):
            intake_from_dict({**desc, "colour": "blue"})
        bad = dict(desc); bad.pop("sigma_a_um")
        with self.assertRaises(FieldError):
            intake_from_dict(bad)

    def test_batch_json_with_md_and_sweep(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli(["--input", str(EXAMPLES / "si_al_contact_md.yaml"), "--json", str(out),
                          "--sweep", "pressure_MPa=1:10:3", "--csv", str(Path(d) / "s.csv"),
                          "--samples", "50"])
            self.assertEqual(rc, 0)
            data = json.loads(out.read_text())
            self.assertEqual(data["kapitza"]["source"], "md")
            self.assertEqual(data["plan"]["md"], "in use")
            self.assertEqual(len(data["sweep"]["points"]), 3)
            hs = [p["h"] for p in data["sweep"]["points"]]
            self.assertTrue(hs[0] < hs[1] < hs[2])           # h rises with pressure
            self.assertTrue((Path(d) / "s.csv").read_text().startswith("pressure_MPa,"))

    def test_batch_of_several_joints(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "two.yaml"
            f.write_text("joints:\n"
                         "  - {material_a: SS304, material_b: SS304, temperature_K: 300, "
                         "pressure_MPa: 1, sigma_a_um: 1, sigma_b_um: 1, medium: vacuum}\n"
                         "  - {material_a: Cu, material_b: Cu, temperature_K: 300, "
                         "joint_type: tim, blt_um: 50, k_tim: 3}\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli(["--input", str(f), "--json", "-", "--samples", "20"])
            data = json.loads(buf.getvalue())
            self.assertEqual(len(data), 2)
            self.assertAlmostEqual(data[1]["R_total_m2K_W"], 50e-6 / 3, delta=1e-9)

    def test_session_sweep_save_load_export(self):
        s = Session(n_samples=50)
        s.start()
        for a in SI_AL:
            s.handle(a)
        s.handle(f"ingest {RUN}")
        s.handle("run")
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("also needs joint_area_mm2", s.handle("sweep flatness_um=0.5,5,50"))
            s.handle("set flatness_um=1")
            s.handle("set joint_area_mm2=900")
            table = s.handle(f"sweep flatness_um=0.5,5,50 csv {Path(d) / 's.csv'}")
            self.assertIn("flatness_um", table)
            self.assertIn("macro constriction", table)
            self.assertIn("didn't work", s.handle("sweep material_a=1:2:3"))
            s.handle(f"save {Path(d) / 's.json'}")
            s.handle(f"export {Path(d) / 'r.json'}")
            self.assertEqual(json.loads((Path(d) / "r.json").read_text())["kapitza"]["source"], "md")
            t = Session(n_samples=50)
            t.start()
            t.handle(f"load {Path(d) / 's.json'}")
            self.assertEqual(t.intake.answers, s.intake.answers)
            t.handle("run")
            self.assertEqual(t.result.kapitza.source, "md")         # MD survives the reload
            self.assertAlmostEqual(t.result.primary.h_joint.si, s.result.primary.h_joint.si,
                                   delta=1e-6 * s.result.primary.h_joint.si)

    def test_fields_listing(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli(["--fields"])
        self.assertIn("flatness_um", buf.getvalue())
