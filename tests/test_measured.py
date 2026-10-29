"""Measured interface conductance as a reference (openkapitza/md/measured.py)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from openkapitza.md import measured as M
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code, Severity

ROOT = Path(__file__).resolve().parents[1]
HEAD = ["id", "interface id", "interface", "interlayer", "ITR (10-9 m2K/W)",
        "measuring temperature (K)", "measurement method", "Film 1",
        "Film 1 preparation method", "Film 1 thickness (nm)", "substrate (Film 2)",
        "substrate details", "id-R"]
ROWS = [
    ["1", "1", "Au/Si", "", "5", "300", "TDTR", "Au", "e-beam", "80", "Si(100)", "", "R12"],
    ["2", "1", "Au/Si", "Ti", "2.5", "300", "TDTR", "Au", "e-beam", "80", "Si(100)", "", "R12"],
    ["3", "2", "Al/sapphire", "", "4", "310", "TDTR", "Al", "sputter", "100", "sapphire", "", "R7"],
    ["4", "3", "Bi/Si", "", "", "300", "3w", "Bi", "", "", "Si", "", "R9"],      # no ITR
]


def _flags(fn, *a):
    f = FlagSet()
    fn(*a, f)
    return f


class TestNames(unittest.TestCase):
    def test_canonical(self):
        for raw, want in (("Si(100)", "Si"), ("sapphire", "Al2O3"), ("Al6061", "Al"),
                          ("Ni200", "Ni"), ("a-SiO2", "SiO2"), ("fused silica", "SiO2"),
                          ("Au film", "Au"), ("4H-SiC", "SiC"), ("Unobtainium", "Unobtainium")):
            self.assertEqual(M.canonical(raw), want, raw)


class TestPackagedData(unittest.TestCase):
    def test_seed_matches_the_benchmark(self):
        with open(ROOT / "tests/benchmarks/kapitza_measured.csv", encoding="utf-8") as fh:
            bench = {r["case_id"]: float(r["h_MW_m2K"]) for r in csv.DictReader(fh)}
        mine = {r.id: r.h for r in M.load() if r.dataset == "project"}
        self.assertEqual(mine, bench)

    def test_al_si_reference(self):
        ref = M.reference("Si", "Al", 300.0)
        self.assertEqual(len(ref.records), 8)
        self.assertAlmostEqual(ref.lo, 114.6)
        self.assertAlmostEqual(ref.hi, 193.0)
        self.assertIn("4 of 8 with an oxide", ref.summary())
        self.assertIsNone(ref.proxy_note)
        self.assertIn("pure-Al", M.reference("Al6061", "Si", 300.0).proxy_note)

    def test_no_data_is_silent(self):
        f = FlagSet()
        self.assertIsNone(M.check_kapitza("Cu", "Ge", 300.0, 2e8, "dmm_debye", f))
        self.assertEqual(len(f), 0)

    def test_temperature_window_widens_and_says_so(self):
        ref = M.reference("Si", "Al", 80.0)
        self.assertTrue(ref.widened)
        self.assertIn("none within 50 K of 80 K", ref.summary())


class TestCheck(unittest.TestCase):
    def check(self, h_MW, source="dmm_debye"):
        f = FlagSet()
        M.check_kapitza("Si", "Al", 300.0, h_MW * 1e6, source, f)
        (flag,) = f.flags
        return flag

    def test_above_range_is_a_caution_with_the_factor_and_a_remedy(self):
        fl = self.check(321)
        self.assertEqual((fl.code, fl.severity), (Code.KAPITZA_ABOVE_MEASURED, Severity.CAUTION))
        self.assertIn("1.7x the highest measured", fl.message)
        self.assertIn("set h_K_MW=151", fl.remedy)

    def test_inside_range_is_info(self):
        fl = self.check(160, "md")
        self.assertEqual((fl.code, fl.severity), (Code.KAPITZA_REFERENCE, Severity.INFO))

    def test_below_range_is_a_warning(self):
        fl = self.check(40, "md")
        self.assertEqual((fl.code, fl.severity), (Code.KAPITZA_BELOW_MEASURED, Severity.WARNING))

    def test_a_users_own_number_is_only_noted(self):
        self.assertEqual(self.check(400, "user").severity, Severity.INFO)
        self.assertEqual(self.check(40, "user").severity, Severity.INFO)

    def test_potential_bias(self):
        f = FlagSet()
        self.assertEqual(M.potential_bias(frozenset({"Al", "Si"}), "meam library.meam", f),
                         "khot2025")
        self.assertEqual(f.flags[0].code, Code.KAPITZA_POTENTIAL_BIAS)
        self.assertIn("Khot", f.flags[0].message)
        g = FlagSet()
        self.assertIsNone(M.potential_bias(frozenset({"Si", "Ge"}), "tersoff SiCGe", g))
        self.assertEqual(len(g), 0)


class TestImport(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.out = self.tmp / "itr.csv"
        self.out.write_text((M.DATA).read_text())          # start from the packaged seed

    def tearDown(self):
        M.load.cache_clear()
        self._td.cleanup()

    def _csv(self) -> Path:
        p = self.tmp / "nims.csv"
        with p.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(HEAD)
            w.writerows(ROWS)
        return p

    def test_csv_import_converts_units_and_keeps_project_rows(self):
        info = M.import_nims(self._csv(), self.out)
        self.assertEqual((info["imported"], info["skipped"], info["kept"]), (3, 1, 8))
        recs = M.load(str(self.out))
        au = [r for r in recs if r.pair == frozenset({"Au", "Si"})]
        self.assertEqual(sorted(r.h for r in au), [200.0, 400.0])      # 1/(5e-9), 1/(2.5e-9)
        self.assertTrue(any(r.pair == frozenset({"Al", "Al2O3"}) for r in recs))
        text = self.out.read_text()
        self.assertIn("CC BY 4.0", text)
        self.assertIn("zenodo.3564173", text)
        # the explicit Ti adhesion layer is left out of the reference, not mixed in
        ref = M.reference("Au", "Si", 300.0, str(self.out))
        self.assertEqual([r.h for r in ref.records], [200.0])

    def test_reimport_replaces_not_duplicates(self):
        M.import_nims(self._csv(), self.out)
        M.import_nims(self._csv(), self.out)
        self.assertEqual(sum(r.dataset == "NIMS-ITR" for r in M.load(str(self.out))), 3)

    def test_xlsx_import(self):
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "ITR dataset"
        ws.append(["Interfacial thermal resistance dataset"])        # a title row above
        ws.append(HEAD)
        for r in ROWS:
            ws.append(r)
        wb.create_sheet("ITR Reference").append(["id-R", "reference"])
        p = self.tmp / "nims.xlsx"
        wb.save(p)
        self.assertEqual(M.import_nims(p, self.out)["imported"], 3)

    def test_unrecognised_table_is_refused(self):
        p = self.tmp / "bad.csv"
        p.write_text("a,b,c\n1,2,3\n")
        with self.assertRaises((ValueError, StopIteration)):
            M.import_nims(p, self.out)


class TestAgent(unittest.TestCase):
    JOINT = {"material_a": "Si", "material_b": "Al", "temperature_K": 300, "pressure_MPa": 5,
             "sigma_a_um": 0.05, "sigma_b_um": 0.2, "slope_a": 0.02, "slope_b": 0.05,
             "medium": "vacuum", "clamping": "uniform"}

    def test_answer_shows_measured_range_and_md_bias(self):
        from openkapitza.agent.io import intake_from_dict
        from openkapitza.agent.session import Session
        s = Session(n_samples=100)
        s.intake = intake_from_dict(self.JOINT)
        out = s.handle("run")
        self.assertIn("Measured Si/Al: 115-193 MW/m²K", out)
        self.assertIn("In use: 321 MW/m²K (DMM placeholder)", out)
        s.handle(f"ingest {ROOT / 'tests/data/md_runs/sial_plumbing'}")
        out = s.handle("run")
        self.assertIn("In use: 445 MW/m²K (MD)", out)
        self.assertIn("Khot et al.", out)
        self.assertIn(Code.KAPITZA_POTENTIAL_BIAS, s.result.flags.codes())

    def test_mcp_must_report_carries_it(self):
        from openkapitza.agent.mcp_server import OpenKapitzaTools
        t = OpenKapitzaTools(n_samples=100)
        t.set_inputs({k: str(v) for k, v in self.JOINT.items()})
        codes = {m["code"] for m in t.run_joint()["must_report"]}
        self.assertIn("KAPITZA_ABOVE_MEASURED", codes)      # cautions are passed on too


if __name__ == "__main__":
    unittest.main()
