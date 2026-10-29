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
# the released ITR_dataset.xlsx layout (Zenodo 10.5281/zenodo.3564173, sheet "ITR")
HEAD = ["id", "interface id", "Interface", "interlayer", "Thermal Boundary Conductance (MW/m2K)",
        "itr (10-9 m2K/W)", "Temperature (K)", "Measurement method", "Film 1",
        "Film preparation method", "Film thickness (nm)", "substrate (Film2)",
        "Substrate details", "Substrate pretreatment", " Interfacial properties ",
        "Reference (id-R)"]
ROWS = [
    [1, 1, "Au/Si", 0, 200, 5, 300, "TDTR", "Au", "e-beam", 80, "Si", "", "", "None", 1],
    [2, 1, "Au/Ti/Si", 1, 400, 2.5, 300, "TDTR", "Au", "e-beam", 80, "Si", "", "", "None", 1],
    [3, 2, "Au/SiO2/Si", 1, 50, 20, 300, "TDTR", "Au", "e-beam", 80, "Si", "", "", "None", 2],
    [4, 3, "Au/Si", 0, 20, 50, 300, "TDTR", "Au", "", 80, "Si", "", "implanted",
     "5.7*1016 protons cm-2", 3],
    [5, 3, "Au/Si", 0, 180, 5.556, 300, "TDTR", "Au", "", 80, "Si", "", "implanted",
     "non-irradiated", 3],
    [6, 4, "Al/ Sapphire", 0, None, 4, 310, "TDTR", "Al", "sputter", 100, "sapphire", "", "",
     "None", 2],                                           # TBC blank: from ITR
    [7, 5, "Al/Si", 0, 193, 5.18, 298, "TDTR", "Al", "", 80, "Si", "", "", "None", 4],  # repeat
    [8, 6, "Bi", 0, None, None, 300, "3 ω", "Bi", "", "", "Si", "", "", "None", 2],     # unusable
]
REFS = [["id-R", "Title", "Authors", "Journal", "DOI"],
        [1, "t1", "A. Smith, B. Jones", "Phys. Rev. B 80, 000001 (2009)", "http://dx.doi.org/x"],
        [2, "t2", "C. Doe and E. Roe", "Appl. Phys. Lett. 90, 111111 (2007)", ""],
        [3, "t3", "F. Poe", "J. Appl. Phys. 1, 222222 (2011)", ""],
        [4, "Effects of surface roughness", "P. E. Hopkins", "PHYSICAL REVIEW B 82, 085307 (2010)",
         "http://dx.doi.org/10.1103/PhysRevB.82.085307"]]


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


def _seed_file(d: Path) -> str:
    """The project's own rows only, so the logic tests do not move with imports."""
    rows = [ln for ln in M.DATA.read_text().splitlines(True)
            if ln.startswith(("#", "id,")) or ",project," in ln]
    p = d / "seed.csv"
    p.write_text("".join(rows))
    return str(p)


class TestPackagedData(unittest.TestCase):
    def test_seed_matches_the_benchmark(self):
        with open(ROOT / "tests/benchmarks/kapitza_measured.csv", encoding="utf-8") as fh:
            bench = {r["case_id"]: float(r["h_MW_m2K"]) for r in csv.DictReader(fh)}
        mine = {r.id: r.h for r in M.load() if r.dataset == "project"}
        self.assertEqual(mine, bench)

    def test_nims_dataset_is_imported_with_attribution(self):
        recs = M.load()
        nims = [r for r in recs if r.dataset == "NIMS-ITR"]
        self.assertGreater(len(nims), 1300)
        self.assertGreater(len({r.pair for r in nims}), 100)
        head = M.DATA.read_text()[:1200]
        for s in ("CC BY 4.0", "zenodo.3564173", "Sci. Data 7, 36 (2020)", "Changes:"):
            self.assertIn(s, head)
        # Hopkins 2010 is in both; it must be counted once
        hop = [r for r in recs if r.pair == frozenset({"Al", "Si"}) and abs(r.h - 193.0) < 0.01]
        self.assertEqual(len(hop), 1)

    def test_al_si_reference_from_the_full_table(self):
        ref = M.reference("Si", "Al", 300.0)
        self.assertGreaterEqual(len(ref.records), 30)
        self.assertGreaterEqual(len(ref.sources), 8)
        self.assertLessEqual(ref.lo, 115)
        self.assertGreaterEqual(ref.hi, 340)                 # the clean interfaces arrived
        self.assertGreater(ref.n_modified, 0)                # irradiated samples left out
        self.assertGreater(ref.n_excluded_layers, 0)         # Ge, GaN, polymer layers left out
        self.assertTrue(all(not r.modified for r in ref.records))
        self.assertIn("clean", ref.summary())

    def test_seed_only_reference(self):
        with tempfile.TemporaryDirectory() as d:
            p = _seed_file(Path(d))
            ref = M.reference("Si", "Al", 300.0, p)
            self.assertEqual(len(ref.records), 8)
            self.assertAlmostEqual(ref.lo, 114.6)
            self.assertAlmostEqual(ref.hi, 193.0)
            self.assertIn("clean 158-193 (n=4), with an oxide 115-144 (n=4)", ref.summary())
            self.assertIsNone(ref.proxy_note)
            self.assertIn("pure-Al", M.reference("Al6061", "Si", 300.0, p).proxy_note)
            self.assertTrue(M.reference("Si", "Al", 80.0, p).widened)
            self.assertIn("none within 50 K of 80 K", M.reference("Si", "Al", 80.0, p).summary())
        M.load.cache_clear()

    def test_no_data_is_silent(self):
        f = FlagSet()
        self.assertIsNone(M.check_kapitza("Cu", "Unobtainium", 300.0, 2e8, "dmm_debye", f))
        self.assertEqual(len(f), 0)


class TestCheck(unittest.TestCase):
    """Against the 8 project rows (115-193 MW/m²K), so the numbers are fixed."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.seed = _seed_file(Path(cls._td.name))

    @classmethod
    def tearDownClass(cls):
        M.load.cache_clear()
        cls._td.cleanup()

    def check(self, h_MW, source="dmm_debye"):
        f = FlagSet()
        M.check_kapitza("Si", "Al", 300.0, h_MW * 1e6, source, f, self.seed)
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
        self.out.write_text(Path(_seed_file(self.tmp)).read_text())   # the project rows

    def tearDown(self):
        M.load.cache_clear()
        self._td.cleanup()

    def _xlsx(self) -> Path:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "ITR"
        ws.append(HEAD)
        for r in ROWS:
            ws.append(r)
        refs = wb.create_sheet("ITR References")
        for r in REFS:
            refs.append(r)
        p = self.tmp / "ITR_dataset.xlsx"
        wb.save(p)
        return p

    def test_real_layout_import(self):
        info = M.import_nims(self._xlsx(), self.out)
        # 8 rows: 1 unusable, 1 repeats Hopkins 2010 (same paper, same value)
        self.assertEqual((info["imported"], info["skipped"], info["duplicates_of_project_rows"],
                          info["kept"], info["modified"]), (6, 1, 1, 8, 1))
        recs = {r.id: r for r in M.load(str(self.out))}
        self.assertEqual(recs["NIMS-2"].interlayer, "Ti")                       # parsed
        self.assertIn("oxide", recs["NIMS-3"].interlayer)                       # SiO2 on Si
        self.assertTrue(recs["NIMS-4"].modified)                                # irradiated
        self.assertFalse(recs["NIMS-5"].modified)                               # "non-irradiated"
        self.assertAlmostEqual(recs["NIMS-6"].h, 250.0)                         # from ITR
        self.assertEqual(recs["NIMS-6"].pair, frozenset({"Al", "Al2O3"}))       # 'Al/ Sapphire'
        self.assertTrue(recs["NIMS-1"].source.startswith("Smith et al., Phys. Rev. B 80"))
        # Ti adhesion layer and damaged sample are left out of the Au/Si reference;
        # the native-oxide row and the non-irradiated control stay
        ref = M.reference("Au", "Si", 300.0, str(self.out))
        self.assertEqual(sorted(r.h for r in ref.records), [50.0, 180.0, 200.0])
        self.assertEqual((ref.n_excluded_layers, ref.n_modified), (1, 1))
        self.assertEqual(len(ref.sources), 3)

    def test_reimport_replaces_not_duplicates(self):
        M.import_nims(self._xlsx(), self.out)
        M.import_nims(self._xlsx(), self.out)
        self.assertEqual(sum(r.dataset == "NIMS-ITR" for r in M.load(str(self.out))), 6)

    def test_csv_import(self):
        p = self.tmp / "nims.csv"
        with p.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(HEAD)
            w.writerows([["" if c is None else c for c in r] for r in ROWS])
        # a CSV has no references sheet, so a repeat cannot be matched to its paper
        # and is kept rather than guessed away
        info = M.import_nims(p, self.out)
        self.assertEqual((info["imported"], info["duplicates_of_project_rows"]), (7, 0))

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
        self.assertIn("Measured Si/Al: ", out)
        self.assertIn("In use: 321 MW/m²K (DMM placeholder)", out)
        self.assertIn(Code.KAPITZA_REFERENCE, s.result.flags.codes())     # within the range
        s.handle(f"ingest {ROOT / 'tests/data/md_runs/sial_plumbing'}")
        out = s.handle("run")
        self.assertIn("In use: 445 MW/m²K (MD)", out)
        self.assertIn("Khot et al.", out)
        self.assertIn(Code.KAPITZA_POTENTIAL_BIAS, s.result.flags.codes())

    def test_mcp_must_report_carries_it(self):
        from openkapitza.agent.mcp_server import OpenKapitzaTools
        t = OpenKapitzaTools(n_samples=100)
        t.set_inputs({k: str(v) for k, v in self.JOINT.items()})
        t.set_inputs({"h_K_MW": "2000"})                    # far above any Al/Si measurement
        t.run_joint()
        codes = {f.code for f in t.s.result.flags}
        self.assertIn("KAPITZA_ABOVE_MEASURED", codes)      # noted for a user's own value
        t.clear_input("h_K_MW")
        t.s.handle(f"ingest {ROOT / 'tests/data/md_runs/sial_plumbing'}")
        codes = {m["code"] for m in t.run_joint()["must_report"]}
        self.assertIn("KAPITZA_POTENTIAL_BIAS", codes)      # cautions are passed on too


if __name__ == "__main__":
    unittest.main()
