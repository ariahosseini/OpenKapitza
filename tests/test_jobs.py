"""The MD job loop: submit, poll, diagnose, auto-resubmit, auto-ingest.

A fake `lmp` plays each way a run can end (it reads a `mode` file in the job
folder); fake `sbatch` / `sacct` stand in for SLURM. The successful run copies
the real Si/Al plumbing output in tests/data/md_runs/, so ingestion is real.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import time
import unittest
from pathlib import Path

from openkapitza.agent.jobs import JobBook, JobError, _get_walltime, diagnose
from openkapitza.agent.session import Session
from openkapitza.compute.runner import JobState

FIX = Path(__file__).resolve().parent / "data" / "md_runs" / "sial_plumbing"

FAKE_LMP = r"""#!/bin/bash
mode=$(cat mode 2>/dev/null || echo ok)
if [ "$mode" = "kill-once" ]; then
  if [ ! -f .killed ]; then touch .killed; echo "starting" > log.lammps; kill -9 $$; fi
  mode=ok
fi
case "$mode" in
  ok)    cp "$FIX"/log.lammps "$FIX"/temperature_profile.dat "$FIX"/reservoir_energy.dat . ;;
  input) printf 'LAMMPS\nERROR: Potential file is missing an entry for: Si Si Si (src/MANYBODY/pair_tersoff.cpp:458)\n' > log.lammps; exit 1 ;;
  lost)  printf 'Step Temp\n100 300\nERROR: Lost atoms: original 4920 current 4911 (src/thermo.cpp:494)\n' > log.lammps; exit 1 ;;
  nan)   printf 'Step Temp E_pair\n100 300 -4.6\n200 nan nan\n' > log.lammps; exit 1 ;;
  empty) printf 'OPENKAPITZA_DONE area_m2 1e-17\n' > log.lammps; printf '# Chunk-averaged data\n# Timestep\n' > temperature_profile.dat ;;
  cut)   printf 'Step Temp\n100 300\n' > log.lammps ;;
esac
"""

FAKE_SBATCH = """#!/bin/bash
n=$(cat "$FAKE_SLURM/count" 2>/dev/null || echo 100); n=$((n+1)); echo $n > "$FAKE_SLURM/count"
echo "$1" >> "$FAKE_SLURM/submitted"
echo "Submitted batch job $n"
"""
FAKE_SACCT = """#!/bin/bash
cat "$FAKE_SLURM/state"
"""


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class JobLoopBase(unittest.TestCase):
    JOBS = ("sial_pp", "bad_input", "lost", "nan", "empty", "cut", "flaky")

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.root = self.tmp / "jobs"
        jobs = []
        for name in self.JOBS:
            wd = self.root / name
            wd.mkdir(parents=True)
            shutil.copy(FIX / "nemd.in", wd / "nemd.in")
            shutil.copy(FIX / "spec.json", wd / "spec.json")
            (wd / "submit.sbatch").write_text("#!/bin/bash\n#SBATCH --job-name=x\n"
                                              "#SBATCH --time=02:00:00\nsrun lmp -in nemd.in\n")
            jobs.append({"name": name, "node_hours": 1.5})
        modes = {"bad_input": "input", "lost": "lost", "nan": "nan", "empty": "empty",
                 "cut": "cut", "flaky": "kill-once"}
        for name, mode in modes.items():
            (self.root / name / "mode").write_text(mode)
        (self.root / "manifest.json").write_text(json.dumps({"jobs": jobs}))
        self.lmp = str(_exe(self.tmp / "fake_lmp", FAKE_LMP))
        self._env = dict(os.environ)
        os.environ["FIX"] = str(FIX)

    def tearDown(self):
        from openkapitza.compute import runner
        for proc in runner._LAUNCHED:          # reap test jobs (ResourceWarning otherwise)
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
        runner._LAUNCHED.clear()
        os.environ.clear()
        os.environ.update(self._env)
        self._td.cleanup()

    def settle(self, book_or_session, timeout=20.0):
        """Poll until nothing is live (fake jobs take well under a second)."""
        t0, events = time.time(), []
        while time.time() - t0 < timeout:
            book = JobBook.open(self.root)
            events += book.refresh()
            if not any(e.status in ("submitted", "running") for e in book.entries.values()):
                return book, events
            time.sleep(0.1)
        self.fail("jobs did not stop")


class TestLocalLoop(JobLoopBase):
    def test_every_outcome_is_classified(self):
        book = JobBook.open(self.root)
        evs = book.submit(["all"], backend="local", lmp=self.lmp)
        self.assertEqual(sum(e.what == "started" for e in evs), len(self.JOBS))
        book, events = self.settle(book)
        st = {n: e.status for n, e in book.entries.items()}
        kind = {n: (e.diagnosis or {}).get("kind") for n, e in book.entries.items()}
        self.assertEqual(st["sial_pp"], "completed")
        self.assertEqual(kind["sial_pp"], "ok")
        self.assertEqual((st["bad_input"], kind["bad_input"]), ("failed", "input"))
        self.assertEqual((st["lost"], kind["lost"]), ("failed", "physics"))
        self.assertEqual((st["nan"], kind["nan"]), ("failed", "physics"))
        self.assertEqual((st["empty"], kind["empty"]), ("failed", "output"))
        self.assertEqual((st["cut"], kind["cut"]), ("failed", "output"))
        # killed once: infrastructure, resubmitted automatically, then finished
        self.assertEqual(st["flaky"], "completed")
        self.assertEqual(book.entries["flaky"].attempts, 2)
        self.assertTrue(any(e.name == "flaky" and e.what == "resubmitted" for e in events))
        self.assertTrue((self.root / "flaky" / "log.lammps.attempt1").is_file())
        # nothing that the input or the physics broke was retried
        for n in ("bad_input", "lost", "nan", "empty", "cut"):
            self.assertEqual(book.entries[n].attempts, 1, n)

    def test_input_error_cites_the_knowledge_base(self):
        book = JobBook.open(self.root)
        book.submit(["bad_input"], backend="local", lmp=self.lmp)
        book, _ = self.settle(book)
        d = book.entries["bad_input"].diagnosis
        self.assertIn("missing an entry for: Si Si Si", d["evidence"])
        self.assertTrue(any("knowledge-base.md" in c for c in d["citations"]), d["citations"])
        self.assertFalse(d["auto_resubmit"])

    def test_physics_failure_proposes_but_does_not_apply(self):
        book = JobBook.open(self.root)
        book.submit(["lost"], backend="local", lmp=self.lmp)
        book, _ = self.settle(book)
        d = book.entries["lost"].diagnosis
        self.assertTrue(any("timestep" in f for f in d["fixes"]))
        self.assertEqual((self.root / "lost" / "nemd.in").read_text(),
                         (FIX / "nemd.in").read_text())       # deck untouched

    def test_book_persists_and_skips_running_or_finished(self):
        book = JobBook.open(self.root)
        book.submit(["sial_pp"], backend="local", lmp=self.lmp)
        self.settle(book)
        again = JobBook.open(self.root)
        self.assertEqual(again.entries["sial_pp"].status, "completed")
        self.assertEqual(again.submit(["sial_pp"])[0].what, "skipped")
        self.assertEqual(again.config["lmp"], self.lmp)

    def test_prepared_run_sh_is_not_overwritten(self):
        (self.root / "sial_pp" / "run.sh").write_text("#!/bin/bash\n# prepared\n")
        book = JobBook.open(self.root)
        book.submit(["sial_pp"], backend="local", lmp=self.lmp)
        self.settle(book)
        self.assertEqual((self.root / "sial_pp" / "run.sh").read_text(), "#!/bin/bash\n# prepared\n")

    def test_selection_and_errors(self):
        book = JobBook.open(self.root)
        with self.assertRaises(JobError):
            book.select(["nope"])
        with self.assertRaises(JobError):
            book.submit(["sial_pp"], backend="local", lmp="/no/such/lmp")
        with self.assertRaises(JobError):
            JobBook.open(self.tmp)                  # no manifest

    def test_cancel(self):
        (self.root / "sial_pp" / "mode").write_text("ok")
        slow = _exe(self.tmp / "slow_lmp", "#!/bin/bash\nsleep 30\n")
        book = JobBook.open(self.root)
        book.submit(["sial_pp"], backend="local", lmp=str(slow))
        ev = book.cancel(["sial_pp"])
        self.assertEqual(ev[0].what, "cancelled")
        self.assertEqual(JobBook.open(self.root).entries["sial_pp"].status, "cancelled")


class TestSessionLoop(JobLoopBase):
    def test_session_submits_watches_and_ingests(self):
        s = Session(lmp=self.lmp)
        out = s.handle(f"jobs {self.root}")
        self.assertIn("No jobs submitted yet", out)
        out = s.handle("submit sial_pp bad_input local")
        self.assertIn("Submitted 2 job(s)", out)
        out = s.handle("watch 0.1")
        self.assertIn("sial_pp: finished", out)
        self.assertIn("Measured:", out)                       # ingested automatically
        self.assertIn("bad_input: failed", out)
        self.assertIn("see: docs/knowledge-base.md", out)
        self.assertEqual(len(s.md), 1)
        self.assertEqual(JobBook.open(self.root).entries["sial_pp"].status, "ingested")

    def test_session_errors_are_friendly(self):
        s = Session()
        self.assertIn("prepare jobs", s.handle("jobs"))
        s.handle(f"jobs {self.root}")
        self.assertIn("which jobs", s.handle("submit"))
        self.assertIn("no prepared job named", s.handle("submit nope local"))
        self.assertIn("plan names no MD jobs", s.handle("submit plan"))

    def test_save_load_keeps_job_folder(self):
        s = Session(lmp=self.lmp)
        s.handle(f"jobs {self.root}")
        p = self.tmp / "s.json"
        s.handle(f"save {p}")
        t = Session()
        t.handle(f"load {p}")
        self.assertEqual(t.jobs_root, str(self.root.resolve()))
        self.assertEqual(t.lmp, self.lmp)


class TestSlurmLoop(JobLoopBase):
    def setUp(self):
        super().setUp()
        self.slurm = self.tmp / "slurm"
        self.slurm.mkdir()
        bindir = self.tmp / "bin"
        bindir.mkdir()
        _exe(bindir / "sbatch", FAKE_SBATCH)
        _exe(bindir / "sacct", FAKE_SACCT)
        os.environ["PATH"] = f"{bindir}:{os.environ['PATH']}"
        os.environ["FAKE_SLURM"] = str(self.slurm)

    def state(self, s: str) -> None:
        (self.slurm / "state").write_text(s + "\n")

    def test_timeout_resubmits_with_longer_walltime_then_ingests(self):
        book = JobBook.open(self.root)
        ev = book.submit(["sial_pp"], backend="slurm")
        self.assertIn("id 101", ev[0].message)
        self.state("RUNNING")
        self.assertEqual(book.refresh(), [])
        self.assertEqual(book.entries["sial_pp"].status, "running")
        self.state("TIMEOUT")
        ev = book.refresh()
        self.assertEqual(ev[0].what, "resubmitted")
        self.assertIn("3.0 h", ev[0].message)
        self.assertAlmostEqual(_get_walltime(self.root / "sial_pp" / "submit.sbatch"), 3.0)
        # second attempt completes with real output
        for f in ("log.lammps", "temperature_profile.dat", "reservoir_energy.dat"):
            shutil.copy(FIX / f, self.root / "sial_pp" / f)
        self.state("COMPLETED")
        ev = book.refresh()
        self.assertEqual((ev[0].what, book.entries["sial_pp"].attempts), ("finished", 2))

    def test_gives_up_after_max_attempts(self):
        book = JobBook.open(self.root)
        book.submit(["sial_pp"], backend="slurm")
        self.state("NODE_FAIL")
        for _ in range(2):
            self.assertEqual(book.refresh()[0].what, "resubmitted")
        ev = book.refresh()
        self.assertEqual(ev[0].what, "failed")
        self.assertIn("after 3 attempts", ev[0].message)
        self.assertEqual(book.entries["sial_pp"].status, "gave_up")

    def test_failed_slurm_job_with_error_is_not_retried(self):
        book = JobBook.open(self.root)
        book.submit(["bad_input"], backend="slurm")
        (self.root / "bad_input" / "log.lammps").write_text(
            "ERROR: Unrecognized pair style 'meam' (src/force.cpp:279)\n")
        self.state("FAILED")
        ev = book.refresh()
        self.assertEqual((ev[0].what, ev[0].diagnosis.kind), ("failed", "input"))
        self.assertEqual(book.entries["bad_input"].attempts, 1)

    def test_slurm_missing_is_a_clear_error(self):
        os.environ["PATH"] = "/usr/bin:/bin"
        with self.assertRaises(JobError) as cm:
            JobBook.open(self.root).submit(["sial_pp"], backend="slurm")
        self.assertIn("SLURM is not available", str(cm.exception))


class TestDiagnose(unittest.TestCase):
    def test_completed_without_marker_is_output_failure(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "log.lammps").write_text("Step Temp\n")
            self.assertEqual(diagnose(Path(d), JobState.COMPLETED).kind, "output")

    def test_fixture_is_ok(self):
        self.assertEqual(diagnose(FIX, JobState.COMPLETED).kind, "ok")


if __name__ == "__main__":
    unittest.main()
