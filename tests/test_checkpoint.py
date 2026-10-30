"""Checkpointed NEMD production and resuming a killed run (md/checkpoint.py)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from openkapitza.md.campaign import write_spec
from openkapitza.md.checkpoint import (
    ResumeError, build_resume_deck, checkpoints, prepare_resume, prune, truncate_energy,
    truncate_profile, usable_checkpoint,
)
from openkapitza.md.nemd import NEMDGeometry, NEMDSpec, build_input_deck, checkpoint_every

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def small_spec() -> NEMDSpec:
    from plan_campaign import PRESETS, base_spec
    spec = base_spec(PRESETS["sige"])
    g = spec.geometry
    spec.geometry = NEMDGeometry(wall_thickness=g.wall_thickness,
                                 reservoir_thickness=g.reservoir_thickness,
                                 bulk_a_length=8e-9, bulk_b_length=8e-9, cross_section=2.2e-9,
                                 bin_width=g.bin_width)
    for k, v in dict(relax_steps=500, equilibrate_steps=2000, steady_steps=2000,
                     production_steps=8000, profile_every=50, profile_frames=8).items():
        setattr(spec, k, v)
    return spec


def fake_outputs(wd: Path, spec: NEMDSpec, upto_step: int, nchunks: int = 3) -> None:
    """reservoir_energy.dat and temperature_profile.dat as a killed run leaves them."""
    dt = spec.timestep_fs * 1e-3
    rows = ["# time[ps] E_hot[eV] E_cold[eV]"]
    for s in range(0, upto_step + 1, spec.profile_every):
        rows.append(f"{s * dt:g} {0.01 * s} {-0.01 * s}")
    rows.append("12.3")                                   # half-written last line
    (wd / "reservoir_energy.dat").write_text("\n".join(rows) + "\n")
    win = 1000
    prof = ["# Chunk-averaged data", "# Timestep Number-of-chunks Total-count",
            "# Chunk Coord1 Ncount temp"]
    for s in range(win, upto_step + 1, win):
        prof.append(f"{s} {nchunks} 300")
        prof += [f"  {i + 1} {5 * i} 100 {300 + i}" for i in range(nchunks)]
    prof.append(f"{upto_step + win} {nchunks} 300")      # an unfinished frame
    prof.append("  1 0 100 300")
    (wd / "temperature_profile.dat").write_text("\n".join(prof) + "\n")


class Deck(unittest.TestCase):
    def test_production_writes_restarts_at_frame_boundaries(self):
        spec = small_spec()
        deck = build_input_deck(spec)
        self.assertEqual(checkpoint_every(spec), 1000)
        self.assertIn("restart         1000 restart.*.bin\nrun             8000\nrestart         0",
                      deck)
        spec.checkpoint_frames = 0
        self.assertNotIn("restart.*.bin", build_input_deck(spec))

    def test_resume_deck_rebuilds_the_production_state(self):
        spec = small_spec()
        r = build_resume_deck(build_input_deck(spec), 4000, 8000, 0.001, 1.5, -1.25)
        lines = r.splitlines()
        self.assertIn("log             log.lammps append", lines)
        self.assertIn("read_restart    restart.4000.bin", lines)
        self.assertTrue(any(l.startswith("pair_coeff") and "Si(D)" in l for l in lines))
        self.assertIn("variable        ehot equal f_hot+1.5", lines)
        self.assertIn("variable        ecold equal f_cold+-1.25", lines)
        self.assertIn("variable        tnow equal step*0.001", lines)
        # outputs append, after a one-step run so the kept checkpoint row is not repeated
        run1 = lines.index("run             1")
        prof = next(i for i, l in enumerate(lines) if "ave/chunk" in l)
        self.assertGreater(prof, run1)
        self.assertIn("append temperature_profile.dat", lines[prof])
        self.assertIn("append reservoir_energy.dat", r)
        self.assertNotIn(" file ", r)
        # only fixes alive at production: walls frozen, NVE, the two reservoirs
        fixes = [l.split()[1] for l in lines if l.startswith("fix")]
        self.assertEqual(sorted(fixes), sorted(["freeze_lo", "freeze_hi", "nve", "hot", "cold",
                                                "profile", "energy"]))
        self.assertNotIn("velocity", r)
        self.assertNotIn("minimize", r)
        # fresh Langevin streams
        hot = next(l for l in lines if l.startswith("fix hot"))
        orig = next(l for l in build_input_deck(spec).splitlines()
                    if l.startswith("fix") and l.split()[1] == "hot")
        self.assertNotEqual(orig.split()[7], hot.split()[7])
        self.assertIn("OPENKAPITZA_DONE", lines[-1])
        self.assertIn("run             8000 upto", lines)

    def test_unknown_production_length_is_refused(self):
        with self.assertRaises(ResumeError):
            build_resume_deck(build_input_deck(small_spec()), 4000, 1234, 0.001, 0, 0)


class Outputs(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.wd = Path(self._td.name)
        self.spec = small_spec()

    def tearDown(self):
        self._td.cleanup()

    def test_truncation_keeps_whole_frames_and_rows_to_the_step(self):
        fake_outputs(self.wd, self.spec, 4300)
        self.assertEqual(truncate_profile(self.wd / "temperature_profile.dat", 4000), 4)
        self.assertEqual(truncate_energy(self.wd / "reservoir_energy.dat", 4000, 0.001), 81)
        e = (self.wd / "reservoir_energy.dat").read_text().splitlines()
        self.assertEqual(e[-1].split()[0], "4")             # the checkpoint row is kept

    def test_newest_checkpoint_with_its_energy_row_is_used(self):
        fake_outputs(self.wd, self.spec, 4300)
        for s in (2000, 3000, 4000, 5000):
            (self.wd / f"restart.{s}.bin").write_bytes(b"x")
        (self.wd / "restart.6000.bin").write_bytes(b"")      # empty: write interrupted
        step, path, row = usable_checkpoint(self.wd, self.spec)
        self.assertEqual((step, row[1]), (4000, 40.0))      # 5000 is past the energy file
        self.assertEqual([s for s, _ in checkpoints(self.wd)], [5000, 4000, 3000, 2000])
        prune(self.wd, keep=2)
        self.assertEqual([s for s, _ in checkpoints(self.wd)], [5000, 4000])

    def test_prepare_resume_writes_the_deck_and_cuts_back(self):
        (self.wd / "nemd.in").write_text(build_input_deck(self.spec))
        fake_outputs(self.wd, self.spec, 4300)
        (self.wd / "job.done").write_text("137\n")
        for s in (3000, 4000):
            (self.wd / f"restart.{s}.bin").write_bytes(b"x")
        self.assertEqual(prepare_resume(self.wd, self.spec), 4000)
        r = (self.wd / "resume.in").read_text()
        self.assertIn("f_hot+40.0", r)
        self.assertFalse((self.wd / "job.done").exists())

    def test_no_checkpoint_means_start_over(self):
        (self.wd / "nemd.in").write_text(build_input_deck(self.spec))
        fake_outputs(self.wd, self.spec, 4300)
        self.assertIsNone(prepare_resume(self.wd, self.spec))
        self.assertFalse((self.wd / "resume.in").exists())


class JobLoop(unittest.TestCase):
    """A local job killed mid-production is resubmitted from its checkpoint."""

    def test_killed_job_resumes(self):
        from openkapitza.agent.jobs import JobBook
        from openkapitza.compute import runner
        spec = small_spec()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "jobs"
            wd = root / "sige"
            wd.mkdir(parents=True)
            (wd / "nemd.in").write_text(build_input_deck(spec))
            write_spec(wd, spec)
            (root / "manifest.json").write_text(json.dumps({"jobs": [{"name": "sige"}]}))
            # what a run killed after its 4,000-step checkpoint leaves behind
            fake_outputs(wd, spec, 4300)
            for st in (3000, 4000):
                (wd / f"restart.{st}.bin").write_bytes(b"x")
            fake = Path(d) / "lmp"
            fake.write_text(                 # first call dies; the second records its arguments
                "#!/bin/bash\n"
                "if [ ! -f .ran ]; then touch .ran; echo starting > log.lammps; kill -9 $$; fi\n"
                "echo \"$@\" > args.txt\n")
            fake.chmod(0o755)
            book = JobBook.open(root)
            book.submit(["sige"], backend="local", lmp=str(fake))
            events, t0 = [], time.time()
            try:
                while time.time() - t0 < 30:
                    book = JobBook.open(root)
                    events += book.refresh()
                    if (wd / "args.txt").exists() and \
                            book.entries["sige"].status not in ("submitted", "running"):
                        break
                    time.sleep(0.1)
            finally:
                for p in runner._LAUNCHED:
                    if p.poll() is None:
                        p.kill()
                    p.wait(timeout=10)
                runner._LAUNCHED.clear()
            res = next(e for e in events if e.what == "resubmitted")
            self.assertIn("resumes from its checkpoint at production step 4,000", res.message)
            self.assertEqual((wd / "args.txt").read_text().split(), ["-in", "resume.in", "-log", "none"])
            self.assertTrue((wd / "resume.in").is_file())
            self.assertFalse(any(wd.glob("*.attempt*")))     # outputs continued, not moved aside

    def test_resume_sbatch_points_at_resume_deck(self):
        from openkapitza.agent.jobs import _resume_script
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "submit.sbatch"
            s.write_text("#!/bin/bash\nsrun lmp -in nemd.in -log log.lammps\n")
            self.assertIn("srun lmp -in resume.in -log none\n", _resume_script(s).read_text())
            s.write_text("#!/bin/bash\nsrun lmp -in nemd.in\n")
            self.assertIn("-in resume.in -log none", _resume_script(s).read_text())


LMP = os.environ.get("OPENKAPITZA_TEST_LMP")
POT = os.environ.get("OPENKAPITZA_TEST_POTENTIALS")


@unittest.skipUnless(LMP and POT, "set OPENKAPITZA_TEST_LMP and OPENKAPITZA_TEST_POTENTIALS")
class EndToEnd(unittest.TestCase):
    """Real LAMMPS: kill a Si/Ge run after its mid-production checkpoint, resume, measure.
    About two minutes on one core."""

    def test_kill_and_resume(self):
        from openkapitza.md.measure import measure_kapitza
        from openkapitza.schema import FlagSet
        spec = small_spec()
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            (wd / "nemd.in").write_text(build_input_deck(spec))
            shutil.copy(Path(POT) / "SiCGe.tersoff", wd)
            p = subprocess.Popen([LMP, "-in", "nemd.in", "-log", "log.lammps", "-screen", "none"],
                                 cwd=wd)
            while p.poll() is None:
                if any(s >= 4000 for s, _ in checkpoints(wd)):
                    time.sleep(0.3)
                    p.send_signal(signal.SIGKILL)
                    break
                time.sleep(0.05)
            p.wait()
            self.assertEqual(prepare_resume(wd, spec), 4000)
            subprocess.run([LMP, "-in", "resume.in", "-log", "none", "-screen", "none"],
                           cwd=wd, check=True)
            log = (wd / "log.lammps").read_text()
            self.assertIn("OPENKAPITZA_RESUMED step 4000", log)
            self.assertIn("OPENKAPITZA_DONE", log)
            t = [float(l.split()[0]) for l in (wd / "reservoir_energy.dat").read_text().splitlines()
                 if l and not l.startswith("#")]
            self.assertEqual(t, [round(0.05 * i, 10) for i in range(161)])   # continuous, no repeats
            r = measure_kapitza(wd, spec, FlagSet())
            self.assertGreater(r.h_K, 1e8)


if __name__ == "__main__":
    unittest.main()
