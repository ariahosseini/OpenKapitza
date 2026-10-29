"""The MD job loop: submit prepared jobs, watch them, diagnose, read results back.

    book = JobBook.open("cluster_jobs")          # a root written by prepare_cluster
    book.submit(["sial_pp"], backend="local", lmp="lmp")
    ...                                          # later, from any process
    events = book.refresh()                      # poll; diagnose what finished

Nothing here waits for a job (the runner contract, compute/runner.py): every
call polls once and returns. State lives in <root>/jobs.json, so a session
that ends, a cron poller and a new session all see the same book.

What happens when a job stops (`refresh`):

  completed, output present   -> ready to ingest (the session does it at once)
  completed, no output        -> "output": the run exited 0 without writing the
                                 profile, or stopped before OPENKAPITZA_DONE
  failed, cluster cause       -> "infrastructure" (timeout, node failure,
                                 preemption, out of memory, killed). Resubmitted
                                 AUTOMATICALLY up to MAX_ATTEMPTS; a timeout gets
                                 walltime x WALLTIME_GROWTH.
  failed, LAMMPS ERROR line   -> "input": the error, the likely cause, and the
                                 best knowledge-base / manual passage. Not
                                 resubmitted: the deck must change.
  failed, physics             -> "physics" (lost atoms, NaN, out-of-range atoms):
                                 proposed fixes only. A smaller timestep or a
                                 different structure changes what is measured,
                                 so a person decides.

The rule: the loop may retry what the cluster broke, never what the physics
or the input broke.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..compute.runner import JobHandle, JobSpec, JobState, LocalRunner, SlurmRunner

BOOK_FILE = "jobs.json"
MAX_ATTEMPTS = 3
WALLTIME_GROWTH = 1.5
WALLTIME_CAP_HOURS = 72.0
DONE_MARKER = "OPENKAPITZA_DONE"
OUTPUTS = ("temperature_profile.dat",)


class JobError(ValueError):
    pass


@dataclass
class Diagnosis:
    kind: str                 # ok | output | infrastructure | input | physics | unknown
    summary: str
    detail: str = ""
    evidence: str = ""        # the log line that decided it
    auto_resubmit: bool = False
    cause: str = ""           # infrastructure: timeout | node | memory | killed
    fixes: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


@dataclass
class JobEntry:
    name: str                              # as in manifest.json ("campaign/r03" etc.)
    workdir: str
    backend: str = ""
    status: str = "prepared"               # prepared | submitted | running | completed |
                                           # failed | cancelled | ingested | gave_up
    attempts: int = 0
    handle: dict | None = None
    diagnosis: dict | None = None
    history: list[str] = field(default_factory=list)
    node_hours: float | None = None
    atoms: int | None = None
    steps: int | None = None

    def log(self, msg: str) -> None:
        self.history.append(f"{time.strftime('%Y-%m-%d %H:%M')} {msg}")


@dataclass
class Event:
    name: str
    what: str                 # finished | failed | resubmitted | gave_up | started
    message: str
    diagnosis: Diagnosis | None = None


class JobBook:
    def __init__(self, root: Path, entries: dict[str, JobEntry], config: dict):
        self.root = Path(root)
        self.entries = entries
        self.config = config               # backend, lmp, local launcher

    # ---- persistence -----------------------------------------------------------
    @classmethod
    def open(cls, root: Path | str) -> "JobBook":
        root = Path(root)
        path = root / BOOK_FILE
        if path.is_file():
            d = json.loads(path.read_text(encoding="utf-8"))
            entries = {k: JobEntry(**v) for k, v in d["entries"].items()}
            book = cls(root, entries, d.get("config", {}))
        else:
            book = cls(root, {}, {})
        book._sync_manifest()
        return book

    def save(self) -> Path:
        path = self.root / BOOK_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"config": self.config,
                                   "entries": {k: asdict(v) for k, v in self.entries.items()}},
                                  indent=2), encoding="utf-8")
        tmp.replace(path)                  # atomic: a poller never reads half a file
        return path

    def _sync_manifest(self) -> None:
        m = self.root / "manifest.json"
        if not m.is_file():
            raise JobError(f"{self.root}: no manifest.json; write the jobs first "
                           "('prepare jobs <dir>')")
        data = json.loads(m.read_text(encoding="utf-8"))
        for j in data.get("jobs", []):
            if j["name"] not in self.entries:
                self.entries[j["name"]] = JobEntry(
                    name=j["name"], workdir=str(self.root / j["name"]),
                    node_hours=j.get("node_hours"), atoms=j.get("atoms"),
                    steps=j.get("steps"))

    # ---- selection -------------------------------------------------------------
    def select(self, names: list[str]) -> list[JobEntry]:
        """Exact names, a prefix ("campaign" -> every rung), or 'all'."""
        if names == ["all"]:
            return list(self.entries.values())
        out, unknown = [], []
        for n in names:
            hit = [e for e in self.entries.values()
                   if e.name == n or e.name.startswith(n.rstrip("/") + "/")]
            (out.extend(hit) if hit else unknown.append(n))
        if unknown:
            raise JobError(f"no prepared job named {', '.join(unknown)}; 'jobs' lists them")
        return list({e.name: e for e in out}.values())

    # ---- submission ------------------------------------------------------------
    def submit(self, names: list[str], backend: str | None = None, lmp: str | None = None,
               launcher: str = "", resubmit: bool = False) -> list[Event]:
        backend = backend or self.config.get("backend") or _default_backend()
        if lmp:
            self.config["lmp"] = lmp
        if launcher:
            self.config["launcher"] = launcher
        self.config["backend"] = backend
        events = []
        for e in self.select(names):
            if e.status in ("submitted", "running") and not resubmit:
                events.append(Event(e.name, "skipped", "already running"))
                continue
            if e.status in ("completed", "ingested") and not resubmit:
                events.append(Event(e.name, "skipped", "already finished"))
                continue
            events.append(self._launch(e, backend))
        self.save()
        return events

    def _launch(self, e: JobEntry, backend: str, walltime_hours: float | None = None) -> Event:
        wd = Path(e.workdir)
        if not (wd / "nemd.in").is_file():
            raise JobError(f"{e.name}: {wd}/nemd.in not found")
        for stale in ("job.done", "log.lammps", "stdout.log", "stderr.log"):
            p = wd / stale
            if p.exists() and e.attempts:
                p.rename(wd / f"{stale}.attempt{e.attempts}")
        if backend == "slurm":
            runner = SlurmRunner()
            if not runner.available():
                raise JobError("SLURM is not available here (no sbatch on PATH). Run on the "
                               "cluster login node, or submit with 'local'.")
            script = wd / "submit.sbatch"
            if walltime_hours:
                _set_walltime(script, walltime_hours)
            handle = runner.submit_script(script, e.name)
        else:
            lmp = self.config.get("lmp") or "lmp"
            if shutil.which(lmp.split()[0]) is None and not Path(lmp.split()[0]).is_file():
                raise JobError(f"LAMMPS executable {lmp!r} not found; give it with "
                               "'submit ... lmp=/path/to/lmp'")
            cmd = f"{self.config.get('launcher', '')} {lmp} -in nemd.in -log log.lammps".strip()
            handle = LocalRunner().submit(JobSpec(name=e.name, command=cmd, workdir=wd))
        e.handle = json.loads(handle.to_json())
        e.backend, e.status, e.diagnosis = backend, "submitted", None
        e.attempts += 1
        e.log(f"submitted ({backend}, attempt {e.attempts}, id {handle.native_id})")
        msg = f"submitted on {backend} (id {handle.native_id})"
        if backend == "local":
            msg += _local_note(e, self.config.get("launcher", ""))
        return Event(e.name, "started", msg)

    def cancel(self, names: list[str]) -> list[Event]:
        events = []
        for e in self.select(names):
            if e.status not in ("submitted", "running") or not e.handle:
                continue
            h = JobHandle(**e.handle)
            (SlurmRunner() if e.backend == "slurm" else LocalRunner()).cancel(h)
            e.status = "cancelled"
            e.log("cancelled")
            events.append(Event(e.name, "cancelled", "cancelled"))
        self.save()
        return events

    # ---- polling -----------------------------------------------------------------
    def refresh(self, auto_resubmit: bool = True) -> list[Event]:
        """Poll every live job once; diagnose and act on what stopped."""
        events = []
        for e in self.entries.values():
            if e.status not in ("submitted", "running") or not e.handle:
                continue
            h = JobHandle(**e.handle)
            runner = SlurmRunner() if e.backend == "slurm" else LocalRunner()
            state = runner.poll(h)
            e.handle = json.loads(h.to_json())
            if state in (JobState.PENDING, JobState.UNKNOWN):
                continue
            if state is JobState.RUNNING:
                if e.status != "running":
                    e.status = "running"
                    e.log("running")
                continue
            d = diagnose(Path(e.workdir), state, h)
            e.diagnosis = asdict(d)
            if d.kind == "ok":
                e.status = "completed"
                e.log("completed")
                events.append(Event(e.name, "finished", d.summary, d))
            elif d.auto_resubmit and auto_resubmit and e.attempts < MAX_ATTEMPTS:
                wall = None
                if d.cause == "timeout" and e.backend == "slurm":
                    wall = min(WALLTIME_CAP_HOURS,
                               _get_walltime(Path(e.workdir) / "submit.sbatch") * WALLTIME_GROWTH)
                e.log(f"failed: {d.summary}; resubmitting")
                self._launch(e, e.backend, walltime_hours=wall)
                extra = f" with walltime {wall:.1f} h" if wall else ""
                events.append(Event(e.name, "resubmitted",
                                    f"{d.summary}. Resubmitted{extra} (attempt {e.attempts} of "
                                    f"{MAX_ATTEMPTS}); the deck has no restart files, so it "
                                    "starts over.", d))
            else:
                e.status = "gave_up" if d.auto_resubmit else "failed"
                e.log(f"failed: {d.summary}")
                why = (f" after {e.attempts} attempts" if d.auto_resubmit else "")
                events.append(Event(e.name, "failed", d.summary + why, d))
        self.save()
        return events

    def mark_ingested(self, name: str) -> None:
        e = self.entries[name]
        e.status = "ingested"
        e.log("ingested")
        self.save()

    # ---- display -----------------------------------------------------------------
    def table(self, only_active: bool = False) -> str:
        rows = [e for e in self.entries.values()
                if not only_active or e.status not in ("prepared",)]
        if not rows:
            return "No jobs submitted yet."
        lines = [f"  {'job':40} {'status':10} {'backend':7} {'try':>3}  note"]
        for e in sorted(rows, key=lambda x: x.name):
            note = (e.diagnosis or {}).get("summary", "") if e.status not in ("prepared",) else \
                (f"{e.node_hours:.1f} node-h" if e.node_hours is not None else "")
            lines.append(f"  {e.name:40} {e.status:10} {e.backend or '-':7} "
                         f"{e.attempts:3d}  {note[:70]}")
        return "\n".join(lines)


# ---- diagnosis -------------------------------------------------------------------
# (pattern in log/stderr, kind, summary, fixes, cause). First match wins.
_RULES: list[tuple[re.Pattern, str, str, list[str], str]] = [
    (re.compile(r"DUE TO TIME LIMIT|TIMEOUT", re.I), "infrastructure",
     "Hit the walltime limit", ["Resubmit with a longer walltime (done automatically)."],
     "timeout"),
    (re.compile(r"NODE_FAIL|node failure|PREEMPT", re.I), "infrastructure",
     "Node failure or preemption", ["Resubmit unchanged (done automatically)."], "node"),
    (re.compile(r"OUT_OF_MEMORY|oom-kill|Out Of Memory|Cannot allocate memory|bad_alloc",
                re.I), "infrastructure", "Out of memory",
     ["Resubmit on more nodes or a high-memory partition (resubmitted unchanged first)."],
     "memory"),
    (re.compile(r"Lost atoms|Out of range atoms|Bond atoms? .* missing|Non-numeric atom coords",
                re.I), "physics",
     "Atoms were lost or flew apart",
     ["Check the relaxed structure (relaxed.data) for overlapping atoms at the interface.",
      "Halve the timestep (timestep_fs) for this pair, re-prepare and resubmit.",
      "Reduce the applied heat flux / temperature difference if the hot side overheats."], ""),
    (re.compile(r"^\s*\d+\s+[^\n]*?(?<![\w.])(-?nan|-?inf)(?![\w.])", re.I | re.M), "physics",
     "Thermo output went to NaN/inf",
     ["Same causes as lost atoms: overlapping atoms, too large a timestep, or a potential "
      "used outside its fitted range."], ""),
]
_ERROR_LINE = re.compile(r"^(ERROR(?: on proc \d+)?:.*)$", re.M)


def _read(p: Path, limit: int = 400_000) -> str:
    if not p.is_file():
        return ""
    data = p.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def diagnose(wd: Path, state: JobState, handle: JobHandle | None = None) -> Diagnosis:
    """Why a stopped job stopped, from its state, logs and outputs."""
    log = _read(wd / "log.lammps")
    err = _read(wd / "stderr.log")
    out = _read(wd / "stdout.log")
    native = (handle.meta.get("sacct_state", "") if handle else "") or ""
    text = "\n".join((native, err, log[-20000:], out[-20000:]))

    if state is JobState.COMPLETED:
        missing = [f for f in OUTPUTS if not (wd / f).is_file()]
        if DONE_MARKER not in log and DONE_MARKER not in out:
            return Diagnosis("output", f"Exited 0 but never reached {DONE_MARKER}",
                             "The run stopped early without an error, or the log was not "
                             "captured. Nothing to measure.", fixes=[
                                 "Look at the end of log.lammps; resubmit if it was cut off."])
        if missing or _header_only(wd / OUTPUTS[0]):
            return _with_citations(Diagnosis(
                "output", "Finished, but the temperature profile is empty",
                "LAMMPS exits 0 when fix ave/chunk never writes a frame (known silent "
                "failure).", fixes=["Check profile_every × profile_frames against the "
                                    "production length."]),
                "temperature_profile.dat contains only its header")
        return Diagnosis("ok", "Finished; output present")

    if state is JobState.CANCELLED and "PREEMPT" not in native.upper():
        return Diagnosis("cancelled", "Cancelled")

    m = _ERROR_LINE.search(log) or _ERROR_LINE.search(out) or _ERROR_LINE.search(err)
    for pat, kind, summary, fixes, cause in _RULES:
        hit = pat.search(text)
        if hit and (kind != "physics" or not m or pat.search(m.group(1))):
            line = _line_of(text, hit.start())
            d = Diagnosis(kind, summary, evidence=line, fixes=list(fixes), cause=cause,
                          auto_resubmit=(kind == "infrastructure"))
            return _with_citations(d, line) if kind == "physics" else d
    if m:
        line = m.group(1).strip()
        d = Diagnosis("input", f"LAMMPS stopped: {line[:120]}", evidence=line,
                      fixes=["Fix the deck or the files it reads, then resubmit. "
                             "Not retried automatically: the same input fails the same way."])
        return _with_citations(d, re.sub(r"^ERROR(?: on proc \d+)?:\s*", "", line))
    rc = None
    if (wd / "job.done").is_file():
        try:
            rc = int((wd / "job.done").read_text().strip())
        except ValueError:
            pass
    if rc in (137, 143) or "Killed" in err or "SIGKILL" in err or "signal 9" in err:
        return Diagnosis("infrastructure", "Killed by a signal (out of memory or the "
                         "scheduler)", evidence=f"exit status {rc}", auto_resubmit=True, cause="killed",
                         fixes=["Resubmit (done automatically); if it repeats, more memory."])
    return Diagnosis("unknown", f"Stopped ({state.value}) with no error in the logs"
                     + (f", exit status {rc}" if rc is not None else ""),
                     fixes=["Read stderr.log and the end of log.lammps."])


def _header_only(p: Path) -> bool:
    if not p.is_file():
        return True
    return not any(ln.strip() and not ln.startswith("#") for ln in
                   p.read_text(errors="replace").splitlines())


def _line_of(text: str, pos: int) -> str:
    a = text.rfind("\n", 0, pos) + 1
    b = text.find("\n", pos)
    return text[a: b if b >= 0 else None].strip()[:200]


def _with_citations(d: Diagnosis, query: str, k: int = 2) -> Diagnosis:
    """Attach the best knowledge-base / manual passages for the error text."""
    try:
        from .. import rag
        hits = rag.search(query, k=k)
    except Exception:                  # retrieval is a help, never a reason to fail
        return d
    d.citations = [h.chunk.citation() + (f"  {h.chunk.url}" if h.chunk.url else "")
                   for h in hits]
    return d


# ---- SLURM script editing -----------------------------------------------------------
_TIME = re.compile(r"^#SBATCH --time=(\S+)\s*$", re.M)


def _get_walltime(script: Path) -> float:
    m = _TIME.search(script.read_text())
    if not m:
        return 24.0
    t = m.group(1)
    days, _, hms = t.rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days or 0) * 24 + parts[0] + parts[1] / 60 + parts[2] / 3600)


def _set_walltime(script: Path, hours: float) -> None:
    from ..compute.runner import _walltime
    s = script.read_text()
    new = f"#SBATCH --time={_walltime(hours)}"
    s = _TIME.sub(new, s) if _TIME.search(s) else s.replace("#!/bin/bash\n", f"#!/bin/bash\n{new}\n", 1)
    script.write_text(s)


def _local_note(e: JobEntry, launcher: str) -> str:
    """'; ~6.2 h here at np=5 (measured 2026-10-28)' when this machine was benchmarked."""
    from ..compute.resources import local_estimate
    if not e.atoms or not e.steps:
        return ""
    try:
        from ..md.campaign import read_spec
        style = read_spec(Path(e.workdir))[0].pair_style
    except Exception:                       # no spec: no estimate, never a failure
        return ""
    est = local_estimate(e.atoms, e.steps, style)
    if est is None:
        return ("; no measured speed for this machine (scripts/benchmark_local.py gives one)")
    m = re.search(r"-np\s+(\d+)", launcher)
    ranks = f"np={m.group(1)}" if m else "serial"
    tip = "" if m and int(m.group(1)) == est["np"] else f"; fastest at np={est['np']}"
    return (f"; ~{est['hours']:.1f} h here at np={est['np']} (measured {est['measured']}), "
            f"running {ranks}{tip}")


def _default_backend() -> str:
    return "slurm" if SlurmRunner().available() else "local"


# ---- command line: a poller for cron / scheduled tasks ---------------------------------
def main(argv: list[str] | None = None) -> int:
    """python -m openkapitza.agent.jobs refresh ROOT [--no-resubmit] [--ingest]"""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m openkapitza.agent.jobs", description=main.__doc__)
    ap.add_argument("cmd", choices=("refresh", "status"))
    ap.add_argument("root", type=Path)
    ap.add_argument("--no-resubmit", action="store_true")
    a = ap.parse_args(argv)
    book = JobBook.open(a.root)
    if a.cmd == "refresh":
        for ev in book.refresh(auto_resubmit=not a.no_resubmit):
            print(f"{ev.name}: {ev.what} — {ev.message}")
    print(book.table(only_active=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
