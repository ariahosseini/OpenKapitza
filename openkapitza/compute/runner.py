"""Job submission, for a local machine or a SLURM cluster.

THE ARCHITECTURAL CONSTRAINT. A NEMD run takes hours to days. Nothing in this
package may block waiting for one. Every runner therefore submits and returns
immediately with a handle; progress is learned by polling, and the handle is
JSON-serialisable so a session that ends can pick the job up again later.

That is what lets the agent layer put a long run behind an interrupt and
resume when an external poller sees it finish, rather than holding a process
open for three days.

Backends are chosen by what is actually present, never assumed: `available()`
probes for sbatch before a SlurmRunner will accept work.
"""

from __future__ import annotations

import enum
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


class JobState(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        return self in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)


@dataclass
class Resources:
    """What a job needs. Ignored by the local backend except for `cpus`."""

    nodes: int = 1
    tasks: int = 1
    cpus_per_task: int = 1
    walltime_hours: float = 24.0
    partition: str | None = None
    account: str | None = None
    constraint: str | None = None
    gpus: int = 0

    @property
    def total_ranks(self) -> int:
        return self.nodes * self.tasks

    @property
    def node_hours(self) -> float:
        return self.nodes * self.walltime_hours


@dataclass
class JobSpec:
    name: str
    command: str                       # shell command to run
    workdir: Path
    resources: Resources = field(default_factory=Resources)
    env: dict[str, str] = field(default_factory=dict)
    modules: list[str] = field(default_factory=list)   # `module load` lines

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)


@dataclass
class JobHandle:
    """Durable reference to a submitted job. Survives the session."""

    job_id: str
    backend: str
    name: str
    workdir: str
    submitted_at: float
    state: str = JobState.PENDING.value
    native_id: str | None = None       # SLURM job id, or local pid
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "JobHandle":
        return cls(**json.loads(text))

    def save(self, path: Path | None = None) -> Path:
        p = Path(path) if path else Path(self.workdir) / "job_handle.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path) -> "JobHandle":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass
class JobResult:
    handle: JobHandle
    state: JobState
    returncode: int | None
    stdout_path: Path | None
    stderr_path: Path | None
    duration_s: float | None = None

    @property
    def ok(self) -> bool:
        return self.state is JobState.COMPLETED and (self.returncode in (0, None))


@runtime_checkable
class JobRunner(Protocol):
    """The seam between the physics and wherever it runs."""

    name: str

    def available(self) -> bool: ...
    def submit(self, spec: JobSpec) -> JobHandle: ...
    def poll(self, handle: JobHandle) -> JobState: ...
    def collect(self, handle: JobHandle) -> JobResult: ...
    def cancel(self, handle: JobHandle) -> None: ...


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------

class LocalRunner:
    """Runs on this machine, detached, so submit() still returns immediately.

    Intended for small Tier-1 screening runs and for exercising the whole
    pipeline without a cluster. A production NEMD campaign does not belong
    here; the cost model will say so.
    """

    name = "local"

    def __init__(self, shell: str = "/bin/bash") -> None:
        self.shell = shell

    def available(self) -> bool:
        return Path(self.shell).exists()

    def submit(self, spec: JobSpec) -> JobHandle:
        spec.workdir.mkdir(parents=True, exist_ok=True)
        out = spec.workdir / "stdout.log"
        err = spec.workdir / "stderr.log"
        done = spec.workdir / "job.done"
        if done.exists():
            done.unlink()

        # The wrapper records an exit status so poll() can tell completed from
        # failed after the process is gone, which a bare pid cannot.
        #
        # Two details matter and both were found the hard way. The command runs
        # in a SUBSHELL, because a command containing `exit N` would otherwise
        # terminate the wrapper before the sentinel line ran, leaving a job
        # that had plainly finished looking like it never reported. And a trap
        # on EXIT writes the sentinel even when the script dies on a signal,
        # so a killed or timed-out job is still distinguishable from a running
        # one.
        # not run.sh: prepared job folders already hold a run.sh for manual use
        script = spec.workdir / "job_wrapper.sh"
        env_lines = "\n".join(
            f"export {k}={shlex.quote(v)}" for k, v in spec.env.items()
        )
        done_q = shlex.quote(str(done))
        script.write_text(
            f"#!{self.shell}\n"
            f"set -o pipefail\n"
            f"__rc=1\n"
            f"trap 'echo $__rc > {done_q}' EXIT\n"
            f"{env_lines}\n"
            f"cd {shlex.quote(str(spec.workdir))}\n"
            f"(\n{spec.command}\n)\n"
            f"__rc=$?\n"
            f"exit $__rc\n",
            encoding="utf-8",
        )
        script.chmod(0o755)

        # Close our copies once Popen has dup'd them into the child, or the
        # parent leaks a descriptor per submitted job (ResourceWarning under
        # -W default, and eventually an fd exhaustion in a long campaign).
        out_fh = open(out, "w", encoding="utf-8")
        err_fh = open(err, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [self.shell, str(script)],
                stdout=out_fh,
                stderr=err_fh,
                cwd=str(spec.workdir),
                start_new_session=True,
            )
        finally:
            out_fh.close()
            err_fh.close()
        _LAUNCHED.append(proc)
        handle = JobHandle(
            job_id=f"local-{uuid.uuid4().hex[:12]}",
            backend=self.name,
            name=spec.name,
            workdir=str(spec.workdir),
            submitted_at=time.time(),
            state=JobState.RUNNING.value,
            native_id=str(proc.pid),
        )
        handle.save()
        return handle

    def poll(self, handle: JobHandle) -> JobState:
        done = Path(handle.workdir) / "job.done"
        if handle.native_id:
            _reap(int(handle.native_id))
        if done.exists():
            try:
                rc = int(done.read_text().strip() or "1")
            except ValueError:
                rc = 1
            state = JobState.COMPLETED if rc == 0 else JobState.FAILED
        elif handle.native_id and _pid_alive(int(handle.native_id)):
            state = JobState.RUNNING
        else:
            # Process gone with no sentinel: killed, or the node went away.
            state = JobState.FAILED
        handle.state = state.value
        handle.save()
        return state

    def collect(self, handle: JobHandle) -> JobResult:
        state = self.poll(handle)
        wd = Path(handle.workdir)
        done = wd / "job.done"
        rc = None
        if done.exists():
            try:
                rc = int(done.read_text().strip())
            except ValueError:
                rc = None
        return JobResult(
            handle=handle,
            state=state,
            returncode=rc,
            stdout_path=wd / "stdout.log" if (wd / "stdout.log").exists() else None,
            stderr_path=wd / "stderr.log" if (wd / "stderr.log").exists() else None,
            duration_s=time.time() - handle.submitted_at,
        )

    def cancel(self, handle: JobHandle) -> None:
        if handle.native_id:
            try:
                os.killpg(os.getpgid(int(handle.native_id)), 15)
            except (ProcessLookupError, PermissionError, ValueError):
                pass
        handle.state = JobState.CANCELLED.value
        handle.save()


# Popen objects for jobs this process launched. A detached job is still
# running when submit() returns, and Python emits a ResourceWarning if the
# Popen is garbage-collected in that state. Holding the reference silences a
# warning that describes intended behaviour, and keeps this process able to
# reap the child in _pid_alive once it exits.
_LAUNCHED: list[subprocess.Popen] = []


def _reap(pid: int) -> None:
    """Collect a finished child this process launched, so its Popen holds the exit
    status and is released (left alone, each finished job stays a zombie with a
    Popen that believes it is still running)."""
    for proc in list(_LAUNCHED):
        if proc.pid == pid and proc.poll() is not None:
            _LAUNCHED.remove(proc)


def _pid_alive(pid: int) -> bool:
    """Is this pid a live process?

    `os.kill(pid, 0)` is not enough. A child this process started and never
    reaped stays in the table as a ZOMBIE, and signalling a zombie succeeds --
    so a job that exited seconds ago is reported as RUNNING forever and any
    loop waiting on it hangs.

    This matters only when the sentinel file is absent, since poll() checks
    that first and the EXIT trap writes it even on a trappable signal. SIGKILL
    cannot be trapped, so that path is reachable and must be handled.

    Three steps, because no single one is portable:

    1. Reap first. If this process is the parent, waitpid() clears the zombie
       outright and the question answers itself.
    2. Ask the OS for the process state: /proc on Linux, `ps` on macOS and the
       BSDs, where /proc does not exist. An earlier version did only the
       /proc check, which silently degraded to the broken os.kill() path on
       macOS -- found by running the suite on a Mac.
    3. Fall back to os.kill(), which is right except for zombies.
    """
    # 0. A job this process launched: let its Popen reap it, so the Popen
    #    learns the exit status (reaping behind its back with waitpid left
    #    every finished job's Popen believing it still ran, and leaked).
    for proc in list(_LAUNCHED):
        if proc.pid == pid:
            if proc.poll() is None:
                return True
            _LAUNCHED.remove(proc)
            return False

    # 1. Reap, if we happen to be the parent.
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except (ChildProcessError, OSError):
        pass

    # 2a. Linux.
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as fh:
            fields = fh.read().rsplit(")", 1)[-1].split()
        return bool(fields) and fields[0] != "Z"
    except (FileNotFoundError, PermissionError, IndexError, OSError):
        pass

    # 2b. macOS and the BSDs: `ps` reports Z for a zombie.
    if sys.platform != "win32":
        try:
            out = subprocess.run(
                ["ps", "-o", "state=", "-p", str(pid)],
                capture_output=True, text=True, timeout=5,
            )
            state = out.stdout.strip()
            if state:
                return not state.startswith("Z")
            if out.returncode != 0:
                return False        # ps knows of no such process
        except (OSError, subprocess.SubprocessError):
            pass

    # 3. Last resort.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------

# sacct state strings mapped onto ours. Anything unmapped stays UNKNOWN rather
# than being guessed at, since treating a live job as failed would throw away
# days of compute.
_SACCT_STATES = {
    "PENDING": JobState.PENDING,
    "CONFIGURING": JobState.PENDING,
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "COMPLETED": JobState.COMPLETED,
    "FAILED": JobState.FAILED,
    "NODE_FAIL": JobState.FAILED,
    "OUT_OF_MEMORY": JobState.FAILED,
    "TIMEOUT": JobState.FAILED,
    "CANCELLED": JobState.CANCELLED,
    "PREEMPTED": JobState.CANCELLED,
}


class SlurmRunner:
    """Submits with sbatch and tracks with sacct."""

    name = "slurm"

    def __init__(self, submit_cmd: str = "sbatch", status_cmd: str = "sacct") -> None:
        self.submit_cmd = submit_cmd
        self.status_cmd = status_cmd

    def available(self) -> bool:
        from shutil import which
        return which(self.submit_cmd) is not None

    def script(self, spec: JobSpec) -> str:
        r = spec.resources
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={spec.name}",
            f"#SBATCH --nodes={r.nodes}",
            f"#SBATCH --ntasks-per-node={r.tasks}",
            f"#SBATCH --cpus-per-task={r.cpus_per_task}",
            f"#SBATCH --time={_walltime(r.walltime_hours)}",
            f"#SBATCH --output={spec.workdir / 'stdout.log'}",
            f"#SBATCH --error={spec.workdir / 'stderr.log'}",
        ]
        if r.partition:
            lines.append(f"#SBATCH --partition={r.partition}")
        if r.account:
            lines.append(f"#SBATCH --account={r.account}")
        if r.constraint:
            lines.append(f"#SBATCH --constraint={r.constraint}")
        if r.gpus:
            lines.append(f"#SBATCH --gres=gpu:{r.gpus}")
        lines.append("")
        lines.append("set -o pipefail")
        lines.extend(f"module load {m}" for m in spec.modules)
        lines.extend(f"export {k}={shlex.quote(v)}" for k, v in spec.env.items())
        lines.append(f"cd {shlex.quote(str(spec.workdir))}")
        lines.append("")
        lines.append(spec.command)
        return "\n".join(lines) + "\n"

    def submit(self, spec: JobSpec) -> JobHandle:
        spec.workdir.mkdir(parents=True, exist_ok=True)
        path = spec.workdir / "submit.sbatch"
        path.write_text(self.script(spec), encoding="utf-8")

        proc = subprocess.run(
            [self.submit_cmd, str(path)],
            capture_output=True, text=True, cwd=str(spec.workdir),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
        m = re.search(r"(\d+)", proc.stdout)
        if not m:
            raise RuntimeError(f"could not parse a job id from: {proc.stdout!r}")
        jid = m.group(1)

        handle = JobHandle(
            job_id=f"slurm-{jid}",
            backend=self.name,
            name=spec.name,
            workdir=str(spec.workdir),
            submitted_at=time.time(),
            state=JobState.PENDING.value,
            native_id=jid,
            meta={"script": str(path), "node_hours": spec.resources.node_hours},
        )
        handle.save()
        return handle

    def submit_script(self, script: Path, name: str) -> JobHandle:
        """sbatch a script that already exists (prepare_cluster writes them)."""
        script = Path(script)
        proc = subprocess.run([self.submit_cmd, str(script)], capture_output=True,
                              text=True, cwd=str(script.parent))
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
        m = re.search(r"(\d+)", proc.stdout)
        if not m:
            raise RuntimeError(f"could not parse a job id from: {proc.stdout!r}")
        handle = JobHandle(job_id=f"slurm-{m.group(1)}", backend=self.name, name=name,
                           workdir=str(script.parent), submitted_at=time.time(),
                           state=JobState.PENDING.value, native_id=m.group(1),
                           meta={"script": str(script)})
        handle.save()
        return handle

    def poll(self, handle: JobHandle) -> JobState:
        proc = subprocess.run(
            [self.status_cmd, "-j", str(handle.native_id),
             "--format=State", "--noheader", "--parsable2"],
            capture_output=True, text=True,
        )
        state = JobState.UNKNOWN
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                token = line.strip().split()[0] if line.strip() else ""
                if token:
                    # keep the raw state: TIMEOUT and FAILED both map to FAILED,
                    # but only one of them is fixed by resubmitting
                    handle.meta["sacct_state"] = token
                    state = _SACCT_STATES.get(token, JobState.UNKNOWN)
                    break
        handle.state = state.value
        handle.save()
        return state

    def collect(self, handle: JobHandle) -> JobResult:
        state = self.poll(handle)
        wd = Path(handle.workdir)
        return JobResult(
            handle=handle,
            state=state,
            returncode=0 if state is JobState.COMPLETED else None,
            stdout_path=wd / "stdout.log" if (wd / "stdout.log").exists() else None,
            stderr_path=wd / "stderr.log" if (wd / "stderr.log").exists() else None,
            duration_s=time.time() - handle.submitted_at,
        )

    def cancel(self, handle: JobHandle) -> None:
        subprocess.run(["scancel", str(handle.native_id)], capture_output=True)
        handle.state = JobState.CANCELLED.value
        handle.save()


def _walltime(hours: float) -> str:
    total = int(round(hours * 3600))
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"


def select_runner(prefer: str | None = None) -> JobRunner:
    """Pick a backend by what is actually present.

    SLURM wins when sbatch exists, because a production run belongs there.
    `prefer` forces a choice, and raises if that backend is unavailable rather
    than silently falling back -- a campaign quietly running on a laptop
    instead of the cluster is worse than an error.
    """
    runners = {"slurm": SlurmRunner(), "local": LocalRunner()}
    if prefer:
        r = runners.get(prefer)
        if r is None:
            raise ValueError(f"unknown backend {prefer!r}")
        if not r.available():
            raise RuntimeError(
                f"backend {prefer!r} requested but not available on this machine"
            )
        return r
    if runners["slurm"].available():
        return runners["slurm"]
    return runners["local"]
