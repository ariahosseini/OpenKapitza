"""Resume a killed NEMD production run from its last restart file.

The NEMD deck (nemd.build_input_deck) writes `restart.<step>.bin` at every
profile-frame boundary of production (NEMDSpec.checkpoint_frames). When a run
dies -- walltime, a node, a laptop reboot -- `prepare_resume` turns the newest
usable checkpoint into `resume.in`:

1. The outputs are cut back to the checkpoint step: profile frames after it
   and reservoir-energy lines after it are dropped (the resumed run writes
   them again, from the next interval).
2. `resume.in` reads the restart file and re-issues what a restart file does
   not carry -- pair coefficients, regions, computes, fixes, the variables they
   use, thermo settings -- by replaying the original deck's command stream up
   to its production `run`, and finishes with `run N upto`.
3. The outputs are appended to, not overwritten (`append` for ave/chunk and
   print); the Langevin tallies restart at zero, so the energy variables carry
   the checkpoint's accumulated values as offsets and reservoir_energy.dat
   stays one continuous series. The time column is written as step*dt
   because `time` is not kept across a restart.
4. The log is appended to log.lammps (`log ... append`; run LAMMPS with
   `-log none` so the launcher does not truncate it first).

Only the production stage is checkpointed: a run killed before production
starts over, which is cheap by comparison. TTM decks are not checkpointed
(fix ttm/thermal keeps electron state the restart file does not hold).
Checked end to end with LAMMPS on a Si/Ge cell (tests/test_checkpoint.py).
"""

from __future__ import annotations

import re
from pathlib import Path

RESUME_DECK = "resume.in"
_RST = re.compile(r"^restart\.(\d+)\.bin$")


class ResumeError(ValueError):
    pass


# ---- checkpoints on disk ----------------------------------------------------------
def checkpoints(wd: Path) -> list[tuple[int, Path]]:
    """(step, path) of every restart file, newest first."""
    out = []
    for p in Path(wd).glob("restart.*.bin"):
        m = _RST.match(p.name)
        if m and p.stat().st_size > 0:
            out.append((int(m.group(1)), p))
    return sorted(out, reverse=True)


def prune(wd: Path, keep: int = 2) -> list[Path]:
    """Delete all but the newest `keep` restart files; returns what was removed."""
    gone = [p for _, p in checkpoints(wd)[keep:]]
    for p in gone:
        p.unlink()
    return gone


def _energy_rows(path: Path) -> list[list[float]]:
    rows = []
    for ln in path.read_text(errors="replace").splitlines():
        parts = ln.split()
        if not parts or parts[0].startswith("#"):
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            break                       # a half-written last line
    return rows


def _row_at(rows: list[list[float]], t: float, dt: float) -> list[float] | None:
    for r in rows:
        if abs(r[0] - t) < 0.25 * dt:
            return r
    return None


def usable_checkpoint(wd: Path, spec) -> tuple[int, Path, list[float]] | None:
    """Newest checkpoint before the end of production whose step is in
    reservoir_energy.dat (so its energy offsets are known)."""
    wd = Path(wd)
    energy = wd / "reservoir_energy.dat"
    if not energy.is_file():
        return None
    dt = spec.timestep_fs * 1e-3
    rows = _energy_rows(energy)
    for step, path in checkpoints(wd):
        if not 0 < step < spec.production_steps:
            continue
        row = _row_at(rows, step * dt, dt)
        if row is not None and len(row) >= 3:
            return step, path, row
    return None


# ---- cutting outputs back ---------------------------------------------------------
def truncate_energy(path: Path, step: int, dt: float) -> int:
    """Keep the header and rows up to and including `step`; returns rows kept.
    The resumed run prints from the next interval on."""
    keep, n = [], 0
    for ln in path.read_text(errors="replace").splitlines():
        parts = ln.split()
        if parts and parts[0].startswith("#"):
            keep.append(ln)
            continue
        try:
            t = float(parts[0])
        except (ValueError, IndexError):
            break
        if t > step * dt + 0.25 * dt:
            break
        keep.append(ln)
        n += 1
    path.write_text("\n".join(keep) + "\n")
    return n


def truncate_profile(path: Path, step: int) -> int:
    """Keep the header and whole frames whose timestep is <= `step`; returns frames kept."""
    lines = path.read_text(errors="replace").splitlines()
    keep, i, frames = [], 0, 0
    while i < len(lines) and lines[i].startswith("#"):
        keep.append(lines[i]); i += 1
    while i < len(lines):
        parts = lines[i].split()
        if lines[i].startswith("#") or len(parts) != 3:
            i += 1
            continue
        ts, n = int(float(parts[0])), int(float(parts[1]))
        block = lines[i:i + 1 + n]
        if ts > step or len(block) < 1 + n:
            break
        keep.extend(block)
        frames += 1
        i += 1 + n
    path.write_text("\n".join(keep) + "\n")
    return frames


# ---- the resume deck --------------------------------------------------------------
def _commands(deck: str) -> list[str]:
    """Logical commands: continuation lines joined, comments and blanks dropped."""
    out, buf = [], ""
    for raw in deck.splitlines():
        ln = _strip_comment(raw).rstrip()
        if ln.endswith("&") or ln.endswith("\\"):
            buf += ln[:-1] + " "
            continue
        ln = (buf + ln).strip()
        buf = ""
        if ln:
            out.append(ln)
    return out


def _strip_comment(line: str) -> str:
    """Drop a '#' comment that is not inside quotes."""
    q = None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch == "#":
            return line[:i]
    return line


_VAR_REF = re.compile(r"v_(\w+)|\$\{(\w+)\}|\$(\w)(?!\w)")


def _refs(text: str) -> set[str]:
    return {a or b or c for a, b, c in _VAR_REF.findall(text)}


def build_resume_deck(deck: str, step: int, production_steps: int, dt_ps: float,
                      e_hot0: float, e_cold0: float, restart_file: str | None = None) -> str:
    """`resume.in` text: the state the original deck had at its production `run`,
    rebuilt on top of `read_restart`."""
    cmds = _commands(deck)
    runs = [i for i, c in enumerate(cmds) if c.split()[0] == "run"]
    prod = next((i for i in reversed(runs) if c_int(cmds[i]) == production_steps), None)
    if prod is None:
        raise ResumeError(f"no 'run {production_steps}' in the deck")
    regions: dict[str, str] = {}
    computes: dict[str, str] = {}
    fixes: dict[str, str] = {}
    variables: dict[str, str] = {}
    settings: dict[str, str] = {}
    pair: list[str] = []
    for c in cmds[:prod]:
        w = c.split()
        k = w[0]
        if k == "region":
            regions[w[1]] = c
        elif k == "compute":
            computes[w[1]] = c
        elif k == "uncompute":
            computes.pop(w[1], None)
        elif k == "fix":
            fixes.pop(w[1], None)
            fixes[w[1]] = c                 # redefinition moves it to the end
        elif k == "unfix":
            fixes.pop(w[1], None)
        elif k == "variable":
            if len(w) > 2 and w[2] == "delete":
                variables.pop(w[1], None)
            else:
                variables[w[1]] = c
        elif k == "pair_style":
            pair = [c]
        elif k == "pair_coeff":
            pair.append(c)
        elif k in ("neighbor", "neigh_modify", "thermo", "thermo_style", "thermo_modify",
                   "timestep"):
            settings[k] = c
        elif k == "restart":
            settings.pop("restart", None)
    tail = [c for c in cmds[prod + 1:] if c.split()[0] in ("print",)]
    tail = [c for c in tail if "OPENKAPITZA_DONE" in c] or tail[-1:]

    # the reservoirs restart their tallies at zero: carry the accumulated energy
    for name, fix_id, e0 in (("ehot", "hot", e_hot0), ("ecold", "cold", e_cold0)):
        if name in variables:
            variables[name] = f"variable        {name} equal f_{fix_id}+{e0!r}"
    if "tnow" in variables:
        variables["tnow"] = f"variable        tnow equal step*{dt_ps!r}"

    used = set()
    pending = set().union(*(_refs(x) for x in (*fixes.values(), *computes.values(),
                                                *settings.values(), *tail)))
    while pending:
        v = pending.pop()
        if v in used or v not in variables:
            continue
        used.add(v)
        pending |= _refs(variables[v].split(None, 3)[-1])

    def fix_line(c: str) -> str:
        w = c.split()
        if len(w) > 3 and w[3] in ("ave/chunk", "ave/time", "ave/histo", "print"):
            c = re.sub(r"\bfile\s+(\S+)", r"append \1", c, count=1)
        if len(w) > 3 and w[3] == "langevin" and len(w) > 7:
            w[7] = str((int(w[7]) + step) % 900_000_000 + 1)   # a fresh random stream
            c = " ".join(w)
        return c

    out_kinds = ("ave/chunk", "ave/time", "ave/histo", "print")
    is_output = lambda c: len(c.split()) > 3 and c.split()[3] in out_kinds
    rf = restart_file or f"restart.{step}.bin"
    out = [
        "# OpenKapitza NEMD -- RESUMED production (generated by md/checkpoint.py).",
        f"# Continues from {rf} (production step {step} of {production_steps}).",
        "# Run with -log none: this appends to log.lammps itself.",
        "log             log.lammps append",
        f"read_restart    {rf}",
        f'print           "OPENKAPITZA_RESUMED step {step}"',
        *regions.values(), *pair,
        *(settings[k] for k in ("neighbor", "neigh_modify") if k in settings),
        *computes.values(),
        *(variables[v] for v in variables if v in used),
        *(fix_line(c) for c in fixes.values() if not is_output(c)),
        *(settings[k] for k in ("thermo", "thermo_style", "thermo_modify") if k in settings),
        "# One step before the output fixes: the checkpoint step's rows are kept",
        "# from the killed run, so the resumed outputs must start after it.",
        "run             1",
        *(fix_line(c) for c in fixes.values() if is_output(c)),
    ]
    ck = next((c for c in cmds[:prod + 1] if c.startswith("restart") and c.split()[1] != "0"), None)
    if ck:
        out.append(ck)
    out += [f"run             {production_steps} upto", "restart         0", *tail]
    return "\n".join(out) + "\n"


def c_int(cmd: str) -> int | None:
    w = cmd.split()
    try:
        return int(w[1])
    except (IndexError, ValueError):
        return None


def prepare_resume(wd: Path | str, spec, deck_name: str = "nemd.in") -> int | None:
    """Cut the outputs back and write resume.in; returns the step resumed from,
    or None when there is no usable checkpoint (the run must start over)."""
    wd = Path(wd)
    hit = usable_checkpoint(wd, spec)
    if hit is None:
        return None
    step, path, row = hit
    dt = spec.timestep_fs * 1e-3
    deck = build_resume_deck((wd / deck_name).read_text(), step, spec.production_steps, dt,
                             row[1], row[2], restart_file=path.name)
    truncate_energy(wd / "reservoir_energy.dat", step, dt)
    prof = wd / "temperature_profile.dat"
    if prof.is_file():
        truncate_profile(prof, step)
    for f in ("job.done",):
        (wd / f).unlink(missing_ok=True)
    (wd / RESUME_DECK).write_text(deck)
    for _, p in checkpoints(wd):
        if p != path and _ > step:
            p.unlink()                   # later, unusable files would confuse the next resume
    return step
