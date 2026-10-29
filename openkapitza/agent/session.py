"""A conversation as a state machine: text in, text out.

Everything a front end needs is here; the terminal wrapper (__main__) only
reads lines and prints replies. The same Session can sit behind a chat UI or
be driven by a script.

    s = Session()
    print(s.start())
    while not s.done:
        print(s.handle(input("> ")))

Commands at any prompt:
    <answer> | <enter>   answer the question / take its default
    why                  why the question matters
    back                 re-ask the previous question
    show                 inputs so far
    set name=value       set or change any field (incl. advanced: z_trunc, h_K_MW)
    run                  run now (asks for anything still required)
    report               full technical report of the last run
    what if name=value   re-run with one change and compare
    sweep name=a:b:n[:log] [csv FILE] [plot FILE]   or  sweep name=v1,v2,...
    prepare jobs <dir>   write the cluster jobs the plan names (needs --potentials)
    ingest <path>        measure finished MD (job folder, campaign, or prepared root)
                         and use its h_K in place of the DMM placeholder
    md                   list ingested MD results;  md clear  to drop them
    submit <jobs|plan|all> [local|slurm] [lmp=PATH] [np=N]
                         submit prepared jobs ('plan' = the ones this joint needs)
    jobs [DIR]           poll once: status, diagnosis of failures, finished jobs
                         ingested automatically; DIR attaches an existing job folder
    watch [SECONDS]      poll until nothing is running (Ctrl-C stops watching only)
    resubmit <job>       after you fixed what a diagnosis named;  cancel <job>
    save FILE / load FILE        the session (inputs, defaults, ingested MD)
    export FILE.json             the last result as JSON
    lookup <question>    search our docs, paper notes and the LAMMPS manual, with
                         citations; prefix  lammps: / papers: / repo:  to narrow
    help, quit
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .explain import _fmt_h, explain
from .fields import BY_NAME, FieldError
from .intake import Intake
from .planner import Plan, plan

HELP = __doc__.split("Commands at any prompt:")[1]


_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_SPACES = dict.fromkeys(map(ord, "\u00a0\u2007\u202f\u2009\u200a\u3000"), " ")


def _clean(text: str) -> str:
    """Typed or pasted text, with the characters copying from a web page brings in
    (non-breaking and thin spaces, zero-width marks) turned into plain spaces."""
    return text.translate(_INVISIBLE).translate(_SPACES).strip()


def _path(text: str) -> str:
    """A path as typed: '~' and $VARS expanded, surrounding quotes and spaces dropped."""
    import os
    return os.path.expandvars(os.path.expanduser(text.strip().strip('"').strip("'")))
_LOOKUP_PREFIX = {"lammps": ("lammps",), "papers": ("papers",), "paper": ("papers",),
                  "repo": ("repo",), "docs": ("repo",)}


def lookup(query: str, k: int = 4) -> str:
    """Retrieved passages with citations. Text informs; it is not a computed result."""
    from .. import rag
    cols = None
    head, sep, rest = query.partition(":")
    if sep and head.strip().lower() in _LOOKUP_PREFIX:
        cols, query = _LOOKUP_PREFIX[head.strip().lower()], rest.strip()
    if not query:
        return ("Usage: lookup <question>   e.g.  lookup fix ave/chunk Nfreq   or  "
                "lookup papers: unloading hysteresis")
    hits = rag.search(query, k=k, collections=cols)
    if not hits:
        return f"Nothing found for '{query}'."
    out = [f"Top {len(hits)} passage(s) for '{query}':"]
    for n, h in enumerate(hits, 1):
        out.append(f"\n[{n}] {h.chunk.citation()}")
        if h.chunk.url:
            out.append(f"    {h.chunk.url}")
        out.append("    " + h.snippet)
    if not rag.has_manual():
        out.append("\n(LAMMPS manual not indexed: only the curated subset was searched. "
                   "Build it with  python -m openkapitza.rag build --lammps-src PATH)")
    return "\n".join(out)
WELCOME = ("OpenKapitza — thermal resistance of a joint between two solids.\n"
           "I'll ask for the materials, surfaces and conditions; press Enter to take a "
           "default, type 'why' for the reason behind a question, 'help' for commands.\n")


@dataclass
class Session:
    intake: Intake = field(default_factory=Intake)
    result: object | None = None
    plan: Plan | None = None
    history: list[str] = field(default_factory=list)      # asked field names
    done: bool = False
    potentials: str | None = None                           # LAMMPS potentials dir
    n_samples: int = 1000
    md: dict = field(default_factory=dict)                  # pair -> ingest.MDResult
    md_used: object | None = None                           # MDResult in the last run
    jobs_root: str | None = None                            # prepared job folder
    lmp: str | None = None                                  # LAMMPS executable (local runs)
    emit: object | None = None                              # callable(str) for live output

    # ---- prompts -------------------------------------------------------------
    def start(self) -> str:
        return WELCOME + "\n" + self.prompt()

    def prompt(self) -> str:
        f = self.intake.next_field()
        if f is None:
            return ("I have what I need. Type 'run' (or press Enter) to calculate, 'show' to "
                    "review, or 'set name=value' to change something.")
        bits = [f.question]
        if f.kind == "choice":
            bits.append(f"  options: {', '.join(f.choice_list())}")
        d = f.default_for(self.intake.answers)
        if d is not None:
            bits.append(f"  [Enter = {d:g}{(' ' + f.unit) if f.unit else ''}]" if isinstance(d, float)
                        else f"  [Enter = {d}]")
        elif not f.required(self.intake.answers):
            bits.append("  [Enter = skip]")
        return "\n".join(bits)

    # ---- input -----------------------------------------------------------------
    def handle(self, text: str) -> str:
        t = _clean(text)
        while t.startswith(">"):                 # a pasted prompt marker ("> jobs ...")
            t = t[1:].lstrip()
        low = t.lower()
        try:
            if low in ("quit", "exit", "q"):
                self.done = True
                return "Bye."
            if low in ("help", "?"):
                return HELP.strip()
            if low == "why":
                f = self.intake.next_field()
                return f.why if f else "Nothing pending; 'run' to calculate."
            if low == "show":
                lines = self.intake.summary_lines()
                return "Inputs so far:\n" + ("\n".join(lines) if lines else "  (none)")
            if low == "back":
                return self._back()
            if low.startswith("set "):
                return self._set(t[4:])
            if low.startswith("what if "):
                return self._what_if(t[8:])
            if low == "report":
                return self._report()
            if low.startswith("prepare jobs"):
                return self._prepare(t[len("prepare jobs"):].strip())
            if low.startswith("ingest "):
                return self._ingest(t[7:].strip())
            if low in ("md", "md clear"):
                return self._md(low)
            if low.startswith("submit"):
                return self._submit(t[6:].split())
            if low == "jobs" or low.startswith("jobs "):
                return self._jobs(t[4:].strip())
            if low == "watch" or low.startswith("watch "):
                return self._watch(t[5:].strip())
            if low.startswith("resubmit "):
                return self._jobs_action("resubmit", t[9:].split())
            if low.startswith("cancel "):
                return self._jobs_action("cancel", t[7:].split())
            if low.startswith("sweep "):
                return self._sweep(t[6:])
            if low.startswith("save "):
                from .io import save_session
                return f"Saved to {save_session(self, _path(t[5:]))}."
            if low.startswith("load "):
                from .io import load_session
                load_session(self, _path(t[5:]))
                return (f"Loaded {t[5:].strip()} ({len(self.md)} MD result(s)).\n"
                        + self.prompt())
            if low.startswith("export "):
                return self._export(t[7:].strip())
            if low == "lookup" or low.startswith("lookup "):
                return lookup(t[6:].strip())
            if low == "run" or (t == "" and self.intake.complete):
                return self._run()
            return self._answer(t)
        except FieldError as e:
            return f"That didn't work: {e}\n{self.prompt()}"

    def _answer(self, t: str) -> str:
        f = self.intake.next_field()
        if f is None:
            return "Nothing pending. 'run', 'show', or 'set name=value'."
        if t == "" or t.lower() == "skip":
            self.intake.accept_default(f.name)
        else:
            self.intake.set(f.name, t)
        self.history.append(f.name)
        return self.prompt()

    def _back(self) -> str:
        if not self.history:
            return self.prompt()
        name = self.history.pop()
        self.intake.unset(name)
        return self.prompt()

    def _set(self, expr: str) -> str:
        if "=" not in expr:
            raise FieldError("use set name=value")
        name, value = (s.strip() for s in expr.split("=", 1))
        if name not in BY_NAME:
            raise FieldError(f"unknown field {name!r}; 'show' lists the fields in use")
        self.intake.set(name, value)
        msg = f"{name} set."
        if self.result is not None:
            msg += " Type 'run' to recalculate."
        return msg + ("\n" + self.prompt() if not self.intake.complete else "")

    # ---- running ---------------------------------------------------------------
    def _compute(self, intake: Intake, *, with_uncertainty: bool = True):
        """Build and run the joint; a user h_K beats ingested MD beats the DMM."""
        from .. import build_joint, run_continuum
        from ..schema import FlagSet
        from .ingest import matches, to_kapitza
        kw = intake.build_kwargs()
        joint = build_joint(**kw)
        flags = FlagSet()
        kap = None
        self.md_used = None
        a, b = joint.material_a.key, joint.material_b.key
        T = joint.temperature.si
        if intake.answers.get("h_K_MW") is not None:
            from ..md.dmm import user_kapitza
            kap = user_kapitza(intake.answers["h_K_MW"] * 1e6)
        elif joint.fill is not None and joint.fill.kind == "bond":
            self._md_into_bond_faces(joint, T, flags)
        else:
            hit = next((r for r in self.md.values() if matches(r, a, b)), None)
            if hit is not None and joint.fill is None:
                spot = (self.result.primary.spot_pressure.si / 1e9
                        if self.result is not None and self.result.stack is None else None)
                kap = to_kapitza(hit, a, b, T, spot, flags)
                self.md_used = hit
        result = run_continuum(joint, kapitza=kap, flags=flags,
                               with_uncertainty=with_uncertainty, n_samples=self.n_samples)
        if self.md_used is not None and joint.fill is None:
            from ..md.measured import potential_bias
            potential_bias(self.md_used.pair, self.md_used.potential, result.flags)
        return result

    def _md_into_bond_faces(self, joint, T: float, flags) -> None:
        """Use ingested MD at a bonded joint's faces where the materials match."""
        from .ingest import matches, to_kapitza
        fill = joint.fill
        faces = (("h_K_a", joint.material_a.key, fill.layers[0].material_key),
                 ("h_K_b", fill.layers[-1].material_key, joint.material_b.key))
        for attr, x, y in faces:
            if not x or not y or getattr(fill, attr) is not None:
                continue
            hit = next((r for r in self.md.values() if matches(r, x, y)), None)
            if hit is not None:
                setattr(fill, attr, to_kapitza(hit, x, y, T, None, flags).h_K)
                self.md_used = hit

    def _run(self) -> str:
        while not self.intake.complete:            # fill remaining defaults/skips
            f = self.intake.next_field()
            try:
                self.intake.accept_default(f.name)
            except FieldError:
                return f"Still needed before I can run:\n{self.prompt()}"
        self.result = self._compute(self.intake)
        self.plan = plan(self.result, md_used=self.md_used)
        return explain(self.result, self.plan) + (
            "\n\nNext: 'what if name=value' to test a change, 'report' for the full "
            "technical report, 'quit' to finish.")

    def _what_if(self, expr: str) -> str:
        if self.result is None:
            raise FieldError("run once first")
        if "=" not in expr:
            raise FieldError("use what if name=value")
        name, value = (s.strip() for s in expr.split("=", 1))
        trial = copy.deepcopy(self.intake)
        trial.set(name, value)
        alt = self._compute(trial)
        h0, h1 = self.result.primary.h_joint.si, alt.primary.h_joint.si
        return (f"With {name} = {value}: {_fmt_h(h1)} (now {_fmt_h(h0)}, "
                f"{(h1 / h0 - 1):+.0%}). The current inputs are unchanged; "
                f"'set {name}={value}' to keep it.")

    def _ingest(self, path: str) -> str:
        path = _path(path)
        from .ingest import IngestError, ingest
        try:
            found = ingest(path)
        except IngestError as e:
            raise FieldError(str(e)) from e
        self.md.update(found)
        lines = ["Measured:"] + [f"  {r.summary()}" for r in found.values()]
        for r in found.values():
            for n in r.notes:
                lines.append(f"    · {n}")
            bad = [f for f in r.flags.flags if f.severity.name in ("WARNING", "CRITICAL")]
            if bad:
                lines.append(f"    {len(bad)} warning(s) on this measurement, e.g.: {bad[0].message}")
        if self.result is not None:
            before = self.result.primary.h_joint.si
            self.result = self._compute(self.intake)
            self.plan = plan(self.result, md_used=self.md_used)
            after = self.result.primary.h_joint.si
            if self.md_used is None:
                lines.append("None of it matches this joint's materials; the DMM placeholder "
                             "stays in use.")
            else:
                lines.append(f"Re-ran the joint with the MD h_K: {_fmt_h(after)} (was "
                             f"{_fmt_h(before)} with the DMM, {after / before - 1:+.1%}).")
                lines.append("'run' shows the full answer with the MD caveats.")
        else:
            lines.append("It will be used when you run a joint with matching materials.")
        return "\n".join(lines)

    def _md(self, low: str) -> str:
        if low == "md clear":
            self.md.clear()
            return "Ingested MD cleared; the DMM placeholder will be used."
        if not self.md:
            return "No MD ingested. 'ingest <path>' after a job finishes."
        return "Ingested MD:\n" + "\n".join(f"  {r.summary()}" for r in self.md.values())

    def _sweep(self, expr: str) -> str:
        from .sweep import parse_spec, plot, run_sweep, table, write_csv
        if not self.intake.complete or self.intake.missing_required():
            raise FieldError("answer the questions (or 'run' once) before sweeping")
        words = expr.split()
        spec, opts = words[0], words[1:]
        name, values = parse_spec(spec)
        pts = run_sweep(self, name, values)
        unit = BY_NAME[name].unit
        out = [table(name, unit, pts)]
        for kind, path in zip(opts[::2], opts[1::2]):
            if kind == "csv":
                out.append(f"CSV: {write_csv(path, name, pts)}")
            elif kind == "plot":
                out.append(f"Plot: {plot(path, name, unit, pts)}")
            else:
                raise FieldError(f"unknown sweep option {kind!r} (csv FILE, plot FILE)")
        return "\n".join(out)

    def _export(self, path: str) -> str:
        path = _path(path)
        import json
        from .io import result_to_dict
        if self.result is None:
            raise FieldError("nothing has run yet")
        d = result_to_dict(self.result, self.plan, self.intake, self.md_used)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2)
        return f"Wrote {path}."

    def _report(self) -> str:
        if self.result is None:
            raise FieldError("nothing has run yet")
        from ..report import render
        return render(self.result)

    def _prepare(self, folder: str) -> str:
        folder = _path(folder) if folder else folder
        if self.plan is None:
            raise FieldError("run first")
        if not self.plan.cluster_jobs:
            return "The plan names no MD jobs for this joint."
        if not folder:
            raise FieldError("give a folder: prepare jobs <dir>")
        if not self.potentials:
            raise FieldError("start the agent with --potentials <lammps>/potentials to "
                             "write MD jobs")
        import json
        import sys
        from pathlib import Path
        scripts = Path(__file__).resolve().parents[2] / "scripts"
        sys.path.insert(0, str(scripts))
        import contextlib
        import io
        import prepare_cluster
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_cluster.main([folder, "--potentials", self.potentials])
        manifest = json.loads((Path(folder) / "manifest.json").read_text())
        self.jobs_root = str(Path(folder).resolve())
        mine = [j for j in manifest["jobs"]
                if any(j["name"] == n or j["name"].startswith(n + "/") for n in self.plan.cluster_jobs)]
        nh = sum(j["node_hours"] for j in mine)
        rows = "\n".join(f"  {j['name']:40} {j['node_hours']:7.2f} node-h" for j in mine)
        return (f"Wrote all prepared jobs to {folder}; the ones for this joint:\n{rows}\n"
                f"  total {nh:.1f} node-hours. Each job folder has submit.sbatch (cluster) "
                f"and run.sh (local). Nothing was submitted: 'submit plan' sends these, "
                f"'submit plan local lmp=PATH' runs them here.")

    # ---- the job loop ----------------------------------------------------------
    def _book(self):
        from .jobs import JobBook, JobError
        if not self.jobs_root:
            raise FieldError("no job folder yet: 'prepare jobs <dir>' first, or 'jobs <dir>' "
                             "for one prepared earlier")
        try:
            return JobBook.open(self.jobs_root)
        except JobError as e:
            raise FieldError(str(e)) from e

    def _submit(self, words: list[str]) -> str:
        from .jobs import JobError
        backend, lmp, launcher, names = None, None, "", []
        for w in words:
            if w in ("local", "slurm"):
                backend = w
            elif w.startswith("lmp="):
                lmp = _path(w[4:])
            elif w.startswith("np="):
                launcher = f"mpirun -np {int(w[3:])}"
            else:
                names.append(w)
        if not names:
            raise FieldError("which jobs? 'submit plan', 'submit all', or job names "
                             "('jobs' lists them)")
        if names == ["plan"]:
            if self.plan is None or not self.plan.cluster_jobs:
                raise FieldError("the current plan names no MD jobs ('run' first)")
            names = list(self.plan.cluster_jobs)
        book = self._book()
        lmp = lmp or self.lmp
        try:
            events = book.submit(names, backend=backend, lmp=lmp, launcher=launcher)
        except JobError as e:
            raise FieldError(str(e)) from e
        if lmp:
            self.lmp = lmp
        lines = [f"  {ev.name}: {ev.message}" for ev in events]
        started = sum(ev.what == "started" for ev in events)
        queued = sum(ev.what == "queued" for ev in events)
        nh = sum(book.entries[ev.name].node_hours or 0 for ev in events if ev.what == "started")
        head = f"Submitted {started} job(s)" + (f", ~{nh:.1f} node-hours" if nh else "") + \
            (f"; {queued} queued to run one after another on this machine" if queued else "") + ":"
        return "\n".join([head] + lines + [
            "Nothing waits on them. 'jobs' polls once; 'watch' polls until they stop. "
            "Finished jobs are measured and used automatically."])

    def _jobs(self, arg: str) -> str:
        if arg:
            from pathlib import Path
            self.jobs_root = str(Path(_path(arg)).resolve())
        book = self._book()
        events = book.refresh()
        out = self._report_events(book, events)
        out.append(book.table(only_active=True))
        return "\n".join(out)

    def _report_events(self, book, events) -> list[str]:
        out = []
        for ev in events:
            out.append(f"{ev.name}: {ev.what} — {ev.message}")
            d = ev.diagnosis
            if d is not None and ev.what in ("failed", "resubmitted") and d.kind != "infrastructure":
                if d.evidence:
                    out.append(f"    log: {d.evidence}")
                out += [f"    fix: {f}" for f in d.fixes]
                out += [f"    see: {c}" for c in d.citations]
            elif d is not None and ev.what == "failed":
                out += [f"    fix: {f}" for f in d.fixes]
        if any(ev.what == "finished" for ev in events):
            out.append(self._auto_ingest(book, [ev.name for ev in events if ev.what == "finished"]))
        return out

    def _auto_ingest(self, book, names: list[str]) -> str:
        """Measure everything finished under the job folder and use it."""
        try:
            msg = self._ingest(str(book.root))
        except FieldError as e:
            return f"Could not measure the finished job(s) yet: {e}"
        for n in names:
            book.mark_ingested(n)
        return msg

    def _watch(self, arg: str) -> str:
        import time
        interval = float(arg) if arg else 60.0
        book = self._book()
        log: list[str] = []
        say = self.emit if callable(self.emit) else log.append
        say(f"Watching {book.root} every {interval:g} s (Ctrl-C stops watching; jobs keep running).")
        try:
            while True:
                book = self._book()
                events = book.refresh()
                for line in self._report_events(book, events):
                    say(line)
                live = [e for e in book.entries.values()
                        if e.status in ("submitted", "running", "queued")]
                if not live:
                    say("Nothing running.")
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            say("Stopped watching; the jobs are still running. 'jobs' to check later.")
        say(book.table(only_active=True))
        return "\n".join(log) if log else ""

    def _jobs_action(self, action: str, names: list[str]) -> str:
        from .jobs import JobError
        book = self._book()
        try:
            events = (book.submit(names, resubmit=True, lmp=self.lmp) if action == "resubmit"
                      else book.cancel(names))
        except JobError as e:
            raise FieldError(str(e)) from e
        return "\n".join(f"  {ev.name}: {ev.message}" for ev in events) or "Nothing to do."
