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
    prepare jobs <dir>   write the cluster jobs the plan names (needs --potentials)
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
        t = text.strip()
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
    def _compute(self, intake: Intake):
        from .. import build_joint, run_continuum
        kw = intake.build_kwargs()
        joint = build_joint(**kw)
        kap = None
        if intake.answers.get("h_K_MW") is not None:
            from ..md.dmm import user_kapitza
            kap = user_kapitza(intake.answers["h_K_MW"] * 1e6)
        return run_continuum(joint, kapitza=kap, n_samples=self.n_samples)

    def _run(self) -> str:
        while not self.intake.complete:            # fill remaining defaults/skips
            f = self.intake.next_field()
            try:
                self.intake.accept_default(f.name)
            except FieldError:
                return f"Still needed before I can run:\n{self.prompt()}"
        self.result = self._compute(self.intake)
        self.plan = plan(self.result)
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

    def _report(self) -> str:
        if self.result is None:
            raise FieldError("nothing has run yet")
        from ..report import render
        return render(self.result)

    def _prepare(self, folder: str) -> str:
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
        mine = [j for j in manifest["jobs"]
                if any(j["name"] == n or j["name"].startswith(n + "/") for n in self.plan.cluster_jobs)]
        nh = sum(j["node_hours"] for j in mine)
        rows = "\n".join(f"  {j['name']:40} {j['node_hours']:7.2f} node-h" for j in mine)
        return (f"Wrote all prepared jobs to {folder}; the ones for this joint:\n{rows}\n"
                f"  total {nh:.1f} node-hours. Each job folder has submit.sbatch (cluster) "
                f"and run.sh (local). Nothing was submitted.")
