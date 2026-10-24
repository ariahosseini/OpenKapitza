"""OpenKapitza as an MCP server: a language model as the front end, the physics unchanged.

    python -m openkapitza.agent.mcp_server [--potentials DIR] [--lmp PATH]

Add it to any MCP client (Claude Desktop, Claude Code, ...), e.g.

    claude mcp add openkapitza -- python -m openkapitza.agent.mcp_server

and describe the joint in plain words. The model turns the description into
fields (`set_inputs`), and the same Session the terminal uses validates them,
runs the physics and decides what is uncertain. The division of labour is the
point:

  the model   understands the description, asks for what is missing, explains
              the answer, drives the job loop
  the server  parses and checks every value (units, ranges, choices), computes
              every number, attaches every warning, and refuses what it cannot
              support

Every result carries `must_report`: the warnings and cautions the answer is not
honest without. The instructions tell the model to pass them on, and not to
invent inputs. Defaults are listed separately so the model can say which values
were assumed. Anything that spends compute (`submit_jobs`) or changes a deck is
for the person to confirm.

Needs the `mcp` package (>= 2.0): pip install "openkapitza[mcp]".
"""

from __future__ import annotations

import argparse
import contextlib
import io
from typing import Any

from .fields import BY_NAME, FIELDS, FieldError
from .session import Session

INSTRUCTIONS = """\
You are the front end of OpenKapitza, which computes the thermal resistance of a
joint between two solids (contact spots, interstitial gas, the atomic interface
from MD, and any filler: grease, pad, solder). You translate; the server computes.

Workflow
1. describe_fields once, to learn the inputs (names, units, choices, defaults).
2. Turn the user's description into set_inputs. Pass values WITH units as the user
   gave them ("0.4 um", "50 psi"); the server converts and checks them. Read the
   reply: rejected values come back with the reason; ask the user, do not guess.
3. Ask for what `missing` lists, in plain words. You may suggest typical values,
   but say they are typical, and let the user accept them. Optional fields can be
   left to their defaults; the result lists every default used — tell the user
   which values were assumed.
4. run_joint. Report the conductance/resistance, the band, where the resistance
   is, and EVERY item in `must_report`, in plain words without softening. Do not
   change, round away or re-derive any number.
5. what_if / sweep for "what if" questions; technical_report for detail.

Rules
- Never invent material properties, measurements or MD results. If the user does
  not know a value and there is no default, say what measurement would give it.
- The server's flags are the critic. If a flag says a result is unreliable, say
  so even if the user hoped otherwise.
- lookup searches the project's knowledge base, paper notes and the LAMMPS manual;
  cite what you use (the citation string).
- MD jobs spend compute. Before submit_jobs, state the jobs and node-hours and get
  the user's yes. job_status polls once; nothing waits on a job. Failures come back
  diagnosed: infrastructure failures are retried automatically; input and physics
  failures need the user's decision — present the proposed fixes, do not apply them.
"""


def _flag(f) -> dict:
    return {"code": f.code.name if hasattr(f.code, "name") else str(f.code),
            "severity": f.severity.name, "message": f.message,
            "remedy": getattr(f, "remedy", "") or ""}


class OpenKapitzaTools:
    """The tool implementations, independent of the MCP plumbing (tested directly)."""

    def __init__(self, potentials: str | None = None, lmp: str | None = None,
                 n_samples: int = 1000):
        self._kw = dict(potentials=potentials, lmp=lmp, n_samples=n_samples)
        self.s = Session(**self._kw)

    # ---- inputs ------------------------------------------------------------------
    def describe_fields(self, include_advanced: bool = False) -> dict:
        a = self.s.intake.answers
        out = []
        for f in FIELDS:
            if f.advanced and not include_advanced:
                continue
            d = f.default_for(a) if f.applies(a) else None
            out.append({
                "name": f.name, "question": f.question, "why": f.why, "kind": f.kind,
                "unit": f.unit, "accepted_units": list(f.units),
                "choices": list(f.choice_list()) if f.kind == "choice" else None,
                "default": d, "default_reason": f.default_reason or None,
                "applies_now": f.applies(a), "required_now": f.applies(a) and f.required(a),
                "advanced": f.advanced})
        return {"fields": out, "note": "Pass values as text with units; the server parses them."}

    def set_inputs(self, values: dict[str, Any]) -> dict:
        accepted, rejected = {}, {}
        for name, v in values.items():
            if name not in BY_NAME:
                rejected[name] = "unknown field (describe_fields lists them)"
                continue
            try:
                accepted[name] = self.s.intake.set(name, str(v))
            except FieldError as e:
                rejected[name] = str(e)
        if accepted and self.s.result is not None:
            self.s.result = None          # stale: the inputs changed
        return {"accepted": accepted, "rejected": rejected, **self._state()}

    def clear_input(self, name: str) -> dict:
        if name not in BY_NAME:
            raise FieldError(f"unknown field {name!r}")
        self.s.intake.unset(name)
        return self._state()

    def show_inputs(self) -> dict:
        return {"inputs": dict(self.s.intake.answers),
                "defaults_accepted": dict(self.s.intake.defaulted), **self._state()}

    def _state(self) -> dict:
        it = self.s.intake
        missing = [{"name": f.name, "question": f.question, "unit": f.unit}
                   for f in it.missing_required()]
        nxt = it.next_field()
        return {"missing_required": missing,
                "next_question": ({"name": nxt.name, "question": nxt.question}
                                  if nxt else None),
                "ready_to_run": not missing}

    # ---- running -------------------------------------------------------------------
    def run_joint(self) -> dict:
        from .explain import explain
        from .io import result_to_dict
        text = self.s._run()
        if self.s.result is None:
            raise FieldError(text)
        r, p = self.s.result, self.s.plan
        d = result_to_dict(r, p, self.s.intake, self.s.md_used)
        flags = list(r.flags.flags) if hasattr(r, "flags") else []
        must = [_flag(f) for f in flags if f.severity.name in ("WARNING", "CRITICAL")]
        return {"explanation": explain(r, p), "result": d,
                "must_report": must,
                "defaults_used": dict(self.s.intake.defaulted),
                "md_used": self.s.md_used.summary() if self.s.md_used else None}

    def what_if(self, field: str, value: str) -> dict:
        return {"answer": self._call(lambda: self.s._what_if(f"{field}={value}"))}

    def sweep(self, field: str, values: str) -> dict:
        """values: 'start:stop:n[:log]' or 'v1,v2,...'"""
        from .sweep import parse_spec, run_sweep, table
        if not self.s.intake.complete or self.s.intake.missing_required():
            raise FieldError("set the inputs (or run once) before sweeping")
        name, vals = parse_spec(f"{field}={values}")
        pts = run_sweep(self.s, name, vals)
        return {"table": table(name, BY_NAME[name].unit, pts),
                "points": [{**p.__dict__, "warnings": sorted(p.warnings)} for p in pts]}

    def technical_report(self) -> dict:
        return {"report": self._call(self.s._report)}

    def lookup(self, query: str, collection: str | None = None, k: int = 4) -> dict:
        from .. import rag
        cols = (collection,) if collection else None
        hits = rag.search(query, k=k, collections=cols)
        return {"hits": [{"citation": h.chunk.citation(), "url": h.chunk.url or None,
                          "source": h.chunk.source, "section": h.chunk.title,
                          "text": h.snippet} for h in hits],
                "full_manual_indexed": rag.has_manual()}

    # ---- MD -----------------------------------------------------------------------
    def prepare_jobs(self, folder: str) -> dict:
        return {"answer": self._call(lambda: self.s._prepare(folder)),
                "jobs_folder": self.s.jobs_root}

    def submit_jobs(self, jobs: list[str], backend: str | None = None,
                    lmp: str | None = None, ranks: int | None = None) -> dict:
        words = list(jobs) + ([backend] if backend else []) + \
            ([f"lmp={lmp}"] if lmp else []) + ([f"np={ranks}"] if ranks else [])
        return {"answer": self._call(lambda: self.s._submit(words))}

    def job_status(self, folder: str | None = None) -> dict:
        text = self._call(lambda: self.s._jobs(folder or ""))
        book = self.s._book()
        return {"answer": text,
                "jobs": [{"name": e.name, "status": e.status, "backend": e.backend,
                          "attempts": e.attempts, "diagnosis": e.diagnosis}
                         for e in book.entries.values() if e.status != "prepared"],
                "md_ingested": [r.summary() for r in self.s.md.values()]}

    def resubmit_job(self, job: str) -> dict:
        return {"answer": self._call(lambda: self.s._jobs_action("resubmit", [job]))}

    def cancel_job(self, job: str) -> dict:
        return {"answer": self._call(lambda: self.s._jobs_action("cancel", [job]))}

    def ingest_md(self, path: str) -> dict:
        return {"answer": self._call(lambda: self.s._ingest(path))}

    # ---- session -------------------------------------------------------------------
    def save_session(self, path: str) -> dict:
        return {"answer": self.s.handle(f"save {path}")}

    def load_session(self, path: str) -> dict:
        from .io import load_session
        load_session(self.s, path)
        return {"loaded": path, **self.show_inputs()}

    def export_json(self, path: str) -> dict:
        return {"answer": self._call(lambda: self.s._export(path))}

    def reset_session(self) -> dict:
        self.s = Session(**self._kw)
        return {"answer": "New session.", **self._state()}

    def command(self, text: str) -> dict:
        """Any terminal command (help lists them), for what no tool covers."""
        return {"answer": self.s.handle(text)}

    @staticmethod
    def _call(fn):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn()


def build_server(potentials: str | None = None, lmp: str | None = None,
                 n_samples: int = 1000):
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    from .. import __version__ as version

    t = OpenKapitzaTools(potentials=potentials, lmp=lmp, n_samples=n_samples)
    srv = MCPServer("openkapitza", title="OpenKapitza",
                    description="Multiscale thermal contact resistance of solid joints",
                    instructions=INSTRUCTIONS, version=version)
    ro = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    spend = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)

    def tool(fn, annotations=None, description=""):
        def wrapped(**kw):
            try:
                return fn(**kw)
            except (FieldError, ValueError, OSError, RuntimeError) as e:
                raise ToolError(str(e)) from e
        wrapped.__name__ = fn.__name__
        wrapped.__doc__ = " ".join(x for x in (description, fn.__doc__) if x)
        import inspect
        sig = inspect.signature(fn)
        wrapped.__signature__ = sig
        wrapped.__annotations__ = {k: v.annotation for k, v in sig.parameters.items()} | \
            {"return": dict}
        srv.tool(annotations=annotations)(wrapped)

    doc = {
        "describe_fields": "Every input: name, question, why it matters, unit, accepted "
                           "units, choices, default. Call once before set_inputs.",
        "set_inputs": "Set inputs from the user's description: {field: value-with-unit}. "
                      "Returns accepted, rejected (with reasons) and what is still missing.",
        "clear_input": "Remove one input so it is asked again or defaults.",
        "show_inputs": "Inputs so far, the defaults accepted, what is missing.",
        "run_joint": "Compute the joint. Returns the plain-language explanation, the full "
                     "result, must_report (warnings to pass on) and the defaults used.",
        "what_if": "Re-run with one changed input and compare; the inputs are not changed.",
        "sweep": "One-parameter sweep. values: 'start:stop:n' (add ':log' for log spacing) "
                 "or 'v1,v2,...'.",
        "technical_report": "The full technical report of the last run.",
        "lookup": "Search the knowledge base, paper notes and LAMMPS manual. collection: "
                  "repo | papers | lammps. Cite the citation strings you use.",
        "prepare_jobs": "Write the MD cluster jobs the plan names into a folder. Nothing is "
                        "submitted.",
        "submit_jobs": "Submit prepared MD jobs: names, 'plan' (this joint's) or 'all'. "
                       "backend: local | slurm. Spends compute: confirm with the user first.",
        "job_status": "Poll the MD jobs once: status, diagnosis of failures; finished jobs are "
                      "measured and used automatically. folder attaches an existing job folder.",
        "resubmit_job": "Resubmit one job after the user fixed what its diagnosis named.",
        "cancel_job": "Cancel one running job.",
        "ingest_md": "Measure finished MD in a folder and use its h_K in the joint.",
        "save_session": "Save inputs, ingested MD and the job folder to a JSON file.",
        "load_session": "Load a saved session.",
        "export_json": "Write the last result as JSON.",
        "reset_session": "Start over with no inputs.",
        "command": "Run any terminal-agent command (e.g. 'help') when no tool fits.",
    }
    for name, d in doc.items():
        fn = getattr(t, name)
        ann = ro if name in ("describe_fields", "show_inputs", "technical_report", "lookup") \
            else spend if name in ("submit_jobs", "resubmit_job", "cancel_job") else None
        tool(fn, ann, d)

    @srv.prompt(title="Analyse a joint")
    def analyse_joint(description: str = "") -> str:
        """Start an analysis from a plain description of the joint."""
        return ("Use the OpenKapitza tools to compute the thermal resistance of this joint. "
                "Follow the server instructions: describe_fields, set_inputs from the "
                "description, ask for anything missing, run_joint, then report the answer "
                "with every must_report item and the defaults used.\n\nJoint: "
                + (description or "(ask me to describe it)"))

    srv._openkapitza_tools = t           # for tests and introspection
    return srv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m openkapitza.agent.mcp_server",
                                 description="OpenKapitza MCP server (stdio)")
    ap.add_argument("--potentials", help="LAMMPS potentials folder (for prepare_jobs)")
    ap.add_argument("--lmp", help="LAMMPS executable for local job runs")
    ap.add_argument("--samples", type=int, default=1000)
    a = ap.parse_args(argv)
    try:
        srv = build_server(a.potentials, a.lmp, a.samples)
    except ImportError as e:
        raise SystemExit(f"{e}\nThe MCP server needs the 'mcp' package (>= 2.0): "
                         'pip install "openkapitza[mcp]"')
    srv.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
