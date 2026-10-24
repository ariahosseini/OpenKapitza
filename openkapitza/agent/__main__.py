"""Terminal wrapper for the agent.

Interactive:
    python -m openkapitza.agent [--potentials DIR]

Scripted conversation (one line per prompt; blank = take the default):
    python -m openkapitza.agent --answers FILE

From a joint description (YAML/JSON; several under `joints:` for a batch):
    python -m openkapitza.agent --input joint.yaml [--json out.json | --json -]
        [--md RUN_DIR ...] [--sweep name=a:b:n[:log]] [--csv out.csv] [--plot out.png]

    --fields      list every field name, its unit and default, and exit
"""

from __future__ import annotations

import argparse
import json
import sys

from .fields import FIELDS, FieldError
from .session import Session


def _fields() -> str:
    rows = []
    for f in FIELDS:
        d = f.default if not callable(f.default) else "(from material)"
        kind = f.unit or ("choice: " + ", ".join(f.choice_list()) if f.kind == "choice" else f.kind)
        rows.append(f"{f.name:18} {kind:40.40} default: {d if d is not None else '—'}"
                    + ("   [advanced]" if f.advanced else ""))
    return "\n".join(rows)


def _batch(a) -> int:
    from .explain import explain
    from .io import intake_from_dict, load_descriptions, result_to_dict
    from .planner import plan
    from .sweep import parse_spec, plot, run_sweep, table, write_csv
    descs = load_descriptions(a.input)
    results, failed = [], 0
    for i, desc in enumerate(descs):
        label = desc.get("label") or f"joint {i + 1}"
        s = Session(potentials=a.potentials, n_samples=a.samples)
        try:
            s.intake = intake_from_dict(desc)
            md_paths = list(a.md or [])
            extra = desc.get("md")
            md_paths += extra if isinstance(extra, list) else ([extra] if extra else [])
            for p in md_paths:
                from .ingest import ingest
                s.md.update(ingest(p))
            s.result = s._compute(s.intake)
            s.plan = plan(s.result, md_used=s.md_used)
        except (FieldError, ValueError) as e:
            failed += 1
            print(f"== {label}: FAILED — {e}", file=sys.stderr)
            continue
        d = result_to_dict(s.result, s.plan, s.intake, s.md_used)
        d["label"] = label
        sweep_expr = a.sweep or desc.get("sweep")
        if sweep_expr:
            name, values = parse_spec(sweep_expr)
            pts = run_sweep(s, name, values)
            d["sweep"] = {"field": name, "points": [{**p.__dict__, "warnings": sorted(p.warnings)} for p in pts]}
            if a.csv:
                write_csv(a.csv if len(descs) == 1 else f"{i + 1}_{a.csv}", name, pts)
            if a.plot:
                from .fields import BY_NAME
                plot(a.plot if len(descs) == 1 else f"{i + 1}_{a.plot}", name,
                     BY_NAME[name].unit, pts)
        results.append(d)
        if a.json != "-":
            print(f"== {label}")
            print(explain(s.result, s.plan))
            if sweep_expr:
                from .fields import BY_NAME
                print()
                print(table(name, BY_NAME[name].unit, pts))
            print()
    if a.json:
        payload = results[0] if len(results) == 1 and len(descs) == 1 else results
        text = json.dumps(payload, indent=2)
        if a.json == "-":
            print(text)
        else:
            with open(a.json, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"JSON written to {a.json}")
    if len(descs) > 1 and a.json != "-":
        print(f"{len(results)} of {len(descs)} joints ran" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m openkapitza.agent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", help="scripted conversation: one answer per line")
    ap.add_argument("--input", help="joint description file (YAML/JSON)")
    ap.add_argument("--json", help="write the result(s) as JSON to this file ('-' = stdout)")
    ap.add_argument("--md", action="append", help="finished MD to ingest (repeatable)")
    ap.add_argument("--sweep", help="name=start:stop:n[:log] or name=v1,v2,...")
    ap.add_argument("--csv", help="sweep table as CSV")
    ap.add_argument("--plot", help="sweep plot (needs matplotlib)")
    ap.add_argument("--potentials", help="LAMMPS potentials folder (for 'prepare jobs')")
    ap.add_argument("--samples", type=int, default=1000, help="Monte Carlo samples")
    ap.add_argument("--fields", action="store_true", help="list the fields and exit")
    a = ap.parse_args(argv)
    if a.fields:
        print(_fields())
        return 0
    if a.input:
        try:
            return _batch(a)
        except FieldError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    if a.json or a.sweep or a.csv or a.plot or a.md:
        ap.error("--json, --md, --sweep, --csv and --plot go with --input")
    s = Session(potentials=a.potentials, n_samples=a.samples)
    print(s.start())
    if a.answers:
        with open(a.answers, encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh if not ln.lstrip().startswith("#")]
        for ln in lines:
            print(f"\n> {ln}")
            print(s.handle(ln))
            if s.done:
                break
        return 0
    while not s.done:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        print(s.handle(line))
    return 0


if __name__ == "__main__":
    sys.exit(main())
