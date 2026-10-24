# Using the agent

The agent asks for a joint, runs the multiscale calculation, and answers in
plain language: the conductance with its band, what carries the resistance,
whether an atomistic (MD) calculation would change the answer, what to watch
and what to measure next. All the logic is in `openkapitza.agent`; the
terminal wrapper only reads lines and prints replies.

## Three ways in

**Conversation.** Enter takes the default shown in brackets; `why` explains a
question; `help` lists the commands.

    python -m openkapitza.agent

**A description file** (YAML or JSON, field names as listed by
`python -m openkapitza.agent --fields`; values may carry units as text):

    python -m openkapitza.agent --input examples/agent/heatsink_screwed.yaml
    python -m openkapitza.agent --input joint.yaml --json result.json
    python -m openkapitza.agent --input joint.yaml --json -        # JSON to stdout

Several joints go under `joints:` as a list; each is run and reported, and
`--json` writes a list. Unlisted fields take their defaults, as pressing Enter
would; an unknown field name is an error, not silently ignored.

**Python.**

    from openkapitza.agent import Session
    s = Session()
    s.start()
    for line in ["Si", "", "Cu", "", "85 C", "tim", "", "50 um", "3.7", "4.2", "grease", "run"]:
        reply = s.handle(line)
    print(reply)
    s.result, s.plan          # the JointResult and the Plan behind the text

## Commands

| command | does |
|---|---|
| `why`, `back`, `show` | explain the question, re-ask the previous one, list the inputs |
| `set name=value` | set or change any field, including advanced ones (`z_trunc`, `h_K_MW`) |
| `run` | calculate (remaining optional fields take their defaults) |
| `what if name=value` | re-run with one change; the inputs are not changed |
| `sweep name=a:b:n[:log]` | one-parameter sweep; `name=v1,v2,...` for explicit values; add `csv FILE` and/or `plot FILE` (plot needs matplotlib) |
| `report` | the full technical report |
| `prepare jobs DIR` | write the cluster jobs the plan names (start with `--potentials`) |
| `ingest PATH` | measure finished MD and use it (below) |
| `md`, `md clear` | list or drop the ingested MD |
| `save FILE`, `load FILE` | the session: inputs, which were defaults, ingested MD |
| `export FILE.json` | the last result as JSON |

From a file, the same sweep is `--sweep name=a:b:n[:log] [--csv FILE] [--plot FILE]`
(or a `sweep:` key in the file).

## The MD loop

1. Run a joint. The plan says whether MD is **not needed**, **optional** or
   **recommended**: it sweeps the interface (Kapitza) term over its uncertainty
   and reports how much the answer moves. For a bonded stack it uses the
   interfaces' share of the stack resistance, including faces with no estimate.
2. If MD is worth it and the pair has a route (Si/Al, Si/Cu, Si/Ge, Si/SiO₂
   today), `prepare jobs DIR` writes the jobs. Every job folder holds a
   `spec.json`, so it can be measured later without the script that wrote it.
3. After the jobs finish, `ingest PATH` with any of:
   - one job folder (its `spec.json` and NEMD output),
   - a campaign root (`plan_campaign`): every rung measured, each axis judged
     for convergence; the most refined rung is quoted with the verdicts,
   - a prepared root (`prepare_cluster`): every finished job measured and
     grouped by material pair; a two-temperature (electron) run is preferred
     over phonon-only, and an EMD job is attached as a cross-check.
4. The joint is re-run with the MD h_K and its error in place of the DMM
   placeholder; the band now carries the MD error, and the MD run's own
   caveats are listed under the MD line ("single run, convergence untested",
   "error 39 %", ...).

Precedence: a measured `h_K_MW` you set beats ingested MD, which beats the DMM.
An MD pair is used only if its elements match the joint (Si, Ge, Al, Cu, SiO₂;
pure Al stands in for Al 6061 with a caution). A temperature more than 25 K
from the run's, or a loaded contact spot against an unloaded MD interface, is
flagged. For a bonded stack, matching MD fills the face conductances.

Example, with the Si/Al MEAM plumbing run kept as a test fixture:

    python -m openkapitza.agent --input examples/agent/si_al_contact_md.yaml

## Examples

`examples/agent/`: three scripted conversations (screwed heat sink, die on a
spreader with grease, solder die attach; run with `--answers FILE`) and two
description files (`heatsink_screwed.yaml`, `si_al_contact_md.yaml`).

## What it does not do yet

Launch or monitor MD (it writes jobs and reads results); routes for pairs
without a validated potential (Phase 4 MLIPs); a chat front end.
