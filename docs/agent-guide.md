# Using the agent

The agent asks for a joint, runs the multiscale calculation, and answers in
plain language: the conductance with its band, where the resistance sits,
whether an atomistic (MD) calculation would change the answer, what to watch,
and what to measure next. All the logic is in `openkapitza.agent`; the
terminal wrapper only reads lines and prints replies, so the same session can
sit behind any front end.

Contents: [ways in](#three-ways-in) · [a session](#a-session) ·
[reading the answer](#reading-the-answer) · [inputs](#inputs) ·
[model options](#model-options) · [commands](#commands) ·
[files and JSON](#description-files-and-json) · [sweeps](#sweeps) ·
[the MD loop](#the-md-loop) · [troubleshooting](#troubleshooting) ·
[limits](#what-it-does-not-do-yet)

## Three ways in

**Conversation.** Enter takes the default shown in brackets; `why` explains a
question; `help` lists the commands.

    python -m openkapitza.agent

**A description file** (YAML or JSON):

    python -m openkapitza.agent --input examples/agent/heatsink_screwed.yaml
    python -m openkapitza.agent --input joint.yaml --json result.json
    python -m openkapitza.agent --input joint.yaml --json -        # JSON to stdout

**Python.**

    from openkapitza.agent import Session
    s = Session()
    s.start()
    for line in ["Si", "", "Cu", "", "85 C", "tim", "0.14 MPa", "50 um", "3.7", "4.2",
                 "grease", "run"]:
        reply = s.handle(line)
    print(reply)
    s.result, s.plan          # the JointResult and the Plan behind the text

## A session

A Si die on a Cu spreader with a thermal grease (`examples/agent/die_spreader_grease.txt`,
run with `--answers`), abridged:

    What is the first solid (side A)?
      options: Si, Ge, Al, Al6061, Cu, SiO2, SS304, Ni200
    > Si
    Is side A crystalline or amorphous?
      [Enter = crystalline]
    >
    ...
    Temperature of the joint?
    > 85 C
    Is it a bare contact, a thermal interface material (grease, pad, PCM), or a
    bonded joint (solder, sinter, die attach)?
    > tim
    Bond-line thickness of the TIM (area average, not the spacer size)?
    > 50 um
    Bulk thermal conductivity of the TIM ...?
    > 3.7
    Boundary resistance R_c1 + R_c2 (the D5470 intercept)? Leave blank if unknown.
    > 4.2
    ...
    > run
    Joint conductance: 56.5 kW/m²K  (resistance 17.7 mm²K/W)
      90 % band from the input uncertainties: 46.9 kW/m²K – 67.7 kW/m²K ...
      Confidence: good
    Where the resistance is:
      - TIM (bulk): 76%
      - TIM boundaries R_c1 + R_c2: 24%
    Atomistic (MD) calculation: not needed — the stack has no solid-solid
    atomic interface in the model.
    > what if blt_um=25
    With blt_um = 25: 91.3 kW/m²K (now 56.5 kW/m²K, +62%). ...

Questions branch: a bare contact asks for roughness, slopes, gas and
clamping; a TIM asks for its thickness and conductivity; a bonded joint asks
for its layers.

## Reading the answer

The screwed heat sink (`examples/agent/heatsink_screwed.yaml`):

    Joint conductance: 6.37 kW/m²K  (resistance 157 mm²K/W)
      90 % band from the input uncertainties: 5 kW/m²K – 8.21 kW/m²K (inputs without a stated uncertainty held fixed)
      over the accommodation-coefficient band: 6.35 kW/m²K – 6.38 kW/m²K
      Confidence: limited — see the warnings

    Heat splits: 49% through the solid contact, 51% through the gas.
    Resistance of the solid path, by part:
      - macro constriction (flatness): 95%
      - constriction (contact spots): 5%

    Atomistic (MD) calculation: not needed — sweeping the interface term moves the answer by 0.5%; ...

    What to watch:
      [WARNING] No Song-Yovanovich coefficients for 'Cu'; using the bulk Vickers hardness ...
      [WARNING] Bolted/screwed joint: the apparent pressure is an average over a face ...
      [WARNING] Out-of-flatness limits the contact to the central 30% of the joint radius; ...
      ...
    Best next measurements:
      - Measure h on the assembled, bolted joint, or at least the face flatness and the pressure map ...
      - Lap the faces flatter or raise the load; or fill the macro gap with a TIM.
      - Run load-dependent Vickers indentation (10-200 gf) to fit c1, c2 ...

- **The number and its band.** The band is a Monte Carlo over every input that
  carries an uncertainty (roughness 10 %, slope 20 %, pressure 5 %, hardness
  30 %, flatness 25 % by default; measured MD h_K with its own error). Inputs
  without one are held fixed, so the band can only be too narrow, never too
  wide. When an accommodation coefficient was defaulted, a second band spans
  its sourced range.
- **Confidence** follows the worst flag: *good* (notes only), *moderate*
  (cautions), *limited* (a warning), *low* (a critical issue — do not quote the
  number unqualified).
- **Where the resistance is.** With gas, first the heat split between the
  solid and gas paths, then the solid path's parts as shares of *its*
  resistance. Without gas, the parts as shares of the total.
- **Atomistic (MD) calculation**: *not needed* (the interface term moves the
  answer < 2 %), *optional* (2–10 %), *recommended* (> 10 %, or a bonded face
  with no estimate that could matter), or *in use* (an ingested MD result
  replaced the placeholder; its own caveats follow). When recommended, the
  route for the pair and the prepared cluster jobs are named.
- **What to watch**: up to six caveats, warnings first. `report` prints all of
  them with their remedies.
- **Best next measurements**: the measurements that would remove the active
  caveats, in order of usual payoff.

## Inputs

Every field, with its unit (other units are accepted and converted: `0.0008 mm`,
`500 kPa`, `45 C`, `40 x 40 mm`, `0.1 Kcm2/W`):

| field | unit | asked when | default |
|---|---|---|---|
| `material_a`, `material_b` | Si, Ge, Al, Al6061, Cu, SiO2, SS304, Ni200 | always | — |
| `phase_a`, `phase_b` | crystalline / amorphous | always | the database's |
| `temperature_K` | K | always | — |
| `joint_type` | bare / tim / bond | always | bare |
| `pressure_MPa` | MPa | always (required for bare) | 0.2 for fills (unused by the stack) |
| `sigma_a_um`, `sigma_b_um` | µm, RMS (Rq ≈ 1.25 Ra) | bare | — |
| `slope_a`, `slope_b` | – (profilometer Δa) | bare, optional | estimated from roughness (flagged) |
| `medium` | vacuum / air / N2 / He / Ar / H2 | bare | air |
| `gas_pressure_kPa` | kPa | bare, in gas | 101.325 |
| `clamping` | uniform / bolted / unknown | bare | unknown |
| `flatness_um` | µm, sum of both faces' max deviation | bare, optional | flat faces assumed (flagged) |
| `joint_area_mm2` | mm² | when flatness given | — |
| `blt_um` | µm, area average | tim | — |
| `k_tim` | W/mK, bulk (D5470 slope) | tim | — |
| `R_c_mm2K_W` | mm²K/W, both boundaries | tim, optional | left out (R then a lower bound) |
| `tim_class` | grease / pad / pcm / gel / adhesive / solder | tim, optional | no range check |
| `layers` | `name thickness_um k [voids] [material]; ...` | bond | — |
| `z_trunc` | number of σ, or `reference` | advanced | fully Gaussian |
| `max_pressure_MPa` | MPa | advanced | first loading |
| `Hv_a_GPa`, `Hv_b_GPa` | GPa | advanced | database |
| `c1_a_GPa` + `c2_a`, `c1_b_GPa` + `c2_b` | GPa, – | advanced | database (none for Cu, Al) |
| `h_K_MW` | MW/m²K, per real contact area | advanced | DMM placeholder |

Advanced fields are never asked; set them with `set name=value` or put them
in a description file. `python -m openkapitza.agent --fields` prints this list
from the code.

## Model options

| you have | set | what changes |
|---|---|---|
| a measured out-of-flatness | `flatness_um`, `joint_area_mm2` | macro constriction in series (Bahrami 2004); gas over the crown profile. Ignoring a crown over-predicted 205 measured joints ~7×. |
| a bolted or screwed joint | `clamping=bolted` | warning: the mean pressure misrepresents a bolted face (CMY was 5–35× high on bolted Al joints). Give flatness too. |
| a light load on bead-blasted SS 304 / Ni 200 | `z_trunc=reference` | truncated-Gaussian contact (Milanez et al. 2004). The reference values go with *their* hardness fit; with your own c1/c2, fit z_trunc to your light-load data. |
| a clamp that relaxed, or assembly at a higher load | `max_pressure_MPa` | Mikic unloading curve: conductance stays above first loading (1.4–3× on Milanez S1). Magnitudes untested against data. |
| your own indentation data | `Hv_*`, or `c1_*` + `c2_*` | replaces the database; removes the hardness fallback warning |
| a measured interface conductance | `h_K_MW` | replaces the DMM placeholder; beats ingested MD |
| finished MD | `ingest PATH` | see [the MD loop](#the-md-loop) |

## Commands

| command | does |
|---|---|
| `why`, `back`, `show` | explain the question, re-ask the previous one, list the inputs |
| `set name=value` | set or change any field, including the advanced ones |
| `run` | calculate (remaining optional fields take their defaults) |
| `what if name=value` | re-run with one change; the inputs are not changed |
| `sweep name=a:b:n[:log]` | one-parameter sweep; `name=v1,v2,...` for explicit values; add `csv FILE` and/or `plot FILE` |
| `report` | the full technical report |
| `prepare jobs DIR` | write the cluster jobs the plan names (start with `--potentials`) |
| `ingest PATH` | measure finished MD and use it |
| `md`, `md clear` | list or drop the ingested MD |
| `save FILE`, `load FILE` | the session: inputs, which were defaults, ingested MD |
| `export FILE.json` | the last result as JSON |
| `help`, `quit` | |

## Description files and JSON

Field names as in [Inputs](#inputs); values may carry units as text. Optional
keys: `label`, `md` (a path or a list, ingested first), `sweep`.

```yaml
label: heat sink to lid
material_a: Al6061
material_b: Cu
temperature_K: 45 C
pressure_MPa: 0.5
sigma_a_um: 0.8
sigma_b_um: 0.4
slope_a: 0.08
slope_b: 0.06
medium: air
clamping: bolted
flatness_um: 25
joint_area_mm2: 40 x 40 mm
```

Several joints go in one file under `joints:` (a list); each is run and
reported, and `--json` writes a list. Unlisted fields take their defaults; an
unknown field name is an error, not ignored. A failed joint is reported and
the others still run (exit status 1).

The JSON holds: `h_joint_W_m2K`, `R_total_m2K_W`, `band_W_m2K` (p05, p50, p95),
`branch`, `contributions` (part, resistance, fraction), `kapitza` (value,
uncertainty, source), `flags` (code, severity, message, remedy), `plan` (md,
reason, route, potential, cluster_jobs, next_measurements), and when present
`accommodation_band_W_m2K`, `macro`, `inputs`, `defaults_used`, `md_used`, `sweep`.

## Sweeps

    sweep pressure_MPa=0.1:5:8:log          8 log-spaced points
    sweep flatness_um=0.5,2,5,10,25,50      explicit values
    sweep blt_um=10:100:10 csv blt.csv plot blt.png

Each point is a full run on a copy of the inputs (without the Monte Carlo, for
speed), so every option and flag applies. The table gives h, R and the part
that carries most of the solid-path resistance; warnings that appear at only
some points are marked on those points, and warnings present everywhere are
listed once below. Plots need matplotlib. From a file: `--sweep`, `--csv`,
`--plot`, or a `sweep:` key.

## The MD loop

1. Run a joint. The plan says whether MD is *not needed*, *optional* or
   *recommended* (see [Reading the answer](#reading-the-answer)).
2. If MD is worth it and the pair has a route (Si/Al, Si/Cu, Si/Ge, Si/SiO₂
   today), `prepare jobs DIR` writes the jobs. Every job folder holds a
   `spec.json`, so it can be measured later without the script that wrote it.
3. After the jobs finish, `ingest PATH` with any of:
   - one job folder (its `spec.json` and NEMD output);
   - a campaign root (`plan_campaign`): every rung measured, each axis judged
     for convergence; the most refined rung is quoted with the verdicts;
   - a prepared root (`prepare_cluster`): every finished job measured and
     grouped by material pair; a two-temperature (electron) run is preferred
     over phonon-only, and an EMD job is attached as a cross-check.
4. The joint is re-run with the MD h_K and its error in place of the DMM
   placeholder; the band carries the MD error, and the MD run's own caveats
   are listed under the MD line ("single run, convergence untested",
   "error 39 %", ...).

Precedence: a measured `h_K_MW` beats ingested MD, which beats the DMM. An MD
pair is used only if its elements match the joint (pure Al stands in for
Al 6061 with a caution). A temperature more than 25 K from the run's, or a
loaded contact spot against an unloaded MD interface, is flagged. For a bonded
stack, matching MD fills the face conductances.

Example, with the Si/Al MEAM plumbing run kept as a test fixture:

    python -m openkapitza.agent --input examples/agent/si_al_contact_md.yaml

## Troubleshooting

- **"That didn't work: ..."** — the answer was rejected and the question is
  asked again. The message says why: an unaccepted unit, a value out of range,
  or a choice that does not exist.
- **Ra or Rq?** Roughness is RMS (Rq). For a Gaussian surface Rq ≈ 1.25 Ra.
- **A Cu or pure Al joint always warns about hardness.** No Vickers
  coefficients are published for annealed Cu or pure Al (knowledge base
  §12.11); the bulk number is used. Give your own with `set c1_a_GPa=...` and
  `set c2_a=...`, or at least `set Hv_a_GPa=...`.
- **"No out-of-flatness given"** on every bare joint is a note, not an error:
  the faces are treated as flat. It becomes a caution for bolted joints. Give
  `flatness_um` and `joint_area_mm2` if you have them.
- **"sweeping flatness_um also needs joint_area_mm2"** — set the area first
  (`set joint_area_mm2=...`).
- **The band looks narrow.** Only inputs with an uncertainty are perturbed;
  `report` lists every input with its ± (none shown = held fixed).
- **Ingested MD is not used.** Its elements do not match the joint's
  materials (`md` lists what is loaded); the reply says so.
- **A filled joint ignores my roughness.** The stack model has no contact
  spots; roughness only feeds the check that the bond line is thicker than the
  asperities.

## Examples

`examples/agent/`: three scripted conversations (screwed heat sink, die on a
spreader with grease, solder die attach; run with `--answers FILE`) and two
description files (`heatsink_screwed.yaml`, `si_al_contact_md.yaml`).

## What it does not do yet

Launch or monitor MD (it writes jobs and reads results); routes for pairs
without a validated potential (Phase 4 MLIPs); a chat front end. Open items
are in [`open-issues.md`](open-issues.md).
