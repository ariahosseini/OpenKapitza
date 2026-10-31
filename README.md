# OpenKapitza — multiscale thermal contact resistance

Total thermal resistance between two solids in contact, for PCB, package and
die-level joints. Three scales, combined on a consistent area basis:

- **contact spots** — Cooper–Mikic–Yovanovich (plastic) or Mikic (elastic)
  constriction from measured roughness, slope and microhardness, with
  light-load truncation, unloading hysteresis and out-of-flatness;
- **gas gap** — conduction through the interstitial gas, with rarefaction and
  sourced accommodation-coefficient bands (skipped in vacuum);
- **the atomic interface** (Kapitza) — a DMM placeholder, replaced by a
  non-equilibrium MD result from LAMMPS when that would change the answer.

Filled and bonded joints (grease, pads, solder, sinter, die attach) go through
a series stack instead. Every number carries its uncertainty and provenance,
and every caveat a stable flag code.

## Quick start

```bash
pip install -e .                          # Python >= 3.9; needs pyyaml, numpy
python -m openkapitza.agent               # guided: asks for the joint, explains the answer
```

Or describe the joint to Claude (or any MCP client) in your own words; the
model fills in the inputs and OpenKapitza does every calculation and check:

```bash
pip install -e ".[mcp]"
claude mcp add openkapitza -- python -m openkapitza.agent.mcp_server
```

From a description file, or in Python:

```bash
python -m openkapitza.agent --input examples/agent/heatsink_screwed.yaml
python -m openkapitza.agent --input joint.yaml --json result.json --sweep pressure_MPa=0.1:5:8:log
```

```python
from openkapitza import build_joint, run_continuum, render

joint = build_joint("SS304", "SS304", sigma_a_um=1.2, sigma_b_um=1.2,
                    slope_a=0.08, slope_b=0.08, pressure_MPa=2.0,
                    temperature_K=300.0, gas="air", gas_pressure_kPa=101.325)
print(render(run_continuum(joint)))
```

The agent answers with the conductance and its band, where the resistance
sits, whether an atomistic calculation would change the answer (and whether
this package has an MD route for the pair), ranked caveats, and the most useful next
measurement. **Start with [`docs/agent-guide.md`](docs/agent-guide.md).**

## Examples

### A bolted steel joint: where the heat goes

The same stainless-steel pair (σ = 1.2 µm, slope 0.08) in vacuum, air and
helium, against contact pressure (`build_joint` + `run_continuum`, the call
shown above):

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/joint_pressure-dark.png"><img alt="Joint conductance against contact pressure for SS304 in vacuum, air and helium" src="docs/figures/joint_pressure-light.png"></picture>

In vacuum all heat crosses at the contact spots, so h rises almost linearly
with load (CMY); in air the gas gap carries most of it at low load, and in
helium (~6× air's conductivity) more. On 570 measured points the contact
model lands within RMS 13–15 % (validation table below).

### The atomic interface: an MD measurement

At each contact spot the interface itself resists heat (Kapitza). OpenKapitza
writes, runs and measures LAMMPS non-equilibrium MD for it: heat flows from a
hot to a cold reservoir and the jump at the interface gives h_K = q / ΔT_K.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/nemd_profile-dark.png"><img alt="Temperature profile through a Si/Ge NEMD cell with a jump at the interface" src="docs/figures/nemd_profile-light.png"></picture>

Before a number is trusted the cell is varied one axis at a time (Si/Ge,
Tersoff; 14 runs on an M3 Pro laptop through the agent's job loop):

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/sige_convergence-dark.png"><img alt="Si/Ge interface conductance against cross-section, length, duration, drive and reservoir size" src="docs/figures/sige_convergence-light.png"></picture>

Four axes agree with the base cell: **483 ± 20 MW/m²K at 16 nm per side**.
Length does not: h rises beyond 16 nm (more long-wavelength phonons), and
the 64 nm cell (hollow) was not yet steady and is being re-run. Hence a
convergence verdict per axis, not a single run ([KB §17](docs/knowledge-base.md)).

### What the interface is made of matters more than anything else

Real contact spots are rarely clean, bonded crystal. The same contact with
the bond removed (van der Waals only), under load, and with adsorbed argon:

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/interface_types-dark.png"><img alt="Interface conductance for bonded, van der Waals and argon-layer Si/Ge contacts" src="docs/figures/interface_types-light.png"></picture>

Losing the bond costs ~25×; 1 GPa, the stress inside a real asperity contact,
recovers ~5×; an adsorbed gas layer halves it again. Small cells, so the
values are preliminary; imposing the flux (orange) cuts the error 10×, and a
20-case sweep is running ([KB §19](docs/knowledge-base.md)). Figures:
`python3 scripts/make_readme_figures.py`.

## What the models cover

| situation | how | `build_joint` option |
|---|---|---|
| bare metal/semiconductor contact | CMY plastic or Mikic elastic, chosen by plasticity index (both if ambiguous); Song–Yovanovich contact hardness | default |
| gas in the gaps | lumped gap model, rarefaction, accommodation band | `gas=`, `gas_pressure_kPa=` |
| light loads | truncated-Gaussian model (Milanez et al. 2004) | `z_trunc=` |
| crowned / warped faces, bolted joints | macro constriction (Bahrami 2004), clamping check | `flatness_um=`, `joint_area_mm2=`, `clamping=` |
| relaxed or re-tightened joint | Mikic unloading curve | `max_pressure_MPa=` |
| your own indentation data | replaces database hardness | `Hv_a_GPa=`, `c1_a_GPa=`, `c2_a=` (and `_b`) |
| grease, pad, PCM | BLT/k + boundary resistances, class-range check | `fill=tim_fill(...)` |
| solder, sinter, die attach | layered stack + Kapitza faces, voids | `fill=bond_fill(...)` |
| measured or MD interface conductance | replaces the DMM placeholder | `run_continuum(kapitza=...)`; agent `ingest` |

## How well it is validated

| test | data | result |
|---|---|---|
| conforming contacts, vacuum | 207 points, Ni 200 / SS 304 (Hegazy, via Bahrami 2004) | geometric mean pred/meas 1.03, RMS 12.7 %, all within 2× |
| more conforming contacts | 158 points incl. Al 6061-T6 and Al/Ni (Antonetti, Milanez, McWaid, Nho) | 0.95, RMS 13.3 %; Al pairs 0.95, 11.1 % |
| crowned contacts | 205 points, steels, brass, Al 2024, Mg, Ni (Bahrami 2004) | with flatness 1.03, RMS 15.3 %; ignoring flatness h ~7× too high |
| light loads | Milanez S1 | RMS 20.8 → 15.5 % with the truncated model (paper: 25.3 → 16.1) |
| grease TIMs | 5 greases, ASTM D5470 (NREL 2017) | stack form reproduces each within 5 % |
| Al/Si interface | 8 TDTR values (Hopkins et al. 2010) | DMM placeholder 1.66× the best interface: an upper estimate |

Not yet validated: the gas path against exact measurements (only digitised
data), Cu or Al joints **in air**, the elastic branch, unloading magnitudes,
and MD interface conductances against experiment (Si/Ge is converged in
cross-section, duration, drive and reservoir, not yet in length; the metal
pairs need a GPU or cluster). Evidence and provenance:
[`tests/benchmarks/SOURCES.md`](tests/benchmarks/SOURCES.md).

## Atomistic side (MD)

LAMMPS non-equilibrium MD with a convergence campaign (cross-section, length,
duration, flux, reservoir), an equilibrium cross-check, two-temperature runs
for electron–phonon resistance on metal sides, spot-pressure loading, and a
structure module for crystalline and amorphous (melt-quench) interfaces.
Routes today: Si/Al and Si/Cu (Jelinek MEAM, passes the elastic gate), Si/Ge
(Tersoff, fails the gate and is reported with that stated), Si/SiO₂ (Tersoff +
melt-quenched a-SiO₂), plus non-bonded and gas-layer contacts
(`structure/nonbonded.py`) and an imposed-flux mode for resistive interfaces.
Si/Ge is converged on every axis but length (above); none is validated
against experiment yet. Long runs checkpoint and resume after a crash. `scripts/prepare_cluster.py`
writes every job, validated and costed (22 jobs, ~70 node-hours);
[`docs/cluster-runbook.md`](docs/cluster-runbook.md) says what to run in what
order; `scripts/benchmark_local.py` times your own machine on the same decks
and says which jobs fit it. The agent submits them (`submit plan`, on SLURM or locally), polls,
retries what the cluster broke (timeouts, node failures), diagnoses input and
physics failures with knowledge-base citations without "fixing" them, and
measures finished runs into the joint automatically.

## Documentation

| file | what it is |
|---|---|
| [`docs/agent-guide.md`](docs/agent-guide.md) | using the agent: conversation, files, sweeps, the MD loop, `lookup`, reading the answer |
| [`docs/open-issues.md`](docs/open-issues.md) | every open item, what closes it, the cluster run list |
| [`docs/knowledge-base.md`](docs/knowledge-base.md) | every defect hit, in Symptom / Cause / Fix / Verify form; literature surveys (§7–§12) |
| [`docs/technical-notes.md`](docs/technical-notes.md) | what Phase 0 corrected in the original spec, design rules, verification history, Phase 1–2 development notes |
| [`docs/agent-instruction-v2.md`](docs/agent-instruction-v2.md) | the specification; read it before changing any correlation, threshold or default |
| [`docs/cluster-runbook.md`](docs/cluster-runbook.md) | cluster procedure |
| [`docs/lammps-rag.md`](docs/lammps-rag.md) | official LAMMPS reference for the physics paths used (generated) |
| [`docs/papers.yaml`](docs/papers.yaml) | our notes on each paper: what it is used for, findings, what is still needed |
| [`tests/benchmarks/SOURCES.md`](tests/benchmarks/SOURCES.md) | provenance of every benchmark number |

## Layout

```
openkapitza/
  agent/      guided use: fields, intake, planner, explain, session, ingest, io, sweep
  rag/        cited keyword search over our docs, paper notes and the LAMMPS manual
  builder.py  the input boundary (build_joint, tim_fill, bond_fill)
  pipeline.py run_continuum: hardness, regime, contact, gap, Kapitza, network
  contact/    hardness, regime, CMY + Mikic, truncated, unloading, macro
  gap/        gas properties, rarefaction, accommodation
  network/    area-scaled series/parallel combination, stacks, Monte Carlo
  md/         DMM, NEMD decks and measurement, campaign, EMD, TTM, elastic, accommodation
  structure/  crystalline and amorphous interface builders, structure analysis
  compute/    local + SLURM job runners, cost model
  schema/     units, Quantity, flags, inputs, data loaders
  data/       materials.yaml, gases.yaml (per-field sources)
scripts/      cluster preparation, structure builders, audits, benchmark builders
examples/     agent sessions and description files
tests/        unit, physics and benchmark tests; benchmarks/ and data/ fixtures
```

```bash
python3 -m unittest discover -s tests      # ~735 tests, ~40 s
```

## Status

Phases 0–2 built. Phase 3 (agent) built: guided input, planning, files and
sweeps, cited lookup, the MD job loop (submit, poll, diagnose, retry, ingest),
and an MCP server; MD runs on a laptop (queued, checkpointed). Under way: the
interface-type sweep, the Si/Ge length axis and the scenario list in
[`docs/open-issues.md`](docs/open-issues.md); then Phase 4 ML potentials.
