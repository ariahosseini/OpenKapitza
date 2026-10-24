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
and any converged MD interface conductance (the jobs are written and costed;
they need a cluster). Evidence and provenance:
[`tests/benchmarks/SOURCES.md`](tests/benchmarks/SOURCES.md).

## Atomistic side (MD)

LAMMPS non-equilibrium MD with a convergence campaign (cross-section, length,
duration, flux, reservoir), an equilibrium cross-check, two-temperature runs
for electron–phonon resistance on metal sides, spot-pressure loading, and a
structure module for crystalline and amorphous (melt-quench) interfaces.
Routes today: Si/Al and Si/Cu (Jelinek MEAM, passes the elastic gate), Si/Ge
(Tersoff, fails the gate and is reported with that stated), Si/SiO₂ (Tersoff +
melt-quenched a-SiO₂). None has a converged, validated h_K yet. `scripts/prepare_cluster.py`
writes every job, validated and costed (22 jobs, ~70 node-hours);
[`docs/cluster-runbook.md`](docs/cluster-runbook.md) says what to run in what
order; the agent's `ingest` reads the results back into a joint.

## Documentation

| file | what it is |
|---|---|
| [`docs/agent-guide.md`](docs/agent-guide.md) | using the agent: conversation, files, sweeps, the MD loop, reading the answer |
| [`docs/open-issues.md`](docs/open-issues.md) | every open item, what closes it, the cluster run list |
| [`docs/knowledge-base.md`](docs/knowledge-base.md) | every defect hit, in Symptom / Cause / Fix / Verify form; literature surveys (§7–§12) |
| [`docs/technical-notes.md`](docs/technical-notes.md) | what Phase 0 corrected in the original spec, design rules, verification history, Phase 1–2 development notes |
| [`docs/agent-instruction-v2.md`](docs/agent-instruction-v2.md) | the specification; read it before changing any correlation, threshold or default |
| [`docs/cluster-runbook.md`](docs/cluster-runbook.md) | cluster procedure |
| [`docs/lammps-rag.md`](docs/lammps-rag.md) | official LAMMPS reference for the physics paths used (generated) |
| [`tests/benchmarks/SOURCES.md`](tests/benchmarks/SOURCES.md) | provenance of every benchmark number |

## Layout

```
openkapitza/
  agent/      guided use: fields, intake, planner, explain, session, ingest, io, sweep
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
python3 -m unittest discover -s tests      # ~620 tests, ~20 s
```

## Status

Phases 0–2 built; Phase 3 (agent) built through guided input, planning, MD
ingestion, files and sweeps. Next: converged MD on a cluster, a chat front end,
and Phase 4 machine-learned potentials for pairs without a validated route.
