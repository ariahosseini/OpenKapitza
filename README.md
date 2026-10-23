# OpenKapitza — Phase 0: continuum core

Multiscale thermal contact resistance. This phase ships the macroscale half:
contact mechanics, gas-gap conductance, and the resistance network that
combines them — with a DMM placeholder standing in for the atomistic Kapitza
term until the Phase-1 NEMD driver lands.

It is a complete, usable tool on its own. For most real joints it returns the
right answer without any MD at all, and it tells you when that is the case.

```python
from openkapitza import build_joint, run_continuum, render

joint = build_joint(
    "SS304", "SS304",
    sigma_a_um=1.2, sigma_b_um=1.2,
    slope_a=0.08, slope_b=0.08,
    pressure_MPa=2.0, temperature_K=300.0,
    gas="air", gas_pressure_kPa=101.325,
)
print(render(run_continuum(joint)))
```

Filled and bonded joints (TIM, solder, sintered Ag, die attach) go through a
series stack instead of the contact-plus-gas network (open issue P0-1):

```python
from openkapitza import bond_fill, tim_fill

grease = build_joint("Al", "Cu", sigma_a_um=1.0, sigma_b_um=1.0, pressure_MPa=0.2,
                     temperature_K=330.0,
                     fill=tim_fill(100, 4.0, tim_class="grease", R_c_mm2K_W=15.0))
die = build_joint("Si", "Cu", sigma_a_um=0.05, sigma_b_um=0.5, pressure_MPa=0.1,
                  temperature_K=330.0,
                  fill=bond_fill([("SAC305", 30, 60, 0.03)], h_K_a_MW=300, h_K_b_MW=300))
print(render(run_continuum(grease)))
```

```
python3 -m unittest discover -s tests
```

---

## What this phase corrects in the original spec

Four changes, in order of how badly the original would have failed.

### 1. The Kapitza term must be scaled by the real contact area

`h_c` from CMY is a conductance per unit **apparent** area that already
contains the constriction resistance of the contact spots. `h_K` from MD is
per unit **real contact** area — an atomically flat patch where the solids
actually touch. Adding `1/h_c + 1/h_K` treats them as the same quantity and
understates the interfacial term by the area fraction ξ = A_r/A_a, typically
10⁻³–10⁻². Two to three orders of magnitude, silently.

```
R_solid = 1/h_c + 1/(ξ · h_K)      series
h_joint = 1/R_solid + h_g           parallel
```

`network/combine.py` implements this, and `TestAreaScaling` pins it.

### 2. MD must be run at the spot pressure, not the apparent pressure

Inside a plastically deforming contact spot the local normal stress is of
order H_c — GPa — not the MPa of apparent contact pressure. A NEMD cell
loaded at the apparent pressure models the wrong state. Every result carries
`spot_pressure` for this reason.

### 3. Plastic is not the only branch

The original hardwired plastic CMY. Two hard, smooth, low-slope surfaces
deform elastically, and Si/Ge at any realistic polish is one of them — the
package puts it at a plasticity index of 0.12, firmly elastic. The Mikic
elastic correlation is now a first-class branch, selected by
ψ = E'·m/H_c, and when ψ lands in the ambiguous band **both** branches are
computed and reported rather than one being picked silently.

### 4. H_c is not Vickers hardness

Contact microhardness depends on indentation load, hence on pressure and
topography. `contact/hardness.py` implements the Song–Yovanovich correlation
when the Vickers coefficients c₁, c₂ are available and falls back to bulk Hv
with a loud flag when they are not. The fallback systematically overestimates
H_c, which understates P/H_c and h_c.

Also fixed, smaller: CMY validity window enforcement (10⁻⁵ ≤ P/H_c ≤ 2×10⁻²),
quadrature combination of σ and m, Knudsen and M/Y regime flags on the gap,
and the quantum-effects screen below ~0.5 θ_D.

---

## Does running MD actually matter?

Every result answers this before anyone spends node-hours finding out. The
network sweeps `h_K` across its plausible band and reports how far the joint
conductance moves:

```
  would MD change this?
    h_K swept by a factor of 2.50 moves h_joint by 0.1 %
    a perfect interface (h_K -> inf) would give 6790 W/m^2.K (+0.0 %)
    NEMD should be run at a spot pressure of ~2 GPa, NOT the apparent 2 MPa
```

For the rough stainless joint above, the interface contributes 0.15 % of the
solid-path resistance. No NEMD campaign can change that answer. The
`KAPITZA_NEGLIGIBLE` flag says so; `KAPITZA_DOMINANT` fires in the opposite
case, where the placeholder is load-bearing and the total should not be
trusted until MD is done.

---

## Design rules

**SI internally, units only at the boundary.** Every physics function takes
and returns plain SI floats. `Quantity` wraps values at the IO boundary and
deliberately implements no arithmetic, so a unit-bearing object cannot
silently propagate through an expression where its unit no longer holds.
Builder keywords carry their unit in the name (`sigma_a_um`, `pressure_MPa`).

**Every number carries its epistemic status.** `Quantity` holds value,
uncertainty, provenance and confidence. `make_derived` caps a result's
confidence at the weakest input that fed it — and an explicit `confidence`
argument is a further cap, never an override. A model can only make things
worse.

**No magic numbers in code.** Material and gas properties live in versioned
YAML with `verified` flags and citation fields. Correlation constants (1.25,
0.95, 1.62) stay in code because changing them means using a different
correlation, not a different material.

**Flags, not vibes.** `FlagSet` accumulates caveats with stable codes. The
Phase-3 agent layer may render flags; it may not invent or suppress them.
`report.py` is deliberately dumb for the same reason — it formats numbers that
already exist and computes nothing, giving the LLM report node a reference
rendering known to contain exactly those numbers.

---

## Verification

`scripts/audit.py` recomputes every model from first principles with literals
and different numerical methods, then compares against the package. Run it
alongside the test suite.

**Algebra: exact.** 21/21 independent recomputations match to machine
precision — combined roughness and slope, `k_s`, `E'`, `H_c`, the plasticity
index, CMY, `xi`, `Y`, all gas properties, the rarefaction parameter and the
whole resistance network.

**Absolute checks against known values:**

| quantity | model | reference | error |
|---|---|---|---|
| air mean free path, 300 K / 1 atm | 67.00 nm | ~67 nm | — |
| air thermal conductivity, 300 K | 0.02623 W/m·K | ~0.0262 | 0.1 % |
| air viscosity, 300 K | 1.8459e-5 Pa·s | 1.846e-5 | 0.00 % |
| Debye C_V (Si, 300 K) | 1.661e6 J/m³·K | 1.630e6 (expt) | 1.9 % |
| h_K (Si/Ge, 300 K) | 334 MW/m²·K | 315–368 (NEMD) | in range |

**Correlations checked against sources.** The CMY prefactor/exponent (1.25,
0.95 — Yovanovich's 1982 recorrelation), the Mikic elastic form, `E'`, and
the plasticity-index thresholds all verified. One source rendered the Mikic
group as `P/(√2·E'·m)`; its own stated simplification (m = 0.1 → prefactor
1.9) proves the correct form is `√2·P/(m·E')`, which is what is implemented —
0.975 vs 1.870 under that check.

**Two defects found and fixed:**

1. **Missing DMM elastic cutoff.** Scattering is elastic, so only frequencies
   present on *both* sides can cross. The integral must stop at the slowest
   branch of the softer solid. Omitting it inflated Si/Ge by 4.5× (1896 vs
   334 MW/m²·K) and broke detailed balance — h(Si→Ge) ≠ h(Ge→Si), which is
   exact for DMM. Both are now regression-tested.
2. **Monte Carlo centring.** The sampler preserved the arithmetic mean, which
   put every input's median below its nominal. Now median-preserving, verified
   at 0.9997 for a single monotonic input.

The residual ~6 % median–nominal gap is *not* a defect: it comes from
`H_c = min(Hv_a, Hv_b)`, where the softer of two uncertain realisations is
systematically softer than nominal. The report explains it when it exceeds 2 %.

## Benchmark: validated against 207 measured points

The contact path is no longer unvalidated. `tests/benchmarks/joint_conductance.csv`
holds 207 measurements — Ni 200 and SS 304, conforming rough flats in a
vacuum, 0.41–8.89 MPa — from Bahrami (2004) Appendix A.2, data originally
Hegazy (1985).

| | |
|---|---|
| geometric mean ratio | **1.030** |
| RMS difference | **12.7 %** |
| within a factor of 2 | **100 %** (207/207) |

Roughness spans 0.48 to 13.94 µm, a factor of 29, with no trend in the ratio.
Ni 200 and SS 304 independently land at gm = 1.030 — two metals differing
fourfold in conductivity agreeing to the same factor. The published figure for
the *extended* CMY correlation on these materials is 3.95–6.43 % RMS; this
package implements the plainer 1982 recorrelation.

**What it does not cover:** every point is in vacuum, so the gap model is
untouched and the area-scaling bridge contributes nothing measurable; eight of
nine specimens are plastic, so the elastic branch has no clean test. Details
and the remaining gaps are in `tests/benchmarks/SOURCES.md`.

### The gap model: a diagnostic, not a validation

`tests/benchmarks/gap_conductance_digitised.csv` holds 113 points **digitised**
from Song & Yovanovich's figures by `scripts/digitize_gap_figures.py`. It is
kept separate from the exact benchmark, tagged as digitised, and excluded from
the 1.030 statistic above, because the extraction has a documented bias: upper
bounds recover within 1–5 %, but lower gas pressures come out 25–45 % high
where the extractor loses the dense overlapping low-pressure cluster.

Run `python3 scripts/compare_gap_digitised.py`. The useful output is not the
aggregate but the **trend**: model/measured falls monotonically from 1.05 in
the rarefied limit to 0.679 in the continuum limit. Since `h_g → k_g/Y` as
`M → 0`, that shortfall cannot come from the rarefaction term or the
accommodation coefficients — it implicates the uniform-gap assumption. Local
gap thickness is Gaussian about `Y` and `1/(t+M)` is convex, so the true area
average exceeds the value at the mean thickness.

`gap_conductance_integral` evaluates that average and closes about a third of
the gap. It is **opt-in**; `k_g/(Y+M)` remains the default, with a test
asserting so. Changing a specified default on digitised evidence — with ~25 %
of the discrepancy still unexplained — would risk being worse than the error
it fixes.

## Honest status of the bundled data

**Every entry in `data/*.yaml` is marked `verified: false` and none has been
checked against a primary source.** Reliability varies sharply:

- **Gas properties look right.** Air at 300 K / 1 atm comes out at 67.0 nm
  mean free path and 0.0262 W/m·K — both match textbook values, which
  exercises the Sutherland fit, molar mass and kinetic-theory relation
  together. Two tests pin this.
- **Solid k, ρ, E, ν, sound velocities, θ_D** are standard handbook
  properties, likely close, but anisotropic constants have been flattened to
  single values.
- **Hardness is the weakest field** and can be off by a factor of two for a
  specific sample. It is loaded at LOW confidence deliberately.
- **Song–Yovanovich c₁/c₂ are null everywhere**, on purpose. They are fitted
  per material from load-dependent indentation data and are not safely
  guessable. Supply your own.

Measured joint-conductance data now ships in `tests/benchmarks/`, transcribed
from a published source by `scripts/build_benchmark_csv.py` so every number is
auditable. It covers the contact path in vacuum only; the gap path has no
*exact* measured comparison, only the digitised diagnostic described above.

**Si/Ge cannot validate Phase 0.** Two polished semiconductors in vacuum
exercise neither the plastic branch nor the gas model. Phase 0 needs metal
pairs with a pressure sweep, measured in vacuum and in gas on the same
specimens. Si/Ge validates Phase 1, where the atomistic physics arrives.

---

## Documentation

| File | What it is |
|---|---|
| [`docs/agent-instruction-v2.md`](docs/agent-instruction-v2.md) | **The spec.** What the agent must do at every step, which physics is fixed by rule rather than judgement, and the validation status of each component. Marks **[CORRECTED]** where v1 would have produced a wrong number and **[NEW]** where something was missing, with the rationale for each at the end. |
| [`docs/knowledge-base.md`](docs/knowledge-base.md) | **The retrieval corpus.** Every defect this project hit, in Symptom / Cause / Fix / Verify form. |
| [`docs/lammps-rag.md`](docs/lammps-rag.md) | **Official LAMMPS reference, 8 physics paths.** Thermal transport, elastic constants and relaxation, hardness/indentation, gas–surface accommodation, surface structure, amorphous melt–quench, electron–phonon (TTM), and gas transport. 20 example scripts verbatim plus 38 command pages, emitted from a version-pinned source tree by `scripts/build_lammps_rag.py`. Generated; rebuild rather than edit. |
| [`tests/benchmarks/SOURCES.md`](tests/benchmarks/SOURCES.md) | **The evidence.** Provenance of every benchmark number, what each dataset does and does not validate, and the digitisation quality audit. |

`docs/agent-instruction-v2.md` is the canonical copy of the instruction; it is
mirrored into the Claude project knowledge base. Read it before changing any
correlation, threshold or default — several constants in this package look
wrong and are not, and the file says why.

## Knowledge base

`docs/knowledge-base.md` is the agent layer's retrieval corpus: every defect
this project hit, in Symptom / Cause / Fix / Verify form, with literal LAMMPS
error text in the headings so lexical and semantic retrieval both hit. It
covers the LAMMPS errors, the silent failures that raise nothing at all, the
physics corrections that change the answer rather than refine it, run-quality
diagnostics, and what LAMMPS actually ships.

Three tests keep it from rotting: every flag code it names must exist, the
corrections that answers depend on must still be present, and entries must
carry a symptom rather than only a fix.

## Layout

```
openkapitza/
  schema/     units, Quantity, flags, inputs, YAML loaders
  contact/    hardness (Song-Yovanovich), regime selector, CMY + Mikic
  gap/        Sutherland gas properties, rarefaction, accommodation
  md/         DMM placeholder, Debye heat capacity, quantum screen
  network/    area-scaled series/parallel combination, Monte Carlo
  pipeline.py end-to-end run_continuum
  report.py   text rendering
  builder.py  the IO boundary
  compute/    JobRunner protocol, local + SLURM backends, cost model
docs/
  knowledge-base.md   retrieval corpus for the agent layer
scripts/
  audit.py            independent recomputation of every Phase-0 model
  verify_extraction.py  extraction against synthetic ground truth
  smoke_lammps.py     end-to-end LAMMPS shakedown
tests/
  test_physics.py     invariant tests, no published data needed
  test_nemd.py        Phase-1 geometry, decks, extraction, compute layer
  test_benchmarks.py  skips until a dataset is supplied
```

## Phase 1 — direct NEMD and the compute layer

> **Status, 22 Oct 2026.** The pipeline is ready for the cluster; no h_K is
> validated yet. Fixed today: the stationarity test now judges drift
> against its own noise (P1-1); the clamped-cell stress is removed by a
> piston z-relaxation (P1-2, P1-7); the profile uses every frame and h_K
> carries a jackknife block error; convergence verdicts take errors; spot
> pressure works as a loaded piston (P1-9); heterostructures (fcc on
> diamond) and MEAM build, and the Jelinek MEAM passes the elastic gate for
> Si, Al and Cu (P1-6). Tracker: [`docs/open-issues.md`](docs/open-issues.md).
> Cluster procedure: [`docs/cluster-runbook.md`](docs/cluster-runbook.md).
> The history below is kept for its reasoning.

**The pipeline has run end to end against real LAMMPS** (30 Sep 2026,
macOS/arm64, Si/Ge with SiCGe.tersoff): deck generation, execution, profile
parsing, reservoir-energy parsing and extraction all completed, exit 0.

**No physics result has been validated.** The shakedown cell is 21 nm long
and 2.17 nm square, run for 60 ps, and it returned h_K = 1348 MW/m²·K
against a published Si/Ge range of 266–368. Nine flags explain the gap:
37 % standard error on ΔT, R² of 0.24/0.11 on the fits, 28 % sensitivity to
the fit window, 15 % reservoir flux imbalance, and apparent conductivities
7 % and 36 % of bulk. The code reported every one of them rather than
presenting the number as an answer. That is the behaviour being verified
here, not the value.

Six defects were found by running on a real machine that 144 Linux-side
tests, an independent numerical audit and synthetic-profile validation had
all missed. They are listed under *What running it actually found* below.

### What IS verified about Phase 1

Every Phase-1 defect found so far was statically checkable in hindsight, so
`scripts/audit_nemd.py` now checks for that whole class. It parses the
rendered deck text back and recomputes each number from the spec by hand,
rather than reusing the builder's own helpers, so a unit or ordering error has
to appear in both places to survive.

Across **288 deck configurations** (cross-section, bulk length, bin width,
lattice constant, run lengths, profile cadence) it verifies: the six region
boundaries against the geometry in Ångström; that the extractor's fit windows
clear both Langevin regions and meet at the interface; that the flux area is
the *snapped* one and provably not the unsnapped value; `fix ave/chunk` Nfreq
divisibility, `Nevery*Nrepeat <= Nfreq`, a named chunk ID that is actually
defined, and a window that fits inside the production run; `reset_timestep 0`
before the final run; non-periodic z; `velocity create` at 2× the setpoint;
and the thermostat setpoints **as written into the deck**, with the group in
the hot region required to carry the higher value.

That last check exists because the first version of the audit compared
`spec.t_hot` against `spec.t_cold` — two properties checked against each
other, never reading the deck — so a template that swapped the two groups
would have passed. A mutation test caught it. `tests/test_nemd_audit.py` now
injects 11 defects into a known-good deck and asserts the audit reports each
one, plus a control asserting the clean deck still passes. An audit nobody
has tried to fool is not evidence.

Extraction is verified separately against 13 synthetic profiles with known
answers (`scripts/verify_extraction.py`): the extrapolated-fit method recovers
ΔT to within 1–13 %, where the naive adjacent-bin difference is off by
11–53 %. Note the 13 % worst case is the smallest jump (ΔT = 1 K) — the method
degrades as the jump shrinks toward the noise floor.

**None of this is physics validation.** A static audit confirms the deck says
what the spec meant; it cannot confirm the spec is right. Closing that gap
needs a converged run — see *Start here once LAMMPS is installed* below. The
convergence machinery itself (`check_series`, `check_flux_linearity`,
`richardson_extrapolate`) is unit-tested against synthetic series but has
never been run against real MD output.

### Reproduced on a second architecture

LAMMPS builds from source in a plain container (`cmake -D BUILD_MPI=no -D
PKG_MANYBODY=yes`), which is worth knowing because it does not need the
host's install and sidesteps a package index. Running the shakedown there —
x86_64 Linux, `stable_30Sep2026`, GCC — reproduced the earlier macOS/arm64
run on every reported diagnostic:

| | macOS arm64 | container x86_64 |
|---|---|---|
| h_K | 1348 | 1348.2 MW/m²·K |
| ΔT standard error | 37 % | 37 % |
| fit R² | 0.24 / 0.11 | 0.2383 / 0.1128 |
| exclusion sensitivity | 28 % | 28.3 % |
| reservoir flux imbalance | 15 % | 14.9 % |
| apparent k, A / B | 7 % / 36 % | 7 % / 36 % |

Agreement this close across architecture and compiler is more than a chaotic
60 ps trajectory would normally give, so read it as "the pipeline is
deterministic given a fixed seed and the diagnostics are dominated by the
deterministic setup" rather than as proof of bitwise portability. It does
confirm the extraction chain is not platform-sensitive, which is what the
macOS shakedown originally called into question.

That run also caught a false positive in the then-new potential check: the
runner copies the potential beside the deck and passes a **bare filename**,
because that is what LAMMPS resolves from its own working directory, but a
bare name is not resolvable from wherever `check_spec` runs — so it reported
`NEMD_POTENTIAL_UNVERIFIED` for a file sitting right there. `pair_file` (for
LAMMPS) and `potential_dir` (for the verifier) are now separate, and
`potential_dir` never enters the deck.

### A diagnostic that was missing entirely

Having an engine made a second run cheap, and comparing two durations of the
same cell exposed the most serious Phase-1 gap found so far:

| run | flux (W/m²) | ΔT (K) | h_K (MW/m²·K) |
|---|---|---|---|
| 60 ps | 1.757e10 | 13.031 | 1348 |
| 20 ps | 2.925e10 | 1.676 | **17453** |

A factor of 13 in h_K — and **both runs raised the identical set of eleven
flags.** Neither the extraction nor `check_run_health` separated them.
`NEMD_UNEQUILIBRATED` existed as a code and never fired, because it compares
mean temperature against the setpoint, and temperature equilibrating is not
the same as heat flow reaching steady state. The reservoirs hit their
setpoints long before the profile between them stops developing; while it
does, the flux is inflated and ΔT has not opened up, so h_K is wrong in
numerator and denominator at once. A flux that depends on how long you watch
is the tell.

`check_flux_stationarity()` now tests what must actually hold at steady state
— constant dE/dt in each reservoir — by fitting the rate over each half of the
averaging window:

| run | hot drift | cold drift |
|---|---|---|
| 60 ps | 19 % | 97 % |
| 20 ps | 76 % | 169 % |

It flags both, which is correct: neither was stationary, and the 60 ps result
was never trustworthy either — it merely had the decency to be the right order
of magnitude. The general lesson is in the knowledge base: a flag set that
does not change when the answer changes by 13× is not measuring the thing that
went wrong.

### Relaxation: measured, and it matters a lot

Two runs of the same cell differing only in `relax_structure`:

| | unrelaxed | relaxed |
|---|---|---|
| Pxx / Pyy | 6.53 / 6.53 GPa | **0.067 / 0.065 GPa** |
| Pzz | 5.76 GPa | 1.51 GPa |
| total pressure | 6.46 GPa | **0.50 GPa** |
| shared in-plane constant | 5.431 Å (Si, imposed) | **5.5968 Å** |
| h_K | 449.6 | **904.9** MW/m²·K |

In-plane stress is 99 % relieved, and the shared constant settles at 5.5968 Å
between Si (5.431) and Ge (5.658) — the strained-layer compromise, rather
than all −4.01 % dumped on the Ge side.

**h_K doubled.** Cell strain is not a second-order correction to this
measurement; it is a leading one. Neither number is a measurement — both runs
still fail the stationarity check — but the size of the shift settles whether
relaxation is worth doing before any convergence campaign. It is.

**Pzz keeps a 1.51 GPa residual** and in-plane relaxation cannot remove it: z
is non-periodic (walls) and cannot be barostatted. Closing that needs each
side built on its own out-of-plane constant set by its Poisson ratio — the
pseudomorphic strained-layer structure, i.e. the Phase-2 structure module.

That residual exposed three further defects, all now fixed:

- `NEMD_CELL_STRESS` tested the **mean** of `Press`, which is (Pxx+Pyy+Pzz)/3.
  Pxx=Pyy=0.067 with Pzz=1.51 averages to 0.55 GPa — under the 1 GPa
  threshold, no flag — while 1.5 GPa of uniaxial stress sat along the
  transport direction. It now checks components and names the worst.
- `check_run_health` averaged **every** thermo row in the log, mixing
  minimisation at 6.5 GPa and 540 K with production at 1.5 GPa and 300 K. It
  reported 3.64 GPa for a cell sitting at 1.51 — a state the run was never in.
  `last_thermo_block()` now restricts the checks to the production block,
  which `reset_timestep 0` makes unambiguous.
- The production `thermo_style` logged **no pressure columns at all**, so the
  stress check could only ever fire on the equilibration block, never where
  the measurement happens. It now logs press, pxx, pyy and pzz.

Throughput, measured: 4,787 atoms for 60 ps took 709 s single-core here. The
converged target (~112,000 atoms, 500 ps) extrapolates to roughly 38 core-hours,
so the container is fine for the small convergence rungs and too slow for the
top of the ladder.

### What running it actually found

Each of these survived every form of verification done beforehand, because
all of it ran on Linux against code I had written, not against LAMMPS.

| # | Defect | Why nothing caught it |
|---|---|---|
| 1 | `_pid_alive` read `/proc`, which does not exist on macOS, so a SIGKILLed job reported RUNNING forever | Linux-only syscall path; the suite hung rather than failed |
| 2 | The registry named `SiGe.sw`, which LAMMPS does not ship | A filename written from assumption |
| 3 | Element labels are `Si(D)`, not `Si` | Assumed the plain chemical symbol |
| 4 | Cross terms exist for only one of three Si parameterizations | Reading labels was not enough; completeness had to be checked |
| 5 | `fix ave/chunk` was missing its chunk ID | LAMMPS argument order, discovered two minutes into a run |
| 6 | The averaging window reached back before the fix existed, so the profile file was created empty with no error | Only visible after a full successful run |

Defects 2–4 are the same error narrowing: treating "what the file probably
contains" as knowledge. The fix was to make the code interrogate the
potential file and refuse combinations it cannot verify.

`validate_deck()` and `check_run_health()` now catch classes 5, 6 and the
cell-stress problem before or immediately after a run, instead of minutes in
or not at all.

### Start here once LAMMPS is installed

```bash
python3 scripts/smoke_lammps.py --lmp /path/to/lmp
```

A deliberately tiny Si/Ge cell — a few minutes on a laptop — that exercises
the whole chain: build deck → run → parse profile → parse reservoir energy →
extract h_K. **It is a plumbing test, not a measurement.** The box is far too
small for a converged number; what matters is that every stage runs and the
result lands in a plausible range rather than orders of magnitude out. The
script prints the corrected DMM value (334 MW/m²·K) and the published NEMD
range (266–368) for orientation only.

Expect the first attempt to fail. Deck syntax, the `SiGe.sw` potential path
and the `fix ave/chunk` column layout are the likely culprits, and the script
prints the relevant log tail for each.

Decks are **self-contained**: the structure is built in place with
`create_atoms` on a diamond lattice, so no external data file is needed. That
uses one lattice constant for both sides, leaving Ge about 4 % strained —
flagged as `NEMD_LATTICE_STRAIN`, since the strain shifts phonon frequencies
and therefore h_K. Phase 2's structure module replaces this with a
strain-minimising supercell. `structure_mode="read_data"` takes an external
file instead.

The periodic cross-section snaps to a whole number of unit cells, and the
flux uses that snapped area — a 5.000 nm request becomes 4.888 nm, a 4.4 %
area difference that would otherwise have gone straight into h_K.

### Extraction is the piece that matters

Differencing the bins either side of the interface does not give the jump. It
gives the jump plus the bulk gradient across the distance to those bins:

```
dT_naive = dT_true + q·(dz_A/k_A + dz_B/k_B)
```

Both terms are positive, so it **overstates** the jump and understates h_K —
and the error is proportional to q, so it masquerades as physical flux
dependence and survives a careless convergence check. `md/profile.py` fits
each side's linear region and extrapolates both to the interface plane.

Against synthetic profiles with a planted 8.000 K jump:

| case | extrapolated fit | adjacent bins |
|---|---|---|
| clean | +0.00 % | +0.20 % |
| noise 0.4 K | +1.08 % | +10.85 % |
| interface curvature 1.5 K | +3.59 % | +33.75 % |
| wide curvature 6 nm | +26.09 % | +68.95 % |

Across a flux sweep the fit's error stays flat at +3.7 % while the naive error
holds near +22 % and climbs with flux — the signature that makes it dangerous.

Quality gating is on the **standard error of the extrapolation**, not R².
With a shallow gradient R² measures signal-to-noise, not linearity, and
rejects perfectly good fits. The standard error of the mean response at the
interface grows with the distance extrapolated, which is the right penalty,
and it gives h_K an honest uncertainty instead of a pass/fail.

`NEMDGeometry` publishes the exact bulk ranges and hands them to the
extractor. Guessing the fit window as a fraction leaves thermostatted bins in
the fit and biases the jump low — found during validation, now flagged.

### Compute layer

`JobRunner` submits and returns immediately; nothing blocks on a run that
takes days. Handles are JSON and survive the session, so a job can be picked
up later — which is what lets the Phase-3 agent put a run behind an interrupt.

Two bugs worth recording, both found by a test hanging rather than failing:
an unreaped child becomes a **zombie** and answers `os.kill(pid, 0)`, so
`poll()` reported RUNNING forever; and a command containing `exit N`
terminated the wrapper before the exit status was recorded. Both are fixed
and regression-tested, including `kill -9 $$`.

The cost model feeds Phase 0's gate — the continuum model says how much h_K
could move the answer, this says what finding out costs. Same 40k-atom system:
0.2 node-hours on Stillinger–Weber, 248 on an MLIP.

### Convergence

Four axes: cross-section, length, duration, flux. Flux linearity is a
*validity* test, not a convergence test — h_K = q/ΔT only means anything if
ΔT ∝ q through the origin. A significant intercept means a fixed offset that
is not the interface.

## Phase 2 — structure module (in progress)

`openkapitza/structure/`: analysis (density, partial RDFs, coordination, bond
angles; exact on ideal crystals), a-SiO₂ melt-quench with a validation gate
against sourced targets and an always-raised quench-rate caveat, and a
Si(100)/a-SiO₂ join whose bonding choices (cut, gap, anneal) and resulting
chemistry (Si–O bonds per nm², coverage, class) are reported.
Heterostructures for crystalline pairs (fcc on diamond) are built by the NEMD
deck itself. End to end: `scripts/build_si_sio2.py`. Status and open items:
[`docs/open-issues.md`](docs/open-issues.md).

## Next
- **Phase 3** — LangGraph agent layer wrapping Phases 0–2.
- **Phase 4** — Tier-2 MLIPs behind a phonon-DOS/elastic-constant gate, and the
  temperature-fluctuation equilibrium cross-check (not Green–Kubo).

### Action items — next session

> **The tracker is now [`docs/open-issues.md`](docs/open-issues.md).** It
> supersedes this block where they differ; the block below is kept for its
> reasoning.

Ordered. The agreed sequence is: confirm Phase 0 and Phase 1, **then** Phase 2.

**Items 1 and 2 are now answered; items 1b and 2b replace them.**

**1 — DONE.** The potentials' lattice constants match experiment to 0.02 %, so
a0 was not the explanation. The *elastic constants* are 8–18 % off and invert
which of Si/Ge is biaxially stiffer (`scripts/measure_potential_elastic.py`).

**1b — the in-plane constant is converged, and the residual gap has a
candidate.** A 50,000-step relaxation holds `lx` at 22.39 Å from step 2,500
onward (a₍∥₎ = 5.5968 Å), with Pxx, Pyy inside ±0.08 GPa. So 73 %-toward-Ge is
real, not a convergence artifact, against 51 % predicted from the potential's
own C_ij. **Leading hypothesis:** the prediction assumes a free-standing
bilayer with σzz = 0, but z is clamped by the walls at Pzz ≈ +1.5 GPa
compressive, which couples through Poisson's ratio to expand the in-plane
constant — pushing a₍∥₎ up, toward Ge, in both the right direction and roughly
the right size. Test by repeating the relaxation with a longer z or a free
surface and seeing whether a₍∥₎ falls toward 5.55 Å.

**2 — ANSWERED, and it changes the method.** The completed 400 ps relaxed run
is still "not equilibrated" in all 23 windows — but the drift does not *decay*.
Fitting drift against window index on the full trace gives a slope of
**−0.41 ± 1.15 %/window (t = −0.35)**: no trend at all. A decaying transient
would trend clearly and negatively. This is the metric's noise floor, not an
unfinished transient, so no run length will fix it.

*Correction to the partial-data reading:* the truncated 294 ps trace gave
**+3.81 ± 1.86 %/window (t = +2.04)**, which I reported as a trend. The last
106 ps removed it. The drift metric is a ratio of two fitted slopes, exactly
the kind of quantity that moves when the tail of the data arrives — so the
conclusion survived but the number quoted for it did not. Worth remembering
before quoting a fitted trend from a run still in progress.

**2b — give the stationarity test a noise floor.** With 4,787 atoms and a
2.2 nm cross-section the reservoir power fluctuates enough that the drift
metric's own uncertainty exceeds the 15 % tolerance, making the test
unsatisfiable regardless of physics. Quantified on the full run: drift spans
3.4–152.7 %, mean 46 %, standard deviation 36 %, so a 15 % tolerance sits
**0.9 standard deviations below the mean** — unreachable by construction. Compare drift against its *statistical
uncertainty* rather than a fixed threshold, and/or widen the cross-section to
cut the noise. Until that is fixed, `check_flux_stationarity` reports "not
steady" for every run, which is useless in a different way from reporting
nothing. Relaxation *did* help the energy balance: imbalance reached 2–6 % in
several windows against a floor of 21 % unrelaxed.

**What the completed 400 ps relaxed run did achieve.** Three flags cleared
against the short unrelaxed run — `NEMD_POOR_FIT`,
`NEMD_EXCLUSION_SENSITIVE` and `NEMD_FLUX_INCONSISTENT` all gone, so the fit
is now well determined, the extracted jump no longer depends on where the
fitting stopped, and the reservoir fluxes agree. h_K came out
**427.8 MW/m²·K** against a published 266–368 and a corrected Debye-DMM
estimate of 334 — still high and still not a measurement (`NEMD_UNEQUILIBRATED`
and `NEMD_CONDUCTIVITY_MISMATCH` remain), but the right order and far from the
1348 and 17453 of the unconverged runs. a₍∥₎ reproduced to 0.01 % against the
shorter relaxation, confirming convergence.

`NEMD_CELL_STRESS` now fires where it previously would not: mean pressure is
0.715 GPa, *below* the 1 GPa threshold, so only the component-based check
added today catches the 1.51 GPa Pzz. That is the fix working on real data.

**3.** Then the convergence campaign (cross-section, length, duration, flux),
~100 core-hours — but only after 2b, or the campaign measures noise.

**4.** Phase 2 structure module, which 1b feeds directly.

**5. MD-derived inputs** — designed in §11 of the agent instruction: when an
MD value may *fill* a missing input, only be *compared* with a measurement, or
must *defer*. Build order, cheapest-first within value:

- **5a. Elastic constants → E, ν.** Built (`md/elastic.py`): a0, C11, C12,
  C44, two strain magnitudes, Voigt–Reuss–Hill E and ν, the 10 % potential
  gate. Reproduces the official `examples/ELASTIC` on Si.sw exactly.
- **5b. Accommodation coefficient α.** Built (`md/accommodation.py`): real
  Ni(100) slab, calibrated gas–solid well, paired-collision estimator,
  measured surface temperature and coverage. Validated against walls with
  imposed α (`scripts/validate_accommodation.py`). He/Ni 0.09–0.12 across
  the adsorption-energy range (classical, so an upper estimate), Ar/Ni
  0.52 ± 0.12.
- **5c. Bulk thermal conductivity k.** Built (`md/conductivity.py`):
  periodic Langevin NEMD after `examples/KAPPA`, 1/k vs 1/L extrapolation.
  Metals return the lattice part only. **Not yet validated**: SW-Si at 500 K
  over 8.7–34.8 nm is noise-limited at 400 ps and the extrapolation is
  undetermined (flagged CRITICAL). Needs ns-scale runs on HPC.
- **5d. Nanohardness, compare-only.** Built (`md/indentation.py`): 3D
  quasi-static sphere indentation, cap-model area, full P(h), pop-in,
  Hertz E*. Ni(100): H = 19.4 ± 1.3 GPa. Never fills c₁, c₂.

Run any of them with `scripts/run_md_inputs.py {accommodation,kappa,indent}`.

Superseded detail from the original item 1:

1. **Measure the potentials' own equilibrium lattice constants.** The relaxed
   in-plane constant came out at 5.5968 Å, which is 73 % of the way from Si
   (5.431) toward Ge (5.658). Si is the stiffer material, so a naive elastic
   argument puts the compromise nearer Si, not Ge. Either the reasoning is
   wrong or `Si(D)`/`Ge` in `SiCGe.tersoff` have equilibrium constants that
   differ from the experimental values the geometry is built from — in which
   case part of the "mismatch" being relaxed away is a potential artifact and
   every strain number downstream inherits it. Two short bulk NPT runs settle
   it. Cheap, and it calibrates everything after it. **Do this first.**
2. **Find the equilibration time.** Both runs fail `check_flux_stationarity`
   (hot/cold drift 19 %/97 % at 60 ps). No convergence rung is worth running
   until drift is under tolerance, or the whole ladder measures transients.
   Sweep `steady_steps` upward and watch the drift fall.
3. **Then the convergence campaign** — cross-section, length, duration, flux.
   ~100 core-hours total; the container does the lower rungs, the top needs
   more cores.
4. **Phase 2 structure module**, which items 1 and 3 feed: the 1.51 GPa Pzz
   residual cannot be relaxed in-plane and needs per-side out-of-plane
   constants from Poisson ratios.

### Open items carried forward

1. **MD gas–surface scattering for the accommodation coefficients.** The one
   piece of the original Phase-0 scope that is not built: §6 of the spec asked
   for the gap model *with* an optional MD accommodation run, and only the
   lookup of defaults exists. This matters more than "optional" suggests —
   the spec itself calls accommodation coefficients the weakest input in the
   whole model, measured values for one gas/solid pair span an order of
   magnitude with surface cleanliness and adsorbed layers, and the defaults
   are hard-flagged LOW confidence. Until this lands, the weakest input in
   the model has no route to improvement. Needs MD, so it sits with Phase 1/4
   work rather than blocking Phase 0.
   **Update:** LAMMPS supplies the *forward* model — `fix wall/reflect/stochastic`
   with `diffusive`/`maxwell`/`ccl` styles takes accommodation coefficients as
   arguments, with an example per model in `examples/wall`. It does not
   *measure* α, which still needs gas atoms fired at a real atomistic surface.
   So this is now a scripting job against an existing facility rather than
   something to build from nothing.
2. **The gap-model continuum deficit.** ~32 % low at M/Y < 0.1, monotone in
   M/Y, cause identified (uniform-gap assumption), opt-in fix implemented,
   default deliberately unchanged, ~25 % of the discrepancy unexplained.
   Settling it needs exact tabulated h_j(P_gas) rather than digitised figures.
3. **Elastic branch and the ξ·h_K bridge have no measured check** — 8 of 9
   benchmark specimens are plastic, and the Kapitza term is negligible at
   these conductances. Neither is a defect; both need different experiments.
