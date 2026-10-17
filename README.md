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

```
python3 -m unittest discover -s tests    # 63 tests
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

No measured joint-conductance data ships with the package. `tests/benchmarks/`
is empty by design — inventing plausible experimental numbers to turn a
validation suite green is worse than having none. See that directory's README
for what to populate it with.

**Si/Ge cannot validate Phase 0.** Two polished semiconductors in vacuum
exercise neither the plastic branch nor the gas model. Phase 0 needs metal
pairs with a pressure sweep, measured in vacuum and in gas on the same
specimens. Si/Ge validates Phase 1, where the atomistic physics arrives.

---

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
tests/
  test_physics.py     63 invariant tests, no published data needed
  test_benchmarks.py  skips until a dataset is supplied
```

## Phase 1 — direct NEMD and the compute layer

**Not yet run against LAMMPS.** No LAMMPS binary exists in CI, so the decks
this generates have never been executed. Everything around that boundary is
verified: extraction against synthetic profiles with a known answer, parsers
round-tripped against LAMMPS-format files, decks checked directive by
directive, job runners driven through real success and failure paths. Treat
the first real run as a shakedown.

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

## Next

- **Phase 2** — structure module: crystalline supercell matching, then
  melt-quench with quench-rate reporting and RDF/bond-angle validation.
- **Phase 3** — LangGraph agent layer wrapping Phases 0–2.
- **Phase 4** — Tier-2 MLIPs behind a phonon-DOS/elastic-constant gate, the
  temperature-fluctuation equilibrium cross-check (not Green–Kubo),
  accommodation-coefficient MD.

Still outstanding from Phase 0: **no comparison against a measured joint
conductance.** `tests/benchmarks/` remains empty by design.
