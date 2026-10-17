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

## Next

- **Phase 1** — `JobRunner` protocol (local + SLURM) and direct NEMD with
  Tier-1 potentials. ΔT from linear fits extrapolated to the interface plane,
  not adjacent bin temperatures. Validate h_K on Si/Ge.
- **Phase 2** — structure module: crystalline supercell matching, then
  melt-quench with quench-rate reporting and RDF/bond-angle validation.
- **Phase 3** — LangGraph agent layer wrapping Phases 0–2.
- **Phase 4** — Tier-2 MLIPs behind a phonon-DOS/elastic-constant gate, the
  temperature-fluctuation equilibrium cross-check, accommodation-coefficient MD.
