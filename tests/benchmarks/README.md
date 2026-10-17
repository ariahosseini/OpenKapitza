# Benchmark datasets

**These files are empty on purpose.** No measured data ships with the package,
because inventing plausible-looking experimental numbers to make a validation
suite go green is worse than having no validation suite at all.

To validate the Phase-0 continuum core, populate `joint_conductance.csv` from a
published source and run `python3 -m unittest tests.test_benchmarks`.

## What to populate it with

Phase 0 contains no atomistic physics, so **Si/Ge cannot validate it**. Two
polished semiconductors in vacuum exercise neither the plastic branch nor the
gas model. The datasets that do exercise this code are joint-conductance
measurements versus contact pressure on metal pairs, ideally the same specimen
pair measured both in vacuum and in a gas:

- stainless/stainless, Ni/Ni or Al/Al, with roughness and slope reported
- a pressure sweep spanning the CMY window, 1e-5 < P/H_c < 2e-2
- vacuum and gas (air, helium, argon) on the same specimens, which separates
  the contact path from the gap path
- stated microhardness, preferably load-dependent so c1/c2 can be fitted

Yovanovich and co-workers' thermal-contact-conductance measurements are the
usual reference set for this. Record the exact source for each row.

## Column schema

| column | unit | notes |
|---|---|---|
| `case_id` | - | free label, unique per row |
| `material_a`, `material_b` | - | keys from `openkapitza/data/materials.yaml` |
| `sigma_a_um`, `sigma_b_um` | um | per-surface RMS roughness |
| `slope_a`, `slope_b` | - | mean absolute asperity slope; blank to force the correlation |
| `pressure_MPa` | MPa | apparent contact pressure |
| `temperature_K` | K | mean joint temperature |
| `gas` | - | gas key, or blank for vacuum |
| `gas_pressure_kPa` | kPa | blank in vacuum |
| `h_measured_W_m2K` | W/(m^2 K) | the measured joint conductance |
| `h_uncertainty_W_m2K` | W/(m^2 K) | blank if not reported |
| `source` | - | citation, required |

## Before you trust a pass

The bundled material properties in `openkapitza/data/materials.yaml` are all
marked `verified: false`, and the hardness values in particular are rough. A
benchmark run against unverified hardness is testing the database as much as
the model. Replace the entries for whichever materials your dataset uses, and
set `verified: true`, before reading anything into the agreement.

Where the source reports its own hardness for the specimens, use that rather
than the bundled value -- it is the single input that most directly sets
P/H_c, and therefore h_c.
