# OpenKapitza knowledge base

Retrieval corpus for the agent layer. Every entry below is something that was
actually got wrong in this project and cost a debugging cycle, or a physics
correction that changes the answer rather than merely refining it.

**How to use.** Each entry is self-contained so a single retrieved chunk is
actionable without its neighbours. Headings carry the literal error text where
one exists, so lexical and semantic retrieval both hit. Before generating a
LAMMPS deck, choosing a potential, interpreting a result or explaining a flag,
retrieve against this file first.

**How to extend.** Add an entry the moment a failure is diagnosed, in the same
Symptom / Cause / Fix / Verify shape. An entry that records only the fix and
not the symptom is not retrievable by the person who next hits it.

---

## 1. LAMMPS errors with a literal message

### ERROR: Potential file is missing an entry for: Si Si Si

**Symptom.** `pair_coeff * * SiCGe.tersoff Si Ge` fails immediately after the
potential file is read. The box has already been built and atoms created.

**Cause.** The element label passed to `pair_coeff` is not a label the file
uses. LAMMPS distinguishes parameterisations in the label itself: the Tersoff
silicon entries are written `Si(B)`, `Si(C)`, `Si(D)`, never bare `Si`.

**Fix.** Read the labels out of the file instead of assuming the chemical
symbol. `openkapitza.md.nemd.potential_elements()` parses them; entries begin
with three labels followed by numeric parameters and may wrap across lines, so
a continuation is recognised by its first token parsing as a float.
`match_species()` maps `Si` onto `Si(D)` by ignoring the parenthesised tag.

**Verify.** The chosen labels are printed before the run; check them against
`grep -v '^#' <file> | head`.

---

### ERROR: Potential file is missing an entry for: Si(B) Si(B) Ge

**Symptom.** As above but with a correct-looking label. The silicon label
exists in the file and so does `Ge`, yet the pair fails.

**Cause.** A file may carry several parameterisations of one element while
providing cross-species terms for only one of them. `SiCGe.tersoff` has three
silicon fits; the Si–Ge cross terms exist for exactly one. A two-element
many-body system needs all **eight** ordered triples over the two labels.

**Fix.** Validate completeness before writing the deck.
`potential_triples()` returns every `(el1, el2, el3)` entry; `complete_pairs()`
returns only label pairs for which all eight required triples are present. Never
select on label existence alone.

**Verify.** `complete_pairs(potential_triples(f), "Si", "Ge")` returns a
non-empty list. An empty list means that file cannot do that pair, whatever
labels it contains.

---

### ERROR: No values from computes, fixes, or variables used in fix ave/chunk command

**Symptom.** Fails minutes into a run, after equilibration, at the line setting
up the profile fix.

**Cause.** `fix ave/chunk` takes the **chunk ID** between `Nfreq` and the
values:

```
fix ID group ave/chunk Nevery Nrepeat Nfreq chunkID value ... keyword args
```

Omitting it makes LAMMPS consume the first value as the chunk ID and find
nothing left to average.

**Fix.** Name the `compute ... chunk/atom` ID explicitly:

```
compute layers gmobile chunk/atom bin/1d z lower 4.0 units box
fix profile gmobile ave/chunk 100 150 15000 layers temp file temperature_profile.dat
```

**Verify.** `validate_deck()` checks this without running anything.

---

## 2. Silent failures — no error is raised

### temperature_profile.dat contains only its header; LAMMPS exits 0

**Symptom.** The run completes, prints its final marker, exits 0. The profile
file exists, carries its three `#` header lines, and holds no data. No warning
anywhere in the log.

**Cause.** `fix ave/chunk` writes on multiples of `Nfreq` and averages
*backwards* over `Nevery*Nrepeat` steps. If production starts at a non-zero
timestep, the window for the first candidate output reaches back before the fix
was defined. Concretely: production started at step 12000 with `Nfreq=60000`;
the only candidate output step inside the run was 60000, whose window spanned
steps 100–60000, requiring 119 samples that never existed. LAMMPS wrote nothing
and said nothing.

**Fix.** `reset_timestep 0` immediately before the production block, so every
averaging window falls inside the run being measured. Recreate the reservoirs
*after* the reset so their accumulators cover the production window only.

**Fix, second part.** Set `Nfreq = production / frames` (default 4 frames)
rather than the whole run. The extractor uses the last frame; the earlier ones
make drift visible when a run has not reached steady state, which a single
whole-run average hides completely.

**Verify.** `validate_deck()` rejects an `ave/chunk` with no preceding
`reset_timestep 0`.

---

### A SIGKILLed job reports RUNNING forever and the waiting loop hangs

**Symptom.** On macOS or BSD, a job killed with SIGKILL never reaches a
terminal state. A polling loop spins until its deadline. On Linux the same code
is fine.

**Cause.** An unreaped child becomes a **zombie**, and `os.kill(pid, 0)`
succeeds on a zombie. The original detection read `/proc/{pid}/stat`, which does
not exist outside Linux, so the check silently degraded to the broken
`os.kill` path. The path is reachable because SIGKILL cannot be trapped, so the
wrapper's `EXIT` trap writes no sentinel file.

**Fix.** Three steps, because no single one is portable: reap with
`waitpid(WNOHANG)` when this process is the parent; then read `/proc` on Linux
or `ps -o state= -p PID` on macOS/BSD, where `Z` means zombie; then fall back to
`os.kill`.

**Verify.** The regression test asserts a terminal state within 15 s, so a
recurrence fails fast instead of consuming the timeout.

---

### A command containing `exit N` leaves no exit status

**Symptom.** A job that plainly finished is reported as failed or unknown; no
sentinel file is written.

**Cause.** The wrapper script writes the exit status on its last line. A
submitted command containing `exit N` terminates the wrapper before that line
runs.

**Fix.** Run the command in a subshell and write the sentinel from a `trap ...
EXIT`, so it is written even when the script dies on a trappable signal.

---

## 3. Physics corrections that change the answer

### h_K and h_c are per different areas — the naive series sum is wrong by 1/ξ

**Symptom.** None. The number looks plausible and is wrong by two to three
orders of magnitude in the interfacial term.

**Cause.** `h_c` from CMY or Mikic is a conductance per unit **apparent** area
and already contains the constriction resistance of the discrete contact spots.
`h_K` from MD is per unit **real contact** area — an atomically flat patch where
the solids actually touch. Adding `1/h_c + 1/h_K` treats them as the same
quantity.

**Fix.**

```
R_solid = 1/h_c + 1/(ξ · h_K)      series: spots, then the interface
h_joint = 1/R_solid + h_g           parallel: solid path and gas path
```

with ξ = A_real/A_apparent, which is `P/H_c` on the plastic branch and
`√2·P/(m·E')` on the elastic branch. Typical ξ is 10⁻³–10⁻², so the naive sum
understates the Kapitza contribution by that factor.

**Verify.** `R_kapitza / R_kapitza_raw == 1/ξ` exactly; pinned by test.

---

### NEMD must be run at the spot pressure, not the apparent contact pressure

**Symptom.** None in the output. The simulated cell is simply not the physical
state of a contact spot.

**Cause.** Inside a plastically deforming contact spot the local normal stress
is of order the contact microhardness H_c — GPa. The apparent contact pressure
is MPa. They differ by three orders of magnitude, and the apparent pressure's
only role is setting the area fraction ξ.

**Fix.** Apply `H_c` as the NEMD cell pressure; `NetworkResult.spot_pressure`
reports it. Offer a pressure sweep, since h_K(σ) is of interest in itself.

**Caveat found later.** A cell already carrying lattice-mismatch strain is under
GPa-scale stress before any pressure is applied (see *6.4 GPa* below).
Deliberate spot pressure and accidental mismatch strain then confound. Relax the
cell, or use a strain-minimising supercell, before loading it.

---

### Plastic CMY is not the default branch; check the plasticity index

**Symptom.** A hard, smooth, low-slope pair returns a contact conductance that
can be off by a large factor.

**Cause.** The plastic Cooper–Mikic–Yovanovich correlation assumes the
asperities yield. Two hard polished surfaces deform elastically. Polished Si/Ge
sits at ψ ≈ 0.12 — firmly elastic, not borderline.

**Fix.** ψ = E'·m/H_c, with 1/E' = (1−ν₁²)/E₁ + (1−ν₂²)/E₂.
ψ ≳ 3 plastic, ψ ≲ 0.33 elastic, in between compute **both** branches and report
both. The thresholds are Mikic's published values, written here in the
reciprocal convention (γ = 1/ψ) so larger means more plastic. Deformation mode
is set by material properties and asperity shape, not by applied load, so ψ does
not move with pressure.

---

### Where the constant 1.62 sits in the Song-Yovanovich relation

**Symptom.** Two published transcriptions disagree. One reads
`P/H_c = [P/(1.62·c1·(σ/m)^c2)]^(1/(1+0.071·c2))` with 1.62 outside the power;
the other has the group as `H_b·(1.62·σ/(m·d_b))^c0`, with it inside.

**Resolution.** Inside. Both sources also give the implicit system the explicit
form approximates; solving it settles the question numerically. Across SS 304
and Ni 200 fits at 0.5–8 MPa, the inside reading tracks the implicit solution
to 0.5–1.4 %, and the outside reading is uniformly 45 % low.

**Rule.** Do not change a verified formula on the strength of a PDF text
extraction. Tables and equations extract badly; the prose around them, and any
implicit definition given alongside, are more reliable. Where an implicit
definition exists, solve it and compare.

**Verify.** `TestSongYovanovichAgainstTheImplicitSystem` pins both directions:
the implementation agrees within 2 %, and the rejected reading is asserted to
be badly wrong so nobody "corrects" it back.

---

### Contact microhardness EXCEEDS bulk Vickers at light load

**Symptom.** Substituting a handbook Vickers number for H_c biases the contact
conductance high, not low.

**Cause.** The indentation size effect. The Vickers exponent c2 is negative, so
`HV = c1·(d_V/d0)^c2` rises as the indent shrinks. At the light loads of a real
joint the microcontacts are small and H_c is therefore larger than the bulk
number. For SS 304 at 2 MPa the fitted value is 3.41 GPa against a bulk 2.0.

**Consequence.** The fallback UNDERestimates H_c, so it OVERestimates P/H_c and
h_c. An earlier version of this project's own flag text asserted the opposite
direction, and the error survived because every material in the database had
null coefficients, so the Song-Yovanovich path never executed.

**Rule.** A code path that no test data exercises is not verified, however
carefully it was written. Check which branches your fixtures actually reach.

---

### H_c is not the handbook Vickers hardness

**Symptom.** A systematic bias in h_c, usually low.

**Cause.** Contact microhardness is load dependent, hence depends on pressure
and topography. A bulk Vickers number overestimates H_c, which understates
P/H_c, and h_c goes as `(P/H_c)^0.95` — the error passes through almost
linearly.

**Fix.** Song–Yovanovich:
`P/H_c = [P / (c₁·(1.62·σ/m [µm])^c₂)]^(1/(1+0.071·c₂))`, with c₁ and c₂ fitted
to load-dependent indentation data on the actual surfaces, H_c from the
**softer** material. When c₁/c₂ are unavailable, fall back to bulk Hv and flag
it loudly — do not guess the coefficients.

---

### `fix indent` reports load, not hardness — and the load is not monotonic

**Symptom.** Expecting `examples/indent` to give a hardness, or a load that
rises steadily with depth.

**Cause.** The fix exposes the force on the indenter (`f_ID[1..3]`) and an
interaction energy. Nothing in it computes contact area, so nothing in it is a
hardness. And materials yield: running the official `in.indent.min` with the
load printed gives 0.049 → 0.150 → 0.135 → 0.185 → 0.128 → 0.136 as the
indenter goes deeper. The drops are pop-in events.

**Fix.** Compute contact area separately (from indenter radius and depth, or
from atoms in contact), and report the whole load–depth curve, because a
single "hardness" depends on where the curve is read. The official example is
2D Lennard-Jones in reduced units — a mechanics template, not a material.

**Never use the result to fill c₁, c₂.** Those come from Vickers microhardness
over micron-scale indents; MD indents nanometres of perfect crystal at extreme
strain rates and typically returns several times the microhardness. Compare
against nanoindentation instead. See §11 of the agent instruction.

### MD thermal conductivity of a metal is the phonon part only

Classical MD has no electrons. In metals the electrons carry most of the heat:
a Wiedemann–Franz estimate gives roughly 60 % in SS 304 and roughly 90 % in
nickel. An MD k for a metal is therefore a lattice-only lower bound and must
never stand in for the total k in kₛ. (For nickel, WF with the Sommerfeld
Lorenz number actually overshoots the measured total — nickel's Lorenz number
is below the ideal value — which is itself a reminder that WF is an estimate.)

### LAMMPS already provides facilities this project called "not modelled"

Searching the official examples beyond the thermal path turned up three
capabilities the project had written off or hand-rolled. Recorded here because
each changes a plan, not just a detail. All are in `docs/lammps-rag.md`.

**1. Accommodation coefficients — `fix wall/reflect/stochastic`.** Styles
`diffusive`, `maxwell` and `ccl` (Cercignani–Lampis) take accommodation
coefficients directly as arguments
(`pos temp velx vely velz accomx accomy accomz`), with `examples/wall`
shipping one script per model. Be precise about what this does and does not
give:

- It is the **forward** model: impose a known α and get the resulting
  gas–surface energy exchange. That is useful for the gap path directly.
- It does **not measure** α. The reflection is from an idealised wall, not a
  real atomistic surface. Measuring α still means firing gas atoms at a real
  slab and comparing incident with reflected energy against the wall
  temperature. There is no canned example for that.

So the unbuilt Phase-0 item changes from "no facility exists" to "the forward
model is built in; the measurement needs a custom script".

**2. Electron–phonon coupling — `fix ttm` / `fix ttm/mod`.** The spec flags
Si/Al as a metal–semiconductor interface where electron–phonon coupling
carries part of the heat and states that nothing in the workflow models it.
LAMMPS implements a two-temperature model, and `examples/ttm` ships an Si
parameterisation (`Si.ttm_mod`) plus `in.ttm.thermal`. The caveat can become a
plan rather than a disclaimer.

**3. Elastic constants — `examples/ELASTIC`.** The official method is
`change_box ... remap units box` followed by `minimize`, applied at plus and
minus the strain, which is the method used in
`scripts/measure_potential_elastic.py` — so that measurement was done the
right way. The official script adds a check this project did not fully meet:
"it is important to verify that the minimization of energy w.r.t atom
positions in the deformed cell is fully converged. One indication of this is
that the elastic constants are insensitive to the choice of `${up}`." Our
measurement used one strain set with a residual check; repeating at a second
magnitude would close that. `compute born/matrix` is the modern alternative
that avoids finite differences entirely.

Also noted: `examples/vashishta` ships a real SiO2 parameterisation and a
quartz data file, which is the Si/SiO2 amorphous path's starting point.

### Two differences from the official LAMMPS thermal scripts

`docs/lammps-thermal-rag.md` holds the official `examples/KAPPA` scripts and
command pages. Reading them against our deck turned up two places where we
differ. Neither is a defect, and both are worth knowing.

**1. Slab thermostatting: static group vs `compute temp/region`.** The manual's
idiom is to apply `fix langevin` and point it at a region-restricted
temperature compute:

```lammps
compute Thot all temp/region hot
fix     hot all langevin ${thi} ${thi} 1.0 59804 tally yes
fix_modify hot temp Thot
```

`fix_langevin.rst` confirms this restricts the thermostat to that region:
"to apply the thermostat only to atoms within a spatial region ... This is not
done by default, but only if the fix_modify command is used to assign a
temperature compute to this fix that includes such a bias term."

We instead use a static group (`group ghot region hot`). For a crystalline
solid the two are near-equivalent, because atoms vibrate about lattice sites
and do not migrate between slabs, and a static group has the advantage of a
**fixed** DOF count — `compute_temp_region.rst` warns that a region
temperature "should probably not be used with Nose-Hoover style fixes ... if
the degrees of freedom included in the computed temperature vary with time."
For a liquid, where atoms cross slab boundaries, the official idiom is the
correct one and ours would be wrong.

**2. `fix ave/chunk` with `units box` and a changing box.** From
`fix_ave_chunk.rst`: "If its *units* keyword is set to *box* or *lattice*,
then the number of bins and size of each bin will be fixed over the Nfreq time
window, which can affect **which atoms are discarded** if the simulation box
size changes."

We use `units box`, and we now run an NPT relaxation that changes the box. We
are safe only because the relaxation finishes before production and the box is
fixed thereafter. Anything that barostats *during* production would silently
drop atoms from the profile. This is also the likely companion to the
`NEMD_FEW_BINS` report of "5 of 53 bins empty" — bins that extend past the
slab. Worth re-checking if a barostat is ever added to the production stage.

### A potential can nail the lattice constant and still invert the stiffness

**Symptom.** The relaxed shared in-plane constant of a coherent Si/Ge cell
came out 5.5968 Å — 73 % of the way from Si (5.432) to Ge (5.657). Si is the
stiffer material, so the compromise should sit *nearer Si*, around 42 %.

**First hypothesis, and it was wrong.** That the potential's own lattice
constants differ from the experimental values the geometry is built from, so
part of the "mismatch" being relaxed away is an artifact. Measured:
Si(D) 5.4320 Å against 5.431, Ge 5.6567 Å against 5.658 — agreement to
0.02 %. The potential is not the problem *there*.

**Actual cause.** The elastic constants, which nobody had checked:

| | a0 | C₁₁ | C₁₂ |
|---|---|---|---|
| Si(D) potential | 5.4320 | 142.5 | 75.4 |
| Si experiment | 5.431 | 165.7 | 63.9 |
| Ge potential | 5.6567 | 138.5 | 44.4 |
| Ge experiment | 5.658 | 128.5 | 48.3 |

Biaxial modulus M = C₁₁ + C₁₂ − 2C₁₂²/C₁₁ — the stiffness that decides how
strain shares between coherently bonded layers — is Si 138 / Ge 154 GPa from
the potential against Si 180 / Ge 141 from experiment. **The potential makes
Ge the stiffer of the two when Si is.** Si/Ge ratio 0.89 against 1.28.

Using the potential's own constants moves the prediction from 42 % to 51 %
toward Ge, in the observed direction. (The remaining gap to 73 % is a separate
question — nonlinear elasticity at ±2–4 % strain, and whether the box had
converged.)

**Fix.** `scripts/measure_potential_elastic.py` measures a0 and C_ij for a
potential and flags an inverted ordering explicitly. Run it before trusting
any strain-dependent result.

**Correction to the first version of this fix.** It originally said "build
geometry from reference elastic constants rather than the potential's". That
cannot work: a relaxed MD cell partitions strain by the *potential's* C_ij no
matter how the starting geometry was built, and a cell built off the
potential's own a0 just starts prestrained. Build MD cells on the potential's
own a0. When its C_ij are known wrong, the honest options are to report that
strain-dependent results inherit the error, or to change potential. Geometry
cannot paper over a potential's physics.

**The general lesson.** Checking a0 alone is not validating a potential. A0 is
the quantity everyone checks and the easiest to fit; this potential reproduces
it to 0.03 % while getting the quantity that actually decides the answer
qualitatively wrong. Validate the property your result depends on, not the
property that is conventional to quote — and h_K moves by a factor of two with
cell strain, so strain partitioning is exactly that property here.

**Scope.** The spec gated only Tier-2 machine-learned potentials on elastic
constants, on the assumption that an established classical potential needs no
checking. It does. The gate now applies to Tier 1.

### DMM needs an elastic cutoff at the softer material's slowest branch

**Symptom.** A Debye-DMM estimate several times larger than published values.
For Si/Ge at 300 K the error is a factor of ~4.5: 1896 against 334 MW/m²·K.

**Cause.** Scattering is elastic, so a phonon can only cross if its frequency
exists on **both** sides. Phonons above the softer material's maximum frequency
have no states to transmit into. Only about a fifth of silicon's spectrum lies
below germanium's transverse cutoff.

**Fix.** Stop the frequency integral at the highest frequency common to all
branches in both solids, set by the slowest branch of the softer material. Take
each branch's cutoff from its own zone boundary, `k_D = (6π²n)^(1/3)`,
`ω = v·k_D`.

**Verify.** **Detailed balance**: h(A→B) must equal h(B→A) exactly. The
pre-fix formula was weighted by one side's heat capacity and failed this
outright. It is the sharpest available check on a DMM implementation.

**Residual caveat.** Even correct, Debye-DMM systematically **over**-predicts:
the Debye construction overestimates each branch's maximum frequency, inflating
the incident flux. It also omits optical branches, which can carry over 40 % of
the conductance for Ge/Si at 300 K.

---

### Never difference adjacent bins to get the interfacial temperature jump

**Symptom.** h_K too low, and the error grows with applied flux — so it
masquerades as physical flux dependence and survives a careless convergence
check.

**Cause.** The bins either side of the interface sit a finite distance from it,
so their difference contains the real jump **plus** the bulk gradient across
that distance: `ΔT_naive = ΔT_true + q·(dz_A/k_A + dz_B/k_B)`. Both terms are
positive.

**Fix.** Fit the linear region on each side and extrapolate **both fits to the
interface plane**. Exclude the thermostatted reservoirs and a near-interface
zone where the phonon distribution is out of local equilibrium.

**Measured on synthetic profiles with a planted 8.000 K jump:** extrapolated fit
+0.00 % clean, +3.6 % with interface curvature; adjacent bins +0.20 % and
+33.8 %. Across a flux sweep the fit error stays flat while the naive error
holds near +22 % and climbs.

**Verify.** Pass the slab bounds explicitly (`bulk_a`, `bulk_b`) rather than
guessing the fit window as a fraction. A reservoir thicker than the guess leaves
thermostatted bins in the fit and biases the jump low while R² collapses.

---

### The equilibrium cross-check is the temperature-fluctuation method, not Green–Kubo

**Symptom.** Implementing "Green–Kubo" for an interface gives bulk κ, not R_K.

**Cause.** Standard Green–Kubo heat-flux autocorrelation is a bulk-conductivity
estimator. It is not an interfacial one.

**Fix.** Autocorrelate ΔT(t) between the two slabs. It decays as exp(−t/τ) with
τ = [C₁C₂/(C₁+C₂)]/(A·h_K). Name it correctly in code and documentation, or
someone will implement the bulk estimator.

---

### Quantum correction: report the uncorrected value as primary

**Symptom.** A "corrected" result that moves the wrong way.

**Cause.** Below roughly 0.5·θ_D classical MD overpopulates high-frequency modes
and overestimates heat capacity. The usual remedy — rescaling the MD temperature
so classical and quantum total energies match, then scaling by
dT_q/dT_c — is not variationally justified and is unreliable for transport.

**Fix.** Report the uncorrected value as the primary result; offer the
correction only as an explicitly labelled bracket. Debye temperatures: Si 645 K,
SiO₂ 470 K (not strictly meaningful for a glass — use only for this screen,
never for transport), Al 428 K, Ge 374 K. **Si/Ge at 300 K is already below
silicon's threshold.**

### The lumped gap form underpredicts, worst in the continuum limit

**Symptom.** `h_g = k_g/(Y + M)` agrees with measurement in the rarefied
regime but falls short as gas pressure rises, reaching ~32 % low at
`M/Y < 0.1`. Raising the accommodation coefficients to compensate makes the
rarefied end worse.

**Cause.** Not the accommodation coefficients, and the direction proves it:
as `M → 0` the rarefaction term drops out and `h_g → k_g/Y`, so a
continuum-limit shortfall cannot come from `M` or from `a1`/`a2` at all. It
comes from the uniform-gap assumption. Local gap thickness is Gaussian about
`Y` with spread `σ`, local conductance goes as `1/(t + M)`, and `1/x` is
convex, so the true area average **exceeds** the value at the mean thickness:

    E[1/(t+M)] > 1/(E[t]+M)

The gain is largest when `M` is small, because then nothing softens the `1/t`
convexity. The lumped form therefore has a pressure-dependent bias, not a
constant offset — which is why a single fudge factor cannot fix it.

**Fix.** `gap_conductance_integral` evaluates the area average directly,

    h_g = k_g * ∫₀^∞ φ((t−Y)/σ)/σ · dt/(t + M)

integrating only over `t > 0` and **not** renormalising, because the `t <= 0`
region is contact area already carried by `h_c`; leaving it out is what
correctly weights `h_g` by the gap's share of the apparent area. Numerically,
substitute `w = ln(u + M/σ)`: the `1/(u+M)` factor cancels exactly against
`du`, the integrand becomes bounded by 1, and the otherwise logarithmic
divergence as `M → 0` needs no hand-placed cutoff.

**Verify.** `σ → 0` must collapse to `k_g/(Y + M)` (a delta at `Y`). Note the
accessible range is narrow: `mean_plane_separation` ties `Y/σ` to `P/H_c`, so
across the whole CMY validity window `Y/σ` only spans ~2.2–5.8.

**Status.** Opt-in, **not** the default — the evidence is digitised figure
data with a documented bias, and ~25 % of the discrepancy is unexplained. See
`tests/benchmarks/SOURCES.md`.

---

## 4. Run-quality diagnostics

### Internal pressure of 6.4 GPa in a Si/Ge cell

**Symptom.** `Press` in the thermo table reads ~64000 bar. A relaxed crystal at
300 K sits near zero.

**Cause.** Building both sides on one lattice constant. Si and Ge differ by
4.01 %, and forcing Ge onto silicon spacing stores that as stress.

**Why it matters.** Stress of this order shifts phonon frequencies enough to
change h_K outright. It is a first-order problem, not a caveat — a converged
number from a strained cell is a converged wrong number.

**Fix.** A strain-minimising supercell (Phase 2), or relax the cell before
imposing the gradient. Until then, report the strain and the pressure alongside
any h_K.

**Verify.** `check_run_health()` flags mean |P| above 1 GPa automatically.

---

### A stationarity test needs a noise floor, or it never passes

**Symptom.** `check_flux_stationarity` reported "not steady" for a 60 ps run,
and still reported it for a relaxed 294 ps run. Every window, every time.

**The tell.** The drift did not *decay*. On the completed 400 ps run, fitting
drift against window index over all 23 windows gives a slope of
**−0.41 ± 1.15 % per window (t = −0.35)** — no trend. A decaying transient
trends clearly and negatively. So the run was not failing to equilibrate; the
metric was failing to resolve. Quantified: drift spans 3.4–152.7 %, mean 46 %,
standard deviation 36 %, so a 15 % tolerance sits 0.9 standard deviations
*below* the mean and is unreachable by construction.

**A second lesson, from getting this wrong once.** The same analysis on a
truncated 294 ps trace gave **+3.81 ± 1.86 %/window (t = +2.04)**, which reads
as a real upward trend and was reported as one. The final 106 ps removed it.
The drift metric is a ratio of two fitted slopes — precisely the kind of
quantity whose apparent trend moves when the tail of the data arrives. Do not
quote a fitted trend from a run still in progress; quote it with its standard
error, or wait.

**Cause.** The drift metric is a ratio of two fitted slopes, each from half a
window of reservoir power. With 4,787 atoms and a 2.2 nm cross-section that
power fluctuates enough that the metric's own statistical uncertainty exceeds
the 15 % tolerance. The test is then unsatisfiable regardless of physics, and
no run length fixes it.

**Fix.** Compare the drift against its own uncertainty rather than a fixed
threshold — bootstrap or block-average the two half-window slopes and ask
whether their difference is significant — and/or widen the cross-section,
since the noise falls as the atom count rises. A fixed tolerance is only
meaningful once the metric's resolution is known to be better than it.

**The general lesson.** A diagnostic that fires on everything is as useless as
one that fires on nothing, and it is more dangerous because it looks
conservative. Before trusting a threshold, measure the metric's noise floor on
data you believe is good. The companion entry below is the opposite failure —
a flag set that could not distinguish a 13x error — and the two together are
the same mistake: a threshold chosen without knowing the metric's resolution.

**What it did catch.** Relaxation genuinely improved the energy balance:
imbalance reached 2–6 % in several windows of the relaxed run, against a floor
of 21 % unrelaxed. The balance check has a usable noise floor; the drift check
does not.

### Identical flags, a 13x different answer

**Symptom.** Two runs of the same cell differing only in duration returned
h_K = 1348 MW/m²·K (60 ps) and 17453 MW/m²·K (20 ps). Between them they
raised the **identical set of eleven flags**. Neither the extraction nor
`check_run_health` distinguished them. Nothing said the second number was
meaningless.

**Cause.** `NEMD_UNEQUILIBRATED` existed as a flag code and never fired,
because it compares mean temperature against the setpoint. **Temperature
equilibrating is not the same as heat flow reaching steady state.** The
Langevin reservoirs reach their setpoints quickly; the thermal profile between
them keeps developing for much longer. While it does, the reservoir power is
still changing, the measured flux is inflated, and ΔT has not opened up — so
h_K = q/ΔT is wrong in both numerator and denominator, in the same direction.

The 20 ps run reported flux 2.925e10 W/m² against the 60 ps run's
1.757e10 W/m² for the same thermostats. A flux that depends on how long you
watch is the tell.

**Fix.** `check_flux_stationarity()` tests the thing that must actually hold
at steady state: dE/dt constant in each reservoir. Fit the rate over the
first and second half of the averaging window and compare. Measured drifts:

| run | hot | cold |
|---|---|---|
| 60 ps | 19 % | 97 % |
| 20 ps | 76 % | 169 % |

It flags both, which is right — neither was stationary. The 60 ps number was
never trustworthy either; it merely had the decency to be the right order of
magnitude.

**Verify.** A synthetic series with constant dE/dt must pass, including with
noise; one with an exponentially decaying rate must fail; too few samples must
flag rather than silently pass. Default tolerance 15 %.

**Generalisation.** A flag set that does not change when the answer changes by
13x is not measuring the thing that went wrong. When two runs disagree, check
whether any diagnostic separates them — if none does, that absence is the
defect, not the disagreement.

### Mean temperature well below the setpoint

**Symptom.** Thermo shows ~268 K against a 300 K target through the production
run.

**Cause — CORRECTED 22 Oct 2026.** This entry used to say "equilibration did
not finish". It was a measurement artefact all along: LAMMPS's default thermo
temperature averages over EVERY atom, including the frozen walls at 0 K. The
walls are 2 nm of a 21 nm cell, 9.5 % of the atoms, and 0.905 × 300 = 271.5 K
— the 268–272 K seen on every run. The mobile atoms were at 300 K.

**Fix.** The deck reports `compute temp` on `gmobile` (`thermo_modify temp
tmobile`, re-applied after the barostat stage, which overrides it). A mean
temperature off the setpoint is meaningful again only on decks with this
line. Lesson: check what a diagnostic measures before explaining what it
says.

**Verify.** `check_run_health()` flags a mean temperature more than 8 % from
the setpoint.

---

### Hot and cold reservoir fluxes disagree

**Symptom.** `flux_from_reservoirs` reports an imbalance above 5 %; 15 % was
seen on a 60 ps run.

**Cause.** At steady state the energy added at one end must equal that removed
at the other. Disagreement means the averaging window includes transient, the
run is too short for the accumulators to converge, or something else is
exchanging energy.

**Fix.** Average longer, start after the transient, confirm the walls are truly
fixed and no other fix adds or removes energy. Take the flux from the measured
accumulators, never from the nominal setpoint.

---

### Apparent conductivity far below bulk

**Symptom.** The fitted gradient implies k = 11 W/m·K for silicon against a
bulk 148 (ratio 0.07).

**Cause.** Expected, not a bug. The slab is shorter than the dominant phonon
mean free paths, so transport is partly ballistic and the apparent conductivity
is size-limited. Silicon is affected more than germanium because its mean free
paths are longer — the observed ordering (0.07 for Si vs 0.36 for Ge) is the
right way round.

**Implication.** The near-interface region is affected too, so h_K is
length-dependent. Run the length convergence series and extrapolate in 1/L
rather than quoting one box.

---

### R² is the wrong gate for fit quality on a shallow gradient

**Symptom.** Good fits rejected; `NEMD_POOR_FIT` firing on valid runs.

**Cause.** When the bulk gradient is shallow against bin-to-bin scatter — common
at low flux, which is exactly where you want to be for linear response — R² is
low even though the slope is well determined. R² measures signal-to-noise, not
linearity.

**Fix.** Gate on the **standard error of the extrapolated value** at the
interface, `se = s·√(1/n + (z₀−z̄)²/Sxx)`. It grows with the distance
extrapolated, which correctly penalises a fit window stopping far short of the
interface, and it yields an honest uncertainty on h_K instead of a pass/fail.

---

### The stationarity test, fixed: judge drift against its own noise

The entry above found the fixed 15 % drift tolerance unreachable (metric
noise 36 %). Fixed in `md/stationarity.py` (open issue P1-1): window powers by
least squares in 20 windows, a linear trend whose error comes from successive
differences of the window powers, and three verdicts — drifting (significant
AND above tolerance), stationary (drift + 2σ inside tolerance), unresolved
(new `NEMD_STATIONARITY_UNRESOLVED`: a statistics limit, fixed by a longer
production window or a wider cross-section, not by equilibrating longer).

**A second trap, inside the fix.** The first replacement took the noise from
residuals about the fitted line. A decaying transient is curved; its
curvature landed in the residuals, inflated the noise, and hid the transient
itself — an exponential decay was caught only 30 % of the time. Successive
differences of a smooth trend are small whatever its shape, so the
von Neumann estimate stays honest: 100 % at moderate noise. Monte Carlo
calibration is pinned in `tests/test_stationarity_stats.py`.

### Profile frames are independent windows, not running averages

`parse_profile` kept only the last of four `fix ave/chunk` frames, on the
stated belief that earlier frames were running averages carrying the
transient. Without `ave running` they are not: each frame averages its own
slice of the production window. Keeping one threw away 75 % of the profile
and measured the jump over a quarter of the window the flux came from. The
default is now the count-weighted mean of all frames; whether the window
holds a transient is the stationarity test's question, asked of the same
window. A test had pinned the wrong belief ("only the last frame is used").

The frames are also the cheapest honest error bar: `measure_kapitza` extracts
the jump per frame and reports a jackknife error on mean(q)/mean(ΔT). On a
real short run it was 2.8x the fit error, which treats bins and time as
independent.

### Two ways a block estimate goes wrong on real data

Both found on the first real run through `measure_kapitza`, both invisible on
clean synthetic data:

- **A short frame can have a negative jump.** `extract_kapitza` rightly
  refuses ΔT ≤ 0 as a final answer, and the per-frame loop caught that
  ValueError and dropped the whole block error *silently*, leaving the
  optimistic fit error as if it were complete. Per-frame jumps are now taken
  unguarded (`raw_delta_T`), averaged, and any failure is flagged with its
  reason.
- **Per-window |ΔE| inflates a noisy flux.** Over a short window the net
  reservoir transfer can be smaller than the tally's noise; taking each
  window's magnitude made the frame fluxes average 2.6x the full-window flux.
  The sign comes from the whole run now, and frame fluxes average exactly to
  the total. The test was rewritten until it failed on the old code (the
  first version's noise was too small to discriminate).

### The clamped cell: frozen walls at the built length put Pzz into the cell

Open issue P1-2: the relaxed in-plane constant sat 73 % of the way from Si to
Ge where the potential's own C_ij predicted 51 %, with Pzz = +1.51 GPa.

**Cause.** The cell is built on one lattice constant (Si's) in all three
directions and both walls are frozen, so its total length is clamped. The Ge
half is compressed along z with nowhere to go, and Poisson coupling pushes the
shared in-plane constant toward Ge. Linear elasticity with the potential's
own constants predicts σzz = −1.63 GPa for the clamp. Measured
(`scripts/test_inplane_relaxation.py`, periodic bilayer):

| | 0 K | 300 K | NEMD cell, 300 K |
|---|---|---|---|
| z clamped | 66.8 %, Pzz 1.46 GPa | 74.0 %, Pzz 1.61 GPa | 73.3 %, Pzz 1.51 GPa |
| z free | 55.9 % | 61.0 % | 62 % with the piston |

The rest of the gap to 51.8 % is temperature (+5 points) and nonlinear
elasticity (+4).

**Fix.** `relax_z` (default on): during relaxation the top wall is a rigid
piston (`setforce 0 0 NULL` + `aveforce NULL NULL 0` + `nve` + viscous
damping), frozen afterwards at its *mean* position over the second half —
the cell breathes with a ~10 ps period, and freezing it mid-swing left up to
±0.3 GPa. Production then has Pzz ≈ Pxx ≈ Pyy. The same piston, loaded with
−P·A/N per wall atom, is how `spot_pressure_GPa` now works; the earlier
`fix npt ... z` version could never run on a non-periodic z.

### Flux linearity is a validity test, not a convergence test

**Cause.** `h_K = q/ΔT` only means something if ΔT is proportional to q
*through the origin*. A significant intercept means a flux-independent
temperature offset that is not the interface.

**Note.** This test alone will **not** catch the naive adjacent-bin error,
whose bias is itself proportional to q — it inflates the slope while leaving the
intercept near zero. Extract correctly rather than relying on this to notice.

---

### Mixing-rule gas–metal wells reach chemisorption strength

**Symptom.** The first accommodation run, Ar on a Ni(100) EAM slab with a
Lorentz–Berthelot gas–solid Lennard-Jones term, condensed: coverage climbed
from 66 to 202 atoms on 72 surface sites, and Ni atoms were lifted out of the
surface as adatoms.

**Cause.** The pair well looks modest (73 meV for Ar–Ni, 21 meV for He–Ni),
but a gas atom in a surface hollow feels many metal atoms at once. A static
scan over a lateral grid of sites gives SUMMED adsorption energies of
**528 meV for Ar and 124 meV for He**, against physisorption energies of
order 80 meV and 6 meV. That is chemisorption strength, from a mixing rule
whose metal LJ parameters were fitted to cohesion, not to physisorption.

**Fix.** `calibrate_well_depth`: one static scan of a single atom over a grid
of lateral sites finds the deepest site, and because LJ energy is linear in ε
at fixed σ, ε_cal = ε_scan × E_target / E_ads. Calibrated wells: 1.04 meV
(He), 11.1 meV (Ar). An uncalibrated run is flagged
`MD_ACCOMMODATION_UNCALIBRATED`; coverage above 0.1 per site is flagged
`MD_ACCOMMODATION_COVERAGE`.

**The general lesson.** A pair parameter is not a binding energy. Check the
summed quantity the physics depends on before trusting a mixing rule.

### A small difference of noisy means: pair the events

**Symptom.** Calibrated He on Ni gave α = 0.076 ± 0.144 from 750 events —
an error twice the value.

**Cause.** α = (E_r − E_i)/(E_s − E_i). For a gas that barely accommodates,
E_r − E_i is a few meV while each energy scatters by ~50 meV. Averaging the
incident and reflected streams separately throws away the fact that each
reflection is strongly correlated with that same atom's incidence.

**Fix.** Pair each incident segment with the same atom's next reflected one
and average E_out − E_in per collision. Same estimand, much less noise: the
same trajectory gave **0.045 ± 0.048**, a 3x smaller error (≈10x fewer
events for a given precision). Incidents that start in the last 10 % of the
run are not counted, and the fraction of the rest left unpaired (trapped
atoms, the well-accommodated ones) is flagged above 1 %. The unpaired
estimate is kept as a cross-check and a >3σ disagreement is a WARNING.

**Validated** against `fix wall/reflect/stochastic` walls with imposed α:
diffusive 0.9947 ± 0.0046, specular exactly 0 (each atom returns its own
energy), Maxwell 0.4 → 0.4015 ± 0.0059, Maxwell 0.8 → 0.8062 ± 0.0056.

### Classical MD over-estimates helium accommodation

Helium (and neon) exchange energy with a surface mainly through elastic and
single-phonon scattering, which classical dynamics does not reproduce;
classical calculations are known to give α "much too large for He and Ne"
(Fan & Manson, arXiv:0804.2269). A classical He value is therefore an UPPER
estimate for the clean surface, flagged `MD_ACCOMMODATION_QUANTUM_GAS`.
Technical surfaces (oxides, adsorbates) sit higher still: uncleaned W and Pt
filaments give roughly 0.15–0.2 for He and 0.5–0.7 for Ar (Caltech thesis,
thesis.caltech.edu/578).

### Bulk k from short NEMD cells: noise, not physics, sets the limit

**Symptom.** SW-Si at 500 K, 400 ps production: k = 13.7, 46.5, 85.7 W/mK at
8.7, 17.4, 34.8 nm. 1/k is far from linear in 1/L; three points give a
negative intercept and the longest two give k_inf = 545 ± 3500.

**Cause.** In the 34.8 nm cell the two halves of the periodic box gave 49.5
and 122 W/mK from the same run. A long cell with a fixed reservoir split has
a shallow gradient, and 400 ps of a 4×4-cell cross-section does not resolve
it. The extrapolation amplifies that noise: 1/k at the longest length is the
point the intercept leans on.

**Fix.** Two guards now in code: an intercept ≤ 0 raises, and an
extrapolated error above half the value is CRITICAL ("undetermined"). The
physics fix is compute: ns-scale production and a wider cross-section for
the long cells, or Green–Kubo. Check the two-halves agreement before
trusting any length; it is the cheapest convergence test the layout gives.

### The flux used the as-built area; relaxation changes it

`measure_kapitza` divided reservoir power by the as-built cross-section, but
the in-plane barostat changes lx, ly during relaxation (Si/Ge: 21.72 → 22.29
Å, +5 % area), so every h_K from a relaxed cell read ~5 % high. The deck now
prints `OPENKAPITZA_BOX lx ly` after relaxation and the flux uses it. Found
while wiring read-in structures, where the built area is not even defined.

### a-SiO₂: the local geometry comes right before the network does

A 10¹⁴ K/s melt-quench with the Munetoh Tersoff already gives the right
Si–O / O–O / Si–Si distances (1.63 / 2.63 / 3.13 Å) and angles (145°, 109°),
yet only 73 % of Si is fourfold (15 % fivefold, 11 % threefold) and the
density is 6 % high. Bond lengths and angles alone would have passed this
glass; coordination and density are what reject it. Check all of them.

### Munetoh Tersoff a-SiO₂ keeps ~14 % fivefold Si

At 10¹³ K/s the glass has 80 % fourfold Si (14 % five-, 7 % threefold) and
2.31 g/cm³; a further 100 ps at 2000 K and re-cooling gives 81 % and 2.35.
A tenfold change in quench rate and an anneal near Tg barely move the
defect fraction, which points at the potential rather than the protocol.
Unconfirmed until a 10¹² K/s run. Every Si/SiO₂ h_K on this potential
carries the defect content in its flags until then.

### Interface bonding: an anneal can leave the interface weakly coupled

Si(100)/a-SiO₂ joined with a 1.6 Å gap and annealed at 1200 K for 20 ps
ended with 2.8 Si(crystal)–O bonds per nm², 20 % of full oxidation
(13.6 per nm²), and 88 % of the top Si layer with no O neighbour — the
weakly coupled class, whose h_K Lampin et al. (2012) find 5x lower than the
strongly coupled one with the same potential and quench rate. The builder
now reports coverage and class for every interface, and flags a preset that
missed its class.

## 5. Environment facts

### `minimize` advances the timestep counter

A `fix ave/time` with Nfreq = N defined after a `minimize` never outputs if
the following run ends on minimize_iterations + N rather than a multiple of
N; reading it then returns 0 with no error. Hit twice in one day. Put
`reset_timestep 0` after `minimize` whenever a later fix keys off step
multiples.

### `fix print` writes a step-0 line

At the start of a run, `fix print` writes the current values (here
`0 0 0` for freshly created tallies after `reset_timestep 0`). Code that
assumed the first sample was at step N was off by one sample. Align series
from the LAST sample, which is always the final step of the run.

### Three-body throughput is ~3-4x lower than first assumed

Measured: Tersoff 4.1e5 (macOS arm64) and 5.6e5 (cloud x86) atom-steps per
core-second; Stillinger–Weber 5.0e5. The cost model had 1.5e6 and 2.0e6 —
more than its 2x walltime safety factor, so cluster jobs would have hit the
wall and lost everything. Corrected in `compute/resources.py`; re-measure
with one short run on any new machine.

### What LAMMPS actually ships

`SiGe.sw` **does not exist** in the distribution. Present for this family:
`Si.sw`, `Si.tersoff`, `SiCGe.tersoff`, `GaN.sw`, `CdTe.sw`. Verify a file
exists before naming it; do not infer a filename from the element pair.

### Mass-mismatch model interface

When no cross-species fit is available, a legitimate fallback is one species'
potential with a different mass on the second atom type — `pair_coeff * * Si.sw
Si Si` with `mass 2` set to germanium's. There is no real chemistry across the
junction, only an acoustic mismatch. It is a standard model system and it
exercises the full pipeline with a genuine temperature jump, **but it is not
Si/Ge** and must not be compared against Si/Ge literature.

### Published Si/Ge reference values at 300 K

NEMD 315–368 MW/m²·K; AEMD sharp interface Si→Ge ~266; a cited DMM prediction
~417. The corrected Debye-DMM in this package returns 334. Use these for
orientation only — they are for unstrained, converged cells.

### Periodic cross-section must be a whole number of unit cells

x and y are periodic, so a box that is not an integer number of cells leaves a
seam where the images meet. Snap the cross-section, and compute the flux from
the **snapped** area — a 5.000 nm request becomes 4.888 nm, a 4.4 % area
difference that otherwise passes straight into h_K.

---

### Benchmark data may be buried in a thesis appendix, not the papers

**Symptom.** The papers that fit a correlation show agreement only as
nondimensionalised plots. PDF table extraction returns empty cells. The
primary dissertations are not openly available.

**Where to look instead.** A later thesis that *validates a model against*
those datasets often tabulates them in full, because the author needed the
numbers. Bahrami's 2004 Waterloo thesis, Appendix A.2, carries 207 conforming-
contact points from Hegazy (1985) as force and measured resistance, plus a
metadata table giving σ, m, the Vickers coefficients, k_s at the test
temperature and the specimen radius.

**Unit traps when using such a table.** Resistance is quoted in K/W, not per
unit area, so the specimen size is needed: `P = F/A`, `h = 1/(R·A)`. Check
whether the quoted radius is radius or diameter by reconstructing the pressure
range and comparing against the range the source states elsewhere — here
treating b_L as the radius gives 0.41–8.89 MPa against a published
0.459–8.769, and as the diameter gives 1.6–35.5.

**Rule.** Benchmark against the properties the source used, not your own
database. These tests ran at 110–145 °C where stainless conductivity is
~19 W/m·K against 16.2 at 300 K; using the database value puts ~19 % into
h_c and the suite grades the database instead of the correlation.

---

## 6. Working rules

### Do not fabricate data to make something run

Applies to potential parameters, hardness coefficients and benchmark values
alike. A fabricated potential produces numbers that look entirely reasonable and
mean nothing, which is worse than failing to run. In this project: `c1`/`c2` are
`null` in `materials.yaml`, `tests/benchmarks/` ships empty, and no `SiGe.sw`
was written. Say what is missing and stop.

### Interrogate files; do not assume their contents

Three consecutive failures in this project were the same error narrowing: a
filename, then an element label, then *which* label. Each was fixed as a
specific instance before the general lesson landed. Read the file and refuse
combinations that cannot be verified.

### ERROR: Cannot use fix npt on a non-periodic z dimension

**Symptom.** A deck dies immediately at the barostat line. Before that, it may
die even earlier with `ERROR: Unknown fix npt keyword: \`.

**Cause.** Two independent bugs in the same never-executed block.

1. The template wrote `\\` in a Python string, which renders as a **literal
   backslash** in the deck. LAMMPS continuation is `&`, not `\`, so the
   backslash was parsed as an `npt` keyword. The neighbouring block used a
   single `\` — a Python line-continuation that joins the lines before LAMMPS
   ever sees them — which is why one worked and the other did not.
2. `fix npt` requires every barostatted dimension to be **periodic**. In this
   workflow z is deliberately non-periodic (`boundary p p f`, walls at both
   ends) so heat cannot short-circuit around the box. Barostatting z is
   therefore always an error here.

**Fix.** For relaxation, barostat the **in-plane** dimensions only:
`couple xy x 0 0 Pdamp y 0 0 Pdamp`. That is not a workaround — the lattice
mismatch strain *is* in-plane, because the cell is built on one shared
in-plane constant. For a **normal load** (spot pressure) there is no barostat
answer: leave the bottom wall fixed, make the top wall mobile in z, and apply
a force per wall atom summing to sigma*A.

**Why it survived.** `spot_pressure_GPa` defaults to None and no fixture ever
set it, so the block was never rendered, let alone run — including by a
576-configuration deck audit whose grid left it at the default. **A config
option nobody exercises is untested code, however many tests surround it.**
Widen audit grids along the option axes, not just the numeric ones.

### A test that asserts deck text can lock in a broken deck

The spot-pressure block above had a dedicated passing test. It asserted that
setting `spot_pressure_GPa` put `"npt"` and the pressure in bar into the deck
— both true — and so it passed for as long as the feature was completely
non-functional. Deck text is not behaviour. A string assertion proves the
generator wrote what the author expected; only the engine can say whether what
the author expected is valid. Where a feature cannot be exercised against the
engine in CI, say so in the test name and treat the feature as unverified
rather than tested.

### A fix that lives in a script is not a fix

`potential_elements`, `potential_triples`, `complete_pairs` and
`match_species` were written to close three defects about potential-file
contents, were unit-tested, and were then called from exactly one place:
`scripts/smoke_lammps.py`. The library never called them. `build_input_deck`
fell back to the bare chemical symbols, so anyone driving the library directly
still got `pair_coeff * * SiCGe.tersoff Si Ge` — the invalid deck the helpers
existed to prevent — and `check_spec` never looked at the potential file at
all.

The defect was *recorded as fixed* in this very file while the library could
still emit it. The shakedown script passed because the script was the thing
holding the fix.

**How it was found.** Running the helpers against a real LAMMPS potentials
directory, then grepping for callers. The grep is the step that matters:
`grep -rn "complete_pairs" openkapitza/` returned nothing.

**Rule.** After fixing a defect, grep for the fix's callers and confirm the
path a user actually takes goes through it. A helper that only the test and
the demo call has not changed the product's behaviour. When a verification
script and the library both need a check, the check belongs in the library and
the script should call it.

**Also.** `match_species` is a trap for this specific job: it returns the
*first* label whose stem matches, which for "Si" in `SiCGe.tersoff` is
`Si(B)` — the parameterisation with **no** Ge cross terms. Use
`resolve_pair_elements`, which returns only a pairing carrying all eight
required triples. Measured against the real file: Si(B) 2/8, Si(C) 2/8,
Si(D) 8/8.

### Verify on a second platform before trusting a test suite

144 passing tests, an independent numerical audit and synthetic-profile
validation all ran on Linux against code written against the same assumptions.
The first run on macOS with real LAMMPS found six defects, including one — the
`/proc` zombie check — that no amount of re-reading the code would have
surfaced.

### Check the deck as text before spending compute

`validate_deck()` catches argument-order and alignment errors in microseconds
that LAMMPS reports only on reaching the offending line — which, for a fix set
up after equilibration, is minutes into a run.

### Run the cheap path first and say whether the expensive one would change the answer

The continuum calculation takes milliseconds; NEMD takes days. Sweep h_K across
its plausible band and report how far the joint conductance moves. If the span
is under ~5 %, the joint is constriction- or gap-limited and no MD campaign can
change the result. Never start a simulation campaign without showing that number
first.

### A result and its caveats are one object

Flags are structured, carry stable codes, and are rendered — never invented,
never suppressed. A number quoted without its flags is a different claim from
the one the calculation supports.

### When a test fails, suspect the invariant before the code

Developing the Gaussian gap integral, two successive "obvious" invariants
failed while the implementation was correct:

1. *Plain Jensen*, `I >= k_g/(Y+M)` — failed in 40 of 105 cases. The Gaussian
   is deliberately **not** renormalised over `t > 0`, because the `t <= 0` tail
   is contact area belonging to `h_c`. Jensen's inequality requires a
   normalised probability measure, so it simply does not apply.
2. *Jensen after dividing by the gas-covered area fraction* — still failed,
   min 0.825. Truncating at `t > 0` removes the low tail, so the truncated
   mean `E[t | t > 0]` is **larger** than `Y`; the bound has to use that larger
   mean, not `Y`.

The true invariant — Jensen on the normalised measure against the truncated
mean, `I/Φ(Y/σ) >= k_g/(E[t|t>0] + M)` — held in 540 of 540 cases with a
minimum ratio of exactly 1.00000, and an independent Monte Carlo agreed with
the quadrature to 0.009 %. Two wrong invariants in a row nearly triggered a
"fix" to correct code. Before editing, write down *why* the invariant should
hold; if the derivation needs an assumption the code deliberately breaks, the
test is wrong. Pin the corrected reasoning in the test docstring so the next
reader does not reach for the same wrong bound.

### Digitised figure data is a diagnostic, never a validation

113 points extracted from Song & Yovanovich's gap figures gave a geometric
mean model/measured of 0.885 with 99 % inside 2×. It would have been easy to
report that as "the gap model is validated". It is not, for two reasons that
must travel with any such number:

- **Known extraction bias.** Upper bounds recovered within 1–3.5 %, but lower
  gas pressures came out 25–45 % high because the extractor loses the dense
  overlapping low-pressure cluster — precisely the rarefied regime of
  interest. Quote the bias next to the statistic, keep digitised points in a
  separate file from exact tabulated ones, and exclude them from the headline.
- **Conditioning.** `h_g` was recovered as `h_j − h_c`, a difference of two
  larger numbers. That is well-conditioned only where `h_g` is comparable to
  or larger than `h_c`, i.e. at high gas pressure. The low-pressure end is
  doubly weak: biased extraction *and* an ill-conditioned subtraction.

What digitised data *is* good for: exposing a **trend**. The ratio fell
monotonically from 1.05 (rarefied) to 0.68 (continuum), and that shape — not
the aggregate number — is what identified a real modelling deficiency. Use
figure extraction to find structure, then demand exact numbers before changing
a default.

### `pgrep -f` in a wait loop matches the loop itself

`until ! pgrep -f job.py; do sleep 5; done` never exits: the loop's own
command line contains "job.py". Wait on an artefact the job writes on exit
(`echo done >> job.log`) instead.

### Long runs checkpoint, or they are lost

A 3-hour quench died at 68 % with a container restart and left nothing to
resume from. Anything longer than about an hour now writes restart files at
stage boundaries with a progress marker, and its runner resumes from the
last completed stage (`structure.amorphous.run_quench`). The same protects
against cluster walltime limits. A ramp must be split so each segment ramps
between its own end temperatures, or a resumed run would restart the ramp.

### Do not change a specified default to chase a residual

The continuum-limit shortfall above is real and physically explained, and the
gap integral demonstrably improves it. It was still added as **opt-in**, with
a test asserting the lumped `k_g/(Y+M)` remains the default, because the only
evidence is digitised data with a documented bias at one end and ~25 % of the
discrepancy is still unexplained. A model change justified by biased data can
be worse than the error it fixes, since it also destroys the agreement that
was previously honest. Implement the alternative, document the evidence, keep
the specified form until exact data settles it.
