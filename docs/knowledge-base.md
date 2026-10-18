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

### Mean temperature well below the setpoint

**Symptom.** Thermo shows ~268 K against a 300 K target through the production
run.

**Cause.** Equilibration did not finish. The system is still converting kinetic
to potential energy, so anything measured on top of it is not a steady state.
Stored strain energy draining into kinetic degrees of freedom is one source.

**Fix.** Lengthen equilibration. Check cell stress at the same time — the two
usually appear together.

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

### Flux linearity is a validity test, not a convergence test

**Cause.** `h_K = q/ΔT` only means something if ΔT is proportional to q
*through the origin*. A significant intercept means a flux-independent
temperature offset that is not the interface.

**Note.** This test alone will **not** catch the naive adjacent-bin error,
whose bias is itself proportional to q — it inflates the slope while leaving the
intercept near zero. Extract correctly rather than relying on this to notice.

---

## 5. Environment facts

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
