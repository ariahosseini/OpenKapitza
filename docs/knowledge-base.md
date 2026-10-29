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

### Lattice matching on the conventional cell misses the primitive surface cells

The first rectangular-supercell matcher enumerated in-plane vectors as integer
combinations of the CONVENTIONAL cubic vectors. For diamond and fcc that lattice
is missing the centring translations a/2⟨110⟩, so the smallest Si(100) cell it
could find was 5.43 × 5.43 Å instead of the primitive 3.84 × 3.84 Å, and Si(111)
was 2× too large. Every match was then found on supercells too big by 2-4× in
area, and some good ones (Si/Al(100) at 8.59 Å, 0.04 %) not at all. Nothing
failed: the matches it did report were valid, only not minimal. Fix:
`Crystal.centring()` derives the pure translations from the basis (any shift
mapping every atom onto a same-species atom), so a conventional cell with any
basis is enumerated as its true lattice. Test: Si(100) primitive length
a/√2, Si(111) a/√2 × a√(3/2), GaN(0001) a × a√3.

### A side group built from one type misses the other species on that side

`PISTON_OFF` grouped side B as `type 2`. On a Si/a-SiO₂ cell (types 1 crystal
Si, 2 glass Si, 3 O) that group is the glass Si only, so `bottom_b` — and the
relaxed interface plane every measurement bins against — ignored the oxygen,
which at a strongly bonded interface is the species closest to the crystal.
Both saved Si/SiO₂ decks carried it. Nothing errors. Now `NEMDSpec.types_a` /
`types_b` list every type per side (Si/SiO₂: (1,) and (2, 3); SiC/GaN: (1, 2)
and (3, 4)), are checked disjoint, and the decks were regenerated.

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
single-phonon scattering, which classical dynamics does not reproduce. Fan
& Manson (arXiv:0804.2269) find classical theory agrees with data for the
HEAVIER rare gases (Ar, Kr, Xe on W), and state in Sec. III that "the present
calculations predict accommodation coefficient values that are much too large
for He and Ne" (checked against the full text, 23 Oct 2026 — it is in the body,
not the abstract). A classical He value is therefore an UPPER estimate for the
clean surface, flagged `MD_ACCOMMODATION_QUANTUM_GAS`.
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
weakly coupled class, whose h_K Chen, Zhang & Li (2012, arXiv:1210.0354) find
5x lower than the strongly coupled one with a Tersoff-type potential at the same
quench rate. The builder
now reports coverage and class for every interface, and flags a preset that
missed its class.

### A strain-mode load stage must tile the relaxation window

The uniaxial-strain spot pressure (P1-10) adds a loaded stage after the
in-plane relaxation, with its own `fix ave/time` for the wall position.
`ave/time` fires only on timesteps that are multiples of its Nfreq, and the
stage starts at step `relax_steps` (the counter is not reset), so unless
`load_steps` divides `relax_steps` the average is never evaluated at the end
of the stage and `f_zav` is read stale. `_load_steps` refuses such a length.

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

## 7. Electron–phonon transport at metal/non-metal interfaces (P1-14 survey, 23 Oct 2026)

Survey of the LAMMPS documentation and source, recent papers, and GitHub,
before building a two-temperature (TTM-MD) treatment. Values below are
quoted from the sources named; nothing here is from memory.

### LAMMPS `fix ttm` family — verified in the source of our build (30 Sep 2026)

Variants: `ttm`, `ttm/grid` (distributed grid), `ttm/mod` (file parameters,
T_e-dependent C_e, vacuum), `ttm/cascade`, and `ttm/thermal` (per-cell
properties file: `ix iy iz C_vol kappa_e gamma_p eta`). All in EXTRA-FIX,
which both our builds include.

- **Periodic in all three dimensions, or a hard error.** `fix_ttm.cpp`:
  "Cannot use non-periodic boundares with fix ttm". Our NEMD deck is
  `boundary p p f`. A TTM run must use `p p p` with a vacuum gap above the
  top wall wider than the potential cutoff, so the slab does not interact
  with its own z-image. The electron grid's diffusion is also periodic, so
  the electron-free cells (below) must separate the metal from its image.
- **No box change while the fix exists** ("Cannot use fix ttm/thermal with
  changing box shape, size, or sub-domains"). Define it after the in-plane
  relaxation and `change_box`; no `fix npt` afterwards. Orthogonal boxes only.
- **Electron-free cells.** In `ttm/thermal` a cell with `kappa_e = 0` and
  `gamma_p = 0` exchanges no heat with its neighbours (the face conductivity
  is the harmonic mean 2ab/(a+b), zero if either side is zero) and applies no
  force to its atoms. `C_vol` must stay > 0: the update divides by it. The
  initial T_e must be > 0 everywhere (`set`, `infile`), so the "zero-T_e
  vacuum" convention of `ttm/mod` is not reachable from input in `ttm/thermal`;
  use k_e = γ_p = 0 instead. The documentation page and the source disagree
  on the property checks (doc: "> 0"; source: "< 0 is an error"); the source
  is what runs.
- **Heat source:** `source P` times per-cell `eta` (≥ 0) heats the electrons
  directly. It cannot be a sink.
- **Tallies:** `f_ID[1]` electronic energy, `f_ID[2]` energy moved electrons →
  atoms on the step (extensive); `outfile N file` writes the T_e grid
  (`ttm`, `ttm/mod`, `ttm/thermal`; not `ttm/grid`). The documentation notes a
  half-step lag between the two energies.
- **gamma_p** is a friction coefficient in mass/time units (metal: g/mol/ps),
  per Duffy & Rutherford: γ_i = m_i g / (3 n k_B) with n the atom number
  density (Wang et al. 2012). Results depend on processor count through the
  random numbers (statistically equivalent).
- Shipped examples (`examples/ttm`): `in.ttm.thermal` with `props.in` (bcc Fe,
  fully periodic, uniform properties); nothing with an interface.

### Physics and methods

- **TTM-NEMD at a metal/non-metal interface** (Wang, Ruan & Roy, PRB 85,
  205311, 2012): electrons on the metal side only; heat injected and removed
  by PHONON thermostats; electron grid adiabatic at its ends. Pure Cu:
  TTM-MD κ = 415 ± 11 W/mK against ~401 measured; plain NEMD 13.6 (phonons
  only). Si/Cu: TTM-MD interface resistance ≥ 25 % above phonon-only MD.
  Adding R_ep in series to R_pp **over-predicts** the full TTM-MD result by
  6 % (g = 2.6e17) and 15 % (g = 5.5e16). Their R_ep carries a factor
  (κ_e/(κ_e+κ_p))^{3/2}, ≈ 1 for good metals. Requirements they state: >
  ~1000 atoms per electron cell; electron cells shorter than the cooling
  length; metal longer than the cooling length so a linear region exists.
  Potential caveat: their Si–Cu Morse cross term is not validated for
  transport.
- **Cooling length** (the decay length of T_e − T_p near the interface):
  l = (g/κ_e + g/κ_p)^(-1/2) ≈ √(κ_p/g) when κ_e ≫ κ_p (Li, Wang, Zhou, Ren
  & Li, arXiv:1412.2465). With the values below: Al √(8–17 / 2.45e17) ≈ 6–8
  nm; Cu √(17.6 / 5.5e16–8.3e16) ≈ 15–18 nm. The metal slab must be several
  cooling lengths long: tens of nm for Cu. Note h_ep = √(g κ_p) = κ_p / l.
- **Cross-interface electron–phonon coupling** (metal electrons straight into
  non-metal phonons): a parallel channel. Lu, Wang & Ruan, PRB 93, 064302
  (2016), Cu/Si TTM-MD with an added interfacial coupling G_ei set equal to
  G_ep (arbitrary, stated as such): TBR 2.8 → 2.29 × 10⁻⁹ m²K/W (−18 %).
  Sadasivam, Waghmare & Fisher (arXiv:1501.02763, first principles, TiSi₂/Si):
  electron coupling to interfacial modes is of the same order as to bulk
  metal phonons, and its conductance comparable to the phonon–phonon one.
  Li et al. (arXiv:1412.2465): the series–parallel network
  σ = 1/(R_m^e + R_ep) + 1/(R_m^p + R_pp); for some pairs (Pb/diamond) the
  interfacial channel dominates. **Status: real but unquantified for our
  pairs; no reliable way to predict G_ei.** Not modelled; to be flagged.
- **Rough interfaces:** Wu, Li, Huang & Tang, PRB 112, 035306 (2025),
  "electron density-corrected" TTM-MD for Cu/GaN with fractal roughness;
  validated against Cu/diamond, Cu/BN, Al/GaN experiments better than plain
  NEMD. Abstract only (paywalled).
- **Spatially varying TTM in LAMMPS for NEMD** (Baer & Walker, J. Mol. Model.
  31, 220, 2025; the per-cell-property approach behind `ttm/thermal`, whose
  face conductivity cites Baer): Ni, frozen ends + Langevin, 1×1×M grid with
  ~1600 atoms per cell, 0.5 fs. Lessons they report: TTM coupling LOWERS the
  MD lattice conductivity (~30 % at κ_e = 0), so the coupled phonon κ is not
  the uncoupled one; short systems need separate electron and lattice
  gradients (a single Fourier fit is wrong); steady state took 0.4–1 ns;
  interface temperature jumps were noisy (some negative) and they prefer a
  series-resistance method including the non-equilibrium region. They quote
  a ~700 Å electron–phonon equilibration length for Ni, far longer than
  √(κ_p/g) ≈ 7 nm from the same inputs; unresolved here.

### Input values, ~300 K (sourced)

| Metal | g (W/m³K) | κ_p lattice (W/mK) | κ_e (W/mK) | C_e |
|---|---|---|---|---|
| Al | 2.45e17 (Lin, Zhigilei & Celli, PRB 77, 075133, 2008) | 10.7 p-p only / **8.1** with p-e (Tong et al., arXiv:1907.01364); literature 5.8–21.1 quoted there; 17.0 MEAM Green–Kubo (Yang et al.) | 225.8 (DFT, Tong et al.) | — |
| Cu | 5.5e16 (Lin et al. 2008, ab initio); 8.31e16 (Zhang et al., arXiv:2202.03089, T_e = 315 K); 2.6e17 (Eesley 1986, as quoted by Wang et al.) | 19.6 / **17.6** (Tong et al.); 13.6 EAM NEMD (Wang et al.) | 360.9 (DFT, Tong et al.); 401 total, Wiedemann–Franz (Lu et al.) | 96.8·T J/m³K (Wang et al.); 5.26e4 J/m³K at 300 K (Lu et al.) |

Tong et al. also list Au, Ag, Ni, Pt, Ti (λ_ep, κ_p, κ_e) and report the
dimensionless λ_ep, not g. T_e-dependent g, C_e, κ_e for many metals and
semiconductors: XTANT-3 data set (github.com/N-Medvedev/XTANT-3_coupling_data,
GPL-3.0, Medvedev & Milov, PRB 102, 064302, 2020 for metals); lowest T_e per
file not yet checked.

**Consequences.** (1) κ_p for the estimate should be the REAL lattice value
(with phonon–electron scattering), not the MD one: MD has no electrons, and
the MEAM Al value (17.0) is 2× the first-principles 8.1, which moves h_ep by
√2. (2) Cu's g spans ~5× → h_ep ~2.2×. Report the estimate as a band.

### Validation targets (300 K)

- Al/Si: phonon-only MEAM NEMD 0.47 GW/m²K, series 0.37 (Yang et al.,
  arXiv:1401.5550). TDTR: 0.35 (Minnich et al.), 0.22 (Zhang et al., Carbon 64,
  61), and Hopkins et al. (PRB 82, 085307, 2010, Table I, now in
  `tests/benchmarks/kapitza_measured.csv`): 0.158–0.193 with the native oxide
  stripped, 0.115–0.144 with it, falling with roughness in both classes.
  Experiment includes oxide and roughness: a bracket, not a test.
- Al/diamond (no electrons in the model): MTP-NEMD 284 ± 22 MW/m²K; TDTR
  21–390 depending on surface (Adnan, Neupane & Feng, IJHMT 235, 126227, 2024).
- Cu/Si: TTM-MD only (Wang 2012; Lu 2016), both on an unvalidated Morse
  cross potential. No direct Cu/Si experiment found in this pass.

### GitHub

No public TTM-MD interface workflow found (the GitHub search API is blocked
from this environment; web search only). Useful: the XTANT-3 parameter set
above. The LAMMPS examples are bulk-only.

### Design decisions taken from this survey

1. Keep the series estimate, but take κ_p from first principles (with p-e
   scattering) by default, report g and κ_p as ranges → an h_ep band; flag
   that the series form over-predicts the coupled resistance by ~6–15 %.
2. TTM-MD deck: `ttm/thermal`, electrons on the metal side only (k_e = γ_p = 0
   elsewhere), electron-free buffer cells at both ends of the metal so the
   periodic electron grid is broken, `boundary p p p` with vacuum headroom,
   fix defined after relaxation. Heat in and out through the existing phonon
   Langevin reservoirs (Wang 2012); electron source as an option later.
   Separate T_e and T_p profiles → R_ep and R_pp separately.
3. Metal slab ≥ several cooling lengths: Al ≳ 30 nm, Cu ≳ 60–90 nm. That is a
   cluster-size cell; here only plumbing.
4. Cross-interface coupling: not modelled; raise a flag with every metal h_K.


### 7.x TTM-NEMD as built (`md/ttm.py`, 23 Oct 2026)

- **Deck.** `NEMDSpec(ttm=TTMSpec("Al"))`: `boundary p p p`; headroom gets the
  vacuum guard (default 12 Å) on top of the piston's travel; after relaxation
  the deck prints `OPENKAPITZA_VACUUM_GAP` and `quit`s if the gap is below the
  guard. `fix twotemp gmetal ttm/thermal` is defined after equilibration (box
  final), on the mobile metal atoms including those in the reservoir.
  `ttm_props.in` (written by `nemd.ttm_properties`, and by `write_campaign`):
  1×1×Nz grid, metal cells with C_vol, κ_e, γ_p; others κ_e = γ_p = 0.
- **Parameters (Al, 300 K):** C_e = 135 J/m³K² × T (Kittel), κ_e 225.8 W/mK,
  g 2.45e17 → γ_p = 2.65 g/mol/ps (electron–lattice time 10 ps). Cooling length
  √(κ_p/g) with the p-p-only κ_p (10.7): 6.6 nm; fit exclusion 3 l_c = 20 nm.
  Cu: 46 nm. **A converged Al cell needs ≳ 30 nm of metal; Cu ≳ 60 nm.**
- **Electron temperature is thermodynamically noisy; never read a snapshot.**
  Equilibrium bulk-Al check (1024 atoms, periodic, no gradient): lattice
  ~292 K, electron snapshots every 2 ps ranged 243–369 K with mean 289 K — no
  bias, but δT_e ≈ T/√(C_e V/k_B) ≈ 40 K, because the electrons' heat capacity
  is ~60x the lattice's smaller and k_e couples the whole metal into one small
  reservoir. In an interface cell with ~20 nm³ of metal that is ~±27 K per
  snapshot, larger than the whole metal-side gradient. The first interface run
  read its snapshots as "electrons 20–40 K hot" before this check. The deck
  now averages the metal's mean T_e in production (`T_e_mean.dat`, from
  `f_twotemp[1]` minus the fixed electron-free cells) on the same windows as
  the lattice profile.
- **Measurement.** `measure_kapitza` starts the metal-side bulk fit
  `exclusion_lc` cooling lengths from the interface. `compare_runs` turns a
  phonon-only/TTM pair of the same cell into h_ep = 1/(1/h_TTM − 1/h_pp) with
  errors in quadrature, and declines to quote it when not resolved.
- **First interface pair (plumbing, 23 Oct 2026).** Si(100)/Al hetero cell, 5,901
  atoms, 1.6 nm wide, 14 nm Al, MEAM, 6 ps relax + 6 ps equilibration + 20 ps
  steady + 40 ps production, exclusion 1 l_c, 2 nm electron cells (~320 atoms).
  Both decks ran end to end; vacuum gap after relaxation 22.3 Å (guard 12).
  Neither run was steady: hot and cold reservoir fluxes differ by 85 % (pp)
  and 102 % (TTM), the jump trends +101 % across production (pp). h_K: pp 340
  ± 99, TTM 1207 ± 917 MW/m²K; `compare_runs` correctly refused to quote h_ep
  (1/h_TTM − 1/h_pp = −2.1 ± 1.1 × 10⁻⁹ m²K/W, wrong sign, not resolved).
  Electrons vs lattice in production: mean of 8 T_e snapshots over the metal
  292.4 K, metal lattice mean 288.6 K — equal within the snapshot noise
  (±27/√8 ≈ ±10 K), so the 315–331 K seen during the steady stage was noise,
  as the bulk check predicted. Steady state needs ~0.4–1 ns (Baer & Walker
  2025), ~10x this run: cluster.
- **Checks raised by `check_spec`:** cross-interface channel not modelled
  (always); metal shorter than 2·exclusion; < 1000 metal atoms per electron
  cell (Wang et al.); electron cell longer than the cooling length; reduced
  exclusion.

### Lessons

- **Check a citation's author list against the paper.** arXiv:1401.5550 is
  N. Yang, T. Luo, ..., G. Chen; this repository called it "Liang et al." in
  five places, and attributed its g to the paper rather than to Lin et al.
  2008.
- **A back-solved input that reproduces a published number is not a check.**
  The estimate's "reproduction" of 0.37 GW/m²K used k_p = 12.3, chosen to hit
  0.37 from 0.47. The paper's own inputs (G_pp 0.46, k_p 17.0) give 0.375 —
  the test now uses those.

## 8. Literature survey across all topics (23 Oct 2026)

Extends §7 to every topic covered so far. Each entry: what the source says,
then what it changes here. Numbers are quoted from the sources named; where
only an abstract was reachable, it says so.

### 8.1 Contact conductance

- **Our constants are the published ones.** Yovanovich, Bahrami & Culham
  (HT2005-72850, 2005): plastic CMY h_c σ/(k_s m) = 1.25 (P/H_c)^0.95; Mikic
  elastic h_e = 1.55 k_s (m/σ)(P/H_e)^0.94 with H_e = E′m/√2; plasticity index
  γ = H_mic/(E′m), γ ≥ 3 predominantly elastic, γ ≤ 0.33 predominantly
  plastic ("for most engineering surfaces the asperity deformation mode is
  plastic"). `contact/regime.py` uses ψ = 1/γ with the same thresholds. Song &
  Yovanovich (J. Thermophys. 2, 43, 1988): the 1.25/0.95 correlation is valid
  for 10⁻⁶ ≤ P/H_c ≤ 2.3×10⁻²; measured pressure exponents 0.94–0.97.
- **CMY under-predicts at LIGHT loads, the packaging regime.** Milanez, Culham
  & Yovanovich (AIAA 2002-0787): bead-blasted SS 304 in vacuum, 16 kPa–3 MPa;
  the Gaussian (CMY) model under-predicts first-loading data by ~40 % (σ =
  0.72 µm) to ~70 % (σ = 1.31 µm) at light loads. Cause: real surfaces lack the
  tallest Gaussian asperities (truncation), so the mean-plane separation is
  smaller than CMY assumes. Milanez, Yovanovich & Mantelli (AIAA Thermophysics
  2003): a truncated-Gaussian model cuts the RMS error from 21–48 % to 5–18 %;
  fitted truncation 3.5–4.2 σ; profilometer R_p/σ is NOT a reliable estimate
  of it; at 15.8 kPa, z_trunc 4.1 → 3.3 changes h_c by ~150 %. Heat-sink
  mounting pressures in electronics are 0.07–0.17 MPa (Electronics Cooling,
  "Calculating interface resistance", web version 2023), i.e. inside this regime.
  **Consequence:** flag CMY below ~0.5 MPa as a likely under-estimate
  (tracker P0-7); a truncated-Gaussian option needs the paper's correlation,
  which was garbled in the retrieved text.
- **Our own benchmark shows the same bias, and where it starts.** Binned by
  Y/σ (Yovanovich separation), measured/predicted over the 207 points: 0.90
  (< 2.8), 0.91 (2.8–3.0), 0.93 (3.0–3.2), 1.05 (3.2–3.4), 1.23 (> 3.4); by
  pressure, 1.36 below 0.5 MPa. The crossover sits just below the fitted
  truncation levels (3.5–4.2σ), as the mechanism predicts. `light_load_check`
  uses these thresholds.
- **Asperity models vs. full theory.** Afferrante, Bottiglione, Putignano,
  Persson & Carbone (Tribol. Lett. 66, 2018): GW-type asperity models
  "cannot correctly capture" contact area and separation; interacting-asperity
  models and Persson theory agree with boundary-element results. Mechanics
  only, no conductance comparison. Komvopoulos (Front. Mech. Eng. 6:36, 2020)
  likewise rejects GW for multiscale (fractal) surfaces. **Consequence:** CMY
  stays (it is calibrated on thermal data, which these are not), but a
  Persson-type alternative is the natural Phase-0 extension if light-load
  errors matter.
- **In electronics, the gap often dominates a bare joint.** Electronics
  Cooling (2023), Al 6063 heat sink on alumina in air: joint resistance
  2.67 → 1.90 cm²K/W over 0.007–0.35 MPa, gap-dominated; silicone grease
  lowers it ~10x (0.34 → 0.21 cm²K/W). Supports P0-1 (filled joints).
- Multiscale TCR for packaging exists (Cui, Li, Xuan & Zhang, Int. J. Therm.
  Sci. 83, 16, 2014: lattice Boltzmann on measured profiles + finite
  differences), but none found that bridges MD Kapitza to CMY as this project
  does.

### 8.2 Gap conductance and accommodation

- **Clean machined/polished surfaces accommodate He far less than technical
  ones.** Trott, Rader, Castañeda, Torczynski & Gallis (AIAA 2007-4039; SAND
  reports 2004-5329, 2005-6084): 304 SS, ±0.02: Ar 0.87, N₂ 0.80 (finish-
  independent), He 0.36 machined / 0.40 polished. Our default He on SS 304 is
  0.63 (Song & Yovanovich 1989, bead-blasted, air-exposed). Both are
  measurements; the spread is surface state.
- **The extracted α depends on the analysis model.** The same Trott data give
  Ar 0.87 (Sherman–Lees) vs 0.94 (GTR), He 0.36 vs 0.38. An α is only
  transferable together with the gap model it was fitted with; ours is the
  Song–Yovanovich form, matching our default's source.
- **Recommendation from the rarefied-gas side** (Roohi, arXiv:2606.14039,
  2026 review): accommodation is uncertain at "several-percent-to-tens-of-
  percent" depending on surface state; report a band, and MD on pristine
  surfaces can predict a real wall worse than a fitted Maxwell α.
  **Consequence:** propagate an α band through h_g by default for He (tracker
  P0-8); keep MD α as compare-only.
- P0-3 (exact tabulated h_j(P_gas) data to settle the continuum-limit
  shortfall): not found in this pass.

### 8.3 NEMD interfacial conductance

- **NEMD and equilibrium MD measure different conductances.** Merabia &
  Termentzidis (arXiv:1209.3485): for LJ interfaces NEMD exceeds EMD by ~5x
  typically, by >10x near mass ratio 1, and they estimate ~3x for Si/Ge. EMD
  gives the Landauer (equilibrium-distribution) conductance and agrees with
  DMM; NEMD agrees with a generalised AMM that accounts for non-equilibrium
  distributions. Use NEMD for steady-state experiments (3ω), EMD for transient
  pump–probe (their suggestion). **Consequence:** P1-8's closing criterion
  ("agree within joint uncertainty") is wrong in principle; change it.
- **Converged Si/Ge is above our quoted band.** Feng, Zhong, Shi & Ruan (PRB
  99, 045301, 2019): Tersoff, Ge lattice-matched to Si (no strain), Langevin
  ±30 K, 0.5 ps damping; G nearly converged at 152 nm total length; 28Si/73Ge
  4.5×10⁸ W/m²K at 300 K; high inelastic share. Our §5 band (NEMD 315–368)
  came from shorter cells. **Consequence:** add 450 to the P1-4 reference set
  with its length, and note our cells (≤ 20 nm) sit on the steep part.
- **Reservoir length is a convergence axis.** Wei et al. (Chin. Phys. B 34,
  116301, 2025): Si/Ge ITR linear in 1/L_sys AND 1/L_bath, extrapolable;
  coupling τ negligible below ~5 ps; size effects from phonons < ~6 THz.
  Balasubramanian & Puri (APL 99, 013116, 2011): R_K falls steeply from 1 to
  40 nm, then levels. **Consequence:** add L_bath to the P1-3 campaign.
- **Spectral and quantum corrections.** Xu, Yang & Zhou (arXiv:2206.05403):
  per-frequency correction with dn_BE/dT on an MD-derived transmission; LJ
  system, 1–40 K only. Rajabpour et al. (arXiv:2407.15404) apply the same
  Landauer-type weighting at 300 K for Si/diamond. Our rule (§3: report
  uncorrected as primary, correction as a labelled bracket) stands; a
  spectral (frequency-resolved) correction is the defensible form to offer.
- **Pressure.** Hohensee, Wilson & Cahill (Nat. Commun. 6, 6578, 2015):
  metal/diamond TDTR to 50 GPa; conductance increases weakly or saturates;
  pressure suppresses extrinsic (weak-bonding) effects. Relevant to P1-9/P1-10:
  expect a strong pressure effect only where the as-built interface is weakly
  bonded.

### 8.4 Interatomic potentials and Tier-2 MLIPs

- **Empirical potentials can be off by ~3x for interface conductance.**
  Rajabpour, Mortazavi, Mirchi, El Hajj, Guo, Zhuang & Merabia
  (arXiv:2407.15404): moment-tensor potential for Si/diamond, trained with
  interface configurations and active learning, gives the infinite-length
  ITC within ~10 % of experiment; Tersoff and Brenner over-estimate by "a
  factor of about 3" (stiffer interface bonding, over-predicted group
  velocities). Interface-localised modes (8–12 THz) carry ~40 % of the ITC.
- **GaN/SiC: MD ~2x experiment.** Mu et al. (arXiv:1905.04171): bonded
  GaN-SiC TDTR 169 MW/m²K as-bonded (3 nm amorphous layer), 229 after
  1000 °C anneal; MBE/MOCVD with AlN 189–200; cited MD ~500. NEP-MLIP work
  on Ga₂O₃/SiC (Liu et al., arXiv:2605.05620) gives 550–700 against
  measurements of 140–162 for real (SiOₓ-containing) interfaces — the gap is
  interface structure, not only the potential.
- **Foundation models are not yet reliable for thermal transport.** Anam et
  al. (arXiv:2509.03401), 2,429 OQMD structures: lattice-k log₁₀ MAE vs DFT
  0.17 (EquiformerV2 fine-tuned) to 1.0 (MACE-MP-0a); dynamic stability 81 %
  (EqV2-FT), 53 % (MACE), 48 % (CHGNet); force RMSE does not predict phonon
  quality. **Consequence:** Tier 2 means a potential trained or fine-tuned on
  the interface, passed through our elastic / phonon / bulk-k gate — never a
  foundation model as shipped.
- **LAMMPS access.** MACE via `pair_style mliap unified` (ML-IAP + KOKKOS +
  Python, GPU only, beta, LAMMPS develop branch; mace-docs). NEP: fastest of
  NEP/DP/MACE/NequIP on GPU (NEP review, arXiv:2501.11191); in LAMMPS
  through the NEP_CPU interface. **Consequence:** the cluster build needs GPU
  nodes for MACE; NEP runs on CPU.
- **Validate MLIPs on interface force constants, not only bulk** (Wyant,
  Rohskopf & Henry, Ge/GaAs SNAP, Comput. Mater. Sci. 2021): interface IFC2s
  return to bulk-like within ~1 nm; mixing rules are unlikely to work.

### 8.5 a-SiO₂ and Si/SiO₂

- **Attribution corrected.** The 0.83 ± 0.25 / 4.27 ± 1.78 × 10⁻⁹ m²K/W
  strong/weak targets are Chen, Zhang & Li (arXiv:1210.0354): Tersoff Si with
  a Tersoff-type SiO₂ set, α-quartz melted at 3000 K, 10¹³ K/s; classes set by
  the join gap L₀/N_c (N_c = 20 strong, 2 weak). Lampin et al. are their
  ref. 45 (0.4 × 10⁻⁹ at 500 K). Fixed in code and docs.
- **Tersoff SiO₂ densities are off; COMB3 is closer.** France-Lanord,
  Soukiassian, Glattli & Wimmer (arXiv:1510.05416): Tersoff α-quartz 2.446 vs
  2.649 g/cm³; COMB3 (variable charge) near experiment, a-SiO₂ k 1.455 vs
  1.3–1.5 measured; a ~1 nm sub-oxide layer reproduces experiment better than
  a sharp interface; Si/SiO₂ resistance halves from 300 to 900 K. COMB3 needs
  charge equilibration (cost) and small timesteps. Consistent with P2-2's
  hypothesis that the potential, not the rate, sets our defects.
- **Open Si/SiOₓ MLIP.** Cvitkovich et al. (J. Chem. Phys. 161, 144706,
  2024): GAP (SOAP) for thermal oxidation of Si; oxide 2.5 g/cm³ (exp.
  2.24–2.36), interface 2.37 (exp. 2.36–2.41), roughness 0.79 Å (exp.
  0.4–0.9), mostly fourfold Si; potential, structures and data on GitHub
  (lukas-cvitkovich/MLFF-SiOx). No thermal validation. A Tier-2 candidate for
  P2-2 / P2-3, after our gate.
- **Modal picture.** Gordiz, Muraleedharan & Henry (arXiv:1807.06633,
  Tersoff): ~650 MW/m²K at 600 K; interfacial modes ~3 % of states but ~18 %
  of conductance; TDTR/FDTR/3ω barely resolve the Si/SiO₂ interface
  resistance, so experiment is weakly constraining.

### 8.6 Lattice matching and interface building

- pymatgen `ZSLGenerator` / `CoherentInterfaceBuilder` (Zur–McGill): general
  (not orthogonal) supercells, strain on the film only, terminations
  enumerated (`termination_ftol`), default gap 2.0 Å. Our matcher deliberately
  differs: rectangular only (the NEMD box stays orthogonal, and `fix ttm`
  requires orthogonal), mismatch split between the sides, registry-based gap.
- **Registry, measured here (23 Oct 2026).** `scan_registry` on Si(100)/Al(100)
  (8.59 Å matched cell, MEAM, 4×4 shifts, thin slabs, box-fixed minimisation):
  relaxed interface energies span 8.6 meV/Å² (~0.14 J/m²); minimum at
  (0, 2.15) Å. Unrelaxed (`run 0`) energies correlate only weakly with the
  relaxed ones (r = 0.37), and the geometric hollow score (mean lateral
  clearance) anti-correlates (r = −0.38) — geometry is not a proxy for
  bonding here. On Si(111)/Al(111) (11.5 × 20 Å cell) the clearance is the
  same at every shift: many atoms per plane average the registry out, as for
  an incommensurate interface. Rule: choose registry by relaxed energy with
  the production potential; use clearance spread only to ask whether it can
  matter.
- **Registry is a separate search.** Tang & Butler (arXiv:2608.19933,
  RegFGW): after ZSL, an optimal-transport score plus Bayesian optimisation
  over the in-plane shift; ZSL "considers only geometric commensurability".
  Ours sets the gap from the registry it is given but does not search
  registries. **Consequence:** add a registry scan (static energy over a
  lateral grid) to P2-1's open items.

### 8.7 MD inputs and LAMMPS

- **Bulk k (5c):** Dong, Fan, Shi, Harju & Ala-Nissila (arXiv:1805.00269),
  Tersoff Si 300 K: Green–Kubo 250 ± 10; NEMD 1/κ–1/L linear only for
  L > 200 nm (260 ± 10); fitting L < 200 nm gives 220. Our 8.7–34.8 nm SW
  cells are an order of magnitude short of the linear regime — consistent with
  P5-1's CRITICAL guard. Their caveat: the heat current for many-body
  potentials must be implemented correctly. LAMMPS `compute heat/flux` says
  the stress/atom route gives "unphysical values" for angle-type terms and
  `compute centroid/stress/atom` was "implemented specifically for such
  cases". **Consequence:** any Green–Kubo path for Tersoff/SW/MEAM uses
  centroid/stress/atom.
- **Nanohardness (5d):** Yang, Zhang, Xiao, Wang & Bai (Sci. Rep. 7, 16053,
  2017): Cu hardness rises then falls with depth below ~100 nm because of tip
  roundness; MD loads are higher (strain rate, contact-area definition); a
  sharp-tip MD gave hardness ~100x experiment. Abram, Chrobak & Nowak (PRL 118,
  095502, 2017): Si pop-ins at ~9–10 nm from extruded Si-III/XII. Supports
  5d staying compare-only.
- **LAMMPS 30 Sep 2026 stable (our build):** C++17 and CMake ≥ 3.20 (KOKKOS:
  C++20); format strings for thermo, `fix ave/time`, `fix ave/chunk` are now
  checked against the data (our decks set none); removed ATC, AWPMD, POEMS,
  ML-RANN, VTK (none used); new `fix baoab`; KOKKOS styles more than doubled.
  4 Jul 2026 feature release: restart-format changes for `fix gjf`,
  `fix press/berendsen` (not used); no changes to ttm, MEAM or heat flux.

### 8.8 Lessons from the survey itself

- **Three citations in this repository were wrong** and two shaped targets:
  "Liang et al." for Yang et al. (§7), "Lampin et al." for Chen, Zhang & Li
  (§8.5), and a verbatim Fan & Manson quote that is not in the abstract and
  could not be checked at the time (the full text, read 23 Oct, confirms it:
  §8.2). Rule: before a source enters code or a target, check the author list
  on the abstract page, and quote only text actually read.

## 9. Filled and bonded joints (P0-1, 23 Oct 2026)

Survey and design for joints that are not bare contact: thermal interface
materials (grease, gel, pad, PCM, adhesive) and bonds (solder, sintered Ag,
die attach). Built as `network/stack.py`; `build_joint(..., fill=tim_fill(...)
| bond_fill(...))` routes `run_continuum` through it.

### Sources

- **Model form.** R_TIM = BLT/k_TIM + R_c1 + R_c2: bulk plus two boundary
  resistances (Mahajan, "Thermal Interface Materials: A Brief Review of Design
  Characteristics and Materials", Electronics Cooling, Feb 2004; Xing, Xu,
  Song & Deng, Nanomaterials 12, 3365, 2022). R_c comes from the real contact
  area of the TIM on rough surfaces: pressure, capillarity, trapped air
  (Prasher). The equations were images on the pages read, so only the form is
  used.
- **BLT** depends on the TIM's yield stress and pressure, with a size-
  dependent yield stress (Prasher, IEEE TCPT 28, 230, 2005; abstract only).
  Not modelled: BLT is an input.
- **Class ranges of the TOTAL R_TIM** (Blazej, Electronics Cooling, Nov 2003,
  K cm²/W): grease 0.2–1 (traditional ~1, newest ~0.2), elastomeric pad 1–3
  (~0.25 mm, ~700 kPa), tape 1–4, PCM 0.3–0.7, gel 0.4–0.8, adhesive < 1 (best
  0.15), solder < 0.05. Worked example there: 0.1 mm grease is 56 % of 0.4
  K cm²/W, so its two boundaries carry ~17 mm²K/W; 0.2 mm pad 90 % of 1.7.
  Datasheets often use K in²/W (× 6.45 → K cm²/W).
- **Bond layers.** SAC (all variants) ≈ 60 W/mK at 25 °C, AuSn 80/20 57 W/mK;
  rule-of-mixtures estimates for solders "can lead to significant errors"
  (Wilson, "Thermal Conductivity of Solders", Electronics Cooling, Aug 2006).
  Pressure-less sintered Ag 80–190 W/mK depending on porosity (17–55 %) and
  sintering (Murali et al., IMAPS 2023; laser-flash 223.9 W/mK at 1–17 %
  porosity quoted there); the data are not monotonic in porosity, so no
  porosity correlation is used. Intermetallic (Cu₆Sn₅, Cu₃Sn) conductivities:
  no source read; user input.
- **Voids.** No closed form. Alavi, De Ceuninck & Daenen (Electronics 13, 2188,
  2024, IGBT solder layer, FE + ML): small voids (10–50 µm) negligible (< 0.05 %
  change); one large void raises R_th more than several small ones of the
  same area; from 5 % void area some layouts, from 7.5 % all, exceed a 20 %
  R_th increase. Shape barely matters (< 1 %).
- **Degradation** (NREL/DuPont, NREL/CP-5400-68014): greases lose performance
  through pump-out and dry-out over −40/125 °C cycling. Not modelled.

### Design

- Filled: bulk t/k per layer plus measured R_c1 + R_c2, or a measured total at
  the working pressure. Missing R_c → bulk only, `STACK_INCOMPLETE` WARNING,
  h_joint an upper bound; the class check is skipped (a partial R against a
  total range would mislead). BLT < 2σ_eff → `STACK_THIN_TIM` (asperities
  bridge the TIM; the parallel solid path is not modelled).
- Bonded: t/(k(1 − f)) per layer, plus the two atomic faces at full contact
  (Kapitza, UNSCALED — xi = 1). h_K from the user/MD, else DMM when both
  neighbours are in `materials.yaml`. A missing face is graded by what it
  could do at 100 MW/m²K (low end of the measured solid–solid interfaces in
  §7–8): > 5 % of R → WARNING and upper bound, else INFO. A 30 µm solder: INFO;
  a 200 nm metal film: WARNING — the thin-film case where this project's MD
  matters directly (P0-5).
- Voids: area-reduction estimate with the direction of its error stated
  (`STACK_VOIDS`, WARNING above 5 %).
- Output: a `StackResult` and, for existing consumers, a `NetworkResult` view
  (xi = 1, no constriction, no gas path); Monte Carlo over layer t, k, R_c,
  R_measured, h_K with stated uncertainties; report section of its own.

## 10. Material data verification (P0-6, 23 Oct 2026)

`materials.yaml` is now sourced PER FIELD (`sources:` under each entry). What
was checked, against what, and what changed:

| Material | Field | Old → new | Source |
|---|---|---|---|
| Si | E, ν | 160 GPa, 0.22 → 162.5, 0.223 | George, EMIS Datareview 1997, Hall's C_ij: Voigt 165.6/0.218, Reuss 159.4/0.228; Hill mean |
| Si | v_L, v_T | 8433, 5843 (kept) | = √(C₁₁/ρ), √(C₄₄/ρ): [100] values, not isotropic averages (Hill isotropic would be ~8950 / 5340) |
| Si, Ge, Al, Cu, Ni | θ_D | kept | Kittel 8th ed.: 645, 374, 428, 343, 450 K |
| SS304 | θ_D | 477 → 470 | no 304 value found; Fe (Kittel) as proxy |
| Si, Ge, Al, Cu | k | 148 → 149, 60 → 60.2, kept, kept | elements data page (CRC/WebElements columns) |
| SiO₂ | k, ρ, E, ν | 1.38 → 1.30, 2200 → 2201, 73 → 72.7, 0.17 → 0.16 | Corning HPFS datasheet, 25 °C |
| SiO₂ | Hv | 8.9 → 5.12 GPa | Corning: KNOOP 522 kg/mm² at 100 g; 8.9 GPa is a nanoindentation-scale hardness, wrong for a microhardness field |
| Cu | E, Hv | 120 → 117 GPa, 0.6 → 0.36 GPa | Aurubis C10100 datasheet (117 GPa; soft ≥ 35 HV; k 391 for OFE at 20 °C); Samsonov via data page: 343–369 MPa |
| Al, Cu, SS304(L) | v_L, v_T, ρ, ν | Al 6420/3040 → 6350/3100, ρ 2710; Cu 4760/2325 → 4680/2260, ν 0.35; SS 5790/3100 → 5640/3070, ρ 7900 | Coppens & Wilson, NPS-61-79-004 (1978), Table 1, after Ewing, Jardetzky & Press (1957) |
| Al | Hv | 0.35 kept | Samsonov 160–350 MPa: top of the range |
| Ni 200 | k, ρ, E, ν | 70 → 70.3, kept, 207 → 205, 0.31 → 0.29 | Special Metals SMC-061 (2006) |

DMM h_K moved by +5 % (Si/Al), −7 % (Si/Cu), −8 % (Al/Cu); Si/Ge unchanged.
The contact benchmark is unaffected (it carries its own properties).

Not found in this pass: Vickers hardness of single-crystal Si and Ge as
numbers (Si: ⟨100⟩ 10–15 % softer than ⟨111⟩, Vabishchevich 2017; instrumented
indentation unreliable for phase-transforming Si, Haberl et al. 2012); SS304
and Ni 200 Vickers (datasheets give HRB/HB only, and a conversion would be our
number, not theirs); SiO₂ θ_D. The rule held: a field gets a citation only if
the cited page states that number.

## 11. Equilibrium (temperature-fluctuation) cross-check as built (P1-8, 23 Oct 2026)

`md/emd_kapitza.py`: the NEMD deck up to the end of equilibration (same cell,
relaxation, piston), then NVE with no reservoirs, logging each slab's kinetic
temperature and total energy. τ from the autocorrelation of ΔT gives
h = C_r/(A τ), C_r = C_aC_b/(C_a+C_b), C = 3Nk_B.

- **Use slab ENERGY, not kinetic temperature.** First Si/Ge run (1,184 mobile
  atoms, 2.5 nm slabs, SiCGe.tersoff, 1 ns): the kinetic-temperature ACF fell
  to ~0.3 within 0.05 ps — each slab swaps energy between its own kinetic and
  potential parts on vibrational time scales — and the interfacial decay was
  buried. Total slab energy (ke/atom + pe/atom) moves only through the
  interface (frozen walls do no work): its ACF decays smoothly.
- **It decays in two stages.** Energy-based run: a fast partial drop
  (τ₁ = 0.87 ps, 42 % of the variance; most likely the per-atom potential-
  energy split across the interface jittering) and the slow interfacial
  relaxation (τ₂ = 32 ps). A single exponential fitted between 1.0 and 0.5
  ACF(0) mixed them (9.5 ps). `h_from_series(model="biexp")` fits both (grid
  over τ₁, τ₂ with linear least squares for the amplitudes; no SciPy) and
  takes τ₂.
- **Statistics.** Synthetic Ornstein–Uhlenbeck checks: τ reads low by ~10 %
  with 143 τ of data and ~5 % with 714 τ; `check` flags series shorter than
  500 τ. The 1 ns local run is 31 τ₂ long: h = 77 ± 269 MW/m²K, unresolved;
  NEMD/EMD 7.5 against the old 577 MW/m²K NEMD value means nothing yet. The
  cluster job (`sige_emd`) runs 16 ns.
- **Comparison.** NEMD and EMD measure different conductances (Merabia &
  Termentzidis): the test is NEMD/EMD inside 1–10 (≈3 expected for Si/Ge),
  not agreement.

## 12. Papers supplied by the user (23 Oct 2026)

Seven full texts that the earlier surveys could not reach. What each changed:

### 12.1 Truncated Gaussian at light load (P0-7) — built

Milanez, Yovanovich & Mantelli, "Thermal Contact Conductance at Low Contact
Pressures", JTHT 18(1), 2004 (AIAA 2003). Built as `contact/truncated.py`,
opt-in through `build_joint(z_trunc=...)`. Equations as printed (pages read as
images; the text layer is garbled):

- h_c = 1.25 k_s (m/σ)(P/H_c)^0.95 (1 + 1/f)^0.9289 √(1 − 1/(1+f)),
  f = (P/H_c) √(2π) z_t exp(z_t²/2)   (Eqs. 10–11)
- P/H_c = [(P/H_c)_G^−p + (P/H_c)_TG^−p]^(−1/p), p = 3.9 + 52 exp(10 c₂);
  (P/H_c)_TG = {P/(c₁(2.178 σ/m)^c₂) · erfc(z_t/√2)^(0.4289 c₂)}^(1/(1+0.5c₂))
  (Eqs. 14–15). The text says Eq. 15 "goes to zero" as z_t → ∞; with c₂ < 0 it
  diverges and its −p power goes to zero, which is what makes Eq. 14 → Eq. 12.

Checks: Table 1's twelve H_c values within 2.2 % (using the combined σ, m
printed on their Figs 3–8 — S3's table values are σ 3.9/m 0.100, its figure
3.07/0.070; the figure values reproduce the table H_c); Fig. 2 sweep 8.4 →
22.6 W/m²K against "approximately 8 to 20"; their S1 data, tabulated in
Bahrami's thesis (A.63), RMS 20.8 → 15.5 % against their 25.3 → 16.1 %.

Findings worth keeping:
1. The Eq. 10 correlation is within 4 % of the exact model only for
   P/H_c ≥ 1e-5; at 1e-6 it is 7 % LOW (exact erfc here; they likely compared
   against Yovanovich's erfc approximations). Their z_t were fitted with Eq. 10,
   so Eq. 10 stays the default.
2. "f > 40 → < 1 %": it is 1.1 % at f = 40.
3. **z_t is not a surface property on its own** — it was fitted together with
   their c₁/c₂ (SS 304: 10.67 GPa/−0.37). With Bahrami's 6.27/−0.23 for the
   same S1 data, plain CMY is already 14.7 % RMS and TG over-corrects to 22.4 %.
   The reference lookup now says so (INFO flag).
4. On Hegazy's 26 light-load points (Y/σ > 3.4, not bead-blasted as far as the
   thesis says, σ/m mostly outside their range) the reference z_t cuts the bias
   1.23 → 1.14 and removes it at Y/σ 3.2–3.4 (1.05 → 1.00); overall RMS
   unchanged. Partial, as expected without the right z_t.
5. Carrying the TG separation (Eq. 9) into the gap model is ours; the paper is
   vacuum-only. It raises h_g; no gas data test it yet (P0-3).

### 12.2 Hardness (P0-6) — Bahrami, Yovanovich & Culham, AIAA 2004-0989

- Table 3 confirms the c₁/c₂ in `materials.yaml` (SS 304 6.271/−0.229, Ni 200
  6.304/−0.264). Bahrami's 2004 thesis rounds both to 6.3/−0.26.
- Al 6061-T6 (Nho): c₁ 1.123 GPa, c₂ −0.0079 (Table 10), Brinell 915 MPa
  (Table 9). c₂ ≈ 0: almost no indentation-size effect, unlike SS and Ni.
- Hot hardness (Nho, Tables 10–13): c₁(T)/c₁(T_rm) = exp[C_T(T − T_rm)],
  C_T = −1.372, −1.675, −1.190 × 10⁻³ /°C (Ni 200, SS 304, Al 6061), valid to
  ~186–190 °C; c₂ independent of T. **Not applied:** on Hegazy's joints at
  110–145 °C, room-temperature c₁ gives pred/meas 1.03 and the softened c₁
  1.19. Either the asperity contact hardness does not follow bulk hot hardness,
  or the thesis's per-row k_s and c₁ already absorb it; we report it as an INFO
  (`HARDNESS_TEMPERATURE`, ≈ +19 % at 137 °C for SS 304) rather than correct.
- c₁, c₂ from bulk Brinell hardness (Sridhar, Eqs. 53–55): c₁/3.178 = 4.0 −
  5.77Z + 4.0Z² − 0.61Z³, c₂ = −0.57 + Z/1.22 − Z²/2.42 + Z³/16.58, Z =
  H_B/3.178 GPa, valid 1.30 ≤ H_B ≤ 7.60 GPa (c₁ RMS 5.3 %, c₂ 20.8 %). Not
  built: the soft packaging metals (Cu, pure Al) are below its range.

### 12.3 Extended benchmark — Bahrami 2004 thesis, Appendix A.2

The thesis (supplied in full) tabulates 36 conforming vacuum pairs; we used 9.
Added `tests/benchmarks/joint_conductance_ext.csv` (174 points, 158 usable
after excluding rows printed too coarsely — R to 2 decimals reaches ±50 % at
0.01 K/W): Antonetti Ni 200, Milanez and McWaid SS 304, Nho Ni 200, Al 6061-T6
and Al/Ni. Parsed from the text and checked against each row's printed %
column. Our CMY reproduces Bahrami's own model column within 3 % for every
same-material set (transcription and area confirmed). Against measurement:
0.950, RMS 13.3 %, all within 2×; Al-containing 0.951, 11.1 %. The light-load
trend repeats on these independent data: meas/pred 1.03 below Y/σ 3.2, 1.15
above. Nho's ground surfaces are anisotropic (outside CMY's assumption).

### 12.4 Bolted joints — Yeh et al., ETFS 25, 349 (2001)

Bolted Al 6061-T6 blocks (63.5 mm square, 4 or 8 bolts, 1–10 N·m), in air,
contact pressure from Fuji film, σ 0.25–0.4 and 1.27 µm, no slopes or
hardness. Results as correlations only: h_c = 1.35 + 0.21 P_c^2.8 (4 bolts),
9.04 + 0.1 P_c^3.79 (8 bolts), kW/m²K with P_c in MPa (2–5 MPa), and 8 bolts
beat 4 at the same mean pressure. CMY at their mean pressure (σ_eff 0.50 µm, m
0.05–0.095, H 1.12 GPa) gives 50–190 kW/m²K against their 3–28: **5–35×
over**. The P^2.8–3.8 dependence and the bolt-count effect say the real
contact is confined near the bolts — flatness and waviness, not asperities,
control it. For screwed heat sinks and lids, the apparent-area pressure into
CMY is the wrong input; a flatness (macro-contact) model or a measured h is
needed. Their Al/RTV/Al data (0.2 mm RTV, k 0.31: 1.16 kW/m²K) imply 2.2 ×
10⁻⁴ m²K/W of boundary resistance for both faces.

### 12.5 Al/Si interface conductance — Hopkins et al., PRB 82, 085307 (2010)

Table I: eight Al(80 nm)/Si(100) interfaces by TDTR at room temperature;
roughness 1.4–10.6 nm by AFM; native oxide stripped (BOE) or regrown.
158–193 MW/m²K clean, 115–144 with oxide; h falls with roughness in both
classes; the oxide adds ~1.74 × 10⁻⁹ m²K/W by their Eq. 4 fit (≈ 2.2 nm of
SiO₂; the plain pairwise difference here is 1.91 × 10⁻⁹). Their sine-
dispersion DMM is ~230; our Debye DMM is 321, i.e. 1.66× the best real
interface and 2.2× the as-received one — the DMM placeholder is an upper
estimate for a real die metallisation. Benchmark in
`tests/benchmarks/kapitza_measured.csv`.

### 12.6 Grease TIMs — DeVoto et al., NREL/CP-5400-68014 (ITherm 2017)

ASTM D5470 at 139 kPa, Ni-plated polished blocks, BLT 29–266 µm set by
measured glass spheres (95th-percentile diameter). Five commercial greases fit
R = BLT/k + R_c with R² 0.991–0.998: k 0.7–3.8 W/mK, R_c (both faces)
4.2–28.3 mm²K/W. The stack's TIM form is exactly this regression. Good greases
reach 12–18 mm²K/W at 29–50 µm, below Blazej's 2003 grease floor (20); the
class range now starts at 10. Pump-out after −40/125 °C cycling is shown as
acoustic images only — no resistance numbers to build a degradation model on.

### 12.7 Accommodation — Fan & Manson, arXiv:0804.2269

Full text read: classical theory with trapping–desorption fits Ar, Kr, Xe on
clean W; "much too large for He and Ne" (Sec. III). §8.2 corrected; our flag's
direction (classical He = upper estimate) stands.

### 12.8 Out-of-flatness as built (P0-9)

From the same thesis, Ch. 3–4: a crowned flat is an equivalent sphere ρ =
b_L²/2δ (δ = combined max out-of-flatness); the macro-contact radius follows
a_L = 1.80 a_H √(α + 0.31τ^0.056)/τ^0.028 (Eq. 3.30) with a_H = (3Fρ/4E′)^⅓,
α = σρ/a_H², τ = ρ/a_H, and the whole face touches above F_c (Eq. 4.23). R_L =
(1 − a_L/b_L)^1.5/(2k_s a_L) adds in series; the micro term is independent of
curvature (Eq. 4.30), so our CMY network stays as the micro part. Reproduces
the thesis's α, τ, B, R_L for all 205 crowned points (A.3–A.38); our CMY + R_L
gives 1.03 / 15.3 % RMS on R_exp. The size of the effect is the lesson: on
those joints plain CMY predicts R at 0.14× measured (h ~7× too high), down to
0.008× — the same direction and order as Yeh's bolted joints (§12.4). A
"flat" joint in the thesis's sense is B ≥ 0.8, roughly δ/σ ≤ 3–30.

Choices: the macro term is applied only when flatness AND joint size are
given; otherwise an INFO flag (CAUTION when bolted) says flat contact was
assumed. In gas, the open macro gap's gas conduction is not modelled, so h_g
is over-estimated there (stated in the flag). For filled joints the crown is
absorbed by the fill; a bond line thinner than δ is warned.

### 12.9 Found while building the agent (P3-1)

Running the scripted heat-sink case end to end exposed two physics defects
that the unit tests had not: (1) the accommodation band re-combined the
network without the macro term, so in air with a crowned face the band read
67 kW/m²K against a 12.6 headline; (2) the gas path was computed for the
asperity gap over the whole face, so a 25 µm crown left 75 % of the heat in a
gas film that is really tens of µm thick. Fixed: the band passes R_macro, and
`crowned_gap_conductance` averages k_g/(y + M) over the crown profile (closed
form; our first-order extension). Same case after the fix: 6.4 kW/m²K, gas
51 %, band 6.35–6.38. Lesson: an end-to-end case through the user-facing path
is a test class of its own — it combines options (gas + flatness + bolted)
that unit tests exercise one at a time.

Also: a bonded layer with no material key gets no face estimate; the planner
first read that as "no interface, MD not needed" while the stack flag said the
faces could carry 99 % of R. The planner now uses the flag's pessimistic share.

### 12.10 Jobs must describe themselves (P3-1, MD ingestion)

Re-measuring the 22 Oct Si/Al plumbing run needed its NEMDSpec, which lived in
a throwaway script. The deck generator had changed since (new comments, a
mobile-atom thermo compute), so the spec could not be recovered by regenerating
decks and comparing; it was rebuilt from the deck's own numbers (geometry from
the region lines, stage lengths from the `run` lines, frames from `ave/chunk`)
and reproduced the recorded 444.6 ± 173.3 MW/m²K exactly. Rule now: every job
written by `prepare_cluster` or `plan_campaign` carries `spec.json`
(`md.campaign.write_spec` / `read_spec`), and `agent.ingest` refuses a folder
without one rather than guessing.

### 12.11 Copper Vickers coefficients — searched, none found (P0-6)

Copper is the most common packaging metal and every Cu joint falls back to
bulk hardness (HARDNESS_FALLBACK). Searched 23 Oct: Song & Yovanovich 1988
(mhtl88-8, Table 2: Zr-4, Zr-2.5Nb, Ni 200, SS 304 only), Yovanovich 2005
review (Table VI: the same four metals), Bahrami et al. 2004 (Tables 3, 10:
those plus Al 6061-T6), Sridhar & Yovanovich 1996 (Wear; fits c₁, c₂ to Brinell
1.30–7.60 GPa, above soft Cu). The one open copper indentation-size-effect
study (Crystals 14, 913, 2024) is 97 % Cu, unannealed, ~100 HV, with Meyer
n ≈ 2.0–2.1 (almost no size effect; HV not even monotonic over 10–100 gf) —
not the annealed packaging copper in materials.yaml (0.36 GPa), so not used.
Two candidates could not be reached: Phil. Mag. A 82(10) (2002), grain-size
effect on the indentation hardness of high-purity Cu and Al; and an "analysis
of indentation size effect in copper and its alloys" study (academia.edu,
venue unverified). Instead of a number, the path for measured data: `Hv_*`,
`c1_*`, `c2_*` overrides in `build_joint` and the agent.

### 12.12 Unloading as built (P0-10)

Mikic's unloading model with the truncated geometry, Milanez et al. Eqs.
16–19, read from page images. Transcription checked analytically before
coding: with u = λ_max + X², I(0) = erfc(λ_max/√2)/2, so the model starts at
the first-loading point; Eq. 19 + Eq. 2 reproduce Eq. 8 and Eq. 1 reproduces
Eq. 6. Two reading errors on the way, both caught by physics: the first factor
of Eq. 18 is exp(−u²), not exp(−u²/2) (with /2 the spot density went negative
immediately); and the printed bracket {λ_max − …} must be {λ_max + X² − …} —
deriving Eq. 18 as the spots lost between λ_max and λ_max + Z² gives that
form, which integrates to n*(λ_max + Z²) and reaches zero when all contacts
open; the printed one keeps ~60 %. The truncated version has an edge where
A_r/A_a = I(Z) − erfc(z_t/√2)/2 reaches zero; the root search is bracketed
inside it. Hysteresis on S1: 1.4× at 1.5 MPa to 2.6–3.1× at 30 kPa. No
tabulated unloading data were available, so magnitudes are untested.

### Still missing

Marmier et al. 1997 (N₂ on Cu, P5-4), Song's 1988 thesis (tabulated gas-gap
data, P0-3), Prasher 2005 (BLT rheology, P0-1), Vandeperre et al. 2007 (Si/Ge
hardness), an elastic-regime dataset (P0-4), and the Ioffe pages (Si/Ge
properties).

## 13. Retrieval (`lookup`) as built (24 Oct 2026)

Pure-Python BM25 (`openkapitza/rag/`), no new dependencies. Chunks are
document sections (Markdown headings 1–4, manual-page sections), one per
LAMMPS error/warning message, one per example script, one per paper note.
Each carries path, line range, section path, version and, for manual pages,
the docs.lammps.org URL. Field weights: title ×3, keywords ×3 (`.. index::`
entries, commands and styles in example scripts, paper topics), body ×1;
k1 1.2, b 0.75. Tokens keep compounds and their parts (`ave/chunk`, `si(d)`,
`NEMD_UNEQUILIBRATED`), `Eq. 18` → `eq18`, crude plural and -ing/-ed folding.
A pasted query found verbatim in a chunk gets +0.6 × the top score.

Corpus: our docs + `papers.yaml` + `lammps-rag.md` = 335 chunks, indexed from
the working tree at each start (0.4 s). Full manual of LAMMPS 30 Sep 2026
(commit 8de817dd): 1,073 pages + 1,005 example files = 10,638 chunks, built in
8 s to a 4.6 MB cache, never committed (GPL text, size).

Question set: 86 questions in `tests/data/rag_questions.yaml`, written before
the search code, phrased as a user would ask. A hit counts if its source (and
section, where given) matches.

| set | n | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|
| repo (manual loaded) | 48 | 0.90 | 0.98 | 0.98 | 0.98 | 0.93 |
| repo (no manual) | 48 | 0.92 | 1.00 | 1.00 | 1.00 | 0.95 |
| curated (manual loaded) | 12 | 0.67 | 0.83 | 0.92 | 1.00 | 0.76 |
| curated (no manual) | 12 | 0.92 | 1.00 | 1.00 | 1.00 | 0.96 |
| manual | 26 | 0.54 | 0.73 | 0.81 | 0.85 | 0.65 |

What changed the manual numbers (first build: R@5 0.73, MRR 0.48):
- **Error entries over-ranked.** Each is one line, so BM25 length
  normalisation rewards it ("Stillinger-Weber pair style" returned five error
  messages). Fix: ×0.6 unless the query matches the message verbatim.
- **Example scripts have almost no prose.** Fix: each `in.*` gets its
  directory README's opening words, plus "example"/"input script", as
  keywords; ×1.4 when the query asks for an example.
- **Howto pages** get "howto / how to / overview" keywords.
- Exact command named ("fix langevin") → that page ×1.3.
- **Rejected, measured:** priors favouring our docs or the curated subset over
  the manual (×1.1–1.2). Manual R@5 fell 0.73 → 0.62–0.65 and repo gained
  nothing; the curated subset repeats manual content, so a prior just puts the
  duplicate first.

Remaining misses at 5 (the case for or against semantic search):
- repo — "how do I build LAMMPS with the packages this project needs": the
  manual's Build pages outrank the runbook. Fair answers, wrong file.
- manual — "hold atoms fixed by setting their force to zero" (`fix setforce`)
  and "thermo output columns pxx pyy pzz custom style" (`thermo_style`): no
  shared words. **Semantic search would likely fix these.**
- manual — "Stillinger-Weber pair style": rank 7, still behind error entries.
- manual — melt/quench and two-temperature example scripts: `examples/melt`
  and `examples/ttm` have no README and the scripts never say "melt" or
  "two-temperature". Our own melt-quench notes and the curated TTM section
  rank first instead: useful answers. Semantic search would not help either:
  the words are not there.

Thresholds in `tests/test_rag.py` sit just under these numbers (repo
0.95/0.88, curated 0.83/0.70, manual 0.75/0.58 for R@5/MRR). The weights were
tuned on this same question set, so the numbers are optimistic. A held-out
set is the honest next check before any further tuning.

## 14. The job loop and the LLM front end (24 Oct 2026)

### The rule: retry what the cluster broke, never what the input or physics broke

`openkapitza/agent/jobs.py`. A stopped job is classified from its scheduler
state, `log.lammps`, stderr and outputs:

- **infrastructure** (TIMEOUT, NODE_FAIL, PREEMPTED, out of memory, killed by
  a signal): resubmitted automatically, at most 3 attempts; a timeout gets
  1.5× the walltime (cap 72 h).
- **input** (a LAMMPS `ERROR:` line): reported with the best two `lookup`
  passages. On the knowledge base's own error the top hit is §1's entry.
- **physics** (lost atoms, NaN in a thermo row): fixes proposed, never applied.
  A smaller timestep or another structure changes what is measured.
- **output** (exit 0 without `OPENKAPITZA_DONE`, or a header-only profile):
  the silent failure of §2.

The decks write no restart files, so a retried job starts over. Restart from a
checkpoint is the obvious next step for multi-day runs.

### Checked end to end on a real run

The Si/Al MEAM plumbing deck was run through the Session with the real LAMMPS
build: `jobs DIR`, `submit sial_pp local`, `watch 20`. It ran 1,840 s serial
(6,680 atoms), was classified "finished; output present", and was ingested
automatically: h_K = 444.6 ± 173 MW/m²K. That is identical to the committed
fixture from 23 Oct, because the deck and seeds are the same. The joint
re-ran from 858 to 905 kW/m²K with the MD caveats attached: 39 % error,
single cell, mean T 260 K against 300 K. The loop is verified; the number is
still a plumbing number, not a converged one.

### Measured on the user's Apple M3 Pro (28 Oct)

`scripts/benchmark_local.py`: 5 performance + 6 efficiency cores, 18 GB,
LAMMPS stable_30Sep2026 built with Homebrew and Open MPI. The rates in
atom-steps per second are total, with per core in brackets:

| potential | np = 4 | np = 5 | np = 8 |
|---|---|---|---|
| Tersoff (Si/Ge, 35,840 atoms) | 4.2e6 (1.05e6) | **5.2e6** (1.04e6) | 3.6e6 (4.5e5) |
| MEAM (Si/Al, 28,836 atoms) | 4.5e5 (1.12e5) | 5.7e5 (1.13e5) | **6.2e5** (7.7e4) |

- **Scaling from 4 to 5 ranks is perfect.** Adding efficiency cores (np = 8)
  makes Tersoff 31 % slower, because every rank waits for the slowest. MEAM,
  with more work per atom, gains 9 %.
- **One M3 performance core is 2.3× the cost model's Tersoff figure** and
  1.5× a cloud container core, for both potentials.
- **Sizes of all 22 prepared jobs here:**
  - Every Tersoff job, including the whole Si/Ge campaign: 0.3–5.3 h each,
    ~33 h in total.
  - Si/Al phonon-only: ~40 h. Si/Al two-temperature: ~40 h, more with the
    TTM fix.
  - Si/Cu: ~51 h.
  - Everything: ~164 h, about a week of continuous running.
- **The job loop launched every local job at once.** With 16 campaign rungs
  that is 80 ranks on 11 cores. Fixed: local jobs queue and run one at a
  time (`LOCAL_SLOTS`).
- Sustained runs may be slower than the 30–130 s benchmark if the laptop
  throttles; the first long job's log will show it.

### Four defects found building it

- **`tests/test_rag.py` broke the documented test command.** It was written
  for pytest; `python3 -m unittest discover` (the README's command, and the
  only runner in a bare install) failed on `import pytest`. Rewritten as
  unittest. Every test file is unittest-style.
- **The walltime increase keyed on wording.** It tested for "timeout" in a
  summary that said "walltime limit", so timeouts were resubmitted with the
  same walltime. Diagnoses now carry a `cause` code. Caught by the fake-SLURM
  test.
- **`LocalRunner` overwrote the prepared `run.sh`.** Its wrapper used the same
  name as the script `prepare_cluster` writes for manual runs. Renamed to
  `job_wrapper.sh`; regression test added.
- **Finished local jobs were never reaped through their `Popen`.** `poll()`
  read the `job.done` sentinel and `_pid_alive` reaped with `os.waitpid`, so
  each job's `Popen` kept believing its process was running. That leaks one
  object per job over a campaign and shows as `ResourceWarning: subprocess N
  is still running` at exit. `poll()` now lets the `Popen` reap its own child.

### MCP server notes

- The installed `mcp` is 2.x: `FastMCP` is now
  `mcp.server.mcpserver.MCPServer`, and `mcp.Client(server)` connects
  in-process for tests. A `ToolError` message reaches the model as the tool
  result, so validation errors read as the agent's own ("must be greater than
  0 um").
- What is enforced is in the server: unit parsing, ranges, choices, physics,
  flags. The instructions (pass on every `must_report` item, say which values
  were defaulted, never invent inputs, confirm before spending compute) are
  guidance to the model, not enforcement. The guide says so.

## 15. ML for interface resistance, and the measured-data check (28 Oct 2026)

### Survey: is an ML quick-estimate layer worth having?

Two families (notes in `docs/papers.yaml`):

- **Descriptor regressions on tables** (Zhan 2017; Wu 2019; Foss & Aksamija
  2023). Good in-distribution. Held out by material, error grows several-fold
  (Foss: RMSE < 5 → 20–30 MW/m²K, failing at descriptor extremes). That is
  exactly the case where OpenKapitza would want a quick estimate.
- **Machine-learned potentials inside MD** (Rajabpour 2024 Si/diamond: Tersoff
  ~3× high, MTP within 10 % of experiment; Khot 2025 Al/Si: DeePMD 380 ± 33
  against ~350–379 measured, a 2NN MEAM 20–30 % high). The evidence for
  accuracy is here.

Two cautions decided the design:

- **Measured and simulated values are different quantities.** A measurement
  includes oxide, roughness, interdiffusion and adhesion layers; ideal-interface
  MD does not. Al/Si: 115–193 MW/m²K on as-prepared surfaces (Hopkins 2010),
  ~350–380 for clean ones. Ga₂O₃/SiC: 550–700 (NEP, ideal) against 140–162
  measured (Liu 2026), so an accurate potential is still 4× off a real
  interface. A surrogate trained on measurements and an MD answer for an ideal
  interface must not be mixed silently.
- **The Kapitza term is often small in the joints this package targets** (14–19 %
  in the Si/Al run of §14). The quick estimate matters mostly for the
  MD/no-MD decision and for bonded faces.

Decision: no separate "ML agent". Three steps in order:
1. A measured-data check in the core (built, below).
2. A Tier-2 potential route for Si/Al (P4-1).
3. A regression surrogate only if it beats the DMM in a leave-one-pair-out
   test on the measured table (pending the data).

### The check as built

`md/measured.py`, called from `run_continuum` for every bare joint:

- **Matching.** Database keys map to measured names (`Al6061` → Al with a note,
  `Ni200` → Ni). Raw table names are normalised (`Si(100)` → Si, `sapphire` →
  Al2O3). Rows with a deliberate interlayer (Ti, Cr...) are left out; native
  oxides are kept and counted.
- **Temperature.** Only points within ±50 K are used; if none, all
  temperatures are used and the summary says so.
- **Outcomes.**
  - Within 1.25× of the measured extremes: INFO.
  - Above: CAUTION. An ideal interface usually conducts better, so it
    over-states a real joint.
  - Below: WARNING. An ideal interface conducting worse than real ones points
    at the model.
  - A user's own h_K is only noted.
- **Remedy, never substitution:** `set h_K_MW=<median>`.
- **Potential bias:** a caution when the MD potential has a published bias for
  the pair (Al/Si MEAM → Khot 2025). Reported, never corrected.
- **MCP:** `must_report` now carries cautions as well as warnings, so the
  model passes the check on.

On the Si/Al joint: the DMM (321 MW/m²K) is 1.7× the highest of the eight
Hopkins points. The MEAM plumbing run (445) is 2.3×, plus the MEAM bias
caution. With only Hopkins' as-prepared surfaces in the table, "highest
measured" is 193; the cleaner ~350–380 measurements arrive with the NIMS
import. The summary names its source count, so it does not overclaim.

### The NIMS dataset, imported

Wu et al., Sci. Data 7, 36 (2020); Zenodo 10.5281/zenodo.3564173; CC BY 4.0.
The user supplied the files (Zenodo is not reachable from this workspace).
`ITR_dataset.xlsx` imports as 1,322 rows over 119 pairs into
`openkapitza/data/itr_measured.csv`; the attribution and the changes made are
in its header. Four facts from the real file, none visible in the paper's
Table 1:

- **The `interlayer` column is a 0/1 flag.** The layer materials are inside
  the `Interface` string (`Au/Ti/a-SiO2/Si`): film first, substrate last,
  interlayers between.
- **SiO₂ often marks a native oxide.** Hopkins 2010's as-received samples are
  "Al/SiO2/Si". An oxide of a neighbour is kept and counted as oxide. Any other
  layer (Ti, Ge, GaN/AlN, polymers: 481 rows) is recorded but left out of the
  pair's range.
- **Some samples were deliberately damaged** (proton irradiation down to 26
  MW/m²K for Al/Si; 20 rows). They are marked `modified` and left out. The
  "non-irradiated" controls stay.
- **Hopkins 2010 is in both tables.** The repeat (same pair, h within 0.5 %,
  T within 5 K, and the same article number) is dropped, so it is not counted
  twice. One row's substrate column contradicts its Interface string (Al/Si
  with substrate "Al"); the Interface string is used.

Si/Al at 300 ± 50 K is now 35–380 MW/m²K: 40 values from 11 papers; clean
35–380 (n = 20), with an oxide 50–191 (n = 20); 65 with other layers and 16
damaged left out. The DMM (321) and the MEAM run (445, 1.17× the top) are
within the 1.25× band. The MEAM bias caution still applies. Part of the spread
is the measurement itself: Ma et al.'s values rise 172 → 218 MW/m²K with the
TDTR pump spot size alone.

### The surrogate test: not worth adding (28 Oct)

`scripts/evaluate_itr_surrogate.py` fits a Gaussian process (Matérn 5/2 +
white noise, standardised descriptors) and boosted trees to the authors'
training table (692 rows, 66 pairs, 35 descriptors), in log10 h:

| model | random 5-fold | leave-pair-out |
|---|---|---|
| Gaussian process | rms 0.18, within 2× 92 % | rms 0.28, within 2× 74 % |
| boosted trees | rms 0.17, within 2× 92 % | rms 0.25, within 2× 80 % |
| training median | — | rms 0.49, within 2× 39 % |

Holding out the pair raises the error by half, as Foss & Aksamija found. That
is the regime OpenKapitza would use it in. Against our Debye DMM, on the rows
our database can compute (Al/Si, Ge/Si; same-material bonded wafers
excluded):

| rows | DMM | GP leave-pair-out |
|---|---|---|
| all (178) | rms 0.39, within 2× 37 % | rms 0.22, within 2× 85 % |
| no interlayer (56) | rms 0.33, within 2× 64 % | rms 0.33, within 2× 59 % |

The GP wins overall only by learning the interlayer flag ("oxide present →
lower"). On clean interfaces, the quantity DMM and MD describe, it is no
better. The measured check already reports the oxide effect with real data for
the pair, so the surrogate adds nothing for the pairs this package can run.
DMM sits a median 1.6× above clean measurements (2.1× with oxides), consistent
with its role as an upper estimate. **Decision: no surrogate.** Revisit if the
material database grows to materials without Debye data (DMM cannot run, a
descriptor model could), or if oxide/bonded interfaces are modelled
explicitly. The training table is not committed; the script reads it from the
Zenodo files.
