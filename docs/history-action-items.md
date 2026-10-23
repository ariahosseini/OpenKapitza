# Archived: README action items (to 23 Oct 2026)

Moved out of the README when `docs/open-issues.md` became the tracker. Kept
verbatim for the reasoning it records; where it disagrees with the tracker,
the tracker is right.

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
