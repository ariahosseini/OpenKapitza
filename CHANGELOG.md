# Changelog

Notable changes, newest first. Dates are commit dates; the version in
`pyproject.toml` changes at a release. Design reasoning lives in
[`docs/knowledge-base.md`](docs/knowledge-base.md) (sections cited as KB §n)
and open work in [`docs/open-issues.md`](docs/open-issues.md). History
before November 2026: [`docs/history-action-items.md`](docs/history-action-items.md)
and `git log`.

## Unreleased (after 0.2.0)

### Added
- **MD drives and measurement.** `flux_mode="ehex"` (drift-corrected
  imposed flux), `interface_tally` (direct interfacial heat flow, reported
  by `measure_kapitza`), stage checkpoints after equilibration and the
  steady stage with automatic resume (KB §28, §32).
- **Interface structures.** Non-bonded (van der Waals) contacts and
  physisorbed gas layers (Ar, N₂, He, Ne); intermixed (`intermix_layers`)
  and stepped (`interface_steps`) Si/Ge interfaces; a rescaled-Tersoff
  strain-free Si/Ge cell; an incoherent 25:24 Si/Ge cell for the cluster
  (KB §19, §26, §30, §33, §34).
- **Contact-model checks** that flag without changing the answer: ballistic
  contact spots, slope measured by optical/AFM instruments, heavily doped
  silicon (KB §27).
- **Quantum bracket** for classical-MD h_K below the Debye temperature,
  graded by T/θ_D (KB §31).
- **Job sets and analysis scripts:** `prepare_nonbonded.py`,
  `prepare_weekend.py`, `prepare_disorder.py`, `analyze_weekend.py`,
  `extract_structure_slices.py`.
- **Docs:** `docs/install.md`, `docs/tutorial.md`; README examples with
  atom pictures and figures, an equations reference; MIT `LICENSE`,
  `CITATION.cff`; CI on Python 3.10–3.13.
- **Builds:** the macOS LAMMPS script adds RIGID, TALLY, ML-SNAP, ML-IAP and
  ML-PACE; the runbook has a KOKKOS-CUDA build for MACE, DeePMD and
  NequIP/Allegro. Local jobs keep the Mac awake.

### Changed
- The gas-gap model is **no longer described as biased low near 1 atm**:
  the trend was a specimen confound (KB §29). The default is unchanged.
- An MD h_K brought into the agent now gets the quantum check; the
  measured-range note says how far outside the range a value is when it is
  inside the tolerance.
- Python ≥ 3.10 (what CI tests).

### Fixed
- `validate_deck` rejects backslash line continuations (LAMMPS uses `&`).
- `papers.yaml` entries with `: ` in the citation are quoted.
