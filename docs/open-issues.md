# Open issues

The one list of everything not finished. Update it in the same commit as the
work that opens or closes an item; a closed item moves to the *Closed* table
with the commit that closed it. Scope is PCB, package and die (agent
instruction, header): items that serve none of those are **parked**, not
deleted.

Status: **open** · **in progress** · **blocked** (by another item) ·
**cluster** (needs long runs; deferred until cluster access) · **parked**
(out of current scope).

Last updated: 22 Oct 2026.

---

## Phase 1 — MD interfacial resistance (current focus)

| ID | Item | Status | Closes when |
|---|---|---|---|
| P1-1 | **Stationarity check has no noise floor.** `check_flux_stationarity` uses a fixed 15 % drift tolerance; on the 400 ps Si/Ge run the metric's own scatter is 36 % (mean 46 %), so it reports "not steady" for every run. | in progress | Drift is judged against its own statistical uncertainty; a known-stationary synthetic series passes and a decaying one fails. |
| P1-2 | **In-plane constant 73 % toward Ge vs 51 % predicted.** Hypothesis: walls clamp z (Pzz ≈ +1.5 GPa), which expands a∥ through Poisson coupling. | open | Relaxation with longer z or a free surface moves a∥ toward ~5.55 Å (hypothesis holds) or does not (find the real cause). |
| P1-3 | **Convergence campaign** over cross-section, length, duration and flux. ~100 core-hours; converged target ~112k atoms, 500 ps. | blocked by P1-1; then **cluster** | Each ladder stays within tolerance over its last two rungs; h_K quoted with the convergence uncertainty. Local pilot first. |
| P1-4 | **Si/Ge h_K = 428 MW/m²K vs published 266–368.** `NEMD_UNEQUILIBRATED`, `NEMD_CONDUCTIVITY_MISMATCH` still raised. | blocked by P1-1, P1-3 | A converged h_K with no WARNING-or-worse flags, compared against the published range. |
| P1-5 | **`SiCGe.tersoff` fails the elastic gate** and inverts the Si/Ge biaxial-stiffness ordering. | open | A gated alternative potential is found, or the result is reported with the inversion stated (current behaviour). |
| P1-6 | **Die-relevant validation pairs.** Phase 1 is validated only on Si/Ge. Needed: Si/Al, Si/SiO₂, Si/Cu; GaN/SiC if a gated potential exists. | open | For each pair: potential gated (a0, C_ij), published h_K located, pilot run within its flags. |
| P1-7 | **Cell stress.** 1.51 GPa Pzz cannot be relaxed in-plane (z non-periodic). Needs per-side out-of-plane constants from Poisson ratios. | open (needs Phase 2 builder) | Structure built with per-side out-of-plane constants; Pzz under the `NEMD_CELL_STRESS` threshold. |
| P1-8 | **Equilibrium cross-check** (temperature-fluctuation method, not Green–Kubo). | open (Phase 4 scope) | Implemented and agrees with NEMD within the joint uncertainty on one pair. |
| P1-9 | **Spot-pressure MD.** §5 says to run at H_c, not the apparent pressure; the deck supports `spot_pressure_GPa`, never exercised on a real run. | open | One run at spot pressure with its effect on h_K reported. |

## Phase 0 — continuum core

Built and validated for the contact path: 207 measured points (Ni 200, SS 304,
vacuum), geometric-mean ratio 1.030, 12.7 % RMS. Return for one packaging pass
after Phase 1.

| ID | Item | Status | Closes when |
|---|---|---|---|
| P0-1 | **No TIM-filled or bonded-joint path.** The gap model handles gas or vacuum only; most package joints are filled (grease, pad, PCM) or bonded (solder, sintered Ag, die attach). | open — packaging pass | Filled-gap and bonded-layer paths in the network, each with flags and a test. |
| P0-2 | **No Cu/Al validation in air.** Benchmark is Ni/SS in vacuum. | open — packaging pass | Exact (not digitised) measured data for Cu or Al joints at ~1 atm air, compared. |
| P0-3 | **Gap model reads ~32 % low toward the continuum limit** (where 1 atm air sits). Cause: uniform-gap assumption; opt-in Gaussian integral closes ~⅓. | open — needs literature | Exact tabulated h_j(P_gas) settles whether to change the default. |
| P0-4 | **Elastic branch has no measured check** (8 of 9 benchmark specimens plastic). | open | Data on an elastic-regime pair (hard smooth surfaces). |
| P0-5 | **ξ·h_K bridge has no measured check.** Kapitza term negligible at the benchmark's conductances. | open | Data where h_K matters — thin bonded die-level layers are the likely case. |
| P0-6 | **Bundled material data all `verified: false`.** Hardness weakest; Song–Yovanovich c₁, c₂ null everywhere on purpose. | open | Packaging materials (Cu, Al, Si, solder alloys) checked against primary sources. |

## Work package 5 — MD-derived inputs (§11)

| ID | Item | Status | Closes when |
|---|---|---|---|
| P5-1 | **Bulk k (5c) not validated.** SW-Si 500 K: 13.7 / 46.5 / 85.7 W/mK at 8.7 / 17.4 / 34.8 nm; extrapolation undetermined (CRITICAL). Target 93 ± 18. | **cluster**, low priority (handbook k exists for packaging materials) | ns-scale runs, wider cross-section, extrapolation within the target's error. |
| P5-2 | **Elastic constants at 0 K only.** | open, low priority | `examples/ELASTIC_T` or `compute born/matrix` at T. |
| P5-3 | **Ar/Ni accommodation from a short run** (0.52 ± 0.12). | parked (1 atm air: α barely moves h_g) | Longer run, ±0.04. |
| P5-4 | **Air (N₂/O₂) accommodation on Cu/Al** — the gases and metals packaging actually uses. Only He, Ne, Ar, N₂ LJ parameters exist; no physisorption target for N₂. | open, low priority | Calibrated N₂ on Cu with a sourced adsorption energy. |
| P5-5 | **Nanohardness box too narrow** (2.1 contact diameters). | parked | Wider slab. |

## Later phases

| ID | Item | Status |
|---|---|---|
| P2-1 | Structure module: crystalline supercell matching; melt-quench amorphous path (Si/SiO₂) with quench-rate reporting and RDF/bond-angle validation. Feeds P1-6, P1-7. | open |
| P3-1 | Agent layer wrapping Phases 0–2. | open |
| P4-1 | Tier-2 MLIPs behind the phonon-DOS / elastic / bulk-k gate. | open |

## Housekeeping

| ID | Item | Status |
|---|---|---|
| H-1 | Repository is public (`github.com/ariahosseini/OpenKapitza`). A deliberate decision, not a default. | waiting on Aria |
| H-2 | README "Action items — next session" partly stale; this file supersedes it as the tracker. | open |

## Cluster run list

Collected here so the cluster session starts from one list.

1. P1-3 convergence campaign (~100 core-hours; Slurm runner already in `openkapitza/compute`).
2. P1-4 converged Si/Ge h_K (part of 1).
3. P1-6 die-relevant pairs at converged size.
4. P5-1 SW-Si bulk k at ns scale (only if bulk k becomes needed).

---

## Closed

| ID | Item | Closed by |
|---|---|---|
| — | Potential lattice constants (a0 within 0.02 %); elastic gate added to Tier 1 | `ac52a65` |
| — | 5a elastic constants with C44, VRH, two strain magnitudes | `35fb50a` |
| — | 5b accommodation coefficient, calibrated and validated (He/Ni 0.09–0.12) | `04f8e57`, `846e8da`, `698bfa4` |
| — | 5d nanoindentation, compare-only | `3c05731`, `2473d90` |
