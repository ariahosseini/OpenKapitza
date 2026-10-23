# Open issues

The one list of everything not finished. Update it in the same commit as the
work that opens or closes an item; a closed item moves to the *Closed* table
with the commit that closed it. Scope is PCB, package and die (agent
instruction, header): items that serve none of those are **parked**, not
deleted.

Status: **open** · **in progress** · **blocked** (by another item) ·
**cluster** (needs long runs; deferred until cluster access) · **parked**
(out of current scope).

Last updated: 22 Oct 2026 (Phase 1 session).

---

## Phase 1 — MD interfacial resistance (current focus)

| ID | Item | Status | Closes when |
|---|---|---|---|
| P1-3 | **Convergence campaign** over cross-section, length, duration, flux. Tooling built (`md/campaign.py`, `scripts/plan_campaign.py`): Si/Ge preset, 12 rungs, ~350 core-hours, every walltime ≤ 2.5 h. | **cluster** | Each axis converged (or extrapolated) with error-aware verdicts; h_K quoted with the convergence uncertainty. |
| P1-4 | **Si/Ge h_K vs published 266–368 MW/m²K.** Last value 428 (clamped cell, last-frame profile, fit error only: all three since fixed). | blocked by P1-3 | Converged h_K from the piston-relaxed cell with no WARNING-or-worse flags. |
| P1-5 | **`SiCGe.tersoff` fails the elastic gate** and inverts the Si/Ge biaxial-stiffness ordering. | open | A gated alternative potential is found, or the result is reported with the inversion stated (current behaviour). |
| P1-6 | **Die-relevant validation pairs.** Si/Al and Si/Cu: `AlSiMgCuFe.meam` (Jelinek et al., PRB 85, 245102, 2012; ships with LAMMPS) has both cross terms; needs the MEAM package (build in progress) and CubicSpec support for two-file MEAM potentials. **Si/Al targets at 300 K:** NEMD 470 MW/m²K phonon-only, 450 with e-ph (Liang et al. 2014, arXiv:1401.5550, Lee 2NN MEAM — a different potential, so a bracket, not a reproduction); TDTR 350 (Minnich 2011), 220 (Zhang 2013), ~200 (Hopkins, roughness-dependent). Si/SiO₂: `SiO.tersoff` plus an amorphous structure (Phase 2). GaN/SiC: no shipped cross potential. | in progress | Per pair: MEAM build, potential gated (a0, C_ij), published h_K located, pilot run within its flags. |
| P1-14 | **Electron–phonon resistance on metal sides.** Phonon-only NEMD misses the e-ph resistance in series at metal/semiconductor interfaces. Al/Si: −4 % (Liang et al. 2014, Duffy–Rutherford coupling); larger for other metals. LAMMPS has `fix ttm`. | open | Two-temperature coupling available for metal sides, or the e-ph term estimated and flagged per pair. |
| P1-8 | **Equilibrium cross-check** (temperature-fluctuation method, not Green–Kubo). | open (Phase 4 scope) | Implemented and agrees with NEMD within the joint uncertainty on one pair. |
| P1-9 | **Spot-pressure MD.** Implemented as a loaded piston (−P·A/N per wall atom), uniaxial stress, flagged `NEMD_SPOT_PRESSURE_MODEL`. Real 2 GPa test (4,960-atom Si/Ge, 20 ps relax): production raw Pzz 1.68 GPa, in-plane −0.08; 6 % of the shortfall is the box-volume artefact now corrected by slab-volume stress columns. | in progress | One h_K at spot pressure reported against P = 0 on a converged cell. |
| P1-13 | **Uniform stress offset between relaxation and production.** Main cause found: the in-plane box was frozen at its instantaneous size while still oscillating under the barostat (~0.1 % strain). Box now set to its second-half mean before the barostat comes off: offset −0.22 → −0.08 GPa (20 ps relax test). | mostly resolved | Residual ≤ 0.05 GPa with the campaign's 50 ps relax / 100 ps equilibration, or its cause identified. |
| P1-10 | **Spot-pressure lateral condition.** Uniaxial stress only; the real asperity sits between uniaxial stress and uniaxial strain. | open | Uniaxial-strain option added; both reported when h_K is pressure-sensitive. |
| P1-11 | **Throughput on the cluster.** Cost model now uses measured 4.5e5 (Tersoff) / 5.0e5 (SW) atom-steps per core-second from this container and a laptop. | **cluster** | One short run on the cluster; update `compute/resources.py`. |
| P1-12 | **Equilibration time for the campaign.** `plan_campaign.py` uses a 100 ps placeholder for `steady_steps`. | blocked by local 400 ps run; then **cluster** | `find_equilibration_time.py` on the 400 ps Si/Ge run, then on the campaign base rung. |

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

Collected here so the cluster session starts from one list. Step-by-step
commands: [`cluster-runbook.md`](cluster-runbook.md).

1. P1-11 throughput check (one short run), then P1-12 equilibration time on the base rung.
2. P1-3 convergence campaign: `python3 scripts/plan_campaign.py write <dir> --potential SiCGe.tersoff --partition ... --account ...` (~350 core-hours), then `... collect <dir>`.
3. P1-4 converged Si/Ge h_K (part of 2).
4. P1-6 die-relevant pairs at converged size (needs a MEAM-enabled LAMMPS build).
5. P5-1 SW-Si bulk k at ns scale (only if bulk k becomes needed).

---

## Closed

| ID | Item | Closed by |
|---|---|---|
| — | Potential lattice constants (a0 within 0.02 %); elastic gate added to Tier 1 | `ac52a65` |
| — | 5a elastic constants with C44, VRH, two strain magnitudes | `35fb50a` |
| — | 5b accommodation coefficient, calibrated and validated (He/Ni 0.09–0.12) | `04f8e57`, `846e8da`, `698bfa4` |
| — | 5d nanoindentation, compare-only | `3c05731`, `2473d90` |
| P1-1 | Stationarity check judged against its own noise (three verdicts, Monte Carlo calibrated) | `d03114b` |
| P1-2 | In-plane constant 73 % vs 51 %: the clamped cell (+13 points), temperature (+5), nonlinearity (+4); measured at 0 K and 300 K | `ebdd7f5`, `5b1e0c1` |
| P1-7 | Cell stress: piston z-relaxation, no Phase 2 builder needed; production Pzz ≈ Pxx ≈ Pyy | `5b1e0c1` |
| — | Profile used only the last of 4 frames; flux unit; frame/flux alignment; per-window flux inflation; silent block-error loss | `442e4a1`, `ebdd7f5`, `5b1e0c1` |
| — | Cost model 3-4x optimistic for three-body potentials | `7e0c89f` |
