# Open issues

The one list of everything not finished. Update it in the same commit as the
work that opens or closes an item; a closed item moves to the *Closed* table
with the commit that closed it. Scope is PCB, package and die (agent
instruction, header): items that serve none of those are **parked**, not
deleted.

Status: **open** · **in progress** · **blocked** (by another item) ·
**cluster** (needs long runs; deferred until cluster access) · **parked**
(out of current scope).

Last updated: 23 Oct 2026 (user-supplied papers: P0-7 truncated Gaussian, extended benchmark incl. Al 6061, Al/Si TDTR, NREL greases, hardness data).

---

## Phase 1 — MD interfacial resistance (current focus)

| ID | Item | Status | Closes when |
|---|---|---|---|
| P1-3 | **Convergence campaign** over cross-section, length, duration, flux. Tooling built (`md/campaign.py`, `scripts/plan_campaign.py`): Si/Ge preset, 12 rungs, ~350 core-hours, every walltime ≤ 2.5 h. | **cluster** | Reservoir-length axis added (1, 2, 4 nm; Wei et al. 2025): 14 rungs. Each axis converged (or extrapolated) with error-aware verdicts; h_K quoted with the convergence uncertainty. |
| P1-4 | **Si/Ge h_K vs published 266–368 MW/m²K.** 400 ps, 2.2 nm, clamped (old deck): 706 from the last frame; **577 ± 67** re-measured with all frames and the jackknife error (block 2.3× the fit error). Small cell, clamped, not converged. Add the converged reference: 450 MW/m²K at ~152 nm, Tersoff, lattice-matched Ge (Feng et al. PRB 99, 045301, 2019); our ≤ 20 nm cells sit on the steep part of the length curve. | blocked by P1-3 | Converged h_K from the piston-relaxed cell with no WARNING-or-worse flags. |
| P1-5 | **`SiCGe.tersoff` fails the elastic gate** and inverts the Si/Ge biaxial-stiffness ordering. Survey + build check (23 Oct): no other shipped Si/Ge potential exists in our build (Tersoff variants only; `SiGeH.sw.quip` needs QUIP). The fix is Tier 2 (P4-1). | open → Tier 2 | A gated Si/Ge MLIP, or results reported with the inversion stated (current behaviour). |
| P1-6 | **Die-relevant validation pairs.** Si/Al and Si/Cu: Jelinek `AlSiMgCuFe.meam` **passes the elastic gate** on Si, Al and Cu (all C_ij within 4 %); heterostructure builder (fcc on diamond, 6:8 supercell, ±0.29 %) and MEAM decks built; MEAM-enabled LAMMPS built here. **Si/Al end-to-end plumbing run** (4 nm slabs, 8 ps production, unconverged): 445 ± 173 MW/m²K phonon-only, ~354 with the e-ph estimate — next to Yang et al.'s 470 (arXiv:1401.5550, NEMD, different MEAM) and TDTR 200–350, but a ±39 % plumbing number, not a validation. **Measured target now tabulated** (`tests/benchmarks/kapitza_measured.csv`, Hopkins et al. PRB 82, 085307, Table I): Al/Si TDTR 158–193 MW/m²K with the native oxide stripped, 115–144 with it (oxide ≈ 1.7–1.9 × 10⁻⁹ m²K/W); our Debye DMM (321) is 1.66× the best one. Si/SiO₂: `SiO.tersoff` plus an amorphous structure (Phase 2). GaN/SiC: no shipped cross potential. | **cluster** for converged runs |
| P1-14 | **Electron–phonon resistance on metal sides.** Phonon-only NEMD misses it. **Estimate built** (`md/electron_phonon.py`): Majumdar–Reddy h_ep = √(g·k_p) in series; reproduces Yang et al.'s Al/Si series value (0.37 GW/m²K) from their own inputs. **Survey done** (knowledge base §7): k_p is the weakest input (Al: 8.1 first-principles with p-e scattering, 17.0 in their MEAM MD); g spans ~5× for Cu (5.5e16–2.6e17); the series formula over-predicts the full TTM-MD resistance by 6–15 % (Wang et al. 2012); `fix ttm/thermal` supports electron-free cells (k_e = γ_p = 0) but requires periodic z and a fixed box; cooling length √(k_p/g) ≈ 6 nm (Al), 15–18 nm (Cu), so metal slabs of tens of nm. **Band built** (`series_band`): first-principles lattice k_p by default, sourced g spread; at h_pp 460 MW/m²K Al/Si → 347 (332–358, single g), Cu/Si at 400 → 301 (283–340); flagged as a lower bound (series over-predicts resistance, cross-interface channel omitted). **TTM-MD deck built** (`md/ttm.py`, KB §7.x): `fix ttm/thermal`, periodic z with a self-checked vacuum guard, electrons on the metal only, production-averaged metal T_e, metal-side fit exclusion, `compare_runs` for h_ep. Plumbing pair on Si/Al (40 ps): runs end to end, not steady (reservoir fluxes 85–102 % apart), h_ep correctly not quoted; electron and lattice means agree within noise. Converged pair needs ≳ 30 nm Al and ~0.5–1 ns. | deck built; **cluster** | TTM-MD run for Si/Al giving h_ep and h_pp separately, and the series estimate reported as a band; converged on the cluster. |
| P1-8 | **Equilibrium cross-check.** **Built** (`md/emd_kapitza.py`, KB §11): same cell as NEMD, NVE fluctuations, slab-ENERGY ΔT (kinetic temperature hides the decay), two-exponential fit with the slow time as the interface, h = C_r/(Aτ), compared as NEMD/EMD expected in 1–10 (≈3 Si/Ge). Local Si/Ge run (1 ns, 31 τ): τ_slow 32 ps, h 77 ± 269 MW/m²K — unresolved, as expected at that length. | built; **cluster** (`sige_emd`, 16 ns) | NEMD/EMD on the converged campaign cell, both with errors. |
| P1-9 | **Spot-pressure MD.** Implemented as a loaded piston (−P·A/N per wall atom), uniaxial stress, flagged `NEMD_SPOT_PRESSURE_MODEL`. Real 2 GPa test (4,960-atom Si/Ge, 20 ps relax): production raw Pzz 1.68 GPa, in-plane −0.08; 6 % of the shortfall is the box-volume artefact now corrected by slab-volume stress columns. | in progress | One h_K at spot pressure reported against P = 0 on a converged cell. |
| P1-13 | **Uniform stress offset between relaxation and production.** Main cause found: the in-plane box was frozen at its instantaneous size while still oscillating under the barostat (~0.1 % strain). Box now set to its second-half mean before the barostat comes off: offset −0.22 → −0.08 GPa (20 ps relax test). | mostly resolved | Residual ≤ 0.05 GPa with the campaign's 50 ps relax / 100 ps equilibration, or its cause identified. |
| P1-10 | **Spot-pressure lateral condition.** `spot_pressure_mode="stress"` (default: in-plane barostat at 0 under load) or `"strain"` (in-plane box relaxed unloaded, then held while the piston loads; lateral stress ~C12/C11·P, checked 0 < Pxx/P < 1 instead of ≈ 0). The flag names the bound and asks for the other. Engine test (2,080-atom Si/Ge, SiCGe.tersoff, 20 ps relax + 20 ps load, 2 GPa): unloaded box 22.2769 Å; **stress** mode 22.4056 Å (+0.58 %), σxx/σyy 0.02/0.03, σzz 1.69 GPa; **strain** mode box identical to unloaded, σxx/σyy 0.87/0.88, σzz 1.81 GPa, lateral/axial 0.48 (≈ C12/C11 of this potential). Both σzz short of 2 GPa: the P1-9 shortfall, smaller under strain. | implemented | Both modes run on a converged cell at one pressure and the bracket reported (cluster, with P1-9). |
| P1-11 | **Throughput on the cluster.** Cost model now uses measured 4.5e5 (Tersoff) / 5.0e5 (SW) atom-steps per core-second from this container and a laptop. | **cluster** | One short run on the cluster; update `compute/resources.py`. |
| P1-12 | **Equilibration time and the steady-state criterion.** Measured on the 400 ps, 2.2 nm Si/Ge run: reservoir-power drift resolvable only to 59 % (2σ) — STATISTICS-LIMITED everywhere; the campaign's largest rungs would reach ~13–27 %. The per-frame jump is far better resolved (11.6 % block error), so `measure_kapitza` now also tests the jump's trend across frames. `plan_campaign.py` uses a 100 ps placeholder for `steady_steps`. | **cluster** | Steady time set from the base rung's per-frame jump trend, not from power drift alone. |

## Phase 0 — continuum core

Built and validated for the contact path: 207 measured points (Ni 200, SS 304,
vacuum), geometric-mean ratio 1.030, 12.7 % RMS (headline set, Hegazy). Extended
set (23 Oct, same thesis appendix): 158 more usable points — Antonetti Ni 200,
Milanez and McWaid SS 304, Nho Ni 200 / Al 6061-T6 / Al–Ni — 0.950, 13.3 % RMS,
all within 2×; Al-containing pairs 0.951, 11.1 %. Return for one packaging pass
after Phase 1.

| ID | Item | Status | Closes when |
|---|---|---|---|
| P0-1 | **Filled and bonded joints.** **Built** (`network/stack.py`, KB §9): `build_joint(fill=tim_fill(...))` or `bond_fill(...)`; TIM = BLT/k + R_c1 + R_c2 (or a measured total), bond = Σ t/(k(1−f)) + two unscaled Kapitza faces (user/MD, else DMM); flags for missing R_c (upper bound), thin TIM vs roughness, voids (direction of error stated), class range, and a missing Kapitza face graded by how much it could matter. **Checked against measured greases** (DeVoto et al., NREL/CP-5400-68014, D5470, 5 greases, BLT 29–266 µm; `tests/test_tim_greases.py`): the stack form is their regression (R² ≥ 0.991), reproduced within 5 % with their k and R_c; grease class range lowered 20 → 10 mm²K/W (three good greases sit at 12–18 at 29–50 µm). Degradation in that report is images only, no numbers. Not modelled: BLT from rheology (Prasher 2005 not supplied), TIM degradation, 3D void maps, solid contacts through a thin TIM. | built; form checked | Bond-line thickness from rheology (Prasher constants) and a measured bonded-joint (solder/sinter) R_th. |
| P0-2 | **No Cu/Al validation in air.** **Al checked in vacuum** (23 Oct): Nho's Al 6061-T6 and Al 6061-T6/Ni 200 joints (Bahrami thesis A.66–A.71; new `Al6061` material) — 72 usable points, pred/meas 0.951, RMS 11.1 % (`tests/test_benchmarks_ext.py`). **Yeh et al. 2001 read: not usable for CMY** — bolted Al 6061, film-averaged pressure, no slopes or hardness, in air; CMY over-predicts their fits 5–35× because flatness/waviness, not asperities, set a bolted joint's real contact (KB §12.4). Still no Cu or Al data **in air** with characterised surfaces. | open — air part only | Exact measured data for Cu or Al joints at ~1 atm air on characterised flats. |
| P0-3 | **Gap model reads ~32 % low toward the continuum limit** (where 1 atm air sits). Cause: uniform-gap assumption; opt-in Gaussian integral closes ~⅓. The supplied Bahrami thesis is vacuum-only; Song's 1988 thesis (the tabulated gas data) is still needed. | open — needs Song 1988 | Exact tabulated h_j(P_gas) settles whether to change the default. |
| P0-4 | **Elastic branch has no measured check** (8 of 9 benchmark specimens plastic). | open | Data on an elastic-regime pair (hard smooth surfaces). |
| P0-5 | **ξ·h_K bridge has no measured check.** Kapitza term negligible at the benchmark's conductances. | open | Data where h_K matters. The bonded-stack path now shows where: a 200 nm metal film makes an omitted face > 5 % of R even at 100 MW/m²K; a 30 µm solder does not. |
| P0-6 | **Bundled material data.** **Per-field verification built** (KB §10): each material lists `sources:` per field; sourced fields load with MEDIUM confidence and their citation, the load-time flag names the rest. Checked and corrected: Si E/ν (Hill average of Hall's C_ij, 162.5 GPa / 0.223), Si sound speeds documented as [100] values; fused silica k 1.38→1.30, ν 0.17→0.16, Hv 8.9→5.12 GPa (Knoop; the old value was nanoindentation-scale); Cu E 117, ν 0.35, Hv 0.6→0.36 GPa, speeds 4680/2260; Al ρ 2710, speeds 6350/3100; Ni 200 k 70.3, E 205, ν 0.29; SS304 θ_D 477→470 (Fe proxy), 304L speeds; Debye temperatures (Kittel) for Si, Ge, Al, Cu, Ni. DMM shifts: Si/Al +5 %, Si/Cu −7 %, Al/Cu −8 %, Si/Ge unchanged. **23 Oct:** SS 304 / Ni 200 c₁, c₂ confirmed against Bahrami, Yovanovich & Culham Table 3; **Al 6061-T6 added** (c₁ 1.123 GPa, c₂ −0.0079, Brinell 915 MPa; Table 9–10, after Nho); hot-hardness fits stored (SS 304, Ni 200, Al 6061) and reported by `HARDNESS_TEMPERATURE` but **not applied** — on Hegazy's 110–145 °C joints the softened c₁ moves pred/meas 1.03 → 1.19 (`tests/test_hardness_data.py`). **Still unsourced:** hardness of Si, Ge, SS304, Ni 200 (bulk Hv); ρ of Si, Ge, Cu; E of Ge, Al, SS304; k of SS304; SiO₂ θ_D; Ge, Ni 200 speeds; Al 6061 k, ρ, E, ν. No solder or TIM entries (stack inputs are per joint). | partly done | Remaining fields sourced (Ioffe for Si/Ge when allowed; handbook for the metals); solder entries if bonded DMM needs them. |
| P0-7 | **CMY under-predicts at light loads.** Flag built (`CMY_LIGHT_LOAD`, Y/σ thresholds from our benchmark). **Truncated-Gaussian option built** (23 Oct, `contact/truncated.py`, `build_joint(z_trunc=...)`): Milanez, Yovanovich & Mantelli (JTHT 18(1) 2004) Eqs. 10–11 (h_c), 14–15 (P/H_c), Eq. 9 separation carried into the gap model (our extension); reference z_trunc for bead-blasted SS 304 / Ni 200 (their Table 1) with uncertainty in the Monte Carlo. Checks: their Table 1 H_c ranges within 2.2 %; their S1 data (thesis A.63) RMS 20.8 → 15.5 % (paper 25.3 → 16.1); on Hegazy's light-load points 1.23 → 1.14. **Two findings:** the Eq. 10 correlation is 7 % low at P/H_c = 1e-6 (paper claims 4 %); and z_trunc only transfers with the c₁/c₂ it was fitted with (with other coefficients it over-corrects, 14.7 → 22.4 % on S1), now an INFO flag. Unloading (hysteresis) model not built. | closed | — |
| P0-8 | **He accommodation spread is surface state.** **Band built**: sourced bands in `gases.yaml` (He 0.36–0.67, Ar 0.87–0.96, N₂/air 0.79–0.87); with any defaulted α the pipeline reports h_g and h_joint at both ends (`JointResult.accommodation_band`, `GAP_ACCOMMODATION_BAND`), widened to include an out-of-band default. Effect: He at 10 kPa on SS 304 h_joint 4.6–8.2 kW/m²K around 7.7; air at 1 atm < 1 %. | closed | — |

## Work package 5 — MD-derived inputs (§11)

| ID | Item | Status | Closes when |
|---|---|---|---|
| P5-1 | **Bulk k (5c) not validated.** SW-Si 500 K: 13.7 / 46.5 / 85.7 W/mK at 8.7 / 17.4 / 34.8 nm; extrapolation undetermined (CRITICAL). Target 93 ± 18. | **cluster**, low priority (handbook k exists for packaging materials) | ns-scale runs, wider cross-section, extrapolation within the target's error. |
| P5-2 | **Elastic constants at temperature.** **Built**: `run_elastic(temperature=T)` / `measure_potential_elastic.py --temperature` (NPT box at its mean, strain on the thermal box, NVT-averaged stress). SW-Si 0 → 300 K: C₁₁ 151.4 → 148.9, C₄₄ 56.4 → 55.2 GPa (the references are room-temperature, so this is the like-for-like gate). | closed | — |
| P5-3 | **Ar/Ni accommodation from a short run** (0.52 ± 0.12). | parked (1 atm air: α barely moves h_g) | Longer run, ±0.04. |
| P5-4 | **Air (N₂/O₂) accommodation on Cu/Al** — the gases and metals packaging actually uses. Only He, Ne, Ar, N₂ LJ parameters exist; no physisorption target for N₂. Source located (Marmier et al., Surf. Sci. 383, 321, 1997: N₂ on Cu(110)) but its values were not reachable here. | open, low priority | Calibrated N₂ on Cu with a sourced adsorption energy. |
| P5-5 | **Nanohardness box too narrow** (2.1 contact diameters). | parked | Wider slab. |

## Phase 2 — structure module (started 22 Oct)

Built: `openkapitza/structure/` — analysis (density, RDF, coordination, angles;
exact on ideal crystals), a-SiO₂ melt-quench with validation gate and
quench-rate flag, Si(100)/a-SiO₂ join with recorded bonding choices;
heterostructures (fcc on diamond) live in `md/nemd.py`. Script:
`scripts/build_si_sio2.py`.

| ID | Item | Status | Closes when |
|---|---|---|---|
| P2-2 | **a-SiO₂ coordination defects (Munetoh Tersoff).** 1536 atoms, melt 6000 K. 10¹⁴ K/s: 73 % fourfold Si, 2.33 g/cm³. 10¹³ K/s: 80 % fourfold (14 % five-, 7 % threefold), 2.31 g/cm³. 10¹³ K/s + 100 ps at 2000 K + re-cool: 81 % fourfold (14 % fivefold), 2.35 g/cm³. Geometry passes throughout; coordination and density fail. The first Si–O shell is fully separated, so the counts are cutoff-independent. **Hypothesis: the potential over-coordinates Si** (defects barely move with rate or anneal). | open; 10¹² K/s **cluster** | 10¹² K/s settles rate vs potential. If the potential: state the defect content with every Si/SiO₂ h_K, and **COMB (Shan et al. 2010, Si/O/Cu) gated: FAILS** — its Si is Tersoff's (−14/+18/−13 %, identical), Cu C₄₄ −31 %; Vashishta SiO₂ cannot describe crystalline Si. Remaining candidates are Tier 2: COMB3 (Tersoff α-quartz density 2.45 vs 2.65 g/cm³; COMB3 near experiment: France-Lanord et al., arXiv:1510.05416) or the open GAP for Si/SiOₓ (Cvitkovich et al., JCP 161, 144706, 2024; github.com/lukas-cvitkovich/MLFF-SiOx), through the gate. |
| P2-3 | **Si/a-SiO₂ h_K.** Targets (300 K): Chen, Zhang & Li 2012 (arXiv:1210.0354; previously misattributed to Lampin et al.), Tersoff-type potential, same rate — strong bonding 0.83 ± 0.25 × 10⁻⁹ m²K/W (≈1.2 GW/m²K), weak 4.27 ± 1.78 × 10⁻⁹ (≈0.23 GW/m²K); experiment > 0.6 GW/m²K (Kimling et al., TR-MOKE). | cell built (`data/structures/Si100_aSiO2_strong_4nm.data`, strong bonding, 117 %); NEMD **cluster** | Converged h_K with bonding descriptors stated, inside the bracket for its bonding class. |
| P2-4 | **Si in `SiO.tersoff` fails the elastic gate** (−14/+18/−13 %): same Tersoff Si(D) as `SiCGe.tersoff`. | open | Reported with every Si/SiO₂ h_K; or a gated Si–O potential. |
| P2-5 | **Quench-rate sensitivity.** One rate is one sample of the rate-dependent glass. | **cluster** | h_K at two rates (10¹³ and 10¹² K/s). |
| P2-6 | **Interface bonding sensitivity.** Both cells built on the same 10¹³ K/s glass, 4 nm per side: strong (15.9 Si(c)–O bonds/nm², 117 % of full Si(100) oxidation, Si⁺¹/⁺²/⁺³ 6/33/15 %) and weak (0 bonds as built, 2.5 Å gap) in `data/structures/`, with decks that write `relaxed.data` so the class actually run is re-measured (`scripts/interface_chemistry.py`). Chen, Zhang & Li: 5x h_K spread between the classes. | cells built; NEMD **cluster** | h_K for both, each with its post-relaxation class. |
| P2-1 | **General crystalline supercell matching.** `structure/crystalline.py`: any (hkl) on cubic or hexagonal crystals (library: Si, Ge, Al, Cu, 3C-SiC, GaN, 4H-SiC), centring derived from the basis so primitive surface cells are found; smallest-area **rectangular** commensurate cell within a strain tolerance (rectangular so the NEMD box stays orthogonal), mismatch split between the sides; slabs with selectable termination (`z_offset`), gap from lateral registry, laid out on the NEMD geometry (`bicrystal_for_nemd`); `scripts/build_bicrystal.py` writes `interface.data` + read-in deck. Per-side type groups (`types_a/types_b`) and per-axis barostat for in-plane anisotropic planes (`couple_xy`). Matches: Si/Al(100) 8.59 Å 0.04 %; Si/Al(111) 11.5 × 20.0 Å 0.59 %; Si/Cu(111) 6.65 × 11.5 Å 1.65 %; 4H-SiC/GaN(0001) 11.1 × 19.2 Å 0.30 %. Engine probe (Si(111)/Al(111), MEAM, 4,920 atoms, 2 ps relax, plumbing only): runs end to end; box +0.2 % as the split strain predicts; Si–Al plane spacing 2.48 → 2.05 Å on relaxation; slab stress −0.12/−0.14/+0.37 GPa; extraction correctly refused (3 nm slabs, 7 bins/side). | built; NEMD **cluster** | Closed for geometry. **Registry scan built** (`scan_registry`, `lammps_static_energy`; `build_bicrystal.py --registry energy`): relaxed MEAM energies on Si(100)/Al(100) span 8.6 meV/Å² over a 4×4 grid, best shift (0, 2.15) Å, not the zero shift used before; the geometric 'hollow' score anti-correlates with it (r = −0.38), so it is a diagnostic only; on Si(111)/Al(111) registry averages out (spread ~0). **Terminations listed** (`terminations`, `--list-terminations`): every distinct cut with bonds broken per face atom (GaN/SiC 1 and 3; Si(100) 2; Si(111) 1 and 3). Remaining: non-rectangular (oblique) cells need a triclinic deck; compound terminations are a user choice (flagged); GaN/SiC has no shipped cross potential. |

## Later phases

| ID | Item | Status |
|---|---|---|
| P3-1 | Agent layer wrapping Phases 0–2. | open |
| P4-1 | Tier-2 MLIPs behind the phonon-DOS / elastic / bulk-k gate. Survey (KB §8.4): Tersoff over-estimates Si/diamond ITC ~3x vs an interface-trained MTP within 10 % of experiment; foundation models as shipped are unreliable for lattice k (MACE-MP-0 log₁₀ MAE ~1.0), so Tier 2 = trained/fine-tuned on the interface. MACE in LAMMPS is GPU-only (ML-IAP + KOKKOS); NEP runs on CPU via NEP_CPU. | open |

## Housekeeping

| ID | Item | Status |
|---|---|---|
| H-1 | Repository is public (`github.com/ariahosseini/OpenKapitza`). A deliberate decision, not a default. | waiting on Aria |
| H-2 | README stale sections. **Closed**: data-status and Phase 2 rewritten; the action-items block moved to `docs/history-action-items.md`. | closed |

## Cluster run list

**All jobs below are written, validated and costed by `scripts/prepare_cluster.py` (dry run: 22 jobs, ~69 node-hours).**

Collected here so the cluster session starts from one list. Step-by-step
commands: [`cluster-runbook.md`](cluster-runbook.md).

1. P1-11 throughput check (one short run), then P1-12 equilibration time on the base rung.
2. P1-3 convergence campaign: `python3 scripts/plan_campaign.py write <dir> --potential SiCGe.tersoff --partition ... --account ...` (~350 core-hours), then `... collect <dir>`.
3. P1-4 converged Si/Ge h_K (part of 2).
4. P1-6 die-relevant pairs at converged size (needs a MEAM-enabled LAMMPS build).
5. P1-14 Si/Al TTM pair: phonon-only and TTM decks of one cell (≥ 30 nm Al, default 3 l_c exclusion, ≥ 1000 atoms per electron cell, ~1 ns steady + production), then `ttm.compare_runs`; report against the series band.
6. Si/SiO₂ NEMD on both built cells `data/structures/Si100_aSiO2_{strong,weak}_4nm.data` (decks alongside; check `relaxed.data` with `interface_chemistry.py`); then P2-2/P2-5 a-SiO₂ at 10¹² K/s (`build_si_sio2.py --quench-rate 1e12`), then Si/SiO₂ h_K for strong and weak bonding (P2-3, P2-6).
7. P5-1 SW-Si bulk k at ns scale (only if bulk k becomes needed).

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
| — | "Mean temperature 9 % below setpoint" on every run was the 0 K frozen walls in LAMMPS's default thermo temperature, not unfinished equilibration; deck now reports mobile-atom temperature | this commit |
