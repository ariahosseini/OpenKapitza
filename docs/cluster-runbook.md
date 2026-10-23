# Cluster runbook

What to do, in order, when cluster access arrives. Every step is a command
against code that already exists; the open-issue IDs refer to
[`open-issues.md`](open-issues.md), whose *Cluster run list* this expands.

Nothing here submits on its own. Each stage writes job directories and a
costed manifest; read it, then submit.

## 0. Build LAMMPS

Packages: `MANYBODY` (Tersoff, SW, EAM), `EXTRA-FIX` (`fix wall/reflect/stochastic`),
`EXTRA-COMPUTE` (`compute born/matrix`), and `MEAM` (Si/Al, Si/Cu through
`AlSiMgCuFe.meam`). With MPI this time.

```bash
cmake ../cmake -D BUILD_MPI=yes -D PKG_MANYBODY=yes -D PKG_EXTRA-FIX=yes \
      -D PKG_EXTRA-COMPUTE=yes -D PKG_MEAM=yes -D CMAKE_BUILD_TYPE=Release
make -j 16
```

Check the build: `lmp -h | grep -A3 "Installed packages"` must list all four.

The 30 Sep 2026 stable release needs C++17 and CMake ≥ 3.20 (KOKKOS: C++20).
For P1-14 (TTM) `EXTRA-FIX` already provides `fix ttm/thermal`. For Tier-2 MACE
later: `PKG_ML-IAP`, `MLIAP_ENABLE_PYTHON`, `PKG_PYTHON`, KOKKOS with CUDA — GPU
nodes only (knowledge base §8.4).

## 1. Test suite and audits on the cluster (minutes)

```bash
python3 -m unittest discover -s tests
OPENKAPITZA_LMP=$(which lmp) OPENKAPITZA_POTENTIALS=<lammps>/potentials \
    python3 -m unittest tests.test_md_elastic tests.test_md_accommodation \
        tests.test_md_conductivity tests.test_md_indentation
python3 scripts/audit_nemd.py          # expect: FAILURES: none
python3 scripts/measure_potential_elastic.py --lmp $(which lmp) \
    --potentials <lammps>/potentials --set sicge
```

A second platform has caught defects before (knowledge base, "verify on a
second platform").

## 1b. Write every job at once

```bash
python3 scripts/prepare_cluster.py <root> --potentials <lammps>/potentials \
    --lmp "$(which lmp)" --partition <p> --account <a> --module <lammps-module>
```

One directory per job (campaign rungs incl. the reservoir-length axis, Si/Ge
spot pressure in both modes, the P1-8 equilibrium run, Si/Al phonon-only and
TTM, Si/Cu, Si/SiO₂ strong and weak), each validated, with `submit.sbatch`
and `run.sh`; `manifest.json` costs them (dry run here: 22 jobs, ~69
node-hours, ~1,800 core-hours with the 2× walltime factor); `commands.txt`
holds the P2-5 quench commands. Update `THROUGHPUT` after step 2 and re-run
it before submitting. The sections below describe each job.

## 2. Throughput (P1-11, ~10 core-minutes)

The cost model's three-body rates are measured on a laptop and a cloud
container. Re-measure on the cluster's nodes before trusting any walltime:
run the base campaign rung for ~20 ps on its planned core count, read
`Loop time` from `log.lammps`, and update `THROUGHPUT` in
`openkapitza/compute/resources.py` (atom-steps per core-second). A job
killed at the wall loses everything; the safety factor is only 2x.

## 3. Equilibration time (P1-12)

Write the campaign (step 4) first, then run only the base rung `r00_*` with
a deliberately short `steady_steps` and a long production, and read where
it becomes stationary:

```bash
python3 scripts/find_equilibration_time.py <campaign>/r00_*/ --window-ps 100
```

`STATISTICS-LIMITED` means the window is too short to certify, not that
the run is unequilibrated (P1-1). Set the preset's `steady_ps` in
`scripts/plan_campaign.py` from the answer, plus margin, and rewrite the
campaign.

## 4. Convergence campaign (P1-3, ~350 core-hours)

```bash
python3 scripts/plan_campaign.py write <campaign> \
    --potential <lammps>/potentials/SiCGe.tersoff \
    --partition <p> --account <a> --lmp "$(which lmp)" --module <lammps-module>
# read <campaign>/manifest.json, then:
for d in <campaign>/r*/; do sbatch "$d/submit.sbatch"; done
# when finished:
python3 scripts/plan_campaign.py collect <campaign>
```

`collect` measures every rung with `measure_kapitza` (all profile frames,
jackknife error, piston-relaxed interface) and judges each axis with
error-aware verdicts. An axis reported `unresolved` needs longer production
on its last two rungs, not larger rungs.

## 5. Si/Ge h_K (P1-4)

From the converged (or extrapolated) campaign. Report against 266–368
MW/m²K, with the `SiCGe.tersoff` stiffness inversion (P1-5) stated next to
the number.

## 6. Die-relevant pairs (P1-6)

Si/Al and Si/Cu through `AlSiMgCuFe.meam` (needs MEAM from step 0). First
gate the potential on each side (a0 and C_ij, within 10 %), then a campaign
on the converged geometry. Si/Al at 300 K brackets: NEMD 470 MW/m²K
phonon-only (Yang et al., arXiv:1401.5550, a different MEAM), TDTR 200–350. The metal
side is phonon-only (P1-14).

## 6b. Electron–phonon on metal sides (P1-14)

Build one Si/Al cell twice, phonon-only and `ttm=TTMSpec("Al")`, from the
same spec (≥ 30 nm Al so the default 3-cooling-length fit exclusion leaves a
bulk region; cross-section wide enough for ≥ 1000 Al atoms per 1 nm electron
cell, i.e. ≳ 4 nm). `write_campaign` stages `ttm_props.in`. Steady state is
~0.5–1 ns. Measure both with `measure_kapitza`, then
`md.ttm.compare_runs(h_pp, err_pp, h_ttm, err_ttm, flags)`; compare with
`md.electron_phonon.series_band` (Al: lower bound by construction). Check
`T_e_mean.dat` against the metal lattice mean (equal far from the interface).

## 7. Si/a-SiO₂ (P2-2 … P2-6)

Glass and interface first, then NEMD on the read-in cell:

```bash
python3 scripts/build_si_sio2.py --lmp $(which lmp) --potentials <lammps>/potentials \
    --out <dir> --quench-rate 1e12 --bonding strong --bulk-nm 8
python3 scripts/build_si_sio2.py ... --bonding weak --glass <dir>/quench/glass.data
```

The quench checkpoints after every segment of its ramp: if a job hits its
walltime, resubmit the same command and it resumes from the last completed
segment. Read the validation table and the bonding line (coverage, class)
before running NEMD. Run both bonding classes on the same glass (P2-6), and repeat
at 10¹³ K/s for the rate sensitivity (P2-5). Bracket: Chen, Zhang & Li 2012 (arXiv:1210.0354),
≈1.2 GW/m²K strong / ≈0.23 weak; experiment > 0.6 GW/m²K.

## 7b. Equilibrium cross-check (P1-8)

`sige_emd/`: same Si/Ge potential, short slabs (2 nm + reservoirs), NVE after
equilibration, slab energies logged every 20 fs for 16 ns (≥ 500 slow
relaxation times; the local run measured τ_slow = 32 ps). Analyse with
`md.emd_kapitza.read_emd` → `h_from_series(..., model="biexp")` → `check(...,
h_nemd=<converged campaign h_K>)`. Use the energy columns, not the kinetic
temperatures, and the two-exponential model (knowledge base §11). Expect NEMD/EMD > 1 (≈3 for Si/Ge per
Merabia & Termentzidis); below 1 means one of the runs is wrong.

## 8. Only if needed

SW-Si bulk k at ns scale (P5-1): only if bulk k for a material without
handbook data becomes an input.
