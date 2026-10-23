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
phonon-only (Liang et al. 2014, a different MEAM), TDTR 200–350. The metal
side is phonon-only (P1-14).

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
at 10¹³ K/s for the rate sensitivity (P2-5). Bracket: Lampin et al. 2012,
≈1.2 GW/m²K strong / ≈0.23 weak; experiment > 0.6 GW/m²K.

## 8. Only if needed

SW-Si bulk k at ns scale (P5-1): only if bulk k for a material without
handbook data becomes an input.
