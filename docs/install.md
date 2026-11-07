# Installing OpenKapitza

Two parts, and most users need only the first:

1. **The Python package** — the continuum model, the agent, the job loop,
   the MCP server. Minutes, any OS with Python ≥ 3.10.
2. **LAMMPS** — only to run molecular dynamics yourself (the Kapitza
   term). Without it the agent still answers, using the Debye-DMM
   placeholder or a value you supply, and says so.

## 1. The Python package

Use a virtual environment so the project's dependencies cannot clash with
anything else on the machine:

```bash
git clone https://github.com/ariahosseini/OpenKapitza.git
cd OpenKapitza
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .                     # pyyaml and numpy
```

`-e` links the checkout into the environment, so a later `git pull` takes
effect without reinstalling. Run `source .venv/bin/activate` in **every new
terminal** before any `python3` command for this project (the prompt shows
`(.venv)`); a `ModuleNotFoundError: No module named 'yaml'` means it was not.

Optional extras, in any combination:

| extra | what it adds |
|---|---|
| `pip install -e ".[mcp]"` | the MCP server, to use OpenKapitza from Claude or another LLM client |
| `pip install -e ".[plots]"` | sweep plots and the README figures (matplotlib) |
| `pip install -e ".[xlsx]"` | importing measured interface data from Excel files |
| `pip install -e ".[all]"` | all three |

**Check it:**

```bash
python3 -m unittest discover -s tests      # ~770 tests, under a minute; LAMMPS ones skip
python3 -m openkapitza.agent               # the guided session
```

## 2. LAMMPS (for molecular dynamics)

OpenKapitza writes LAMMPS inputs and runs the `lmp` executable as a separate
program. It is validated against **stable_30Sep2026** with these packages:

| package | used for |
|---|---|
| MANYBODY | Tersoff, Stillinger–Weber, EAM |
| MEAM | Si/Al, Si/Cu (Jelinek MEAM) |
| EXTRA-FIX, EXTRA-COMPUTE | two-temperature MD (`fix ttm/thermal`), elastic constants |
| RIGID | `fix ehex`, the drift-free imposed heat flux |
| TALLY | the direct interfacial heat flow |
| ML-SNAP, ML-IAP, ML-PACE | machine-learned potentials that run on a CPU (optional) |

### macOS (Apple silicon or Intel)

One script builds it, checks every package and prints the two paths you
need:

```bash
bash scripts/mac_setup_lammps.sh            # into ~/lammps-openkapitza
```

- **macOS 15 or newer:** it uses Homebrew (installs cmake, open-mpi, libomp).
- **macOS 14 or older:** Homebrew no longer ships prebuilt packages there;
  use Miniforge instead, as the script explains, then re-run it inside
  `conda activate openkapitza`.
- If Homebrew says *"Your Command Line Tools are too outdated"*, update them
  from Software Update; the LAMMPS build itself usually still succeeds.
- ML-PACE downloads a library while it configures; if that fails the script
  finishes without it and says so.

**Long runs on a laptop:** the job loop keeps the Mac awake while a job runs
(`caffeinate`), but closing the lid still sleeps most Macs. Keep it open and
on power, or start the agent with `caffeinate -ims python3 -m openkapitza.agent`.

### Linux workstation

```bash
git clone --depth 1 --branch stable_30Sep2026 https://github.com/lammps/lammps.git
cd lammps && mkdir build && cd build
cmake ../cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_MPI=yes \
      -D PKG_MANYBODY=yes -D PKG_MEAM=yes -D PKG_EXTRA-FIX=yes -D PKG_EXTRA-COMPUTE=yes \
      -D PKG_RIGID=yes -D PKG_TALLY=yes \
      -D PKG_ML-SNAP=yes -D PKG_ML-IAP=yes -D PKG_ML-PACE=yes
make -j 8
./lmp -h | grep -A6 "Installed packages"   # every package above must be listed
```

Needs a C++17 compiler, CMake ≥ 3.20 and an MPI library (Open MPI or MPICH).

### HPC cluster, and GPUs

See [`cluster-runbook.md`](cluster-runbook.md): the same CPU build with the
cluster's modules, job scripts written by the agent (`submit ... slurm`),
and §0b for the separate KOKKOS-CUDA build that MACE, DeePMD and
NequIP/Allegro need. Apple-silicon GPUs cannot run these inside LAMMPS.

### Check LAMMPS with OpenKapitza

```bash
export OPENKAPITZA_LMP=~/lammps-openkapitza/build/lmp
export OPENKAPITZA_POTENTIALS=~/lammps-openkapitza/potentials
python3 -m unittest tests.test_checkpoint tests.test_md_accommodation \
    tests.test_md_conductivity tests.test_md_elastic tests.test_md_indentation
```

These run short LAMMPS jobs (minutes) and skip themselves without the two
variables. Then, in the agent, prepared MD jobs run with
`submit <jobs> local lmp=$OPENKAPITZA_LMP np=4` (see the
[agent guide](agent-guide.md#running-the-jobs)); a timed benchmark of your
machine is `python3 scripts/benchmark_local.py --lmp ... --potentials ...`.

## Optional: the full LAMMPS manual for `lookup`

`lookup` searches our docs and a curated LAMMPS subset out of the box. To
search the whole manual of the LAMMPS you run (never committed):

```bash
python -m openkapitza.rag build --lammps-src ~/lammps-openkapitza
```

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `No module named 'yaml'` (or `numpy`) | the virtual environment is not active: `source .venv/bin/activate` |
| `pip`: *externally-managed environment* | same: activate `.venv` first |
| agent: *LAMMPS executable 'lmp' not found* | give the path: `submit ... lmp=/path/to/lmp` |
| a job's CPU use falls to a few % overnight | the Mac slept: keep the lid open and on power (see above) |
| `ERROR: Unrecognized fix style 'ehex'` or `compute ... heat/flux/virial/tally` | LAMMPS built without RIGID or TALLY: rebuild with the script |
