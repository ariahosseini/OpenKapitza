#!/usr/bin/env bash
# Build the LAMMPS this project uses (stable_30Sep2026) on macOS, natively.
#
#   bash scripts/mac_setup_lammps.sh [INSTALL_DIR]        # default ~/lammps-openkapitza
#
# Two ways to get the tools, chosen automatically:
#   conda   an ACTIVE conda environment (Miniforge, conda-forge) holding cmake,
#           openmpi and llvm-openmp. Use this on macOS 14 or older, where Homebrew
#           no longer ships prebuilt packages:
#             conda create -n openkapitza -c conda-forge python=3.12 cmake openmpi \
#                   llvm-openmp git make
#             conda activate openkapitza
#   brew    Homebrew on macOS 15 or newer: installs cmake, open-mpi and libomp.
# Both need the Xcode command-line tools (Apple's compiler). Clones LAMMPS at the release tag, checks the
# commit is the one OpenKapitza was validated against, and builds with the
# packages the decks use: MANYBODY (Tersoff, SW), MEAM, EXTRA-FIX (ttm/thermal),
# EXTRA-COMPUTE, RIGID (fix ehex) and TALLY (compute heat/flux/virial/tally; KB 20),
# the CPU machine-learned potentials ML-SNAP, ML-IAP (SNAP/linear; no Python) and
# ML-PACE (ACE; cmake downloads its library from GitHub, and the build continues
# without it if that fails), plus OPENMP when Apple's compiler accepts Homebrew's
# libomp (MPI runs do not need it; the build continues without it if that fails).
# Not here, by design: MACE, NequIP/Allegro and DeePMD need libtorch/TensorFlow
# and, to be practical, a CUDA GPU; build those on the cluster (cluster-runbook.md).
#
# Prints, at the end, the two paths the benchmark and the agent need:
#   LMP=<install>/build/lmp   POTENTIALS=<install>/potentials
set -euo pipefail

DEST="${1:-$HOME/lammps-openkapitza}"
TAG="stable_30Sep2026"
COMMIT="8de817dd79bfe4525d5d39246a212d833e6dee07"
JOBS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)"

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "this script is for macOS (on Linux, follow docs/cluster-runbook.md)"
xcode-select -p >/dev/null 2>&1 || die "Xcode command-line tools missing: run 'xcode-select --install', then re-run"
MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  MODE=conda
  say "Using the active conda environment: $CONDA_PREFIX"
  for tool in cmake mpicxx mpirun git; do
    command -v "$tool" >/dev/null || die "$tool not in the conda environment: conda install -c conda-forge cmake openmpi llvm-openmp git make"
  done
  [[ -f "$CONDA_PREFIX/lib/libomp.dylib" ]] || say "llvm-openmp not found in the environment; OpenMP will be skipped"
  OMP="$CONDA_PREFIX"
  MPI_PREFIX="$CONDA_PREFIX"
  # conda's Open MPI wrappers name conda's own clang; use Apple's (Xcode CLT)
  export OMPI_CC=clang OMPI_CXX=clang++
else
  MODE=brew
  if (( MACOS_MAJOR < 15 )); then
    die "macOS $MACOS_MAJOR: Homebrew no longer ships prebuilt packages for it, so every
       package would compile from source. Use Miniforge instead (no OS update needed):
         1. install Miniforge: https://conda-forge.org/download/
         2. conda create -n openkapitza -c conda-forge python=3.12 cmake openmpi llvm-openmp git make
         3. conda activate openkapitza
         4. re-run this script in that terminal"
  fi
  command -v brew >/dev/null || die "Homebrew not found: install it from https://brew.sh, or use Miniforge (see the top of this script)"
  say "Homebrew packages (cmake, open-mpi, libomp, git)"
  brew install cmake open-mpi libomp git
  OMP="$(brew --prefix libomp)"
  MPI_PREFIX="$(brew --prefix open-mpi)"
fi

if [[ -d "$DEST/.git" ]]; then
  say "Using the existing checkout in $DEST"
  git -C "$DEST" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG" 2>/dev/null || true
  git -C "$DEST" checkout -q "$TAG"
else
  say "Cloning LAMMPS $TAG into $DEST"
  git clone -q --depth 1 --branch "$TAG" https://github.com/lammps/lammps.git "$DEST"
fi
HEAD="$(git -C "$DEST" rev-parse HEAD)"
if [[ "$HEAD" != "$COMMIT" ]]; then
  die "checkout is at $HEAD, expected $COMMIT ($TAG). Remove $DEST and re-run."
fi

common=(
  -S "$DEST/cmake" -B "$DEST/build"
  -D CMAKE_BUILD_TYPE=Release
  -D BUILD_MPI=yes
  -D PKG_MANYBODY=yes -D PKG_MEAM=yes -D PKG_EXTRA-FIX=yes -D PKG_EXTRA-COMPUTE=yes
  -D PKG_RIGID=yes -D PKG_TALLY=yes -D PKG_ML-SNAP=yes -D PKG_ML-IAP=yes
  -D CMAKE_PREFIX_PATH="$MPI_PREFIX"
  -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++
)
omp=(
  -D BUILD_OMP=yes -D PKG_OPENMP=yes
  -D OpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include"
  -D OpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include"
  -D OpenMP_C_LIB_NAMES=omp -D OpenMP_CXX_LIB_NAMES=omp
  -D OpenMP_omp_LIBRARY="$OMP/lib/libomp.dylib"
)

# Try the richest configuration first, then drop the optional parts one at a time.
WITH_OMP=no; WITH_PACE=no
for attempt in "omp pace" "omp" "pace" ""; do
  extra=()
  [[ "$attempt" == *omp* ]] && extra+=("${omp[@]}") || extra+=(-D BUILD_OMP=no)
  [[ "$attempt" == *pace* ]] && extra+=(-D PKG_ML-PACE=yes)
  say "Configuring (${attempt:-MPI only, no ML-PACE})"
  rm -rf "$DEST/build"
  if cmake "${common[@]}" "${extra[@]}" > "$DEST/cmake-config.log" 2>&1; then
    [[ "$attempt" == *omp* ]] && WITH_OMP=yes
    [[ "$attempt" == *pace* ]] && WITH_PACE=yes
    break
  fi
  cp "$DEST/cmake-config.log" "$DEST/cmake-config.failed-${attempt// /-}.log"
  [[ -z "$attempt" ]] && die "cmake failed; see $DEST/cmake-config.log"
done
[[ $WITH_PACE == yes ]] || say "ML-PACE skipped (its download or configure failed; see $DEST/cmake-config.failed-*.log)"

say "Building with $JOBS parallel jobs (several minutes)"
cmake --build "$DEST/build" -j "$JOBS" > "$DEST/build.log" 2>&1 \
  || die "build failed; the end of $DEST/build.log says why"

LMP="$DEST/build/lmp"
say "Checking the build"
"$LMP" -h | grep -A6 "Installed packages" || true
for pkg in MANYBODY MEAM EXTRA-FIX EXTRA-COMPUTE RIGID TALLY ML-SNAP ML-IAP; do
  "$LMP" -h | grep -q "$pkg" || die "package $pkg missing from the build"
done
mpirun --version | head -1

cat <<EOF

Done. LAMMPS $TAG ($COMMIT), tools from $MODE, OpenMP: $WITH_OMP, ML-PACE: $WITH_PACE

  LMP=$LMP
  POTENTIALS=$DEST/potentials

Next, the timed benchmark (about 15-25 minutes):

  python3 scripts/benchmark_local.py --lmp "$LMP" --potentials "$DEST/potentials"
EOF
