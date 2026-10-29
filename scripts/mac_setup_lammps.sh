#!/usr/bin/env bash
# Build the LAMMPS this project uses (stable_30Sep2026) on macOS, natively.
#
#   bash scripts/mac_setup_lammps.sh [INSTALL_DIR]        # default ~/lammps-openkapitza
#
# Needs Homebrew (https://brew.sh) and Xcode command-line tools. Installs cmake,
# open-mpi and libomp with brew, clones LAMMPS at the release tag, checks the
# commit is the one OpenKapitza was validated against, and builds with the
# packages the decks use: MANYBODY (Tersoff, SW), MEAM, EXTRA-FIX (ttm/thermal),
# EXTRA-COMPUTE, plus OPENMP when Apple's compiler accepts Homebrew's libomp
# (MPI runs do not need it; the build continues without it if that step fails).
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
command -v brew >/dev/null || die "Homebrew not found: install it from https://brew.sh, then re-run"
xcode-select -p >/dev/null 2>&1 || die "Xcode command-line tools missing: run 'xcode-select --install', then re-run"

say "Homebrew packages (cmake, open-mpi, libomp, git)"
brew install cmake open-mpi libomp git

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

OMP="$(brew --prefix libomp)"
common=(
  -S "$DEST/cmake" -B "$DEST/build"
  -D CMAKE_BUILD_TYPE=Release
  -D BUILD_MPI=yes
  -D PKG_MANYBODY=yes -D PKG_MEAM=yes -D PKG_EXTRA-FIX=yes -D PKG_EXTRA-COMPUTE=yes
  -D CMAKE_PREFIX_PATH="$(brew --prefix open-mpi)"
)
omp=(
  -D BUILD_OMP=yes -D PKG_OPENMP=yes
  -D OpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include"
  -D OpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include"
  -D OpenMP_C_LIB_NAMES=omp -D OpenMP_CXX_LIB_NAMES=omp
  -D OpenMP_omp_LIBRARY="$OMP/lib/libomp.dylib"
)

say "Configuring (with OpenMP)"
rm -rf "$DEST/build"
WITH_OMP=yes
if ! cmake "${common[@]}" "${omp[@]}" > "$DEST/cmake-config.log" 2>&1; then
  say "OpenMP configuration failed (see $DEST/cmake-config.log); building with MPI only"
  rm -rf "$DEST/build"
  cmake "${common[@]}" -D BUILD_OMP=no > "$DEST/cmake-config.log" 2>&1 \
    || die "cmake failed; see $DEST/cmake-config.log"
  WITH_OMP=no
fi

say "Building with $JOBS parallel jobs (several minutes)"
cmake --build "$DEST/build" -j "$JOBS" > "$DEST/build.log" 2>&1 \
  || die "build failed; the end of $DEST/build.log says why"

LMP="$DEST/build/lmp"
say "Checking the build"
"$LMP" -h | grep -A6 "Installed packages" || true
for pkg in MANYBODY MEAM EXTRA-FIX EXTRA-COMPUTE; do
  "$LMP" -h | grep -q "$pkg" || die "package $pkg missing from the build"
done
mpirun --version | head -1

cat <<EOF

Done. LAMMPS $TAG ($COMMIT), OpenMP: $WITH_OMP

  LMP=$LMP
  POTENTIALS=$DEST/potentials

Next, the timed benchmark (about 15-25 minutes):

  python3 scripts/benchmark_local.py --lmp "$LMP" --potentials "$DEST/potentials"
EOF
