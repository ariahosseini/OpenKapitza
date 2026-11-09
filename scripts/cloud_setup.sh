#!/usr/bin/env bash
# Set up a rented Linux machine (Ubuntu 22.04/24.04) for the OpenKapitza CPU job set.
#
#   git clone https://github.com/ariahosseini/OpenKapitza.git && cd OpenKapitza
#   bash scripts/cloud_setup.sh [INSTALL_DIR]      # default ~/lammps-openkapitza
#
# Does, in order, and stops at the first problem:
#   1. system packages: compilers, cmake, Open MPI, Python venv (apt; needs sudo)
#   2. LAMMPS stable_30Sep2026 at the validated commit, built with MPI and the
#      packages the decks use (MANYBODY, MEAM, EXTRA-FIX, EXTRA-COMPUTE, RIGID,
#      TALLY, ML-SNAP, ML-IAP; OPENMP and ML-PACE when they configure), tuned
#      for this CPU (-march=native: rebuild if you move to another CPU type)
#   3. a Python venv in .venv with OpenKapitza installed, and the test suite
#   4. the job set (scripts/prepare_cloud.py) in ~/ok_cloud, and a smoke copy
#      in ~/ok_smoke run to the end on this machine (scripts/run_smoke.py):
#      every deck must pass before you start the real runs
#
# Environment: NP (ranks per job, default: physical cores / SLOTS), SLOTS
# (jobs side by side, default 4), SKIP_SMOKE=1, SKIP_APT=1, JOBS_DIR, SMOKE_DIR,
# VENV_FLAGS / PIP_FLAGS (passed to python3 -m venv / pip install; offline tests).
# Safe to re-run: finished steps are reused.
set -euo pipefail

DEST="${1:-$HOME/lammps-openkapitza}"
TAG="stable_30Sep2026"
COMMIT="8de817dd79bfe4525d5d39246a212d833e6dee07"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
JOBS_DIR="${JOBS_DIR:-$HOME/ok_cloud}"
SMOKE_DIR="${SMOKE_DIR:-$HOME/ok_smoke}"
SLOTS="${SLOTS:-4}"

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "this script is for Linux (macOS: scripts/mac_setup_lammps.sh)"
[[ -f "$REPO/pyproject.toml" ]] || die "run it from an OpenKapitza clone (scripts/cloud_setup.sh)"

# Physical cores (not hyperthreads): MPI runs gain nothing from SMT siblings.
PHYS="$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
[[ "$PHYS" -ge 1 ]] || PHYS="$(nproc)"
NP="${NP:-$(( PHYS / SLOTS ))}"
(( NP >= 1 )) || NP=1
say "$PHYS physical cores: $SLOTS jobs side by side, $NP ranks each"

# ---- 1. system packages ------------------------------------------------------
if [[ "${SKIP_APT:-0}" == 1 ]]; then
  say "Skipping system packages (SKIP_APT=1)"
elif command -v apt-get >/dev/null; then
  say "System packages (apt)"
  SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq build-essential cmake git \
    libopenmpi-dev openmpi-bin python3-venv python3-dev curl > /dev/null
else
  say "No apt-get: install a C++ compiler, cmake >= 3.20, Open MPI and python3-venv yourself"
fi
for tool in cmake mpicxx mpirun git python3; do
  command -v "$tool" >/dev/null || die "$tool missing"
done

# ---- 2. LAMMPS -----------------------------------------------------------------
LMP="$DEST/build/lmp"
lmp_help() { "$LMP" -h 2>/dev/null || true; }   # read whole: grep -q + pipefail = SIGPIPE
if [[ -x "$LMP" ]] && lmp_help | grep -c TALLY >/dev/null; then
  say "Reusing the LAMMPS build in $DEST/build"
else
  if [[ -d "$DEST/.git" ]]; then
    git -C "$DEST" fetch -q --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG" 2>/dev/null || true
    git -C "$DEST" checkout -q "$TAG"
  else
    say "Cloning LAMMPS $TAG"
    git clone -q --depth 1 --branch "$TAG" https://github.com/lammps/lammps.git "$DEST"
  fi
  HEAD="$(git -C "$DEST" rev-parse HEAD)"
  [[ "$HEAD" == "$COMMIT" ]] || die "LAMMPS checkout is at $HEAD, expected $COMMIT ($TAG)"
  common=(-S "$DEST/cmake" -B "$DEST/build" -D CMAKE_BUILD_TYPE=Release -D BUILD_MPI=yes
          -D CMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native"
          -D PKG_MANYBODY=yes -D PKG_MEAM=yes -D PKG_EXTRA-FIX=yes -D PKG_EXTRA-COMPUTE=yes
          -D PKG_RIGID=yes -D PKG_TALLY=yes -D PKG_ML-SNAP=yes -D PKG_ML-IAP=yes)
  WITH=""
  for attempt in "omp pace" "omp" ""; do
    extra=()
    [[ "$attempt" == *omp* ]] && extra+=(-D BUILD_OMP=yes -D PKG_OPENMP=yes) || extra+=(-D BUILD_OMP=no)
    [[ "$attempt" == *pace* ]] && extra+=(-D PKG_ML-PACE=yes)
    say "Configuring LAMMPS (${attempt:-MPI only})"
    rm -rf "$DEST/build"
    if cmake "${common[@]}" "${extra[@]}" > "$DEST/cmake-config.log" 2>&1; then
      WITH="$attempt"; break
    fi
    [[ -z "$attempt" ]] && die "cmake failed; see $DEST/cmake-config.log"
  done
  say "Building LAMMPS on $(nproc) threads (about 5-10 minutes)"
  cmake --build "$DEST/build" -j "$(nproc)" > "$DEST/build.log" 2>&1 \
    || die "LAMMPS build failed; the end of $DEST/build.log says why"
fi
for pkg in MANYBODY MEAM EXTRA-FIX EXTRA-COMPUTE RIGID TALLY ML-SNAP ML-IAP; do
  grep -q -- "$pkg" <<< "$(lmp_help)" || die "package $pkg missing from the LAMMPS build"
done
say "LAMMPS ok: $(lmp_help | grep -A2 'Installed packages' | tail -1 | xargs)"

# ---- 3. Python -----------------------------------------------------------------
if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  say "Python venv"
  python3 -m venv ${VENV_FLAGS:-} "$REPO/.venv"
fi
PY="$REPO/.venv/bin/python"
"$PY" -m pip install -q ${PIP_FLAGS:-} --upgrade pip || say "pip not upgraded (continuing)"
"$PY" -m pip install -q ${PIP_FLAGS:-} -e "$REPO[plots]" || die "pip install failed"
say "Test suite (a minute or two)"
(cd "$REPO" && "$PY" -m unittest discover -s tests > "$REPO/.cloud_tests.log" 2>&1) \
  || die "tests failed; see $REPO/.cloud_tests.log"
tail -1 "$REPO/.cloud_tests.log"

# ---- 4. job set and smoke run -----------------------------------------------------
POT="$DEST/potentials"
if [[ -f "$JOBS_DIR/jobs.json" ]]; then
  say "$JOBS_DIR already has submitted jobs: leaving it alone"
  SKIP_SMOKE=1
else
say "Writing the job set into $JOBS_DIR"
"$PY" "$REPO/scripts/prepare_cloud.py" "$JOBS_DIR" --potentials "$POT" --np "$NP" --slots "$SLOTS"
if [[ "${SKIP_SMOKE:-0}" != 1 ]]; then
  say "Smoke run: every deck, shortened, run to the end here (about 10-20 minutes)"
  rm -rf "$SMOKE_DIR"
  "$PY" "$REPO/scripts/prepare_cloud.py" "$SMOKE_DIR" --potentials "$POT" --smoke \
    --np "$NP" --slots "$SLOTS" > /dev/null
  "$PY" "$REPO/scripts/run_smoke.py" "$SMOKE_DIR" --lmp "$LMP" --np "$NP" --parallel "$SLOTS" \
    || die "a smoke run failed (above; details in $SMOKE_DIR/smoke_report.json). Do not start the real runs."
  RATE="$("$PY" -c "import json;print(json.load(open('$SMOKE_DIR/smoke_report.json'))['core_rate'] or 1.0)")"
  "$PY" "$REPO/scripts/prepare_cloud.py" "$JOBS_DIR" --potentials "$POT" --np "$NP" \
    --slots "$SLOTS" --core-rate "$RATE" | tail -2
fi
fi

cat <<EOF

Ready. LAMMPS $TAG ($COMMIT; extras: ${WITH:-reused build}), $SLOTS slots x $NP ranks.

Start the runs inside tmux, so they survive a dropped SSH connection:

  tmux new -s ok
  source $REPO/.venv/bin/activate
  python3 -m openkapitza.agent
  > jobs $JOBS_DIR
  > submit all local lmp=$LMP np=$NP slots=$SLOTS
  > watch

Detach with Ctrl-b d; reattach with 'tmux attach -t ok'. The queue advances by
itself, so closing the agent does not stop it. Bring the results home with
docs/cloud-runbook.md section 5, and stop the machine when 'jobs' shows nothing
running.
EOF
