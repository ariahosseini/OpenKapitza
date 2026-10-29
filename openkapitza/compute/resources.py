"""Estimating what a run will cost before submitting it.

This feeds the gate that Phase 0 established: the continuum model already
says how much h_K could possibly move the joint conductance, and this says
what finding out would cost. Those two numbers together are what makes an
NEMD campaign a decision rather than a reflex.

The throughput figures are coarse, order-of-magnitude rates for timesteps per
core-second. They vary by an order of magnitude with build, hardware, cutoff
and neighbour-list settings, so the estimate is a planning aid, never a
promise -- calibrate against a short run on the target machine before relying
on it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..schema.flags import Code, FlagSet, Severity
from .runner import Resources

# Atom-timesteps per core-second. Rough, build dependent, calibrate locally.
# Atom-steps per core-second. The three-body figures are MEASURED, and were
# 3-4x lower than first assumed (sw 2.0e6, tersoff 1.5e6). That gap is larger
# than the walltime safety factor, so jobs sized on the old figures would have
# been killed at the wall with nothing saved:
#   tersoff  4.1e5  Si/Ge 4,787 atoms, macOS arm64, 30 Sep 2026
#            5.6e5  Si/Ge 4,960 atoms, cloud x86, NPT, 2 jobs on 2 cores, 22 Oct
#   sw       5.0e5  SW-Si 2,048 atoms, cloud x86, NVE + Langevin, 22 Oct
#   meam     9.3e4  Si/Al (Jelinek MEAM) 6,680 atoms, cloud x86, NPT/NVT, 22 Oct
# Re-measure on the cluster with one short run before the campaign.
THROUGHPUT = {
    "sw": 5.0e5,          # Stillinger-Weber, 3-body (measured, see above)
    "tersoff": 4.5e5,     # Tersoff, 3-body (measured, see above)
    "meam": 9.3e4,        # MEAM (measured, see above): ~5x slower than Tersoff
    "eam": 5.0e6,         # EAM, pair-ish
    "lj": 2.0e7,          # Lennard-Jones
    "mlip": 2.0e3,        # machine-learned: ~3 orders slower, often GPU-bound
}

# Beyond this, warn before anyone submits.
HIGH_COST_NODE_HOURS = 500.0
# A local machine has no business running more than this.
LOCAL_MAX_NODE_HOURS = 8.0


@dataclass
class CostEstimate:
    n_atoms: int
    n_timesteps: int
    potential: str
    core_seconds: float
    node_hours: float
    wall_hours: float
    cores_used: int

    def summary(self) -> str:
        return (
            f"{self.n_atoms:,} atoms x {self.n_timesteps:,} steps on "
            f"{self.potential} ~ {self.node_hours:,.1f} node-hours "
            f"({self.wall_hours:,.1f} h wall on {self.cores_used} cores)"
        )


def estimate_cost(
    n_atoms: int,
    n_timesteps: int,
    potential: str,
    *,
    cores: int = 64,
    cores_per_node: int = 64,
    parallel_efficiency: float = 0.7,
    flags: FlagSet | None = None,
) -> CostEstimate:
    """Estimate node-hours for one NEMD run."""
    key = potential.lower()
    rate = THROUGHPUT.get(key)
    if rate is None:
        rate = THROUGHPUT["sw"]
        if flags is not None:
            flags.add(
                Code.COMPUTE_COST_HIGH,
                Severity.INFO,
                f"No throughput figure for potential '{potential}'; assuming a "
                f"three-body classical rate.",
                remedy="Calibrate with a short run on the target machine.",
            )

    core_seconds = n_atoms * n_timesteps / rate
    wall_seconds = core_seconds / max(1.0, cores * parallel_efficiency)
    nodes = max(1, -(-cores // cores_per_node))
    node_hours = nodes * wall_seconds / 3600.0

    est = CostEstimate(
        n_atoms=n_atoms,
        n_timesteps=n_timesteps,
        potential=key,
        core_seconds=core_seconds,
        node_hours=node_hours,
        wall_hours=wall_seconds / 3600.0,
        cores_used=cores,
    )

    if flags is not None and node_hours > HIGH_COST_NODE_HOURS:
        flags.add(
            Code.COMPUTE_COST_HIGH,
            Severity.WARNING,
            f"Estimated {node_hours:,.0f} node-hours for this run.",
            remedy=(
                "Check the continuum sensitivity result first: if h_K cannot "
                "move the joint conductance, this buys nothing. Otherwise "
                "consider a smaller cross-section, a Tier-1 potential for "
                "screening, or fewer convergence points."
            ),
            node_hours=node_hours,
        )
    return est


def check_backend_fit(
    est: CostEstimate, backend: str, flags: FlagSet
) -> None:
    """Warn when a run is being sent somewhere it does not belong."""
    if backend == "local" and est.node_hours > LOCAL_MAX_NODE_HOURS:
        flags.add(
            Code.COMPUTE_BACKEND_UNAVAILABLE,
            Severity.WARNING,
            f"This run is estimated at {est.node_hours:,.1f} node-hours but is "
            f"queued on the local backend.",
            remedy=(
                "Local execution is for screening and pipeline checks. Submit "
                "to a cluster, or cut the system size or run length."
            ),
            node_hours=est.node_hours,
        )
    if est.potential == "mlip" and backend == "local":
        flags.add(
            Code.COMPUTE_COST_HIGH,
            Severity.WARNING,
            "A machine-learned potential is roughly three orders of magnitude "
            "slower than a classical one and is normally GPU-bound.",
            remedy=(
                "Use a Tier-1 potential for screening and reserve the MLIP "
                "for a final run on a GPU partition."
            ),
        )


def resources_for(
    est: CostEstimate,
    *,
    cores_per_node: int = 64,
    safety_factor: float = 2.0,
    partition: str | None = None,
    account: str | None = None,
) -> Resources:
    """Turn an estimate into a SLURM request.

    The walltime carries a safety factor because the throughput figures are
    coarse and a job killed at the wall loses everything it had done.
    """
    nodes = max(1, -(-est.cores_used // cores_per_node))
    return Resources(
        nodes=nodes,
        tasks=min(est.cores_used, cores_per_node),
        cpus_per_task=1,
        walltime_hours=max(0.5, est.wall_hours * safety_factor),
        partition=partition,
        account=account,
    )


# ---- measured rates for THIS machine (scripts/benchmark_local.py) -------------------
DEFAULT_CALIBRATION = Path(os.environ.get(
    "OPENKAPITZA_THROUGHPUT", Path.home() / ".config" / "openkapitza" / "throughput.json"))


def load_calibration(path: Path | str | None = None) -> dict | None:
    """The benchmark's measured rates, or None if this machine was never measured."""
    p = Path(path) if path else DEFAULT_CALIBRATION
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def local_estimate(n_atoms: int, n_timesteps: int, potential: str,
                   calibration: dict | None = None) -> dict | None:
    """Hours for a run on the measured machine at its fastest rank count.

    Uses the measured total rate (atom-steps per second at the best `np`), so
    parallel efficiency is already in it. None when the machine or the
    potential was not measured: then fall back to `estimate_cost`.
    """
    cal = calibration if calibration is not None else load_calibration()
    if not cal:
        return None
    s = cal.get("styles", {}).get(potential.lower().split("/")[0])
    if not s:
        return None
    secs = n_atoms * n_timesteps / s["atom_steps_per_s"]
    return {"hours": secs / 3600.0, "np": s["best_np"], "machine": cal.get("processor") or
            cal.get("host", "this machine"), "measured": cal.get("date")}
