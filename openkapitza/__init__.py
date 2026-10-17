"""OpenKapitza -- multiscale thermal contact resistance.

Phase 0: the continuum core. Contact mechanics, gas-gap conductance and the
resistance network, with a DMM placeholder standing in for the atomistic
Kapitza term until the Phase-1 NEMD driver lands.

Quick start
-----------
    from openkapitza import build_joint, run_continuum, render

    joint = build_joint(
        "SS304", "SS304",
        sigma_a_um=1.2, sigma_b_um=1.2, slope_a=0.08, slope_b=0.08,
        pressure_MPa=2.0, temperature_K=300.0,
        gas="air", gas_pressure_kPa=101.325,
    )
    print(render(run_continuum(joint)))
"""

from .builder import build_joint
from .pipeline import JointResult, run_continuum
from .report import render
from .schema import Confidence, FlagSet, Provenance, Quantity, Severity

__version__ = "0.1.0"

__all__ = [
    "build_joint",
    "JointResult",
    "run_continuum",
    "render",
    "Confidence",
    "FlagSet",
    "Provenance",
    "Quantity",
    "Severity",
    "__version__",
]
