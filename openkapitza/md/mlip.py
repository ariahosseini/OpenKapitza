"""Tier-2 machine-learned potentials in the NEMD decks (Phase 4, P4-1; KB 36).

The agent instruction (section 4) tiers the potentials: Tier 1 classical
(Tersoff, SW, EAM, MEAM; `nemd.TIER1_POTENTIALS`), Tier 2 machine-learned.
This module is the Tier-2 plumbing. It does NOT supply a model: none of the
potentials shipped with LAMMPS covers an interface we study (the Si and Ge
SNAP fits of Zuo et al. 2020 are single-element), and a model trained here
without reference data would be the fabricated potential nemd.py warns
against. What it does:

- describes a model (`MLIPModel`, or a small YAML file): kind, files, the
  elements in type order, what it was trained on, and where it can run;
- writes its LAMMPS pair section, which goes into `NEMDSpec.pair_block`
  unchanged, so every deck, check and measurement works as for Tier 1;
- checks it against the instruction's rules before a run (`check_model`):
    * the LAMMPS packages it needs and whether it needs a CUDA GPU (MACE,
      NequIP/Allegro and DeePMD are GPU codes in practice; an Apple GPU
      cannot run them inside LAMMPS);
    * training coverage: a model whose training set has no configurations of
      THIS interface is extrapolating at the one place h_K is decided --
      said outright, as the instruction requires;
    * foundation (universal) models: positioned for screening, known to
      soften at high temperature (so a melt-quench with one needs a melting
      check);
    * the gate: phonon DOS, elastic constants and bulk k against a reference
      before any production run. Results are recorded in `gate`; a model
      without them is flagged, and production is not to be quoted.

Pair sections (LAMMPS stable_30Sep2026; check each project's current
instructions when building, they change faster than LAMMPS):
  snap    pair_style snap                     pair_coeff * * X.snapcoeff X.snapparam El...
  pace    pair_style pace                     pair_coeff * * model.yace El...
  mace    pair_style mliap unified M.pt 0     pair_coeff * * El...      (ML-IAP + KOKKOS)
  deepmd  pair_style deepmd graph.pb          pair_coeff * * El...      (DeePMD plugin)
  nequip  pair_style nequip                   pair_coeff * * model.pth El...
  allegro pair_style allegro                  pair_coeff * * model.pth El...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..schema.flags import Code, FlagSet, Severity

KINDS = {
    #          (packages / plugin,                       needs a CUDA GPU in practice)
    "snap": (("ML-SNAP",), False),
    "pace": (("ML-PACE",), False),
    "mace": (("ML-IAP", "PYTHON", "KOKKOS (CUDA)"), True),
    "deepmd": (("DeePMD-kit LAMMPS plugin",), True),
    "nequip": (("pair_nequip_allegro (libtorch)",), True),
    "allegro": (("pair_nequip_allegro (libtorch)",), True),
}
GATE_ITEMS = ("phonon_dos", "elastic", "bulk_k")


@dataclass
class MLIPModel:
    kind: str                                  # one of KINDS
    files: tuple[str, ...]                     # model file(s), in pair_coeff order
    elements: tuple[str, ...]                  # one per atom type, in type order
    trained_on: tuple[str, ...] = ()           # e.g. ("bulk Si", "bulk Ge", "Si/Ge interface")
    foundation: bool = False                   # a universal / foundation model
    source: str = ""                           # citation or URL
    gate: dict = field(default_factory=dict)   # {"phonon_dos": "pass", ...} once run

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown MLIP kind {self.kind!r}; one of {sorted(KINDS)}")
        n = {"snap": 2}.get(self.kind, 1)
        if len(self.files) != n:
            raise ValueError(f"{self.kind} takes {n} model file(s), got {len(self.files)}")

    @property
    def packages(self) -> tuple[str, ...]:
        return KINDS[self.kind][0]

    @property
    def needs_gpu(self) -> bool:
        return KINDS[self.kind][1]


def pair_block(model: MLIPModel) -> str:
    """The LAMMPS pair section for this model (goes into NEMDSpec.pair_block)."""
    els = " ".join(model.elements)
    f = model.files
    if model.kind == "snap":
        style, coeff = "snap", f"* * {f[0]} {f[1]} {els}"
    elif model.kind == "pace":
        style, coeff = "pace", f"* * {f[0]} {els}"
    elif model.kind == "mace":
        style, coeff = f"mliap unified {f[0]} 0", f"* * {els}"
    elif model.kind == "deepmd":
        style, coeff = f"deepmd {f[0]}", f"* * {els}"
    else:                                          # nequip, allegro
        style, coeff = model.kind, f"* * {f[0]} {els}"
    return f"pair_style      {style}\npair_coeff      {coeff}\n"


def load_model(path: Path | str) -> MLIPModel:
    """An MLIPModel from a YAML description (keys as the dataclass fields)."""
    import yaml
    d = yaml.safe_load(Path(path).read_text())
    for k in ("files", "elements", "trained_on"):
        if k in d:
            d[k] = tuple(d[k])
    return MLIPModel(**d)


def _covers(model: MLIPModel, el_a: str, el_b: str) -> bool:
    """True if the training set names an interface of this pair."""
    want = {el_a.lower(), el_b.lower()}
    for item in model.trained_on:
        s = item.lower().replace("-", "/")
        if "interface" in s and all(e in s for e in want):
            return True
    return False


def check_model(model: MLIPModel, el_a: str, el_b: str, flags: FlagSet, *,
                backend: str = "local", has_gpu: bool = False) -> None:
    """Flag what the instruction requires before a Tier-2 run (section 4)."""
    missing = [e for e in {el_a, el_b} if e not in model.elements]
    if missing:
        flags.add(Code.MLIP_TRAINING_COVERAGE, Severity.CRITICAL,
                  f"The {model.kind} model has no {', '.join(missing)} "
                  f"(elements: {', '.join(model.elements)}).",
                  remedy="Use a model whose element set includes both sides.")
        return
    if not _covers(model, el_a, el_b):
        flags.add(Code.MLIP_TRAINING_COVERAGE, Severity.WARNING,
                  f"The {model.kind} model's training set "
                  f"({'; '.join(model.trained_on) or 'not stated'}) has no {el_a}/{el_b} "
                  f"interface configurations: at the interface, where h_K is decided, "
                  f"it is extrapolating.",
                  remedy=("Add interface configurations (several registries, strains and "
                          "temperatures; active learning) and retrain, or quote the result "
                          "as unvalidated."))
    if model.foundation:
        flags.add(Code.MLIP_FOUNDATION, Severity.CAUTION,
                  "A foundation (universal) model: positioned for screening, with an "
                  "accuracy-transferability trade-off; such models are known to soften at "
                  "high temperature, which corrupts a melt-quench.",
                  remedy="Fine-tune on the interface before quoting h_K; check melting "
                         "before building an amorphous side with it.")
    done = {k for k in GATE_ITEMS if str(model.gate.get(k, "")).lower() == "pass"}
    if done != set(GATE_ITEMS):
        flags.add(Code.MLIP_UNGATED, Severity.WARNING,
                  f"Tier-2 gate incomplete: {', '.join(sorted(set(GATE_ITEMS) - done))} "
                  f"not passed. Production results are not to be quoted before the gate.",
                  remedy="Run phonon DOS, elastic constants (scripts/"
                         "measure_potential_elastic.py) and bulk k against references, and "
                         "record them in the model's `gate`.")
    if model.needs_gpu and (backend == "local" and not has_gpu):
        flags.add(Code.MLIP_NEEDS_GPU, Severity.WARNING,
                  f"{model.kind} runs inside LAMMPS through {', '.join(model.packages)}; "
                  f"in practice that needs a CUDA GPU, and an Apple GPU cannot run it.",
                  remedy="Submit to a GPU cluster (docs/cluster-runbook.md section 0b), or "
                         "use a CPU model (SNAP, ACE) for small tests.")


# Crystals the gate builds for each element (lattice, a0 guess in A, mass).
GATE_CRYSTALS = {"Si": ("diamond", 5.431, 28.0855), "Ge": ("diamond", 5.658, 72.63),
                 "Al": ("fcc", 4.0495, 26.9815), "Cu": ("fcc", 3.6149, 63.546)}


def cubic_spec_for(model: MLIPModel, element: str):
    """The single-element crystal the gate runs this model on (md/elastic.CubicSpec)."""
    from .elastic import CubicSpec
    if element not in GATE_CRYSTALS:
        raise ValueError(f"no gate crystal for {element!r}; one of {sorted(GATE_CRYSTALS)}")
    lattice, a0, mass = GATE_CRYSTALS[element]
    f = model.files
    style, args = {
        "snap": ("snap", f"{f[0]} {f[-1]} {element}"),
        "pace": ("pace", f"{f[0]} {element}"),
        "mace": (f"mliap unified {f[0]} 0", element),
        "deepmd": (f"deepmd {f[0]}", element),
        "nequip": ("nequip", f"{f[0]} {element}"),
        "allegro": ("allegro", f"{f[0]} {element}"),
    }[model.kind]
    return CubicSpec(f"{element}({model.kind})", lattice, a0, mass, style, f[0], element,
                     element, extra_files=tuple(f[1:]), coeff_args=args)

