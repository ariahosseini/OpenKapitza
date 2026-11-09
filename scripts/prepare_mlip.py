"""Write a Si/Ge NEMD job that uses a Tier-2 machine-learned potential (P4-1, KB 36).

    python3 scripts/prepare_mlip.py ~/ok_jobs --model my_model.yaml --model-dir DIR \
        [--name NAME] [--backend local|slurm] [--gpu] [--smoke]

The cell is the campaign's base Si/Ge cell (r00), so the result is directly
comparable with the Tersoff value. --smoke cuts every stage to a few hundred
steps: a check that the model runs in the deck, and a timing (Colab test). The model description (YAML; see
examples/mlip/) says what the model is, what it was trained on and how far
it has been gated. Before anything is written the model is checked against
the instruction's Tier-2 rules (md/mlip.check_model) and every flag is
printed; a CRITICAL one stops the script. The model files are copied from
--model-dir into the job folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from openkapitza.md.campaign import write_spec                     # noqa: E402
from openkapitza.md.mlip import check_model, load_model, pair_block   # noqa: E402
from openkapitza.md.nemd import build_input_deck, relax_steps, validate_deck  # noqa: E402
from openkapitza.schema.flags import FlagSet, Severity             # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--backend", default="local", choices=("local", "slurm"))
    ap.add_argument("--gpu", action="store_true", help="the target machine has a CUDA GPU")
    ap.add_argument("--smoke", action="store_true", help="a few hundred steps per stage")
    a = ap.parse_args(argv)
    from plan_campaign import PRESETS, base_spec
    model = load_model(a.model)
    spec = base_spec(PRESETS["sige"])
    if a.smoke:
        from prepare_cloud import _shorten
        _shorten(spec)
    flags = FlagSet()
    check_model(model, spec.element_a, spec.element_b, flags, backend=a.backend,
                has_gpu=a.gpu)
    for f in flags.flags:
        print(f"[{Severity(int(f.severity)).name}] {f.code}: {f.message}")
    if any(int(f.severity) >= int(Severity.CRITICAL) for f in flags.flags):
        print("Stopped: a CRITICAL flag.")
        return 1
    spec.pair_block = pair_block(model)
    spec.pair_style = spec.pair_block.split()[1]
    deck = build_input_deck(spec)
    problems = validate_deck(deck)
    if problems:
        raise SystemExit(f"deck problems: {problems}")
    name = a.name or f"sige_{model.kind}"
    wd = a.root.expanduser() / "mlip" / name
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    for f in model.files:
        shutil.copy(a.model_dir.expanduser() / f, wd / f)
    (wd / "nemd.in").write_text(deck)
    write_spec(wd, spec, "nemd", mlip={"kind": model.kind, "files": list(model.files),
                                       "trained_on": list(model.trained_on),
                                       "gate": model.gate, "source": model.source},
               flags=[str(f.code) for f in flags.flags])
    steps = relax_steps(spec) + spec.equilibrate_steps + spec.steady_steps + spec.production_steps
    man = a.root.expanduser() / "manifest.json"
    data = json.loads(man.read_text()) if man.is_file() else {"jobs": []}
    data["jobs"] = [j for j in data.get("jobs", []) if j["name"] != f"mlip/{name}"] + [
        {"name": f"mlip/{name}", "atoms": int(spec.geometry.estimate_atoms(4.99e28, 4.42e28)),
         "steps": steps, "potential": "mlip"}]
    man.write_text(json.dumps(data, indent=2))
    print(f"\nwritten {wd}; in the agent: jobs {a.root}, then submit mlip/{name} "
          f"{a.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
