"""The potential gate in one command: elastic constants, phonons, bulk k (KB 40).

    # a Tier-1 set
    python3 scripts/gate_potential.py --lmp LMP --potentials DIR --set sicge
    # a Tier-2 model, per element, writing the verdicts into its YAML
    python3 scripts/gate_potential.py --lmp LMP --model m.yaml --model-dir DIR \
        --elements Si Ge [--write]

The agent instruction (section 4) gates every potential before production:
elastic constants (md/elastic.py, against single-crystal references within
GATE_TOLERANCE), phonons (md/phonon_dos.py: optical peak within 5 %, TA
within 10 %), and bulk thermal conductivity (md/conductivity.py; hours, so
only with --with-k, and then only reported: classical MD over-populates the
high modes below the Debye temperature, so k has no clean pass line).
With --write a model's `gate:` block is set to pass/fail per check (pass
only if every element passes); `gate_detail:` keeps the numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from openkapitza.md.elastic import gate as elastic_gate, run_elastic    # noqa: E402
from openkapitza.md.phonon_dos import (VDOSSpec, gate as phonon_gate,   # noqa: E402
                                       peaks, run_vdos)
from openkapitza.schema import FlagSet                                   # noqa: E402


def gate_crystal(lmp, crystal, potdir, ref: str, *, quick: bool, with_k: bool) -> dict:
    out = {"crystal": crystal.name}
    el = run_elastic(Path(lmp), crystal, Path(potdir), FlagSet())
    g = elastic_gate(el, ref)
    out["elastic"] = {"pass": g.passes, "a0_A": round(el.a0, 4),
                      **{f"err_{k}": round(v, 4) for k, v in g.errors.items()}}
    spec = VDOSSpec(crystal)
    if quick:
        crystal.cells = 3
        spec = VDOSSpec(crystal, equil_steps=2000, prod_steps=6000)
    f, gdos = run_vdos(lmp, spec, potdir)
    pg = phonon_gate(peaks(f, gdos), ref)
    out["phonon_dos"] = {"pass": pg.passes, **{k: round(v, 3) for k, v in pg.peaks.items()},
                         **{f"err_{k}": round(v, 4) for k, v in pg.errors.items()}}
    if with_k:
        from openkapitza.md.conductivity import KappaSpec, run_conductivity
        r = run_conductivity(Path(lmp), KappaSpec(crystal), Path(potdir), FlagSet())
        out["bulk_k"] = {"pass": None, "k_inf_W_mK": round(r.k_inf, 1)}
    return out


def report(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        e, p = r["elastic"], r["phonon_dos"]
        lines.append(
            f"{r['crystal']:14} elastic {'PASS' if e['pass'] else 'FAIL'} "
            f"(a0 {e['err_a0']:+.1%}, C11 {e['err_c11']:+.1%}, C12 {e['err_c12']:+.1%}, "
            f"C44 {e['err_c44']:+.1%})   "
            f"phonons {'PASS' if p['pass'] else 'FAIL'} (optical {p['optical']:.2f} THz "
            f"{p['err_optical']:+.1%}, TA {p['ta']:.2f} {p['err_ta']:+.1%})"
            + (f"   k {r['bulk_k']['k_inf_W_mK']} W/mK (reported)" if "bulk_k" in r else ""))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmp", required=True)
    ap.add_argument("--potentials", type=Path, help="Tier-1: LAMMPS potentials folder")
    ap.add_argument("--set", help="Tier-1 set (as in measure_potential_elastic.py)")
    ap.add_argument("--model", type=Path, help="Tier-2: model YAML (md/mlip.py)")
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--elements", nargs="*", default=[])
    ap.add_argument("--with-k", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smaller, shorter phonon run")
    ap.add_argument("--write", action="store_true", help="write verdicts into --model")
    a = ap.parse_args(argv)
    rows = []
    if a.set:
        from measure_potential_elastic import SETS
        for c in SETS[a.set]:
            rows.append(gate_crystal(a.lmp, c, a.potentials, c.reference, quick=a.quick,
                                     with_k=a.with_k))
    if a.model:
        from openkapitza.md.mlip import cubic_spec_for, load_model
        model = load_model(a.model)
        for el in a.elements or sorted(set(model.elements)):
            rows.append(gate_crystal(a.lmp, cubic_spec_for(model, el), a.model_dir, el,
                                     quick=a.quick, with_k=a.with_k))
        if a.write:
            import yaml
            d = yaml.safe_load(a.model.read_text())
            d["gate"] = {
                "phonon_dos": "pass" if all(r["phonon_dos"]["pass"] for r in rows) else "fail",
                "elastic": "pass" if all(r["elastic"]["pass"] for r in rows) else "fail",
                "bulk_k": (d.get("gate") or {}).get("bulk_k", "") if not a.with_k
                else "reported",
            }
            d["gate_detail"] = rows
            a.model.write_text(yaml.safe_dump(d, sort_keys=False))
            print(f"gate written into {a.model}")
    print(report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
