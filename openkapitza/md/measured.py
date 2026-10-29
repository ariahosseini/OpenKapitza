"""Measured interface conductance as a reference for the Kapitza term.

What this is for: putting the h_K in use (the DMM placeholder, an MD result,
or the user's number) next to what has been MEASURED for the same pair, so a
reader sees at once whether the number is plausible.

What it is not: a replacement for h_K. A measured value belongs to a real
interface -- native oxide, roughness, interdiffusion, an adhesion layer, a
particular deposition -- and is often well below an ideal-interface MD or DMM
value for the same pair (Al/Si: 115-193 MW/m^2K by TDTR on as-prepared
surfaces, Hopkins et al. 2010, vs ~350-380 for cleaner interfaces). So the
check reports the RANGE with its conditions and flags a value outside it; it
never substitutes the measurement. Using a measured value is the user's call
(`h_K_MW`), and the flag says how.

Data: `openkapitza/data/itr_measured.csv`. It is seeded with the rows this
project has read itself (tests/benchmarks/kapitza_measured.csv), plus whatever
`python -m openkapitza.md.measured import FILE` has added from the NIMS
interfacial-thermal-resistance dataset (Wu et al., Sci. Data 7, 36 (2020);
Zenodo 10.5281/zenodo.3564173; CC BY 4.0).
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..schema.flags import Code, FlagSet, Severity

DATA = Path(__file__).resolve().parents[1] / "data" / "itr_measured.csv"
COLUMNS = ("id", "material_a", "material_b", "raw_a", "raw_b", "interlayer",
           "temperature_K", "method", "preparation", "film_thickness_nm",
           "h_MW_m2K", "h_sd_MW_m2K", "dataset", "source")
T_WINDOW_K = 50.0
OUTSIDE = 1.25            # a value beyond 1.25x the measured extremes is flagged

# database key -> the name measured data uses for it, and a note if it is a proxy
CANONICAL = {"Al6061": ("Al", "Al 6061 compared with pure-Al measurements"),
             "Ni200": ("Ni", None), "SS304": ("SS304", None)}

# raw names in published tables -> canonical
_ALIASES = {
    "si": "Si", "silicon": "Si", "c-si": "Si", "ge": "Ge", "germanium": "Ge",
    "al": "Al", "aluminum": "Al", "aluminium": "Al", "cu": "Cu", "copper": "Cu",
    "ni": "Ni", "nickel": "Ni", "au": "Au", "gold": "Au", "pt": "Pt", "ti": "Ti",
    "cr": "Cr", "w": "W", "pb": "Pb", "bi": "Bi", "ag": "Ag", "mo": "Mo",
    "sio2": "SiO2", "a-sio2": "SiO2", "silica": "SiO2", "fused silica": "SiO2",
    "thermal oxide": "SiO2", "glass": "SiO2",
    "al2o3": "Al2O3", "sapphire": "Al2O3", "diamond": "C(diamond)", "c": "C(diamond)",
    "gan": "GaN", "sic": "SiC", "4h-sic": "SiC", "6h-sic": "SiC", "aln": "AlN",
    "gaas": "GaAs", "graphene": "graphene", "hbn": "hBN", "h-bn": "hBN",
}
_ORIENT = re.compile(r"\s*[\(\[]\s*-?\d{3,4}\s*[\)\]]\s*|\s*\b(single[- ]crystal|"
                     r"substrate|wafer|film|thin film|bulk|polycrystalline)\b\s*", re.I)


def canonical(name: str) -> str:
    """'Si(100)' -> 'Si', 'sapphire' -> 'Al2O3', 'Al6061' -> 'Al'; unknown kept, cleaned."""
    if name in CANONICAL:
        return CANONICAL[name][0]
    s = _ORIENT.sub(" ", name or "").strip()
    low = re.sub(r"\s+", " ", s.lower())
    return _ALIASES.get(low, _ALIASES.get(low.replace(" ", ""), s))


@dataclass
class Record:
    id: str
    a: str
    b: str
    h: float                         # MW/m^2K
    h_sd: float | None
    T: float | None
    method: str
    preparation: str
    interlayer: str
    thickness_nm: float | None
    dataset: str
    source: str

    @property
    def pair(self) -> frozenset:
        return frozenset((self.a, self.b))

    @property
    def oxide_or_layer(self) -> bool:
        """An oxide or interlayer is recorded ('none...' or blank means a clean interface)."""
        s = self.interlayer.strip().lower()
        return bool(s) and not s.startswith(("none", "no ", "-"))


def _num(s) -> float | None:
    try:
        v = float(str(s).strip())
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=4)
def load(path: str | None = None) -> tuple[Record, ...]:
    p = Path(path) if path else DATA
    if not p.is_file():
        return ()
    out = []
    with p.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(r for r in fh if not r.startswith("#")):
            h = _num(row.get("h_MW_m2K"))
            if h is None or h <= 0:
                continue
            out.append(Record(
                id=row["id"], a=canonical(row["material_a"]), b=canonical(row["material_b"]),
                h=h, h_sd=_num(row.get("h_sd_MW_m2K")), T=_num(row.get("temperature_K")),
                method=row.get("method", ""), preparation=row.get("preparation", ""),
                interlayer=row.get("interlayer", ""),
                thickness_nm=_num(row.get("film_thickness_nm")),
                dataset=row.get("dataset", ""), source=row.get("source", "")))
    return tuple(out)


@dataclass
class Reference:
    pair: tuple[str, str]
    records: list[Record]
    T: float
    widened: bool = False            # no point within the window: all temperatures used
    proxy_note: str | None = None
    n_excluded_layers: int = 0       # rows with an explicit interlayer, left out

    @property
    def values(self) -> list[float]:
        return [r.h for r in self.records]

    @property
    def lo(self) -> float:
        return min(self.values)

    @property
    def hi(self) -> float:
        return max(self.values)

    @property
    def median(self) -> float:
        return statistics.median(self.values)

    @property
    def sources(self) -> list[str]:
        return sorted({r.source for r in self.records})

    def summary(self) -> str:
        a, b = self.pair
        n, ns = len(self.records), len(self.sources)
        temps = sorted({round(r.T) for r in self.records if r.T})
        tdesc = (f"{temps[0]}-{temps[-1]} K" if len(temps) > 1 else f"{temps[0]} K") if temps \
            else "temperature not stated"
        methods = ", ".join(sorted({r.method for r in self.records if r.method})) or "method n/a"
        ox = sum(r.oxide_or_layer for r in self.records)
        cond = (f"; {ox} of {n} with an oxide or interlayer" if 0 < ox < n
                else "; all with an oxide or interlayer" if ox == n else "")
        s = (f"Measured {a}/{b}: {self.lo:.0f}-{self.hi:.0f} MW/m²K (median {self.median:.0f}; "
             f"{n} value{'s' if n > 1 else ''} from {ns} source{'s' if ns > 1 else ''}, "
             f"{methods}, {tdesc}{cond})")
        if self.widened:
            s += f"; none within {T_WINDOW_K:.0f} K of {self.T:.0f} K, so all temperatures shown"
        return s


def reference(a_key: str, b_key: str, T: float, path: str | None = None) -> Reference | None:
    """Measured values for the pair near temperature T, or None if there are none."""
    a, b = canonical(a_key), canonical(b_key)
    notes = [CANONICAL[k][1] for k in (a_key, b_key) if k in CANONICAL and CANONICAL[k][1]]
    rows = [r for r in load(path) if r.pair == frozenset((a, b))]
    if not rows:
        return None
    plain = [r for r in rows if not _explicit_layer(r)]
    excluded = len(rows) - len(plain)
    rows = plain or rows                        # all have layers: show them, they say so
    near = [r for r in rows if r.T is not None and abs(r.T - T) <= T_WINDOW_K]
    widened = not near
    return Reference(pair=(a, b), records=near or rows, T=T, widened=widened,
                     proxy_note="; ".join(notes) or None, n_excluded_layers=excluded if plain else 0)


def _explicit_layer(r: Record) -> bool:
    """A deliberate interlayer (Ti, Cr adhesion layer, SAM...), not a native oxide."""
    s = r.interlayer.strip().lower()
    return bool(s) and not s.startswith(("none", "no ", "-")) and "native" not in s


def check_kapitza(a_key: str, b_key: str, T: float, h_K_W: float, source: str,
                  flags: FlagSet, path: str | None = None) -> Reference | None:
    """Flag an h_K (W/m^2K) that sits outside the measured range for the pair."""
    ref = reference(a_key, b_key, T, path)
    if ref is None:
        return None
    v = h_K_W / 1e6
    what = {"md": "The MD h_K", "user": "Your h_K", "dmm_debye": "The DMM placeholder"}.get(
        source, f"h_K ({source})")
    tail = f" {ref.proxy_note}." if ref.proxy_note else ""
    use = (f"If your interface is prepared like the measured ones, 'set h_K_MW={ref.median:.0f}' "
           f"uses the measured median.")
    if v > ref.hi * OUTSIDE:
        sev = Severity.INFO if source == "user" else Severity.CAUTION
        flags.add(Code.KAPITZA_ABOVE_MEASURED, sev,
                  f"{what} ({v:.0f} MW/m²K) is {v / ref.hi:.1f}x the highest measured value. "
                  f"{ref.summary()}. Measured interfaces carry oxide, roughness or "
                  f"interdiffusion; an ideal interface usually conducts better, so this "
                  f"over-states h_K for a real joint by up to that factor.{tail}",
                  remedy=use, measured_lo=ref.lo, measured_hi=ref.hi, value=v)
    elif v < ref.lo / OUTSIDE:
        sev = Severity.INFO if source == "user" else Severity.WARNING
        flags.add(Code.KAPITZA_BELOW_MEASURED, sev,
                  f"{what} ({v:.0f} MW/m²K) is below every measured value "
                  f"({ref.lo / v:.1f}x under the lowest). {ref.summary()}. An ideal interface "
                  f"conducting worse than real ones points at the model or the run, not the "
                  f"joint.{tail}",
                  remedy="Check the potential and the run (convergence, temperature); " + use,
                  measured_lo=ref.lo, measured_hi=ref.hi, value=v)
    else:
        flags.add(Code.KAPITZA_REFERENCE, Severity.INFO,
                  f"{what} ({v:.0f} MW/m²K) lies within the measured range. {ref.summary()}.{tail}",
                  measured_lo=ref.lo, measured_hi=ref.hi, value=v)
    return ref


# ---- known biases of the potentials this package routes to ---------------------------
# (pair, potential substring) -> (direction, size, what, source). Only published
# comparisons that were read; a bias is reported, never corrected for.
BIASES = {
    (frozenset({"Al", "Si"}), "meam"): (
        "high", "20-30 %",
        "A classical 2NN MEAM over-estimated Al/Si conductance by 20-30 % against a "
        "DFT-trained neural-network potential that matched experiment (380 ± 33 vs ~350-379 "
        "MW/m²K at 300 K; Khot et al., J. Appl. Phys. 137, 115301 (2025)). Whether theirs is "
        "the Jelinek AlSiMgCuFe parameterisation used here is not verified.",
        "khot2025"),
}


def potential_bias(pair: frozenset, potential: str, flags: FlagSet) -> str | None:
    """Add a caution when the potential behind an MD h_K has a published bias for the pair."""
    pair = frozenset(canonical(x) for x in pair)
    for (p, pot), (direction, size, text, key) in BIASES.items():
        if p == pair and pot in (potential or "").lower():
            flags.add(Code.KAPITZA_POTENTIAL_BIAS, Severity.CAUTION, text,
                      remedy=f"Read the MD h_K as likely {size} {direction}; a Tier-2 "
                             "(machine-learned) potential trained on the interface removes it.")
            return key
    return None


# ---- importing the NIMS dataset ----------------------------------------------------------
_HEAD = {
    "interface": ("interface",), "interlayer": ("interlayer",),
    "itr": ("itr",), "temperature": ("measuring temperature", "temperature"),
    "method": ("measurement method", "measuring method", "method"),
    "film": ("film 1", "film1"), "film_prep": ("film 1 preparation", "preparation method"),
    "thickness": ("film 1 thickness", "thickness"),
    "substrate": ("substrate (film 2)", "substrate", "film 2"),
    "ref": ("id-r", "reference id", "ref"), "id": ("id",),
}


def _find(headers: list[str], key: str) -> int | None:
    low = [h.strip().lower() for h in headers]
    for cand in _HEAD[key]:
        for i, h in enumerate(low):
            if h == cand:
                return i
    for cand in _HEAD[key]:
        for i, h in enumerate(low):
            if h.startswith(cand) and not (key == "film" and "prep" in h or
                                           key == "film" and "thick" in h):
                return i
    return None


def _rows_from(path: Path) -> list[list]:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        name = next((n for n in wb.sheetnames if "itr" in n.lower() and "ref" not in n.lower()),
                    wb.sheetnames[0])
        return [[("" if c is None else c) for c in r] for r in wb[name].iter_rows(values_only=True)]
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


def import_nims(src: Path | str, out: Path | str = DATA, scale: float = 1e-9) -> dict:
    """Convert the NIMS ITR table (Wu et al. 2020) into itr_measured.csv rows.

    The paper gives ITR in units of 1e-9 m^2K/W (`scale`); h = 1/ITR. Rows already
    imported (dataset 'NIMS-ITR') are replaced, the project's own rows kept.
    """
    rows = _rows_from(Path(src))
    hi = next(i for i, r in enumerate(rows) if any("itr" in str(c).lower() for c in r))
    headers = [str(c) for c in rows[hi]]
    col = {k: _find(headers, k) for k in _HEAD}
    if col["itr"] is None or (col["film"] is None and col["interface"] is None):
        raise ValueError(f"{src}: no ITR / material columns found in {headers}")
    new, skipped = [], 0
    for r in rows[hi + 1:]:
        g = lambda k: (str(r[col[k]]).strip() if col[k] is not None and col[k] < len(r) else "")
        itr = _num(g("itr"))
        if not itr or itr <= 0:
            skipped += 1
            continue
        a, b = g("film"), g("substrate")
        if (not a or not b) and "/" in g("interface"):
            a, b = (x.strip() for x in g("interface").split("/", 1))
        if not a or not b:
            skipped += 1
            continue
        h = 1.0 / (itr * scale) / 1e6
        new.append({"id": f"NIMS-{g('id') or len(new) + 1}", "material_a": canonical(a),
                    "material_b": canonical(b), "raw_a": a, "raw_b": b,
                    "interlayer": g("interlayer"), "temperature_K": g("temperature"),
                    "method": g("method"), "preparation": g("film_prep"),
                    "film_thickness_nm": g("thickness"), "h_MW_m2K": f"{h:.4g}",
                    "h_sd_MW_m2K": "", "dataset": "NIMS-ITR",
                    "source": f"Wu et al., Sci. Data 7, 36 (2020), Zenodo 10.5281/zenodo.3564173; "
                              f"original reference {g('ref')}"})
    out = Path(out)
    keep = [r for r in _read_raw(out) if r.get("dataset") != "NIMS-ITR"]
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write(_HEADER_COMMENT)
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for row in keep + new:
            w.writerow({k: row.get(k, "") for k in COLUMNS})
    load.cache_clear()
    return {"imported": len(new), "skipped": skipped, "kept": len(keep), "path": str(out),
            "pairs": len({frozenset((x["material_a"], x["material_b"])) for x in new})}


_HEADER_COMMENT = (
    "# Measured interface conductance, h = 1/ITR, MW/m^2K. Reference only: never substituted\n"
    "# for h_K (openkapitza/md/measured.py). dataset 'project': rows read by this project\n"
    "# (tests/benchmarks/SOURCES.md). dataset 'NIMS-ITR': Wu, Zhan, Hou, Fang & Xu,\n"
    "# 'Physical and chemical descriptors for predicting interfacial thermal resistance',\n"
    "# Sci. Data 7, 36 (2020), doi:10.1038/s41597-020-0373-2; data Zenodo\n"
    "# 10.5281/zenodo.3564173, CC BY 4.0 (attribution as above; converted, not altered).\n")


def _read_raw(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(r for r in fh if not r.startswith("#")))


def main(argv: list[str] | None = None) -> int:
    """python -m openkapitza.md.measured import FILE | show A B [T]"""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m openkapitza.md.measured", description=main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("import", help="add the NIMS ITR table (xlsx or csv)")
    i.add_argument("file")
    i.add_argument("--scale", type=float, default=1e-9,
                   help="ITR unit in m^2K/W (the paper's table: 1e-9)")
    s = sub.add_parser("show", help="measured values for a pair")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("T", nargs="?", type=float, default=300.0)
    a = ap.parse_args(argv)
    if a.cmd == "import":
        print(import_nims(a.file, scale=a.scale))
    else:
        ref = reference(a.a, a.b, a.T)
        print(ref.summary() if ref else f"No measured values for {a.a}/{a.b}.")
        for r in (ref.records if ref else []):
            print(f"  {r.h:8.1f}  {r.T or '':>5}  {r.method:8} {r.preparation[:50]:50} {r.source[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
