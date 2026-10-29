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
           "temperature_K", "method", "preparation", "condition", "modified",
           "film_thickness_nm", "h_MW_m2K", "h_sd_MW_m2K", "dataset", "source")
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
    condition: str = ""
    modified: bool = False          # deliberately damaged (irradiated, implanted)

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
                dataset=row.get("dataset", ""), source=row.get("source", ""),
                condition=row.get("condition") or "",
                modified=(row.get("modified") or "").strip().lower() == "yes"))
    return tuple(out)


@dataclass
class Reference:
    pair: tuple[str, str]
    records: list[Record]
    T: float
    widened: bool = False            # no point within the window: all temperatures used
    proxy_note: str | None = None
    n_excluded_layers: int = 0       # rows with an explicit interlayer, left out
    n_modified: int = 0              # deliberately damaged samples, left out

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
        """Distinct publications (a dataset row number is not a source)."""
        return sorted({r.source.split("; via ")[0] for r in self.records})

    def summary(self) -> str:
        if getattr(self, "_summary", None) is None:     # cached: Reference objects are
            self._summary = self._make_summary()        # shared through the lookup cache
        return self._summary

    def _make_summary(self) -> str:
        a, b = self.pair
        n, ns = len(self.records), len(self.sources)
        temps = sorted({round(r.T) for r in self.records if r.T})
        tdesc = (f"{temps[0]}-{temps[-1]} K" if len(temps) > 1 else f"{temps[0]} K") if temps \
            else "temperature not stated"
        methods = ", ".join(sorted({r.method for r in self.records if r.method})) or "method n/a"
        oxd = [r.h for r in self.records if r.oxide_or_layer]
        cln = [r.h for r in self.records if not r.oxide_or_layer]
        if oxd and cln:
            cond = (f"; clean {min(cln):.0f}-{max(cln):.0f} (n={len(cln)}), with an oxide "
                    f"{min(oxd):.0f}-{max(oxd):.0f} (n={len(oxd)})")
        else:
            cond = "; all with an oxide or interlayer" if oxd else ""
        left = [f"{self.n_excluded_layers} with other interlayers" if self.n_excluded_layers else "",
                f"{self.n_modified} damaged samples" if self.n_modified else ""]
        left = [x for x in left if x]
        if left:
            cond += f"; left out: {', '.join(left)}"
        s = (f"Measured {a}/{b}: {self.lo:.0f}-{self.hi:.0f} MW/m²K (median {self.median:.0f}; "
             f"{n} value{'s' if n > 1 else ''} from {ns} source{'s' if ns > 1 else ''}, "
             f"{methods}, {tdesc}{cond})")
        if self.widened:
            s += f"; none within {T_WINDOW_K:.0f} K of {self.T:.0f} K, so all temperatures shown"
        return s


def reference(a_key: str, b_key: str, T: float, path: str | None = None) -> Reference | None:
    """Measured values for the pair near temperature T, or None if there are none.

    Cached (the Monte Carlo calls this once per sample); `load.cache_clear()` and
    `reference.cache_clear()` after changing the table in-process.
    """
    # keyed on the loaded table's identity: reloading (load.cache_clear) gives a new one
    return _reference(a_key, b_key, round(float(T), 1), path, id(load(path)))


reference.cache_clear = lambda: _reference.cache_clear()     # type: ignore[attr-defined]


@lru_cache(maxsize=256)
def _reference(a_key: str, b_key: str, T: float, path: str | None,
               _table_id: int) -> Reference | None:
    a, b = canonical(a_key), canonical(b_key)
    _table = load(path)
    notes = [CANONICAL[k][1] for k in (a_key, b_key) if k in CANONICAL and CANONICAL[k][1]]
    rows = [r for r in _table if r.pair == frozenset((a, b))]
    if not rows:
        return None
    n_mod = sum(r.modified for r in rows)
    rows = [r for r in rows if not r.modified] or rows
    plain = [r for r in rows if not _explicit_layer(r)]
    excluded = len(rows) - len(plain)
    rows = plain or rows                        # all have layers: show them, they say so
    near = [r for r in rows if r.T is not None and abs(r.T - T) <= T_WINDOW_K]
    widened = not near
    return Reference(pair=(a, b), records=near or rows, T=T, widened=widened,
                     proxy_note="; ".join(notes) or None, n_excluded_layers=excluded if plain else 0,
                     n_modified=n_mod)


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
# The released ITR_dataset.xlsx (sheet "ITR"): id | interface id | Interface | interlayer
# (0/1) | Thermal Boundary Conductance (MW/m2K) | itr (10-9 m2K/W) | Temperature (K) |
# Measurement method | Film 1 | Film preparation method | Film thickness (nm) | substrate
# (Film2) | Substrate details | Substrate pretreatment | Interfacial properties |
# Reference (id-R); sheet "ITR References": id-R | Title | Authors | Journal | DOI.
_HEAD = {
    "id": ("id",), "interface": ("interface",), "interlayer": ("interlayer",),
    "tbc": ("thermal boundary conductance",), "itr": ("itr",),
    "temperature": ("temperature", "measuring temperature"),
    "method": ("measurement method", "measuring method"),
    "film": ("film 1",), "film_prep": ("film preparation method", "film 1 preparation"),
    "thickness": ("film thickness", "film 1 thickness"),
    "substrate": ("substrate (film2)", "substrate (film 2)", "substrate"),
    "pretreat": ("substrate pretreatment",), "props": ("interfacial properties",),
    "ref": ("reference (id-r)", "id-r", "reference"),
}
_DAMAGE = re.compile(r"\d[\d.*eE+]*\s*protons|(?<!non-)\birradiated\b|ion dose|implanted to",
                     re.I)


def _find(headers: list[str], key: str) -> int | None:
    low = [re.sub(r"\s+", " ", str(h)).strip().lower() for h in headers]
    for cand in _HEAD[key]:                      # exact, then prefix
        if cand in low:
            return low.index(cand)
    for cand in _HEAD[key]:
        for i, h in enumerate(low):
            if h.startswith(cand):
                return i
    return None


def _sheet(path: Path, want: str) -> list[list]:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        names = wb.sheetnames
        if want == "refs":
            name = next((n for n in names if "ref" in n.lower()), None)
            if name is None:
                return []
        else:
            name = next((n for n in names if "itr" in n.lower() and "ref" not in n.lower()),
                        names[0])
        return [[("" if c is None else c) for c in r] for r in wb[name].iter_rows(values_only=True)]
    if want == "refs":
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


def _references(rows: list[list]) -> dict[str, str]:
    """id-R -> 'First author et al., Journal (DOI)'."""
    if not rows:
        return {}
    h = [str(c).strip().lower() for c in rows[0]]
    get = lambda r, k: str(r[h.index(k)]).strip() if k in h and h.index(k) < len(r) else ""
    out = {}
    for r in rows[1:]:
        rid = get(r, "id-r")
        if not rid:
            continue
        auth = re.split(r",| and ", get(r, "authors"))[0].strip()
        surname = auth.split()[-1] if auth else "?"
        doi = re.sub(r"\s+", "", get(r, "doi"))
        out[_key(rid)] = f"{surname} et al., {re.sub(chr(10), ' ', get(r, 'journal')).strip()}" + \
            (f" ({doi})" if doi else "")
    return out


def _key(x) -> str:
    v = _num(x)
    return str(int(v)) if v is not None and v == int(v) else str(x).strip()


def _cite_tokens(text: str) -> set[str]:
    """Article numbers / DOI fragments ('085307', '10.1103/physrevb.82.085307' -> {'085307'})."""
    return set(re.findall(r"(?<![\d.])\d{5,7}(?![\d])", text or ""))


def _is_oxide_of(layer: str, sides: tuple[str, str]) -> bool:
    """SiO2 on Si, AlOx on Al: a native or grown oxide of a neighbour."""
    m = re.match(r"(?:a-)?([A-Z][a-z]?)O", layer)
    return bool(m) and any(m.group(1) == re.match(r"[A-Z][a-z]?", s).group(0)
                           for s in sides if re.match(r"[A-Z][a-z]?", s))


def import_nims(src: Path | str, out: Path | str = DATA) -> dict:
    """Convert the NIMS ITR table (Wu et al. 2020) into itr_measured.csv rows.

    Materials come from the `Interface` string (film/.../substrate; middle parts
    are interlayers). An oxide of a neighbour is recorded as an oxide (kept);
    any other interlayer is recorded and left out of the reference range.
    Deliberately damaged samples are marked `modified`. Rows that repeat one of
    the project's own (same pair, h within 0.5 %, T within 5 K) are dropped.
    Re-importing replaces earlier NIMS rows.
    """
    src = Path(src)
    rows = _sheet(src, "itr")
    hi = next(i for i, r in enumerate(rows)
              if any(str(c).strip().lower().startswith(("itr", "thermal boundary")) for c in r))
    headers = [str(c) for c in rows[hi]]
    col = {k: _find(headers, k) for k in _HEAD}
    if (col["tbc"] is None and col["itr"] is None) or col["interface"] is None:
        raise ValueError(f"{src}: no conductance / ITR or Interface column in {headers}")
    refs = _references(_sheet(src, "refs"))
    out = Path(out)
    own = [r for r in _read_raw(out) if r.get("dataset") != "NIMS-ITR"]
    own_keys = [(frozenset((canonical(r["material_a"]), canonical(r["material_b"]))),
                 _num(r["h_MW_m2K"]), _num(r["temperature_K"]), _cite_tokens(r["source"]))
                for r in own]
    new, skipped, dupes = [], 0, 0
    for r in rows[hi + 1:]:
        g = lambda k: (re.sub(r"\s+", " ", str(r[col[k]])).strip()
                       if col[k] is not None and col[k] < len(r) else "")
        h = _num(g("tbc")) if col["tbc"] is not None else None
        if h is None:
            itr = _num(g("itr"))
            h = 1e3 / itr if itr and itr > 0 else None      # 1e-9 m2K/W -> MW/m2K
        parts = [x.strip() for x in g("interface").split("/") if x.strip()]
        if not h or h <= 0 or len(parts) < 2:
            skipped += 1
            continue
        a, b = canonical(parts[0]), canonical(parts[-1])
        mids = [canonical(x) for x in parts[1:-1]]
        layer = "/".join(mids)
        if mids and all(_is_oxide_of(m, (a, b)) for m in mids):
            layer += " (native or grown oxide)"
        T = _num(g("temperature"))
        pair = frozenset((a, b))
        rid = _key(g("ref"))
        cite = _cite_tokens(refs.get(rid, ""))
        if any(p == pair and hv and abs(h / hv - 1) < 0.005 and ct & cite and
               (T is None or tv is None or abs(T - tv) <= 5) for p, hv, tv, ct in own_keys):
            dupes += 1                       # the same measurement from the same paper
            continue
        cond = g("props")
        new.append({"id": f"NIMS-{_key(g('id')) or len(new) + 1}", "material_a": a,
                    "material_b": b, "raw_a": parts[0], "raw_b": parts[-1],
                    "interlayer": layer, "temperature_K": g("temperature"),
                    "method": g("method"),
                    "preparation": "; ".join(x for x in (g("film_prep"), g("pretreat"))
                                             if x and x.lower() != "none")[:200],
                    "condition": "" if cond.lower() == "none" else cond[:120],
                    "modified": "yes" if _DAMAGE.search(cond) else "",
                    "film_thickness_nm": g("thickness"), "h_MW_m2K": f"{h:.4g}",
                    "h_sd_MW_m2K": "", "dataset": "NIMS-ITR",
                    "source": f"{refs.get(rid, 'reference ' + rid)}; via Wu et al., Sci. Data 7, "
                              f"36 (2020), NIMS ITR dataset row {_key(g('id'))}"})
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write(_HEADER_COMMENT)
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for row in own + new:
            w.writerow({k: row.get(k, "") or "" for k in COLUMNS})
    load.cache_clear()
    return {"imported": len(new), "skipped": skipped, "duplicates_of_project_rows": dupes,
            "kept": len(own), "path": str(out),
            "pairs": len({frozenset((x["material_a"], x["material_b"])) for x in new}),
            "with_other_interlayers": sum(bool(x["interlayer"]) and "oxide" not in x["interlayer"]
                                          for x in new),
            "modified": sum(x["modified"] == "yes" for x in new)}


_HEADER_COMMENT = (
    "# Measured interface conductance, h = 1/ITR, MW/m^2K. Reference only: never substituted\n"
    "# for h_K (openkapitza/md/measured.py). dataset 'project': rows read by this project\n"
    "# (tests/benchmarks/SOURCES.md). dataset 'NIMS-ITR': Wu, Zhan, Hou, Fang & Xu,\n"
    "# 'Physical and chemical descriptors for predicting interfacial thermal resistance',\n"
    "# Sci. Data 7, 36 (2020), doi:10.1038/s41597-020-0373-2; data Zenodo\n"
    "# 10.5281/zenodo.3564173, CC BY 4.0. Changes: conductance taken from its TBC column;\n"
    "# materials and interlayers parsed from the Interface string and names normalised;\n"
    "# damaged samples marked; rows repeating a 'project' row from the same paper dropped.\n")


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
    i.add_argument("file", help="ITR_dataset.xlsx from Zenodo 10.5281/zenodo.3564173")
    s = sub.add_parser("show", help="measured values for a pair")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("T", nargs="?", type=float, default=300.0)
    a = ap.parse_args(argv)
    if a.cmd == "import":
        print(import_nims(a.file))
    else:
        ref = reference(a.a, a.b, a.T)
        print(ref.summary() if ref else f"No measured values for {a.a}/{a.b}.")
        for r in (ref.records if ref else []):
            print(f"  {r.h:8.1f}  {r.T or '':>5}  {r.method:8} {r.preparation[:50]:50} {r.source[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
