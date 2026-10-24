"""Turn source files into citable chunks.

A chunk is the unit of retrieval and of citation: one section of a document,
one manual-page section, one error message, one example script, one paper
note. Each carries where it came from precisely enough to open the source at
that place (path, line range, section path, version, URL).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAX_WORDS = 450            # longer sections are split at paragraph breaks
_FENCE = re.compile(r"^\s*(```|~~~)")
_MD_HEAD = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")


@dataclass
class Chunk:
    id: str
    collection: str                 # "repo" | "papers" | "lammps"
    source: str                     # repo-relative path, or lammps/doc/src/x.rst, lammps/examples/...
    title: str                      # section path, " > " separated
    text: str
    start_line: int = 1
    end_line: int = 1
    keywords: list[str] = field(default_factory=list)
    kind: str = "doc"               # doc | kb | example | error | howto | paper | curated
    version: str = ""
    url: str = ""

    def citation(self) -> str:
        loc = f"{self.source}:{self.start_line}" + (
            f"-{self.end_line}" if self.end_line > self.start_line else "")
        ver = f" [{self.version}]" if self.version else ""
        return f"{loc} — {self.title}{ver}"

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(lines: list[str], start: int) -> list[tuple[int, int, str]]:
    """(first line, last line, text) pieces of at most ~MAX_WORDS, cut at blank lines."""
    pieces, buf, buf_start, words = [], [], start, 0
    in_fence = False
    for i, ln in enumerate(lines):
        if _FENCE.match(ln):
            in_fence = not in_fence
        buf.append(ln)
        words += len(ln.split())
        if words >= MAX_WORDS and not ln.strip() and not in_fence:
            pieces.append((buf_start, start + i, "\n".join(buf).strip()))
            buf, buf_start, words = [], start + i + 1, 0
    if any(x.strip() for x in buf):
        pieces.append((buf_start, start + len(lines) - 1, "\n".join(buf).strip()))
    return pieces


def _emit(out, collection, source, path_titles, lines, start, kind, version, url, keywords=()):
    title = " > ".join(path_titles) if path_titles else Path(source).name
    pieces = _split_long(lines, start)
    for n, (a, b, text) in enumerate(pieces, 1):
        if not text:
            continue
        t = title + (f" (part {n})" if len(pieces) > 1 else "")
        out.append(Chunk(id=f"{source}:{a}", collection=collection, source=source, title=t,
                         text=text, start_line=a, end_line=b, keywords=list(keywords),
                         kind=kind, version=version, url=url))


def markdown_chunks(path: Path, source: str, collection: str = "repo", kind: str = "doc",
                    version: str = "") -> list[Chunk]:
    """Split a Markdown file at headings (levels 1-4), ignoring '#' inside code fences."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[Chunk] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    body_start = 1
    in_fence = False

    def flush():
        if any(x.strip() for x in body):
            _emit(out, collection, source, [t for _, t in stack], body, body_start, kind,
                  version, "", keywords=_code_spans(" ".join(t for _, t in stack)))

    for i, ln in enumerate(lines, 1):
        if _FENCE.match(ln):
            in_fence = not in_fence
        m = None if in_fence else _MD_HEAD.match(ln)
        if m:
            flush()
            level, text = len(m.group(1)), m.group(2).strip()
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, text)]
            body, body_start = [], i + 1
        else:
            body.append(ln)
    flush()
    return out


def _code_spans(s: str) -> list[str]:
    return re.findall(r"`([^`]+)`", s)


# ---- LAMMPS manual (.rst) ----------------------------------------------------------
_RST_UNDER = re.compile(r"^([=\-\"^~*+#])\1{2,}\s*$")
_ROLE = re.compile(r":(?:doc|ref|math|cite|file|class|numref|term):`([^`<]*?)(?:\s*<[^>]*>)?`")


def _rst_clean(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        s = ln.rstrip()
        if re.match(r"^\s*\.\. (index|_[\w\-]+|image|figure|raw|only|toctree|table-of-contents)", s):
            continue
        s = re.sub(r"^\s*\.\. (code-block|parsed-literal|math|note|warning|admonition)::.*$", "", s)
        s = _ROLE.sub(r"\1", s)
        s = re.sub(r"``([^`]+)``", r"\1", s)
        s = re.sub(r"`([^`<]+?)\s*<[^>]+>`_+", r"\1", s)
        out.append(s)
    return out


def rst_chunks(path: Path, source: str, version: str, url: str) -> list[Chunk]:
    """One chunk per section of a manual page; error pages one per message."""
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    keywords = [m.group(1).strip() for ln in raw
                if (m := re.match(r"^\.\. index::\s*(.+)$", ln))]
    if path.stem in ("Errors_messages", "Errors_warnings"):
        return _error_chunks(raw, source, version, url)
    kind = "howto" if path.stem.startswith("Howto") else "doc"
    # find headings: a text line followed by an underline at least as long
    heads = []
    for i in range(len(raw) - 1):
        if raw[i].strip() and _RST_UNDER.match(raw[i + 1]) and \
                len(raw[i + 1].strip()) >= len(raw[i].strip()) - 2 and \
                not _RST_UNDER.match(raw[i]):
            heads.append(i)
    out: list[Chunk] = []
    page = raw[heads[0]].strip() if heads else path.stem
    bounds = heads + [len(raw)]
    pre = raw[: heads[0]] if heads else raw
    if heads and any(x.strip() and not x.startswith("..") for x in pre):
        _emit(out, "lammps", source, [page], _rst_clean(pre), 1, kind, version, url, keywords)
    for j, h in enumerate(heads):
        sec = raw[h].strip()
        body = raw[h + 2: bounds[j + 1]]
        titles = [page] if sec == page else [page, sec]
        _emit(out, "lammps", source, titles, _rst_clean(body), h + 3, kind, version,
              url, keywords)
    return out


def _error_chunks(raw: list[str], source: str, version: str, url: str) -> list[Chunk]:
    out = []
    starts = [i for i, ln in enumerate(raw) if re.match(r"^\*[^*\s].*\*\s*$", ln)]
    for j, i in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else len(raw)
        msg = raw[i].strip().strip("*")
        body = [ln.strip() for ln in raw[i + 1: end] if ln.strip()]
        text = msg + "\n" + " ".join(body)
        out.append(Chunk(id=f"{source}:{i + 1}", collection="lammps", source=source,
                         title=f"{'Error' if 'messages' in source else 'Warning'}: {msg}",
                         text=text, start_line=i + 1, end_line=end, keywords=[msg],
                         kind="error", version=version, url=url))
    return out


# ---- LAMMPS example scripts -----------------------------------------------------------
_STYLED = {"fix": 3, "compute": 3, "pair_style": 1, "bond_style": 1, "angle_style": 1,
           "kspace_style": 1, "atom_style": 1, "min_style": 1, "run_style": 1,
           "dump": 3, "region": 2, "fix_modify": 0}


def example_chunks(path: Path, source: str, version: str) -> list[Chunk]:
    """An input script (or an example README): commands and styles as keywords."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    keys: list[str] = []
    for ln in lines:
        toks = ln.split("#", 1)[0].split()
        if not toks:
            continue
        cmd = toks[0]
        keys.append(cmd)
        k = _STYLED.get(cmd)
        if k and len(toks) > k:
            keys.append(f"{cmd} {toks[k]}")
    keys = sorted(set(keys))
    out: list[Chunk] = []
    title = source.split("lammps/", 1)[-1]
    _emit(out, "lammps", source, [title], lines, 1, "example", version, "", keys)
    return out


# ---- our paper notes ---------------------------------------------------------------
def paper_chunks(path: Path, source: str) -> list[Chunk]:
    import yaml
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    text_lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for e in entries:
        key = e["key"]
        line = next((i for i, ln in enumerate(text_lines, 1) if ln.strip() == f"- key: {key}"), 1)
        parts = [e.get("cite", "")]
        for f in ("used_for", "findings", "needed_for"):
            if e.get(f):
                parts.append(f"{f.replace('_', ' ')}: {e[f]}")
        parts.append(f"status: {e.get('status', '')}; knowledge base: "
                     + ", ".join(f"§{k}" for k in e.get("kb", [])))
        out.append(Chunk(id=f"{source}:{key}", collection="papers", source=source,
                         title=f"{key} — {e.get('cite', '')[:90]}", text="\n".join(parts),
                         start_line=line, end_line=line, keywords=list(e.get("topics", [])) + [key],
                         kind="paper"))
    return out
