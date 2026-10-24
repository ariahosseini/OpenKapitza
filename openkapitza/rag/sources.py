"""What gets indexed, and where the built manual index lives.

repo     our docs, knowledge base, benchmark sources, spec, README -- indexed
         from the working tree every time (small, always current)
papers   docs/papers.yaml, our own notes on each paper (no copyrighted text)
lammps   docs/lammps-rag.md (the committed, curated subset) always; the FULL
         manual (doc/src/*.rst), Howto pages, error messages and example
         scripts when built from a LAMMPS checkout with `python -m
         openkapitza.rag build --lammps-src PATH`. That index goes to a local
         cache, never into the repository (it is GPL text, and large).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .chunk import Chunk, example_chunks, markdown_chunks, paper_chunks, rst_chunks

ROOT = Path(__file__).resolve().parents[2]

REPO_DOCS = (
    ("docs/knowledge-base.md", "kb"),
    ("docs/agent-guide.md", "doc"),
    ("docs/agent-instruction-v2.md", "doc"),
    ("docs/open-issues.md", "doc"),
    ("docs/technical-notes.md", "doc"),
    ("docs/cluster-runbook.md", "doc"),
    ("tests/benchmarks/SOURCES.md", "doc"),
    ("README.md", "doc"),
)
CURATED = "docs/lammps-rag.md"
PAPERS = "docs/papers.yaml"


def cache_dir() -> Path:
    env = os.environ.get("OPENKAPITZA_RAG_DIR")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "openkapitza" / "rag"


def repo_version() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "working tree"
    except (OSError, subprocess.SubprocessError):
        return "working tree"


def repo_chunks(root: Path = ROOT) -> list[Chunk]:
    ver = repo_version()
    out: list[Chunk] = []
    for rel, kind in REPO_DOCS:
        p = root / rel
        if p.is_file():
            out += markdown_chunks(p, rel, "repo", kind, ver)
    p = root / CURATED
    if p.is_file():
        for c in markdown_chunks(p, CURATED, "lammps", "curated", ver):
            c.version = "LAMMPS 30 Sep 2026 (curated subset)"
            out.append(c)
    p = root / PAPERS
    if p.is_file():
        out += paper_chunks(p, PAPERS)
    return out


def lammps_version(src: Path) -> str:
    v = src / "src" / "version.h"
    m = re.search(r'LAMMPS_VERSION\s+"([^"]+)"', v.read_text()) if v.is_file() else None
    return f"LAMMPS {m.group(1)}" if m else "LAMMPS (unknown version)"


def _readme_summary(d: Path, words: int = 80) -> str:
    """The opening words of an example directory's README: what the scripts are for
    (an input script itself has little prose for keyword search to find)."""
    for name in ("README", "README.md", "README.txt"):
        p = d / name
        if p.is_file():
            txt = p.read_text(encoding="utf-8", errors="replace")
            return " ".join(txt.split()[:words])
    return ""


def manual_chunks(src: Path) -> tuple[list[Chunk], dict]:
    """Every manual page, Howto, error message and example script of a checkout."""
    src = Path(src)
    doc = src / "doc" / "src"
    if not doc.is_dir():
        raise FileNotFoundError(f"{doc} not found: give the root of a LAMMPS source checkout")
    ver = lammps_version(src)
    out: list[Chunk] = []
    n_pages = n_ex = 0
    for rst in sorted(doc.glob("*.rst")):
        cs = rst_chunks(rst, f"lammps/doc/src/{rst.name}", ver,
                        f"https://docs.lammps.org/{rst.stem}.html")
        if rst.stem.startswith("Howto"):
            for c in cs:
                c.keywords += ["howto", "how to", "overview"]
        out += cs
        n_pages += 1
    ex = src / "examples"
    for p in sorted(ex.rglob("*")):
        if not p.is_file() or p.stat().st_size > 400_000:
            continue
        if p.name.startswith("in.") or p.name in ("README", "README.md", "README.txt"):
            if p.name.startswith("in.") and p.suffix in (".gz", ".bz2", ".png", ".jpg"):
                continue
            rel = p.relative_to(src).as_posix()
            cs = example_chunks(p, f"lammps/{rel}", ver)
            if p.name.startswith("in."):
                about = _readme_summary(p.parent)
                for c in cs:
                    c.keywords += ["example", "input script"] + ([about] if about else [])
            out += cs
            n_ex += 1
    meta = {"lammps_version": ver, "pages": n_pages, "example_files": n_ex,
            "chunks": len(out), "source": str(src)}
    return out, meta
