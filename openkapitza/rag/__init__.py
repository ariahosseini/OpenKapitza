"""Retrieval over this project's docs, our paper notes, and the LAMMPS manual.

    from openkapitza.rag import search
    for hit in search("fix ave/chunk Nfreq must be a multiple of Nevery", k=3):
        print(hit.chunk.citation())
        print(hit.snippet)

Our docs and paper notes are indexed from the working tree on first use. The
full LAMMPS manual and examples are added once built from a checkout:

    python -m openkapitza.rag build --lammps-src /path/to/lammps

Every hit carries a citation (path, lines, section, version, and for manual
pages the docs.lammps.org URL). Retrieved text informs; it does not enter the
physics without a source check (see docs/papers.yaml and materials.yaml).
"""

from __future__ import annotations

from pathlib import Path

from .chunk import Chunk
from .index import Hit, Index, tokenize
from .sources import cache_dir, manual_chunks, repo_chunks

MANUAL_FILE = "lammps-manual.json.gz"

_INDEX: Index | None = None
_META: dict = {}


def build_manual(src: Path | str, out_dir: Path | None = None) -> dict:
    """Index a LAMMPS checkout's manual and examples into the local cache."""
    chunks, meta = manual_chunks(Path(src))
    idx = Index(chunks)
    path = (out_dir or cache_dir()) / MANUAL_FILE
    idx.save(path, meta)
    meta["path"] = str(path)
    reset()
    return meta


def load(refresh: bool = False) -> Index:
    """The combined index: repo + papers + curated LAMMPS, plus the manual if built."""
    global _INDEX, _META
    if _INDEX is not None and not refresh:
        return _INDEX
    chunks = repo_chunks()
    tfs = [Index._tf(c) for c in chunks]
    meta = {"repo_chunks": len(chunks), "manual": None}
    path = cache_dir() / MANUAL_FILE
    if path.is_file():
        m, mchunks, mtfs = Index.load_parts(path)
        chunks += mchunks
        tfs += mtfs
        meta["manual"] = m
    _INDEX, _META = Index(chunks, tfs), meta
    return _INDEX


def reset() -> None:
    global _INDEX
    _INDEX = None


def info() -> dict:
    load()
    return dict(_META)


def has_manual() -> bool:
    return (cache_dir() / MANUAL_FILE).is_file()


def search(query: str, k: int = 5, collections: tuple[str, ...] | None = None,
           sources: tuple[str, ...] | None = None) -> list[Hit]:
    return load().search(query, k=k, collections=collections, sources=sources)


__all__ = ["Chunk", "Hit", "Index", "build_manual", "has_manual", "info", "load", "reset",
           "search", "tokenize"]
