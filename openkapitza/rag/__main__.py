"""python -m openkapitza.rag build --lammps-src PATH | query TEXT... | stats | eval [--misses]"""

from __future__ import annotations

import argparse
import sys

from . import build_manual, info, search
from .sources import cache_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m openkapitza.rag", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="index a LAMMPS checkout's manual and examples")
    b.add_argument("--lammps-src", required=True, help="root of a LAMMPS source checkout")
    q = sub.add_parser("query", help="search and print cited passages")
    q.add_argument("text", nargs="+")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--collection", choices=("repo", "papers", "lammps"), action="append")
    sub.add_parser("stats", help="what is indexed")
    e = sub.add_parser("eval", help="recall@k and MRR on tests/data/rag_questions.yaml")
    e.add_argument("--misses", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "build":
        m = build_manual(a.lammps_src)
        print(f"Indexed {m['lammps_version']}: {m['pages']} manual pages, "
              f"{m['example_files']} example files, {m['chunks']} chunks -> {m['path']}")
    elif a.cmd == "query":
        cols = tuple(a.collection) if a.collection else None
        for n, h in enumerate(search(" ".join(a.text), k=a.k, collections=cols), 1):
            print(f"[{n}] {h.score:.1f}  {h.chunk.citation()}")
            if h.chunk.url:
                print(f"     {h.chunk.url}")
            print(f"     {h.snippet}\n")
    elif a.cmd == "stats":
        m = info()
        print(f"repo + papers + curated chunks: {m['repo_chunks']}")
        man = m["manual"]
        print(f"manual: {man['lammps_version']}, {man['chunks']} chunks ({cache_dir()})"
              if man else f"manual: not built (cache {cache_dir()})")
    elif a.cmd == "eval":
        from .evaluate import evaluate, report
        print(report(evaluate(), misses=a.misses))
    return 0


if __name__ == "__main__":
    sys.exit(main())
