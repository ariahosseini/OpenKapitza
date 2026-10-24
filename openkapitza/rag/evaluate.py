"""Score retrieval on the question set: recall@k and mean reciprocal rank per set.

    python -m openkapitza.rag eval            # all sets the index can answer
    python -m openkapitza.rag eval --misses   # also list the questions that missed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import has_manual, search

QUESTIONS = Path(__file__).resolve().parents[2] / "tests" / "data" / "rag_questions.yaml"
DEPTH = 10


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def matches(hit, expect: list[dict]) -> bool:
    c = hit.chunk
    for e in expect:
        src = e["source"]
        # "examples/ttm/" names a directory: any file in it counts
        if not (src in c.source if src.endswith("/") else c.source.endswith(src)):
            continue
        t = e.get("title")
        if t is None or t.lower() in c.title.lower():
            return True
    return False


def rank_of(q: dict, depth: int = DEPTH) -> int | None:
    for r, h in enumerate(search(q["q"], k=depth), 1):
        if matches(h, q["expect"]):
            return r
    return None


@dataclass
class SetScore:
    name: str
    n: int = 0
    ranks: list = field(default_factory=list)
    misses: list = field(default_factory=list)

    def recall(self, k: int) -> float:
        return sum(1 for r in self.ranks if r is not None and r <= k) / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        return sum(1.0 / r for r in self.ranks if r) / self.n if self.n else 0.0


def evaluate(questions: list[dict] | None = None, sets: tuple[str, ...] | None = None
             ) -> dict[str, SetScore]:
    qs = questions if questions is not None else load_questions()
    if sets is None:
        sets = ("repo", "curated") + (("manual",) if has_manual() else ())
    out = {s: SetScore(s) for s in sets}
    for q in qs:
        s = out.get(q["set"])
        if s is None:
            continue
        r = rank_of(q)
        s.n += 1
        s.ranks.append(r)
        if r is None or r > 5:
            s.misses.append((q["q"], r))
    return out


def report(scores: dict[str, SetScore], misses: bool = False) -> str:
    lines = [f"{'set':8s} {'n':>3s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR':>6s}"]
    for s in scores.values():
        lines.append(f"{s.name:8s} {s.n:3d} {s.recall(1):6.2f} {s.recall(3):6.2f} "
                     f"{s.recall(5):6.2f} {s.recall(10):6.2f} {s.mrr:6.2f}")
        if misses:
            for q, r in s.misses:
                lines.append(f"    miss@5 (rank {r or '>10'}): {q}")
    return "\n".join(lines)
