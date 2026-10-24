"""Keyword retrieval: BM25 over title, keywords and body, plus an exact-phrase bonus.

Why keyword search first: most lookups in this project are for exact strings
-- LAMMPS command and style names (`fix ave/chunk`, `ttm/thermal`), literal
error messages, flag codes, equation numbers, element labels like `Si(D)` --
which embedding search handles poorly. The question set in
tests/data/rag_questions.yaml measures whether that is enough; semantic search
is to be added only if it shows misses that keywords cannot fix.

Tokens: lower case; compounds joined by '/', '-', '.', '_' are kept whole AND
split into parts ("ave/chunk" -> ave/chunk, ave, chunk); "Eq. 18" becomes
"eq18"; plurals are folded crudely (no full stemmer: it mangles technical
words). Field weights: title x3, keywords x3, body x1.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .chunk import Chunk

K1, B = 1.2, 0.75
W_TITLE, W_KEYWORDS, W_BODY = 3.0, 3.0, 1.0
PHRASE_BONUS = 0.6          # relative to the top BM25 score, for an exact phrase hit
# No collection prior: measured, favouring our docs or the curated subset over
# the full manual cost 0.08-0.11 manual recall@5 and gained nothing elsewhere
# (knowledge-base.md section 13).
ERROR_PRIOR = 0.6           # one-line error entries are short, so BM25 over-rewards
                            # them; full weight back when the message matches verbatim
EXAMPLE_PRIOR = 1.4         # when the query asks for an example / input script
_WANTS_EXAMPLE = re.compile(r"\b(example|examples|input script|sample script|in\.\w+)\b", re.I)

_STOP = frozenset("""a an and are as at be been but by can could do does for from had has
have how i if in into is it its may me my no not of on or our should so such than that the
their them then there these they this to too us was we were what when where which while who
why will with would you your vs via per also only just any all each one two""".split())
_TOKEN = re.compile(r"[a-z0-9_]+(?:\([a-z0-9]+\))?(?:[/\-.][a-z0-9_]+(?:\([a-z0-9]+\))?)*")
_EQ = re.compile(r"\beq(?:uation)?s?\.?\s*\(?(\d+[a-z]?)")


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().replace("²", "2").replace("₂", "2").replace("³", "3")


def _plural(t: str) -> str:
    """Crude folding: plurals, and -ing/-ed onto a root of 4+ letters
    (melting -> melt, quenched -> quench, binning -> bin). Not a stemmer."""
    if t.isdigit():
        return t
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is", "os")):
        return t[:-1]
    for suf in ("ing", "ed"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            r = t[: -len(suf)]
            if len(r) >= 4 and r[-1] == r[-2] and r[-1] not in "lsz":
                r = r[:-1]
            return r
    return t


def tokenize(text: str) -> list[str]:
    s = _fold(text)
    out = [f"eq{m.group(1)}" for m in _EQ.finditer(s)]
    for m in _TOKEN.finditer(s):
        tok = m.group(0).strip("._-")
        if not tok:
            continue
        parts = [p for p in re.split(r"[/\-._()]", tok) if p]
        if len(parts) > 1:
            out.append(tok)
        for p in parts:
            if p in _STOP or (len(p) == 1 and not p.isdigit()):
                continue
            out.append(_plural(p))
    return out


def _norm_phrase(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w/ .:\-]", " ", _fold(s))).strip()


@dataclass
class Hit:
    chunk: Chunk
    score: float
    snippet: str


class Index:
    def __init__(self, chunks: list[Chunk], tfs: list[dict] | None = None):
        self.chunks = chunks
        self.tfs = tfs if tfs is not None else [self._tf(c) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avglen = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        self.post: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, tf in enumerate(self.tfs):
            for t, f in tf.items():
                self.post[t].append((i, f))
        n = len(chunks)
        self.idf = {t: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.post.items()}
        self._norm = [None] * n

    @staticmethod
    def _tf(c: Chunk) -> dict:
        tf: Counter = Counter()
        for t in tokenize(c.title):
            tf[t] += W_TITLE
        for t in tokenize(" ".join(c.keywords)):
            tf[t] += W_KEYWORDS
        for t in tokenize(c.text):
            tf[t] += W_BODY
        return dict(tf)

    def _norm_text(self, i: int) -> str:
        if self._norm[i] is None:
            c = self.chunks[i]
            self._norm[i] = _norm_phrase(c.title + " \n " + c.text)
        return self._norm[i]

    def search(self, query: str, k: int = 5, collections: tuple[str, ...] | None = None,
               sources: tuple[str, ...] | None = None) -> list[Hit]:
        q = query.strip()
        q = re.sub(r"^(error|warning)\s*:\s*", "", q, flags=re.I)
        terms = tokenize(q)
        if not terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        for t in set(terms):
            idf = self.idf.get(t)
            if idf is None:
                continue
            qtf = terms.count(t)
            for i, f in self.post[t]:
                denom = f + K1 * (1 - B + B * self.lens[i] / self.avglen)
                scores[i] += idf * f * (K1 + 1) / denom * (1 + 0.2 * (qtf - 1))
        if collections or sources:
            scores = {i: s for i, s in scores.items()
                      if (not collections or self.chunks[i].collection in collections)
                      and (not sources or any(src in self.chunks[i].source for src in sources))}
        if not scores:
            return []
        top = sorted(scores.items(), key=lambda kv: -kv[1])[: max(50, 4 * k)]
        # exact-phrase bonus: the whole query (pasted error messages, titles) or
        # any "quoted" segment, found verbatim (normalised) in title or text
        phrases = [_norm_phrase(p) for p in re.findall(r'"([^"]{6,})"', query)]
        whole = _norm_phrase(q)
        if len(whole) >= 12:
            phrases.append(whole)
        wants_ex = bool(_WANTS_EXAMPLE.search(query))
        ql = " " + _norm_phrase(q) + " "

        def adjust(i: int, s: float) -> tuple[float, bool]:
            c = self.chunks[i]
            phrase = bool(phrases) and any(p in self._norm_text(i) for p in phrases)
            if c.kind == "error" and not phrase:
                s *= ERROR_PRIOR
            if c.kind == "example" and wants_ex:
                s *= EXAMPLE_PRIOR
            # the query names this manual page's command exactly ("fix langevin")
            if c.url and c.kind != "error":
                page = _norm_phrase(c.title.split(" > ")[0].replace(" command", ""))
                if len(page) > 3 and f" {page} " in ql:
                    s *= 1.3
            return s, phrase

        adj = [(i, *adjust(i, s)) for i, s in top]
        best = max(a for _, a, _ in adj)
        top = [(i, a + (PHRASE_BONUS * best if ph else 0.0)) for i, a, ph in adj]
        top.sort(key=lambda kv: -kv[1])
        return [Hit(self.chunks[i], s, snippet(self.chunks[i].text, terms)) for i, s in top[:k]]

    # ---- persistence ---------------------------------------------------------------
    def save(self, path: Path, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"meta": meta, "chunks": [c.to_dict() for c in self.chunks], "tfs": self.tfs}
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(data, fh)

    @staticmethod
    def load_parts(path: Path) -> tuple[dict, list[Chunk], list[dict]]:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        return data["meta"], [Chunk(**c) for c in data["chunks"]], data["tfs"]


def snippet(text: str, terms: list[str], width: int = 300) -> str:
    """The ~width-character window of `text` holding the most query terms."""
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= width:
        return flat
    tset = set(terms)
    words = flat.split(" ")
    marks = [1 if any(t in tset for t in tokenize(w)) else 0 for w in words]
    best, best_i, run, i0 = -1, 0, 0, 0
    span = max(8, width // 7)
    run = sum(marks[:span])
    best, best_i = run, 0
    for i in range(1, max(1, len(words) - span)):
        run += marks[i + span - 1] - marks[i - 1]
        if run > best:
            best, best_i = run, i
    s = " ".join(words[best_i: best_i + span])
    return ("… " if best_i else "") + s[:width] + (" …" if best_i + span < len(words) else "")
