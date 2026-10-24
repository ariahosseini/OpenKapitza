"""Retrieval: tokenizer, chunkers, index, citations, the lookup command, and the
scored question set (tests/data/rag_questions.yaml).

Thresholds sit a little under what was measured on 24 Oct 2026 (knowledge-base
section 13), so a change that makes retrieval worse fails here. The manual set
runs only when the full LAMMPS manual has been built into the local cache.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from openkapitza import rag
from openkapitza.rag.chunk import Chunk, example_chunks, markdown_chunks, rst_chunks
from openkapitza.rag.evaluate import evaluate, load_questions, matches
from openkapitza.rag.index import Index, tokenize

ROOT = Path(__file__).resolve().parents[1]

# measured 24 Oct 2026: repo R@5 0.98 / MRR 0.93 with the manual loaded (1.00 / 0.95
# without); curated 0.92 / 0.76 (1.00 / 0.96); manual 0.81 / 0.65
THRESHOLDS = {"repo": (0.95, 0.88), "curated": (0.83, 0.70), "manual": (0.75, 0.58)}


# ---- tokenizer -------------------------------------------------------------------------
# ---- chunkers ----------------------------------------------------------------------------
# ---- index -------------------------------------------------------------------------------
def _toy():
    mk = lambda i, title, text, **kw: Chunk(id=str(i), collection=kw.pop("collection", "repo"),
                                            source=f"s{i}.md", title=title, text=text, **kw)
    return Index([
        mk(0, "Thermostats", "Langevin damping and Nose-Hoover chains for temperature control."),
        mk(1, "Chunk averaging", "fix ave/chunk bins atoms into layers for a temperature profile."),
        mk(2, "Error: Nfreq is not a multiple of Nevery",
           "Nfreq is not a multiple of Nevery\nFix the arguments.", kind="error"),
        mk(3, "Paper", "Unloading hysteresis of contact conductance.", collection="papers"),
    ])


# ---- agent command ---------------------------------------------------------------------
# ---- the question set ------------------------------------------------------------------


class TestRag(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_tokenizer_keeps_compounds_and_parts(self):
        t = tokenize("fix ave/chunk with Si(D) and NEMD_UNEQUILIBRATED")
        for tok in ("ave/chunk", "ave", "chunk", "si(d)", "nemd_unequilibrated", "nemd"):
            assert tok in t



    def test_tokenizer_equation_numbers(self):
        assert "eq18" in tokenize("the printed Eq. 18 has a typo")
        assert "eq18" in tokenize("equation (18)")



    def test_tokenizer_folding(self):
        t = tokenize("melting quenched binning potentials R² 1990s")
        for tok in ("melt", "quench", "bin", "potential", "r2", "1990"):
            assert tok in t
        assert "the" not in tokenize("the")



    def test_markdown_chunks_ignore_headings_in_fences(self):
        tmp_path = self.tmp
        p = tmp_path / "x.md"
        p.write_text("# Top\nintro\n## A\ntext a\n```\n# not a heading\n```\n## B\ntext b\n")
        cs = markdown_chunks(p, "x.md")
        assert [c.title for c in cs] == ["Top", "Top > A", "Top > B"]
        a = cs[1]
        assert "# not a heading" in a.text and (a.start_line, a.end_line) == (4, 7)
        assert a.citation().startswith("x.md:4-7 — Top > A")



    def test_rst_chunks_and_errors(self):
        tmp_path = self.tmp
        page = tmp_path / "fix_demo.rst"
        page.write_text(".. index:: fix demo\n\nfix demo command\n================\n\n"
                        "Syntax\n\"\"\"\"\"\"\n\nfix ID group demo N\n\nDescription\n"
                        "\"\"\"\"\"\"\"\"\"\"\"\n\nSee :doc:`fix ave/chunk <fix_ave_chunk>` and "
                        "``Nevery``.\n")
        cs = rst_chunks(page, "lammps/doc/src/fix_demo.rst", "LAMMPS test", "https://x/fix_demo.html")
        titles = [c.title for c in cs]
        assert "fix demo command > Syntax" in titles and "fix demo command > Description" in titles
        desc = cs[titles.index("fix demo command > Description")]
        assert "See fix ave/chunk and Nevery." in desc.text and ":doc:" not in desc.text
        assert desc.keywords == ["fix demo"] and desc.url == "https://x/fix_demo.html"
        err = tmp_path / "Errors_messages.rst"
        err.write_text("*Bad thing happened*\n   Because of X.\n\n*Other %s failure*\n   Y.\n")
        es = rst_chunks(err, "lammps/doc/src/Errors_messages.rst", "v", "u")
        assert [e.title for e in es] == ["Error: Bad thing happened", "Error: Other %s failure"]
        assert all(e.kind == "error" for e in es)



    def test_example_chunks_keywords(self):
        tmp_path = self.tmp
        p = tmp_path / "in.demo"
        p.write_text("units metal\npair_style sw\nfix 1 all ave/chunk 10 10 100 c v_t\n# fix nope\n")
        (c,) = example_chunks(p, "lammps/examples/demo/in.demo", "v")
        assert {"pair_style sw", "fix ave/chunk", "units"} <= set(c.keywords)
        assert "fix nope" not in c.keywords and c.kind == "example"



    def test_bm25_ranks_and_filters(self):
        idx = _toy()
        assert idx.search("temperature profile layers", k=1)[0].chunk.id == "1"
        assert idx.search("fix ave/chunk", k=1)[0].chunk.id == "1"
        assert idx.search("hysteresis", k=3, collections=("repo",)) == []
        assert idx.search("hysteresis", k=3, collections=("papers",))[0].chunk.id == "3"
        assert idx.search("the of and") == []



    def test_phrase_bonus_lifts_pasted_error_message(self):
        hits = _toy().search("ERROR: Nfreq is not a multiple of Nevery", k=2)
        assert hits[0].chunk.id == "2"



    def test_save_load_roundtrip(self):
        tmp_path = self.tmp
        idx = _toy()
        idx.save(tmp_path / "i.json.gz", {"v": 1})
        meta, chunks, tfs = Index.load_parts(tmp_path / "i.json.gz")
        again = Index(chunks, tfs)
        assert meta == {"v": 1} and len(again.chunks) == 4
        assert again.search("Langevin", k=1)[0].chunk.id == "0"



    def test_repo_index_covers_all_three_collections(self):
        cols = {c.collection for c in rag.load().chunks}
        assert {"repo", "papers", "lammps"} <= cols
        kinds = {c.kind for c in rag.load().chunks if c.collection != "lammps" or c.kind == "curated"}
        assert {"kb", "doc", "paper", "curated"} <= kinds



    def test_every_hit_is_citable(self):
        for h in rag.search("contact microhardness Vickers", k=5):
            c = h.chunk
            assert c.source and c.start_line >= 1 and c.title
            assert c.citation().startswith(c.source + ":")
            assert h.snippet



    def test_paper_notes_have_no_copyrighted_text_markers(self):
        """papers.yaml holds our own notes: no PDF dumps, no long verbatim passages."""
        import yaml
        entries = yaml.safe_load((ROOT / "docs/papers.yaml").read_text())
        assert len(entries) >= 20
        for e in entries:
            assert {"key", "cite", "status", "topics"} <= set(e)
            assert e["status"] in ("full", "partial", "needed")
            for f in ("used_for", "findings", "needed_for"):
                assert len(str(e.get(f, ""))) < 1500, (e["key"], f)



    def test_manual_cache_is_never_in_the_repo(self):
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True,
                                 text=True).stdout.split()
        assert not [f for f in tracked if f.endswith(".json.gz") or f.startswith("lammps/")]



    def test_cache_dir_env(self):
        from openkapitza.rag.sources import cache_dir
        old = os.environ.get("OPENKAPITZA_RAG_DIR")
        os.environ["OPENKAPITZA_RAG_DIR"] = str(self.tmp)
        try:
            assert cache_dir() == self.tmp
        finally:
            if old is None:
                del os.environ["OPENKAPITZA_RAG_DIR"]
            else:
                os.environ["OPENKAPITZA_RAG_DIR"] = old



    def test_session_lookup(self):
        from openkapitza.agent.session import Session
        s = Session()
        out = s.handle("lookup why can't I add 1/h_c and 1/h_K in series")
        assert "docs/knowledge-base.md:" in out and "[1]" in out
        out = s.handle("lookup papers: unloading hysteresis")
        assert "docs/papers.yaml" in out and "knowledge-base" not in out
        assert "Usage" in s.handle("lookup")
        assert s.intake.answers == {}          # lookup never touches the inputs



    def test_question_set_is_well_formed(self):
        qs = load_questions()
        assert len(qs) >= 80
        assert {q["set"] for q in qs} == {"repo", "curated", "manual"}
        for q in qs:
            assert q["q"].strip() and q["expect"]
            assert all("source" in e for e in q["expect"])



    def test_matches_directory_expectation(self):
        from openkapitza.rag.index import Hit
        c = Chunk(id="x", collection="lammps", source="lammps/examples/ttm/in.ttm", title="t", text="")
        assert matches(Hit(c, 1.0, ""), [{"source": "examples/ttm/"}])
        assert not matches(Hit(c, 1.0, ""), [{"source": "examples/melt/"}])



    def _scores(self, name):
        s = evaluate(sets=(name,))[name]
        r5_min, mrr_min = THRESHOLDS[name]
        assert s.n > 0
        assert s.recall(5) >= r5_min, f"{name} recall@5 {s.recall(5):.2f} < {r5_min}: {s.misses}"
        assert s.mrr >= mrr_min, f"{name} MRR {s.mrr:.2f} < {mrr_min}"



    def test_retrieval_scores_repo(self):
        self._scores("repo")



    def test_retrieval_scores_curated(self):
        self._scores("curated")



    @unittest.skipUnless(rag.has_manual(), "LAMMPS manual not built "
                         "(python -m openkapitza.rag build --lammps-src PATH)")

    def test_retrieval_scores_manual(self):
        self._scores("manual")


if __name__ == "__main__":
    unittest.main()
