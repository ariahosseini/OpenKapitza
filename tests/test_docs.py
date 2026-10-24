"""The user-facing docs must not drift from the code: every agent field and
command is documented, every relative link resolves, the guide's anchors exist,
and the examples it names are present."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from openkapitza.agent.fields import FIELDS
from openkapitza.agent.session import HELP

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "docs" / "agent-guide.md"
LINK = re.compile(r"\]\(([^)]+)\)")


def slug(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


class Docs(unittest.TestCase):
    def test_every_agent_field_is_documented(self):
        text = GUIDE.read_text()
        missing = [f.name for f in FIELDS if f"`{f.name}`" not in text and f.name not in text]
        self.assertEqual(missing, [])

    def test_every_session_command_is_documented(self):
        text = GUIDE.read_text()
        for word in ("why", "back", "show", "set", "run", "report", "what if", "sweep",
                     "prepare jobs", "ingest", "md clear", "save", "load", "export", "lookup"):
            self.assertIn(word, HELP, f"{word!r} missing from Session HELP")
            self.assertIn(f"`{word}", text, f"{word!r} missing from the guide")

    def test_relative_links_resolve(self):
        files = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))
        broken = []
        for f in files:
            if f.name == "lammps-rag.md":          # generated corpus
                continue
            for target in LINK.findall(f.read_text()):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path = target.split("#")[0]
                if path and not (f.parent / path).exists():
                    broken.append(f"{f.relative_to(ROOT)} -> {target}")
        self.assertEqual(broken, [])

    def test_guide_anchors_exist(self):
        text = GUIDE.read_text()
        heads = {slug(h) for h in re.findall(r"^#+ (.+)$", text, flags=re.M)}
        anchors = re.findall(r"\]\(#([^)]+)\)", text)
        self.assertTrue(anchors)
        self.assertEqual([a for a in anchors if a not in heads], [])

    def test_named_examples_exist(self):
        text = GUIDE.read_text() + (ROOT / "README.md").read_text()
        for name in set(re.findall(r"examples/agent/[\w.]+", text)):
            self.assertTrue((ROOT / name).exists(), name)

    def test_readme_stays_short(self):
        self.assertLess(len((ROOT / "README.md").read_text().splitlines()), 200)


if __name__ == "__main__":
    unittest.main()
