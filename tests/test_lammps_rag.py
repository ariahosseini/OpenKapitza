"""Keep the LAMMPS RAG corpus honest.

`docs/lammps-rag.md` is generated from a LAMMPS source tree by
`scripts/build_lammps_rag.py`. These tests check the properties that make it
useful for retrieval, so a generator change or a LAMMPS upgrade that quietly
empties a section fails here rather than at the point someone trusts a page
that is not there.

Two failure modes are guarded specifically:

  - **Silent emptying.** A renamed `.rst` or moved example would otherwise
    produce a section with a heading and no content. The generator records its
    own coverage counts in the output, and a test reads them back.
  - **Scope creep.** A corpus that grows to cover the whole manual retrieves
    everything and discriminates nothing, so the out-of-scope test pins what
    must stay absent.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

RAG = Path(__file__).resolve().parent.parent / "docs" / "lammps-rag.md"

TOPIC_HEADINGS = (
    "Thermal transport",
    "Elastic constants and cell relaxation",
    "Hardness and indentation",
    "Gas-surface scattering and accommodation coefficients",
    "Surface structure and roughness",
    "Amorphous structures and melt-quench",
    "Electron-phonon coupling (two-temperature model)",
    "Gas transport properties",
)

# One script per topic that has them, as a canary for that section.
CANARY_SCRIPTS = (
    "KAPPA/in.langevin",
    "ELASTIC/in.elastic",
    "indent/in.indent",
    "wall/in.wall.ccl",
    "melt/in.melt",
    "vashishta/in.vashishta.sio2",
    "ttm/in.ttm",
)

# Commands whose absence would mean a physics path lost its reference.
CANARY_COMMANDS = (
    "fix langevin", "fix ave/chunk", "compute temp/region",
    "fix box/relax", "minimize", "compute born/matrix",
    "fix indent",
    "fix wall/reflect/stochastic",
    "displace_atoms", "delete_atoms",
    "pair_style vashishta",
    "fix ttm",
    "fix viscosity",
)


class RagFixture(unittest.TestCase):
    def setUp(self):
        if not RAG.is_file():
            self.skipTest(f"{RAG.name} not generated in this checkout")
        self.text = RAG.read_text()


class TestProvenance(RagFixture):
    def test_records_source_and_version(self):
        """A corpus without a source and version is unauditable."""
        self.assertIn("build_lammps_rag.py", self.text)
        self.assertRegex(self.text, r"version \*\*\d+ \w+ \d{4}\*\*")

    def test_says_it_is_generated_not_hand_edited(self):
        self.assertIn("do not edit", self.text.lower())

    def test_notes_that_scripts_are_verbatim(self):
        self.assertIn("verbatim", self.text)


class TestCoverage(RagFixture):
    def test_all_topics_present(self):
        for heading in TOPIC_HEADINGS:
            with self.subTest(heading=heading):
                self.assertIn(heading, self.text)

    def test_canary_scripts_present(self):
        for script in CANARY_SCRIPTS:
            with self.subTest(script=script):
                self.assertIn(f"`{script}`", self.text)

    def test_canary_commands_present(self):
        for command in CANARY_COMMANDS:
            with self.subTest(command=command):
                self.assertIn(f"`{command}`", self.text)

    def test_generator_reports_no_missing_sources(self):
        """The generator writes its coverage into the corpus; read it back.

        This is the test that catches a renamed .rst after a LAMMPS upgrade,
        which would otherwise leave a heading with nothing under it.
        """
        self.assertIn("## Generation coverage", self.text)
        scripts = re.search(r"example scripts: \*\*(\d+) found\*\*, (\d+) missing",
                            self.text)
        pages = re.search(r"command pages: \*\*(\d+) found\*\*, (\d+) missing",
                          self.text)
        self.assertIsNotNone(scripts, "coverage line for scripts missing")
        self.assertIsNotNone(pages, "coverage line for pages missing")
        self.assertEqual(int(scripts.group(2)), 0,
                         f"{scripts.group(2)} example scripts missing")
        self.assertEqual(int(pages.group(2)), 0,
                         f"{pages.group(2)} command pages missing")
        self.assertGreaterEqual(int(scripts.group(1)), 18)
        self.assertGreaterEqual(int(pages.group(1)), 35)


class TestRetrievalQuality(RagFixture):
    def test_scripts_are_code_not_prose(self):
        """Verbatim scripts are the point; a summary would not be runnable."""
        for marker in ("fix_modify      hot temp Thot",
                       "fix ywalls all wall/reflect/stochastic ccl"):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_accommodation_coefficient_arguments_are_documented(self):
        """The reason the gas-surface section exists.

        An earlier generator took only the FIRST syntax block and silently
        dropped the nested parsed-literal that defines these arguments.
        """
        self.assertIn("accomx accomy accomz", self.text)

    def test_admonitions_are_captured(self):
        count = len(re.findall(r"\*\*(?:NOTE|WARNING):\*\*", self.text))
        self.assertGreaterEqual(count, 40, f"only {count} admonitions captured")

    def test_each_topic_states_why_it_is_here(self):
        """Rationale is what makes a retrieved chunk actionable."""
        self.assertGreaterEqual(self.text.count("*Why this is here:*"),
                                len(TOPIC_HEADINGS))

    def test_records_the_line_continuation_lesson(self):
        """One of this project's own LAMMPS defects; cheap to pin."""
        self.assertIn("continuation is `&`", self.text)


class TestScope(RagFixture):
    def test_scope_stays_bounded(self):
        """Pages deliberately OUT of scope; their presence means creep."""
        for out_of_scope in ("#### `pair_style lj/cut`", "#### `read_data`",
                             "#### `dump`", "#### `fix rigid`",
                             "#### `angle_style harmonic`",
                             "#### `fix bond/react`"):
            with self.subTest(command=out_of_scope):
                self.assertNotIn(out_of_scope, self.text)

    def test_size_is_reasonable(self):
        kb = len(self.text.encode()) / 1024
        self.assertGreater(kb, 60, "suspiciously small; a section may be empty")
        self.assertLess(kb, 500, "too large; scope has probably crept")


if __name__ == "__main__":
    unittest.main(verbosity=2)
