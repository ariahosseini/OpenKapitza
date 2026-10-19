"""Keep the LAMMPS thermal RAG corpus honest.

`docs/lammps-thermal-rag.md` is generated from a LAMMPS source tree by
`scripts/build_lammps_thermal_rag.py`. These tests check the properties that
make it useful for retrieval, so a generator change or a LAMMPS upgrade that
quietly empties a section fails here rather than at the point someone trusts
a missing page.

The corpus is deliberately scoped to the thermal-transport path only. A test
asserts that scope, because the obvious failure mode for a RAG corpus is
growing until it retrieves everything and discriminates nothing.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

RAG = Path(__file__).resolve().parent.parent / "docs" / "lammps-thermal-rag.md"

# The five official methods. If one disappears, retrieval silently loses a
# whole approach rather than returning something wrong.
SCRIPTS = ("in.langevin", "in.heat", "in.ehex", "in.mp", "in.heatflux")

COMMANDS = (
    "fix langevin", "fix heat", "fix ehex", "fix thermal/conductivity",
    "compute heat/flux", "compute temp/region", "compute ke/atom",
    "compute chunk/atom", "fix ave/chunk", "fix ave/time",
    "fix ave/correlate",
)


class TestCorpusExists(unittest.TestCase):
    def setUp(self):
        if not RAG.is_file():
            self.skipTest(f"{RAG.name} not generated in this checkout")
        self.text = RAG.read_text()

    def test_records_its_provenance(self):
        """A corpus without a source and version is unauditable."""
        self.assertIn("Generated", self.text)
        self.assertIn("build_lammps_thermal_rag.py", self.text)
        self.assertRegex(self.text, r"version \*\*\d+ \w+ \d{4}\*\*")

    def test_says_it_is_generated_not_hand_edited(self):
        self.assertIn("do not edit", self.text.lower())

    def test_all_five_official_methods_are_present(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                self.assertIn(f"`{script}`", self.text)

    def test_scripts_are_present_as_code_not_description(self):
        """Verbatim scripts are the point; a summary would not be runnable."""
        for marker in ("fix_modify      hot temp Thot",
                       "compute         layers all chunk/atom bin/1d z"):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_every_command_page_is_present(self):
        for command in COMMANDS:
            with self.subTest(command=command):
                self.assertIn(f"### `{command}`", self.text)

    def test_syntax_blocks_are_captured(self):
        """Without a syntax block a page cannot answer "what are the args"."""
        self.assertGreaterEqual(self.text.count("**Syntax**"), len(COMMANDS) - 1)

    def test_admonitions_are_captured(self):
        """The notes and warnings are the highest-value part of the manual."""
        count = len(re.findall(r"\*\*(?:NOTE|WARNING):\*\*", self.text))
        self.assertGreaterEqual(count, 10, f"only {count} admonitions captured")

    def test_kappa_extraction_recipes_are_included(self):
        """How to turn the output into a number, in the source's own words."""
        self.assertIn("kappa = dQ * dZ/dTemp", self.text)

    def test_carries_the_sources_own_caveat(self):
        """The README warns these runs are not converged; that must travel."""
        self.assertIn("No guarantee is made", self.text)

    def test_no_stray_rst_underlines_leaked(self):
        """Quoted prose must read as prose, not as half-converted markup."""
        for line in self.text.splitlines():
            stripped = line.strip()
            if stripped and set(stripped) <= set('"') and len(stripped) > 2:
                self.fail(f"stray rst underline leaked: {line!r}")

    def test_scope_stays_narrow(self):
        """A corpus that grows to cover everything discriminates nothing.

        These are commands deliberately OUT of scope: structural, I/O and
        force-field pages that would dilute thermal retrieval.
        """
        for out_of_scope in ("### `pair_style lj/cut`", "### `read_data`",
                             "### `minimize`", "### `velocity`",
                             "### `dump`", "### `fix npt`"):
            with self.subTest(command=out_of_scope):
                self.assertNotIn(out_of_scope, self.text)

    def test_is_large_enough_to_be_useful_and_small_enough_to_retrieve(self):
        kb = len(self.text.encode()) / 1024
        self.assertGreater(kb, 20, "suspiciously small; a section may be empty")
        self.assertLess(kb, 400, "too large; scope has probably crept")


if __name__ == "__main__":
    unittest.main(verbosity=2)
