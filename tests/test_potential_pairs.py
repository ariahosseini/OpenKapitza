"""The potential file must support the pair before a deck is written.

Three separate defects in this project were the same error narrowing: a
filename assumed (`SiGe.sw`, which LAMMPS does not ship), then a label assumed
(`Si`, where the file says `Si(D)`), then *which* label assumed (`Si(B)`,
which exists but carries no Si-Ge cross terms). Each failed only when LAMMPS
reached the `pair_coeff` line -- minutes into a run.

The interrogation helpers that fix this were written, tested, and then used
only from `scripts/smoke_lammps.py`. The library never called them, so
`build_input_deck` fell back to the bare chemical symbols and anyone driving
the library directly still got a deck LAMMPS rejects, with nothing to warn
them. Found by checking the helpers against a real LAMMPS potentials
directory and then grepping for callers. These tests pin the wiring.

The fixtures below are synthetic but reproduce the real structure of
`SiCGe.tersoff`: several parameterisations of one element, cross terms for
only one of them. Verified against the genuine file, whose labels are
Si(B), Si(C), C, Si(D), Ge -- with all eight Si/Ge triples present for
Si(D) only, and two each for Si(B) and Si(C).
"""

from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

from openkapitza.md.nemd import (
    NEMDGeometry,
    NEMDSpec,
    build_input_deck,
    check_spec,
    complete_pairs,
    match_species,
    potential_elements,
    potential_triples,
    resolve_pair_elements,
)
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code, Severity

# Numeric tail of a Tersoff entry; the values are irrelevant to label parsing.
PARAMS = "3.0 1.0 1.8 0.3 1.2 -0.6 1.0 1.0e-6 1.7 2.1 2.7 1.0 180.0 0.0"


def _entry(a: str, b: str, c: str) -> str:
    return f"{a} {b} {c} {PARAMS}\n"


def _write_potential(tmp: Path, name: str, triples) -> Path:
    path = tmp / name
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# synthetic test potential\n")
        for a, b, c in triples:
            fh.write(_entry(a, b, c))
    return path


def _all_triples(*labels):
    return list(itertools.product(labels, repeat=3))


class PotentialFixture(unittest.TestCase):
    """A file where only Si(D) carries the Ge cross terms."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)

        # Entry ORDER matters and mirrors the real file: Si(B) first, so the
        # first label whose stem is "Si" is the one WITHOUT the cross terms.
        # That ordering is what makes match_species the wrong tool here.
        triples = [
            ("Si(B)", "Si(B)", "Si(B)"),
            ("Si(C)", "Si(C)", "Si(C)"),
            ("C", "C", "C"),
        ]
        triples += _all_triples("Si(D)", "Ge")           # the complete pairing
        self.mixed = _write_potential(tmp, "SiCGe.tersoff", triples)

        # A single-element file, like Si.sw: cannot do Si/Ge at all.
        self.single = _write_potential(tmp, "Si.sw", [("Si", "Si", "Si")])

    def tearDown(self):
        self._tmp.cleanup()

    def _spec(self, pair_file, pair_elements=None, a="Si", b="Ge"):
        return NEMDSpec(
            geometry=NEMDGeometry(), element_a=a, element_b=b,
            lattice_a=5.431e-10, lattice_b=5.658e-10,
            mass_a=28.0855, mass_b=72.63,
            pair_style="tersoff", pair_file=str(pair_file),
            temperature=300.0, pair_elements=pair_elements,
        )

    def _potential_flags(self, spec):
        flags = FlagSet()
        check_spec(spec, flags)
        return [f for f in flags.flags
                if f.code in (Code.NEMD_POTENTIAL_INCOMPLETE,
                              Code.NEMD_POTENTIAL_UNVERIFIED)]


class TestFixtureMatchesReality(PotentialFixture):
    def test_labels_are_not_bare_symbols(self):
        self.assertIn("Si(D)", potential_elements(self.mixed))
        self.assertNotIn("Si", potential_elements(self.mixed))

    def test_only_one_parameterisation_is_complete(self):
        """Counts measured against the real SiCGe.tersoff: 2, 2, 8 of 8.

        The 2 is the element's own self-triple plus (Ge,Ge,Ge); there is no
        Si(B)-Ge or Si(C)-Ge cross term at all.
        """
        triples = potential_triples(self.mixed)
        for si, expected in (("Si(B)", 2), ("Si(C)", 2), ("Si(D)", 8)):
            need = set(itertools.product([si, "Ge"], repeat=3))
            self.assertEqual(len(need & triples), expected, si)
        self.assertEqual(
            potential_elements(self.mixed),
            ["Si(B)", "Si(C)", "C", "Si(D)", "Ge"],
            "label order must mirror the real file",
        )

    def test_match_species_picks_the_WRONG_one(self):
        """Documents why match_species must not be used to write a deck."""
        picked = match_species(potential_elements(self.mixed), "Si")
        self.assertEqual(picked, "Si(B)")
        self.assertNotIn(
            (picked, "Ge"),
            [tuple(p) for p in complete_pairs(potential_triples(self.mixed),
                                              "Si", "Ge")],
        )


class TestResolvePairElements(PotentialFixture):
    def test_resolves_to_the_complete_pairing(self):
        self.assertEqual(
            resolve_pair_elements(self.mixed, "Si", "Ge"), ("Si(D)", "Ge"))

    def test_raises_when_no_pairing_is_complete(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_pair_elements(self.single, "Si", "Ge")
        # The message must say what IS available; that is the next question.
        self.assertIn("Si", str(ctx.exception))

    def test_raises_on_a_missing_file(self):
        with self.assertRaises(OSError):
            resolve_pair_elements("/nonexistent/SiGe.sw", "Si", "Ge")


class TestDeckUsesResolvedLabels(PotentialFixture):
    def test_deck_writes_the_complete_labels_not_bare_symbols(self):
        deck = build_input_deck(self._spec(self.mixed))
        line = next(l for l in deck.splitlines() if l.startswith("pair_coeff"))
        self.assertIn("Si(D) Ge", line)
        self.assertNotRegex(line, r"\bSi Ge\b")

    def test_explicit_pair_elements_are_respected(self):
        deck = build_input_deck(self._spec(self.mixed, ("Si(D)", "Ge")))
        line = next(l for l in deck.splitlines() if l.startswith("pair_coeff"))
        self.assertIn("Si(D) Ge", line)

    def test_deck_still_generates_without_the_potential_file(self):
        """Decks are often written away from the potentials directory."""
        deck = build_input_deck(self._spec("/nonexistent/SiGe.sw"))
        self.assertIn("pair_coeff", deck)


class TestCheckSpecFlags(PotentialFixture):
    def test_incomplete_requested_pairing_is_critical(self):
        flags = self._potential_flags(self._spec(self.mixed, ("Si(B)", "Ge")))
        self.assertTrue(flags, "the Si(B) trap raised nothing")
        self.assertEqual(flags[0].code, Code.NEMD_POTENTIAL_INCOMPLETE)
        self.assertEqual(flags[0].severity, Severity.CRITICAL)

    def test_impossible_pair_is_critical(self):
        flags = self._potential_flags(self._spec(self.single))
        self.assertTrue(flags)
        self.assertEqual(flags[0].severity, Severity.CRITICAL)

    def test_unreadable_file_warns_rather_than_failing(self):
        flags = self._potential_flags(self._spec("/nonexistent/SiGe.sw"))
        self.assertTrue(flags)
        self.assertEqual(flags[0].code, Code.NEMD_POTENTIAL_UNVERIFIED)
        self.assertEqual(flags[0].severity, Severity.WARNING)

    def test_valid_pairing_raises_nothing(self):
        self.assertEqual(self._potential_flags(self._spec(self.mixed, ("Si(D)", "Ge"))), [])

    def test_auto_resolved_spec_raises_nothing(self):
        self.assertEqual(self._potential_flags(self._spec(self.mixed)), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
