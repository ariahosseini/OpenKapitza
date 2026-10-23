"""Elastic constants from MD, and the isotropic E and nu the continuum needs.

Pure-Python tests pin the averaging, the central differences, the
insensitivity check, the gate and the section-11 confidence cap. One engine
test runs only when a LAMMPS binary is supplied:

    OPENKAPITZA_LMP=/path/to/lmp OPENKAPITZA_POTENTIALS=/path/to/potentials \\
        python3 -m unittest tests.test_md_elastic

It reproduces the published Stillinger-Weber silicon constants
(C11 151.4, C12 76.4, C44 56.4 GPa). The calculator was also checked against
the official examples/ELASTIC script on the same potential, which returned
151.424 / 76.422 / 56.449 GPa; ours gives 151.42 / 76.42 / 56.45.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from openkapitza.md.elastic import (
    GATE_TOLERANCE, CubicSpec, ElasticResult, assemble, biaxial_modulus,
    born_stable, build_elastic_deck, constants_from_rows, gate,
    parse_elastic_output, run_elastic, to_quantities, voigt_reuss_hill,
)
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code
from openkapitza.schema.quantities import Confidence, Provenance

SW = CubicSpec("Si (SW)", "diamond", 5.431, 28.0855, "sw", "Si.sw", "Si", "Si")
NI = CubicSpec("Ni", "fcc", 3.52, 58.6934, "eam", "Ni_u3.eam", None, "Ni")


def _rows_from(c11, c12, c44, magnitudes=(0.002, 0.005), offset=0.3):
    """Synthetic stress rows for exact constants, with a residual offset."""
    rows = []
    for m in magnitudes:
        for s in (1.0, -1.0):
            e = s * m
            rows.append(("xx", e, offset + c11 * e, offset + c12 * e,
                         offset + c12 * e, 0.0))
            rows.append(("yz", e, offset, offset, offset, offset + c44 * e))
    return rows


class TestVoigtReussHill(unittest.TestCase):
    def test_isotropic_crystal_round_trips_exactly(self):
        """For C44 = (C11-C12)/2 the bounds coincide and E, nu come back."""
        E, nu = 200.0, 0.30
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        iso = voigt_reuss_hill(lam + 2 * mu, lam, mu)
        self.assertAlmostEqual(iso.youngs, E, places=9)
        self.assertAlmostEqual(iso.poisson, nu, places=12)
        self.assertAlmostEqual(iso.shear_voigt, iso.shear_reuss, places=9)
        self.assertAlmostEqual(iso.zener, 1.0, places=12)

    def test_silicon_reference_gives_textbook_polycrystal_values(self):
        """VRH of experimental Si: E ~ 162.7 GPa, nu ~ 0.223."""
        iso = voigt_reuss_hill(165.7, 63.9, 79.6)
        self.assertAlmostEqual(iso.youngs, 162.7, delta=0.2)
        self.assertAlmostEqual(iso.poisson, 0.223, delta=0.001)

    def test_voigt_is_an_upper_and_reuss_a_lower_bound(self):
        iso = voigt_reuss_hill(246.5, 147.3, 124.7)       # strongly anisotropic Ni
        self.assertGreater(iso.shear_voigt, iso.shear)
        self.assertGreater(iso.shear, iso.shear_reuss)
        self.assertGreater(iso.voigt_reuss_spread, 0.15)

    def test_unstable_constants_are_refused_not_averaged(self):
        self.assertFalse(born_stable(100.0, 120.0, 50.0))
        with self.assertRaises(ValueError):
            voigt_reuss_hill(100.0, 120.0, 50.0)

    def test_biaxial_modulus(self):
        self.assertAlmostEqual(biaxial_modulus(165.7, 63.9), 180.3, delta=0.1)


class TestDeck(unittest.TestCase):
    def test_relaxes_the_box_before_straining(self):
        deck = build_elastic_deck(SW, "xx", 0.002)
        self.assertLess(deck.index("box/relax"), deck.index("change_box"))

    def test_shear_needs_a_triclinic_box_and_relaxes_atoms(self):
        """The post-strain minimisation IS the Kleinman relaxation for C44."""
        deck = build_elastic_deck(SW, "yz", 0.002)
        self.assertIn("change_box      all triclinic", deck)
        self.assertGreater(deck.count("minimize"), 1)

    def test_funcfl_eam_uses_per_pair_syntax_and_no_label(self):
        self.assertIn("pair_coeff      1 1 Ni_u3.eam", build_elastic_deck(NI, "xx", 0.002))

    def test_manybody_styles_need_a_label(self):
        bad = CubicSpec("x", "diamond", 5.4, 28.0, "tersoff", "f.tersoff", None)
        with self.assertRaises(ValueError):
            build_elastic_deck(bad, "xx", 0.002)

    def test_unknown_component_rejected(self):
        with self.assertRaises(ValueError):
            build_elastic_deck(SW, "xy", 0.002)


class TestExtraction(unittest.TestCase):
    def test_parse_reads_lattice_constant_and_stresses(self):
        text = ("junk\nELASTIC_A0 5.430900\n"
                "ELASTIC_STRESS xx 0.00200000 0.3 0.15 0.15 0.0\nmore junk\n")
        a0, rows = parse_elastic_output(text)
        self.assertAlmostEqual(a0, 5.4309)
        self.assertEqual(rows[0][0], "xx")
        self.assertAlmostEqual(rows[0][2], 0.3)

    def test_central_difference_recovers_exact_constants_despite_offset(self):
        """A residual stress after box/relax must cancel, not leak into C_ij."""
        c11, c12, c44 = constants_from_rows(_rows_from(151.4, 76.4, 56.4), 0.002)
        self.assertAlmostEqual(c11, 151.4, places=9)
        self.assertAlmostEqual(c12, 76.4, places=9)
        self.assertAlmostEqual(c44, 56.4, places=9)

    def test_missing_strain_is_an_error(self):
        rows = [r for r in _rows_from(151.4, 76.4, 56.4) if r[0] == "xx"]
        with self.assertRaises(ValueError):
            constants_from_rows(rows, 0.002)


class TestInsensitivityCheck(unittest.TestCase):
    def test_consistent_magnitudes_pass(self):
        flags = FlagSet()
        r = assemble(SW, 5.43, _rows_from(151.4, 76.4, 56.4), (0.002, 0.005), flags)
        self.assertLess(r.spread, 1e-9)
        self.assertNotIn(Code.MD_ELASTIC_UNCONVERGED, flags.codes())

    def test_magnitude_dependence_is_flagged(self):
        rows = [r for r in _rows_from(151.4, 76.4, 56.4, (0.002,))]
        rows += [r for r in _rows_from(140.0, 76.4, 56.4, (0.005,))]
        flags = FlagSet()
        r = assemble(SW, 5.43, rows, (0.002, 0.005), flags)
        self.assertIn(Code.MD_ELASTIC_UNCONVERGED, flags.codes())
        self.assertAlmostEqual(r.c11, 151.4, places=6)   # smallest magnitude reported


class TestGateAndHandoff(unittest.TestCase):
    def _result(self, c11, c12, c44, a0=5.432):
        return ElasticResult("t", a0, c11, c12, c44, {}, potential="SiCGe.tersoff")

    def test_tersoff_silicon_fails_the_gate(self):
        """The measured SiCGe.tersoff Si(D) constants: -14 / +18 / -13 %."""
        report = gate(self._result(142.5, 75.4, 69.0), "Si")
        self.assertFalse(report.passes)
        self.assertGreater(abs(report.errors["c12"]), GATE_TOLERANCE)

    def test_reference_values_pass_their_own_gate(self):
        self.assertTrue(gate(self._result(165.7, 63.9, 79.6, 5.431), "Si").passes)

    def test_quantities_carry_md_provenance_and_the_potential(self):
        q = to_quantities(self._result(165.7, 63.9, 79.6), potential_passes_gate=True)
        self.assertIs(q["E"].provenance, Provenance.MD_COMPUTED)
        self.assertEqual(q["E"].citation, "SiCGe.tersoff")
        self.assertIn("Voigt-Reuss-Hill", q["E"].note)

    def test_failed_gate_caps_confidence_low(self):
        q = to_quantities(self._result(142.5, 75.4, 69.0), potential_passes_gate=False)
        self.assertEqual(q["E"].confidence, Confidence.LOW)
        self.assertEqual(q["nu"].confidence, Confidence.LOW)

    def test_passed_gate_allows_medium_and_no_higher(self):
        q = to_quantities(self._result(165.7, 63.9, 79.6), potential_passes_gate=True)
        self.assertEqual(q["E"].confidence, Confidence.MEDIUM)


@unittest.skipUnless(os.environ.get("OPENKAPITZA_LMP")
                     and os.environ.get("OPENKAPITZA_POTENTIALS"),
                     "set OPENKAPITZA_LMP and OPENKAPITZA_POTENTIALS to run")
class TestOnRealEngine(unittest.TestCase):
    def test_reproduces_published_stillinger_weber_constants(self):
        flags = FlagSet()
        r = run_elastic(Path(os.environ["OPENKAPITZA_LMP"]), SW,
                        Path(os.environ["OPENKAPITZA_POTENTIALS"]), flags)
        self.assertAlmostEqual(r.c11, 151.4, delta=0.2)
        self.assertAlmostEqual(r.c12, 76.4, delta=0.2)
        self.assertAlmostEqual(r.c44, 56.4, delta=0.2)    # relaxed; unrelaxed is ~117
        self.assertNotIn(Code.MD_ELASTIC_UNCONVERGED, flags.codes())


class MultiFilePotential(unittest.TestCase):
    """MEAM needs a library file and a parameter file, staged together."""

    def test_meam_pair_coeff_and_files(self):
        spec = CubicSpec("Al", "fcc", 4.05, 26.98, "meam", "library.meam", "AlS", None,
                         extra_files=("AlSiMgCuFe.meam",),
                         coeff_args="library.meam AlS SiS MgS CuS FeS AlSiMgCuFe.meam AlS")
        self.assertEqual(spec.files, ("library.meam", "AlSiMgCuFe.meam"))
        self.assertEqual(spec.pair_coeff(),
                         "pair_coeff      * * library.meam AlS SiS MgS CuS FeS "
                         "AlSiMgCuFe.meam AlS")

    def test_missing_extra_file_is_reported(self):
        import tempfile
        from openkapitza.md.elastic import stage_potential
        spec = CubicSpec("Al", "fcc", 4.05, 26.98, "meam", "library.meam", "AlS", None,
                         extra_files=("AlSiMgCuFe.meam",))
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            Path(src, "library.meam").write_text("x")
            with self.assertRaises(FileNotFoundError):
                stage_potential(spec, Path(src), Path(dst))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class FiniteTemperatureDeck(unittest.TestCase):
    """P5-2: thermal box, strain without minimisation, averaged stress."""

    def test_deck_structure(self):
        from openkapitza.md.elastic import CubicSpec, build_elastic_deck
        spec = CubicSpec("Si", "diamond", 5.431, 28.0855, "sw", "Si.sw", "Si", "Si")
        d = build_elastic_deck(spec, "xx", 0.01, temperature=300.0)
        self.assertIn("npt temp 300.00 300.00 0.1 iso 0.0 0.0 1.0", d)
        self.assertNotIn("minimize", d)
        self.assertNotIn("box/relax", d)
        i_strain = d.index("x scale 1.0100000000")
        i_reset = d.index("reset_timestep  0")
        i_avg = d.index("fix             sav all ave/time")
        self.assertLess(i_strain, i_reset); self.assertLess(i_reset, i_avg)
        self.assertIn("ELASTIC_STRESS xx 0.01000000 $(f_sav[1])", d)
        d0 = build_elastic_deck(spec, "xx", 0.01)
        self.assertIn("minimize", d0)                       # 0 K path unchanged

    def test_charge_style_for_variable_charge_potentials(self):
        from openkapitza.md.elastic import CubicSpec, build_elastic_deck
        spec = CubicSpec("Si(COMB)", "diamond", 5.431, 28.0855, "comb", "ffield.comb",
                         "Si", "Si", charge=True)
        for T in (None, 300.0):
            d = build_elastic_deck(spec, "yz", 0.005, temperature=T)
            self.assertIn("atom_style      charge", d)
            self.assertIn("set             group all charge 0.0", d)
