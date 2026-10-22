"""Section 11 of the agent instruction, enforced: when MD may stand in.

`openkapitza/md/fill_policy.py` decides whether an MD-computed number may fill
a missing continuum input, may only be compared with a measurement, or must be
refused. Every calculator in work package 5 routes through it, so these tests
pin the decisions that matter most:

  - MD never silently overwrites a measurement.
  - MD may not fill total thermal conductivity for a metal (no electrons).
  - MD nanohardness can never pose as the microhardness Song-Yovanovich needs.
  - An MD accommodation coefficient DOES beat the generic default, without a
    special case, because the default carries CORRELATION_DEFAULT provenance.
  - A requested MD conductivity for a metal still carries the phonon-only
    warning. This one was a gap found in review: the request branch used to
    return the value silently.
"""

from __future__ import annotations

import unittest

from openkapitza.md.fill_policy import (
    POLICIES, Mode, cap_md, md_eligible, policy_for, select,
)
from openkapitza.schema import FlagSet, Quantity, default_accommodation
from openkapitza.schema.flags import Code
from openkapitza.schema.quantities import Confidence, Provenance


def q(value, unit="-", prov=Provenance.USER_MEASURED, conf=None, cite=None):
    return Quantity.parse(value, unit, provenance=prov, confidence=conf,
                          citation=cite)


def md(value, unit="-", conf=Confidence.HIGH):
    # Deliberately HIGH so the tests prove the cap, not the default, limits it.
    return q(value, unit, Provenance.MD_COMPUTED, conf, cite="test.potential")


class TestPolicyTable(unittest.TestCase):
    def test_material_properties_are_computable(self):
        for key in ("accommodation", "h_K", "elastic", "thermal_conductivity",
                    "nanohardness"):
            self.assertTrue(policy_for(key).computable, key)

    def test_surface_finish_and_test_conditions_are_not(self):
        for key in ("roughness", "slope", "pressure", "gas_pressure",
                    "temperature", "microhardness"):
            self.assertFalse(policy_for(key).computable, key)

    def test_every_policy_states_a_reason(self):
        for key, pol in POLICIES.items():
            self.assertTrue(pol.reason.strip(), f"{key} has no reason")

    def test_unknown_key_is_an_error_not_a_default(self):
        with self.assertRaises(KeyError):
            policy_for("viscosity_of_unicorns")


class TestEligibility(unittest.TestCase):
    def test_metal_conductivity_is_not_fillable(self):
        ok, why = md_eligible("thermal_conductivity", is_metal=True)
        self.assertFalse(ok)
        self.assertIn("phonon", why)

    def test_nonmetal_conductivity_is_fillable(self):
        self.assertTrue(md_eligible("thermal_conductivity", is_metal=False)[0])

    def test_nanohardness_never_fills(self):
        self.assertFalse(md_eligible("nanohardness")[0])

    def test_h_K_is_an_output_not_a_fill(self):
        self.assertFalse(md_eligible("h_K")[0])


class TestConfidenceCaps(unittest.TestCase):
    def test_elastic_capped_low_without_a_passed_gate(self):
        self.assertEqual(cap_md("elastic", md(200.0, "GPa")).confidence,
                         Confidence.LOW)

    def test_elastic_capped_low_when_gate_failed(self):
        capped = cap_md("elastic", md(200.0, "GPa"), potential_passes_gate=False)
        self.assertEqual(capped.confidence, Confidence.LOW)

    def test_elastic_may_reach_medium_only_when_gate_passed(self):
        capped = cap_md("elastic", md(200.0, "GPa"), potential_passes_gate=True)
        self.assertEqual(capped.confidence, Confidence.MEDIUM)

    def test_accommodation_capped_at_medium(self):
        self.assertEqual(cap_md("accommodation", md(0.5)).confidence,
                         Confidence.MEDIUM)

    def test_cap_never_raises_confidence(self):
        low = md(0.5, conf=Confidence.LOW)
        self.assertEqual(cap_md("accommodation", low).confidence, Confidence.LOW)


class TestFillMode(unittest.TestCase):
    def test_measurement_beats_md(self):
        flags = FlagSet()
        sel = select("accommodation", [q(0.8), md(0.5)], flags)
        self.assertEqual(sel.source, Provenance.USER_MEASURED)
        self.assertNotIn(Code.MD_FILLED, flags.codes())

    def test_md_beats_the_real_accommodation_default(self):
        """The 'no special case needed' claim, checked against the real default."""
        flags = FlagSet()
        default = default_accommodation("Ar", "Ni200", FlagSet())
        self.assertIs(default.provenance, Provenance.CORRELATION_DEFAULT)
        sel = select("accommodation", [default, md(0.45)], flags)
        self.assertIs(sel.source, Provenance.MD_COMPUTED)
        self.assertAlmostEqual(sel.value.si, 0.45)
        self.assertIn(Code.MD_FILLED, flags.codes())

    def test_literature_beats_md(self):
        flags = FlagSet()
        lit = q(150.0, "GPa", Provenance.LITERATURE)
        sel = select("elastic", [lit, md(140.0, "GPa")], flags)
        self.assertIs(sel.source, Provenance.LITERATURE)

    def test_md_k_for_a_metal_is_refused_and_flagged(self):
        flags = FlagSet()
        sel = select("thermal_conductivity", [md(5.0, "W/mK")], flags,
                     is_metal=True)
        self.assertIsNone(sel.value)
        self.assertIn(Code.MD_FILL_REFUSED, flags.codes())

    def test_md_k_for_a_nonmetal_fills_at_low(self):
        flags = FlagSet()
        sel = select("thermal_conductivity", [md(150.0, "W/mK")], flags)
        self.assertIs(sel.source, Provenance.MD_COMPUTED)
        self.assertEqual(sel.value.confidence, Confidence.LOW)

    def test_md_nanohardness_cannot_be_offered_as_microhardness(self):
        with self.assertRaises(ValueError):
            select("microhardness", [md(20.0, "GPa")], FlagSet())

    def test_fill_flag_names_the_potential(self):
        flags = FlagSet()
        select("accommodation", [md(0.5)], flags)
        flag = next(f for f in flags.flags if f.code == Code.MD_FILLED)
        self.assertIn("test.potential", flag.message)


class TestCompareMode(unittest.TestCase):
    def test_measurement_is_used_md_is_reported(self):
        flags = FlagSet()
        sel = select("elastic", [q(160.0, "GPa"), md(140.0, "GPa")], flags,
                     mode=Mode.COMPARE)
        self.assertIs(sel.source, Provenance.USER_MEASURED)
        self.assertAlmostEqual(sel.ratio_md_to_measured, 140.0 / 160.0)

    def test_large_disagreement_is_flagged(self):
        flags = FlagSet()
        select("elastic", [q(160.0, "GPa"), md(80.0, "GPa")], flags,
               mode=Mode.COMPARE)
        self.assertIn(Code.MD_DISAGREES_WITH_MEASUREMENT, flags.codes())

    def test_compare_without_a_measurement_is_an_error(self):
        with self.assertRaises(ValueError):
            select("elastic", [md(140.0, "GPa")], FlagSet(), mode=Mode.COMPARE)


class TestRequestMode(unittest.TestCase):
    def test_request_overrides_a_measurement_but_flags_it(self):
        flags = FlagSet()
        sel = select("elastic", [q(160.0, "GPa"), md(140.0, "GPa")], flags,
                     mode=Mode.REQUEST)
        self.assertIs(sel.source, Provenance.MD_COMPUTED)
        self.assertIn(Code.MD_OVERRIDES_MEASUREMENT, flags.codes())

    def test_requested_metal_k_still_carries_the_phonon_warning(self):
        """Gap found in review: this used to come back with no warning."""
        flags = FlagSet()
        sel = select("thermal_conductivity", [md(5.0, "W/mK")], flags,
                     mode=Mode.REQUEST, is_metal=True)
        self.assertIs(sel.source, Provenance.MD_COMPUTED)
        self.assertIn(Code.MD_K_METAL_PHONON_ONLY, flags.codes())

    def test_request_without_md_is_an_error(self):
        with self.assertRaises(ValueError):
            select("elastic", [q(160.0, "GPa")], FlagSet(), mode=Mode.REQUEST)


if __name__ == "__main__":
    unittest.main(verbosity=2)
