"""Mutation tests: prove the NEMD deck audit actually has teeth.

`scripts/audit_nemd.py` passes 288 deck configurations. That is only
reassuring if the audit would FAIL on a broken deck, and an audit whose
patterns do not match the deck text passes everything while checking nothing.
That is not hypothetical here: the first version of the audit silently missed
a swapped hot/cold setpoint because it compared `spec.t_hot` against
`spec.t_cold` -- two properties, checked against each other, never reading the
deck at all. A mutation test caught it.

So each test below injects one defect into a known-good deck and asserts the
audit reports something. Most of the injected defects are ones that actually
happened in this project and were found only when real LAMMPS read the deck.
"""

from __future__ import annotations

import re
import unittest
from unittest import mock

import scripts.audit_nemd as audit_mod
from openkapitza.md.nemd import NEMDGeometry, NEMDSpec, build_input_deck


def _spec() -> NEMDSpec:
    # cross_section deliberately not a whole multiple of the lattice constant,
    # so snapped and unsnapped areas differ and the area check can bite.
    return NEMDSpec(
        geometry=NEMDGeometry(cross_section=7.3e-9),
        element_a="Si", element_b="Ge",
        lattice_a=5.431e-10, lattice_b=5.658e-10,
        mass_a=28.0855, mass_b=72.63,
        pair_style="sw", pair_file="Si.sw",
        temperature=300.0,
    )


def _audit(deck: str) -> list[str]:
    """Run the audit against a supplied deck instead of a generated one."""
    spec = _spec()
    with mock.patch.object(audit_mod, "build_input_deck", lambda s, _d=deck: _d):
        return audit_mod.audit_one(spec, "mutation")


def _hot_region_sub(deck: str, fn) -> str:
    return re.sub(
        r"^region\s+hot\s+block INF INF INF INF (\S+) (\S+) units box",
        fn, deck, flags=re.MULTILINE,
    )


class TestAuditHasTeeth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = build_input_deck(_spec())

    def assertCaught(self, mutated: str, label: str):
        self.assertNotEqual(mutated, self.deck,
                            f"mutation {label!r} was a no-op -- it patched nothing, "
                            f"so the test proves nothing")
        self.assertTrue(_audit(mutated),
                        f"audit did NOT catch injected defect: {label}")

    # --- the control -----------------------------------------------------
    def test_clean_deck_passes(self):
        """Without this, every test below could pass by always failing."""
        self.assertEqual(_audit(self.deck), [])

    # --- defects that actually happened ---------------------------------
    def test_catches_unsnapped_flux_area(self):
        """The 4.4 % error straight into h_K."""
        g = _spec().geometry
        self.assertCaught(
            re.sub(r"(area_m2\s+)[0-9.eE+-]+", r"\g<1>%.6e" % g.area, self.deck),
            "unsnapped flux area")

    def test_catches_region_in_lattice_counts(self):
        """5 nm would have been built as 9 A."""
        self.assertCaught(
            _hot_region_sub(
                self.deck,
                lambda m: "region          hot      block INF INF INF INF "
                          f"{float(m.group(1)) / 5.431:.4f} "
                          f"{float(m.group(2)) / 5.431:.4f} units box"),
            "region written in lattice counts")

    def test_catches_ave_chunk_missing_chunk_id(self):
        # Patterned, not literal: the window follows spec.profile_frames, and a
        # literal string silently became a no-op when the default changed 4 -> 8.
        mutated = re.sub(r"(ave/chunk \d+ \d+ \d+) layers", r"\1", self.deck, count=1)
        self.assertCaught(mutated, "fix ave/chunk missing its chunk ID")

    def test_catches_window_outrunning_production(self):
        """Produced an empty profile file and exit code 0."""
        self.assertCaught(
            re.sub(r"^run\s+2000000\s*$", "run             100000",
                   self.deck, flags=re.MULTILINE),
            "averaging window outruns the production run")

    def test_catches_reset_timestep_removed(self):
        self.assertCaught(
            self.deck.replace("reset_timestep  0", "# reset_timestep  0"),
            "reset_timestep 0 commented out")

    # --- defects that would be silent and wrong -------------------------
    def test_catches_periodic_z(self):
        """Heat short-circuits around the box; h_K comes out meaningless."""
        self.assertCaught(
            self.deck.replace("boundary        p p f", "boundary        p p p"),
            "z made periodic")

    def test_catches_swapped_reservoir_setpoints(self):
        """The blind spot that motivated this file. Heat flows the wrong way."""
        mutated = (self.deck
                   .replace("hot  ghot  langevin 320.00 320.00",
                            "hot  ghot  langevin 280.00 280.00")
                   .replace("cold gcold langevin 280.00 280.00",
                            "cold gcold langevin 320.00 320.00"))
        self.assertCaught(mutated, "swapped hot/cold setpoints")

    def test_catches_ramping_thermostat(self):
        self.assertCaught(
            self.deck.replace("ghot  langevin 320.00 320.00",
                              "ghot  langevin 320.00 340.00"),
            "thermostat ramps instead of holding a setpoint")

    def test_catches_deck_setpoints_disagreeing_with_spec(self):
        self.assertCaught(
            self.deck.replace("ghot  langevin 320.00 320.00",
                              "ghot  langevin 350.00 350.00"),
            "deck setpoints disagree with the spec")

    def test_catches_langevin_overlapping_the_fit_window(self):
        """Folds the damping into the bulk gradient and biases the jump."""
        self.assertCaught(
            _hot_region_sub(
                self.deck,
                lambda m: "region          hot      block INF INF INF INF "
                          f"{m.group(1)} {float(m.group(2)) + 40.0:.4f} units box"),
            "hot Langevin region overlapping the fit window")

    def test_catches_velocity_create_at_one_times_setpoint(self):
        """Equipartition halves it; seeding at 1x starts the run cold."""
        self.assertCaught(
            self.deck.replace("create 600.00", "create 300.00"),
            "velocity create at 1x the setpoint")


class TestBarostatAndRelaxChecks(unittest.TestCase):
    """The spot-pressure block barostatted z, which is non-periodic here.

    LAMMPS rejects that outright ("Cannot use fix npt on a non-periodic z
    dimension"), and the block had never been run because nothing set
    `spot_pressure_GPa` -- including every configuration in the audit's own
    grid until it was widened. Proven against a real LAMMPS build.
    """

    @classmethod
    def setUpClass(cls):
        cls.deck = build_input_deck(_spec())

    def assertCaught(self, mutated: str, label: str):
        self.assertNotEqual(mutated, self.deck, f"mutation {label!r} was a no-op")
        self.assertTrue(_audit(mutated), f"audit did NOT catch: {label}")

    def test_catches_npt_barostatting_non_periodic_z(self):
        self.assertCaught(
            self.deck.replace(
                "couple xy x 0.0 0.0 1.0000 y 0.0 0.0 1.0000",
                "z 60000.00 60000.00 1.0000"),
            "npt barostats non-periodic z")

    def test_catches_npt_coupling_non_periodic_z(self):
        self.assertCaught(self.deck.replace("couple xy", "couple xyz"),
                          "npt couples z")

    def test_catches_relax_fix_left_running(self):
        """An un-unfixed barostat would dilate the box through production."""
        self.assertCaught(
            self.deck.replace("unfix           relax", "# unfix relax"),
            "relax fix never unfixed")

    def test_catches_missing_minimize(self):
        self.assertCaught(
            re.sub(r"^minimize\s.*$", "# minimize", self.deck, flags=re.MULTILINE),
            "minimize removed while relax_structure=True")

    def test_catches_relaxation_after_the_reservoirs(self):
        """Relaxing with the gradient already imposed measures nothing."""
        self.assertCaught(
            self.deck.replace("fix             relax", "fix             zzrelax")
                     .replace("unfix           relax", "unfix           zzrelax"),
            "relaxation moved after the reservoirs")

    def test_spot_pressure_refuses_instead_of_emitting_a_bad_deck(self):
        spec = _spec()
        spec.spot_pressure_GPa = 6.0
        with self.assertRaises(NotImplementedError):
            build_input_deck(spec)

    def test_relax_can_be_turned_off(self):
        spec = _spec()
        spec.relax_structure = False
        deck = build_input_deck(spec)
        self.assertNotRegex(deck, r"(?m)^minimize\s")
        self.assertNotRegex(deck, r"(?m)^fix\s+relax\s")
        self.assertEqual(audit_mod.audit_one(spec, "no-relax"), [])


class TestAuditSweepIsClean(unittest.TestCase):
    def test_sweep_of_configurations_passes(self):
        """A smaller version of the script's own grid, kept in the suite."""
        for cross_nm in (3.0, 5.0, 7.3):
            for bulk_nm in (6.0, 20.0):
                for bin_nm in (0.3, 0.5):
                    spec = NEMDSpec(
                        geometry=NEMDGeometry(
                            bulk_a_length=bulk_nm * 1e-9,
                            bulk_b_length=bulk_nm * 1e-9,
                            cross_section=cross_nm * 1e-9,
                            bin_width=bin_nm * 1e-9,
                        ),
                        element_a="Si", element_b="Ge",
                        lattice_a=5.431e-10, lattice_b=5.658e-10,
                        mass_a=28.0855, mass_b=72.63,
                        pair_style="sw", pair_file="Si.sw",
                        temperature=300.0,
                    )
                    label = f"cross={cross_nm} bulk={bulk_nm} bin={bin_nm}"
                    self.assertEqual(audit_mod.audit_one(spec, label), [], label)


if __name__ == "__main__":
    unittest.main(verbosity=2)
