"""Anisotropic cell stress must not be averaged away.

Found while measuring the in-plane relaxation. On a relaxed Si/Ge cell the
NPT stage drove Pxx and Pyy from 6.53 GPa to 0.067 GPa -- but Pzz stayed at
1.51 GPa, because z is non-periodic (walls at both ends, so heat cannot
short-circuit the box) and cannot be barostatted. `Press` is (Pxx+Pyy+Pzz)/3,
so those components average to 0.55 GPa: under a 1 GPa threshold, no flag,
while 1.5 GPa of uniaxial stress sits along the TRANSPORT direction -- the
component h_K is most sensitive to.

Two further defects surfaced from the same measurement:

  - `check_run_health` averaged every thermo row in the log. With relaxation
    enabled that mixes minimisation rows at 6.5 GPa and 540 K with production
    rows at 1.5 GPa and 300 K, and reported 3.64 GPa for a cell sitting at
    1.51 GPa -- a state the run was never in.
  - The production `thermo_style` logged no pressure columns at all, so the
    stress check could only ever fire on the equilibration block, never where
    the measurement happens.
"""

from __future__ import annotations

import unittest

from openkapitza.md import NEMDGeometry, NEMDSpec, build_input_deck
from openkapitza.md.nemd import check_run_health, last_thermo_block
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code

GPA = 1e4          # bar per GPa, LAMMPS metal units


def _rows(n, *, step0=0, temp=300.0, pxx=0.0, pyy=0.0, pzz=0.0):
    return [
        {
            "Step": float(step0 + i),
            "Temp": temp,
            "Press": (pxx + pyy + pzz) / 3.0 * GPA,
            "Pxx": pxx * GPA, "Pyy": pyy * GPA, "Pzz": pzz * GPA,
        }
        for i in range(n)
    ]


class TestAnisotropicStress(unittest.TestCase):
    def test_isotropic_below_threshold_does_not_flag(self):
        flags = FlagSet()
        check_run_health(_rows(10, pxx=0.5, pyy=0.5, pzz=0.5), 300.0, flags)
        self.assertNotIn(Code.NEMD_CELL_STRESS, flags.codes())

    def test_anisotropic_hidden_by_the_mean_is_flagged(self):
        """The real case: mean 0.6 GPa, Pzz 1.8 GPa.

        The scalar-only check passed this. It must not.
        """
        flags = FlagSet()
        check_run_health(_rows(10, pxx=0.0, pyy=0.0, pzz=1.8), 300.0, flags)
        self.assertIn(Code.NEMD_CELL_STRESS, flags.codes())

    def test_flag_names_the_component_and_the_transport_direction(self):
        flags = FlagSet()
        check_run_health(_rows(10, pzz=1.8), 300.0, flags)
        flag = next(f for f in flags.flags if f.code == Code.NEMD_CELL_STRESS)
        self.assertIn("zz", flag.message)
        self.assertIn("transport direction", flag.message)
        self.assertIn("Phase-2", flag.remedy)

    def test_in_plane_residual_is_also_caught(self):
        """Not only z: a failed in-plane relaxation must flag too."""
        flags = FlagSet()
        check_run_health(_rows(10, pxx=2.0, pyy=2.0, pzz=0.0), 300.0, flags)
        self.assertIn(Code.NEMD_CELL_STRESS, flags.codes())

    def test_scalar_fallback_when_components_absent(self):
        """Older logs have Press only; that path must still work."""
        rows = [{"Step": float(i), "Temp": 300.0, "Press": 6.4 * GPA}
                for i in range(10)]
        flags = FlagSet()
        check_run_health(rows, 300.0, flags)
        self.assertIn(Code.NEMD_CELL_STRESS, flags.codes())

    def test_relaxed_cell_measurement_does_not_false_positive(self):
        """Pxx=Pyy=0.067, Pzz=1.51 -- flags, and for the right reason."""
        flags = FlagSet()
        check_run_health(_rows(10, pxx=0.067, pyy=0.065, pzz=1.51), 300.0, flags)
        flag = next(f for f in flags.flags if f.code == Code.NEMD_CELL_STRESS)
        self.assertIn("zz", flag.message)
        self.assertIn("1.51", flag.message)


class TestLastThermoBlock(unittest.TestCase):
    def test_splits_on_reset_timestep(self):
        rows = _rows(4, step0=0, temp=540.0) + _rows(3, step0=0, temp=300.0)
        block = last_thermo_block(rows)
        self.assertEqual(len(block), 3)
        self.assertTrue(all(r["Temp"] == 300.0 for r in block))

    def test_single_block_is_returned_whole(self):
        rows = _rows(6, step0=0)
        self.assertEqual(len(last_thermo_block(rows)), 6)

    def test_empty_is_safe(self):
        self.assertEqual(last_thermo_block([]), [])

    def test_health_check_ignores_the_relaxation_stage(self):
        """Minimisation at 6.5 GPa must not contaminate a relaxed production run."""
        rows = (_rows(5, step0=0, temp=540.0, pxx=6.5, pyy=6.5, pzz=6.5)
                + _rows(5, step0=0, temp=300.0, pxx=0.05, pyy=0.05, pzz=0.05))
        flags = FlagSet()
        check_run_health(rows, 300.0, flags)
        self.assertNotIn(Code.NEMD_CELL_STRESS, flags.codes())
        self.assertNotIn(Code.NEMD_UNEQUILIBRATED, flags.codes())

    def test_use_last_block_can_be_disabled(self):
        rows = (_rows(5, step0=0, temp=300.0, pzz=6.5)
                + _rows(5, step0=0, temp=300.0, pzz=0.05))
        flags = FlagSet()
        check_run_health(rows, 300.0, flags, use_last_block=False)
        self.assertIn(Code.NEMD_CELL_STRESS, flags.codes())


class TestProductionThermoLogsComponents(unittest.TestCase):
    def test_production_block_reports_the_stress_components(self):
        """Without these columns the check has nothing to test where it counts."""
        deck = build_input_deck(NEMDSpec(
            geometry=NEMDGeometry(), element_a="Si", element_b="Ge",
            lattice_a=5.431e-10, lattice_b=5.658e-10, temperature=300.0,
        ))
        production = next(l for l in deck.splitlines()
                          if l.startswith("thermo_style") and "f_hot" in l)
        for column in ("press", "pxx", "pyy", "pzz"):
            self.assertIn(column, production)


if __name__ == "__main__":
    unittest.main(verbosity=2)
