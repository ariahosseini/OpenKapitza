"""One run directory in, one h_K with an honest error out (md.measure).

Pins: all frames enter the profile; per-frame block estimates use each
frame's own flux window; the reported error is max(fit, block); and
NEMD_ERROR_UNDERSTATED fires when frame-to-frame scatter exceeds the fit's
claim -- the fit treats bins and time as independent, which they are not.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from openkapitza.md.measure import frame_fluxes, measure_kapitza
from openkapitza.md.nemd import NEMDGeometry, NEMDSpec
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code

EV = 1.602176634e-19


def _spec(frames=8):
    g = NEMDGeometry(wall_thickness=1e-9, reservoir_thickness=2e-9,
                     bulk_a_length=20e-9, bulk_b_length=20e-9,
                     cross_section=5.0e-9, bin_width=0.5e-9)
    return NEMDSpec(geometry=g, element_a="Si", element_b="Ge",
                    lattice_a=5.431e-10, lattice_b=5.658e-10, temperature=300.0,
                    production_steps=800_000, profile_every=100,
                    profile_frames=frames)


def _write_run(wd: Path, spec: NEMDSpec, *, dT=5.0, k_a=50.0, k_b=30.0,
               frame_noise=0.0, bin_noise=0.0, seed=1, step_zero=False, ramp=0.0):
    """Planted-jump profile per frame plus linear, balanced reservoir tallies."""
    rng = np.random.default_rng(seed)
    g = spec.geometry
    area = spec.snapped_area
    n = g.n_bins
    z = (np.arange(n) + 0.5) * g.bin_width
    dt_s = spec.timestep_fs * 1e-15
    q = 2.0e9                                        # W/m^2
    rate_eV_per_step = q * area * dt_s / EV
    window = spec.production_steps // spec.profile_frames
    with open(wd / "temperature_profile.dat", "w") as fh:
        fh.write("# a\n# b\n# c\n")
        for k in range(spec.profile_frames):
            jump = dT * (1.0 + frame_noise * rng.standard_normal())
            T = np.where(z < g.interface_z,
                         320.0 - (q / k_a) * (z - g.z_bulk_a_start),
                         320.0 - (q / k_a) * (g.interface_z - g.z_bulk_a_start)
                         - jump - (q / k_b) * (z - g.interface_z))
            T = T + bin_noise * rng.standard_normal(n)
            fh.write(f"{(k + 1) * window} {n} {n * 80}\n")
            for i, (zz, tt) in enumerate(zip(z, T), start=1):
                fh.write(f"{i} {zz * 1e10:.6f} 80 {tt:.6f}\n")
    with open(wd / "reservoir_energy.dat", "w") as fh:
        fh.write("# time[ps] E_hot[eV] E_cold[eV]\n")
        first = 0 if step_zero else spec.profile_every
        P = spec.production_steps
        for step in range(first, P + 1, spec.profile_every):
            # Offset `time`: alignment must not depend on it.
            t_ps = 5000.0 + step * spec.timestep_fs * 1e-3
            # Integral of rate*(1 + ramp*(s/P - 0.5)) ds from 0 to step.
            e = rate_eV_per_step * (step + ramp * (step * step / (2 * P) - 0.5 * step))
            fh.write(f"{t_ps} {-e} {e}\n")
    return q


class Measurement(unittest.TestCase):
    def setUp(self):
        self.wd = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.wd, ignore_errors=True)

    def test_clean_run_recovers_planted_h(self):
        spec = _spec()
        q = _write_run(self.wd, spec, dT=5.0)
        m = measure_kapitza(self.wd, spec, FlagSet())
        self.assertAlmostEqual(m.h_K / (q / 5.0), 1.0, places=4)
        self.assertEqual(len(m.per_frame_h_K), 8)
        for h in m.per_frame_h_K:
            self.assertAlmostEqual(h / (q / 5.0), 1.0, places=4)

    def test_frame_flux_alignment_ignores_absolute_time(self):
        spec = _spec()
        q = _write_run(self.wd, spec)
        window = spec.production_steps // spec.profile_frames
        ends = [(k + 1) * window for k in range(spec.profile_frames)]
        fl = frame_fluxes(self.wd / "reservoir_energy.dat", ends, window,
                          spec.timestep_fs, spec.snapped_area, spec.profile_every,
                          spec.production_steps)
        np.testing.assert_allclose(fl, q, rtol=1e-6)

    def test_alignment_with_a_step_zero_line(self):
        """Real LAMMPS writes '0 0 0' at production start; both forms must agree."""
        spec = _spec()
        q = _write_run(self.wd, spec, step_zero=True)
        window = spec.production_steps // spec.profile_frames
        ends = [(k + 1) * window for k in range(spec.profile_frames)]
        fl = frame_fluxes(self.wd / "reservoir_energy.dat", ends, window,
                          spec.timestep_fs, spec.snapped_area, spec.profile_every,
                          spec.production_steps)
        np.testing.assert_allclose(fl, q, rtol=1e-6)

    def test_ramping_power_is_attributed_to_the_right_frame(self):
        """Power rising in time: frame k must get the mean power of ITS window."""
        spec = _spec()
        ramp = 0.5
        q = _write_run(self.wd, spec, ramp=ramp, step_zero=True)
        window = spec.production_steps // spec.profile_frames
        ends = [(k + 1) * window for k in range(spec.profile_frames)]
        fl = frame_fluxes(self.wd / "reservoir_energy.dat", ends, window,
                          spec.timestep_fs, spec.snapped_area, spec.profile_every,
                          spec.production_steps)
        P = spec.production_steps
        expect = [q * (1 + ramp * ((e - window / 2) / P - 0.5)) for e in ends]
        np.testing.assert_allclose(fl, expect, rtol=1e-6)

    def test_frame_scatter_beyond_fit_error_is_flagged_and_used(self):
        spec = _spec()
        _write_run(self.wd, spec, dT=5.0, frame_noise=0.15, bin_noise=0.002, seed=3)
        flags = FlagSet()
        m = measure_kapitza(self.wd, spec, flags)
        self.assertIn(Code.NEMD_ERROR_UNDERSTATED, flags.codes())
        self.assertGreater(m.block_se, 2 * m.fit_se)
        self.assertEqual(m.uncertainty, m.block_se)
        # The block error should be about 15 % / sqrt(8) of h.
        self.assertAlmostEqual(m.block_se / m.h_K, 0.15 / math.sqrt(8), delta=0.03)

    def test_too_few_frames_says_so(self):
        spec = _spec(frames=2)
        _write_run(self.wd, spec)
        flags = FlagSet()
        m = measure_kapitza(self.wd, spec, flags)
        self.assertIsNone(m.block_se)
        self.assertIn(Code.NEMD_ERROR_UNDERSTATED, flags.codes())


if __name__ == "__main__":
    unittest.main()
