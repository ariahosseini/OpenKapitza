"""Phase-1 tests: NEMD geometry, deck generation, extraction, compute layer.

LAMMPS is not available in CI, so nothing here runs a simulation. Everything
else is exercised: the extraction is validated against synthetic profiles
whose jump is known by construction, the deck is checked directive by
directive, the parsers round-trip against LAMMPS-format files written here,
and the job runners are driven through real success and failure paths.

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from openkapitza.compute import (
    CostEstimate,
    JobHandle,
    JobSpec,
    JobState,
    LocalRunner,
    Resources,
    SlurmRunner,
    check_backend_fit,
    estimate_cost,
    resources_for,
    select_runner,
)
from openkapitza.md.convergence import (
    ConvergenceSeries,
    check_flux_linearity,
    check_series,
    richardson_extrapolate,
)
from openkapitza.md.nemd import (
    NEMDGeometry,
    NEMDSpec,
    build_input_deck,
    check_spec,
    parse_profile,
    parse_reservoir_energy,
)
from openkapitza.md.profile import (
    TemperatureProfile,
    extract_kapitza,
    flux_from_reservoirs,
    naive_adjacent_bin_delta_T,
)
from openkapitza.schema import Code, FlagSet, Severity


# ---------------------------------------------------------------------------
# Synthetic profile with a known answer
# ---------------------------------------------------------------------------

def synth_profile(
    *, delta_T=8.0, k_a=148.0, k_b=60.0, flux=2.0e9, L=40e-9, n_bins=160,
    noise=0.0, bend_amp=0.0, bend_len=3e-9, reservoir_bins=16, seed=0,
):
    """Profile with a planted jump, optionally contaminated like a real run."""
    rng = np.random.default_rng(seed)
    z = np.linspace(0, 2 * L, n_bins, endpoint=False) + L / n_bins
    interface = L
    T = np.empty_like(z)
    a = z < interface
    T_a_int = 310.0
    T[a] = T_a_int + (flux / k_a) * (interface - z[a])
    T[~a] = (T_a_int - delta_T) - (flux / k_b) * (z[~a] - interface)

    if bend_amp:                       # non-equilibrium curvature near interface
        d = np.abs(z - interface)
        T += np.where(a, 1.0, -1.0) * bend_amp * np.exp(-d / bend_len)
    if reservoir_bins:                 # thermostatted ends, pinned flat
        T[:reservoir_bins] = T[reservoir_bins] + 1.5
        T[-reservoir_bins:] = T[-reservoir_bins - 1] - 1.5
    if noise:
        T = T + rng.normal(0.0, noise, T.shape)

    dz = 2 * L / n_bins
    bulk_a = (reservoir_bins * dz, interface)
    bulk_b = (interface, 2 * L - reservoir_bins * dz)
    prof = TemperatureProfile(z=z, T=T, interface_z=interface, area=(5e-9) ** 2)
    return prof, bulk_a, bulk_b


class TestExtractionGroundTruth(unittest.TestCase):
    """The planted jump must come back out."""

    def test_clean_profile_recovers_exactly(self):
        p, ba, bb = synth_profile()
        f = FlagSet()
        r = extract_kapitza(p, 2e9, f, bulk_a=ba, bulk_b=bb)
        self.assertAlmostEqual(r.delta_T.si, 8.0, places=6)
        self.assertAlmostEqual(r.h_K.si, 2e9 / 8.0, places=0)
        self.assertEqual(f.codes(), set())

    def test_recovers_under_noise(self):
        for seed in range(5):
            p, ba, bb = synth_profile(noise=0.3, seed=seed)
            f = FlagSet()
            r = extract_kapitza(p, 2e9, f, bulk_a=ba, bulk_b=bb)
            self.assertAlmostEqual(r.delta_T.si / 8.0, 1.0, delta=0.10)

    def test_tolerates_near_interface_curvature(self):
        p, ba, bb = synth_profile(bend_amp=1.5, noise=0.2)
        f = FlagSet()
        r = extract_kapitza(p, 2e9, f, bulk_a=ba, bulk_b=bb)
        self.assertAlmostEqual(r.delta_T.si / 8.0, 1.0, delta=0.10)

    def test_beats_the_naive_adjacent_bin_estimate(self):
        """The whole reason the extrapolation exists."""
        p, ba, bb = synth_profile(bend_amp=1.5, noise=0.2)
        f = FlagSet()
        r = extract_kapitza(p, 2e9, f, bulk_a=ba, bulk_b=bb)
        fit_err = abs(r.delta_T.si - 8.0) / 8.0
        naive_err = abs(naive_adjacent_bin_delta_T(p) - 8.0) / 8.0
        self.assertLess(fit_err, naive_err / 3.0,
                        "extrapolated fit should be far closer than adjacent bins")

    def test_naive_error_grows_with_flux_but_fit_error_does_not(self):
        """The naive bias is proportional to q, so it mimics flux dependence.

        This is what makes it dangerous: it survives a careless convergence
        check by looking like real physics.
        """
        fit_errs, naive_errs = [], []
        for q in (5e8, 4e9):
            p, ba, bb = synth_profile(flux=q, bend_amp=1.0, noise=0.1)
            f = FlagSet()
            r = extract_kapitza(p, q, f, bulk_a=ba, bulk_b=bb)
            fit_errs.append(abs(r.delta_T.si - 8.0) / 8.0)
            naive_errs.append(abs(naive_adjacent_bin_delta_T(p) - 8.0) / 8.0)
        # the fit's error is essentially flux independent
        self.assertAlmostEqual(fit_errs[0], fit_errs[1], delta=0.02)
        # the naive error is large at both
        self.assertGreater(min(naive_errs), 0.10)

    def test_small_jump_is_harder_and_says_so(self):
        p, ba, bb = synth_profile(delta_T=0.5, noise=0.3, seed=11)
        f = FlagSet()
        r = extract_kapitza(p, 2e9, f, bulk_a=ba, bulk_b=bb)
        self.assertIsNotNone(r.delta_T.uncertainty)
        self.assertGreater(r.delta_T.uncertainty, 0.0)


class TestExtractionGuards(unittest.TestCase):
    def test_guessed_fit_window_is_flagged(self):
        p, _, _ = synth_profile()
        f = FlagSet()
        extract_kapitza(p, 2e9, f)
        self.assertIn(Code.NEMD_FIT_WINDOW_GUESSED, f.codes())

    def test_declared_window_is_not_flagged(self):
        p, ba, bb = synth_profile()
        f = FlagSet()
        extract_kapitza(p, 2e9, f, bulk_a=ba, bulk_b=bb)
        self.assertNotIn(Code.NEMD_FIT_WINDOW_GUESSED, f.codes())

    def test_reservoir_bins_left_in_the_fit_bias_the_jump_low(self):
        """Declaring a bulk range that overlaps the reservoir must not pass."""
        p, ba, bb = synth_profile(reservoir_bins=30)
        bad_a = (8 * (80e-9 / 160), ba[1])       # claims a much thinner reservoir
        bad_b = (bb[0], 80e-9 - 8 * (80e-9 / 160))
        f = FlagSet()
        r = extract_kapitza(p, 2e9, f, bulk_a=bad_a, bulk_b=bad_b)
        self.assertLess(r.delta_T.si, 8.0, "pinned bins drag the jump down")
        self.assertLess(r.fit_a.r2, 0.99)

    def test_inverted_jump_is_refused(self):
        p, ba, bb = synth_profile(delta_T=-5.0)
        f = FlagSet()
        with self.assertRaises(ValueError):
            extract_kapitza(p, 2e9, f, bulk_a=ba, bulk_b=bb)
        self.assertIn(Code.NEMD_NONPHYSICAL_JUMP, f.codes())

    def test_negative_flux_rejected(self):
        p, ba, bb = synth_profile()
        with self.assertRaises(ValueError):
            extract_kapitza(p, -1.0, FlagSet(), bulk_a=ba, bulk_b=bb)

    def test_size_limited_conductivity_is_flagged(self):
        p, ba, bb = synth_profile(k_a=148.0)
        f = FlagSet()
        extract_kapitza(p, 2e9, f, bulk_a=ba, bulk_b=bb, k_expected_a=10.0)
        self.assertIn(Code.NEMD_CONDUCTIVITY_MISMATCH, f.codes())

    def test_profile_validation(self):
        with self.assertRaises(ValueError):      # too few bins
            TemperatureProfile(z=np.arange(4.0), T=np.ones(4), interface_z=2.0, area=1.0)
        with self.assertRaises(ValueError):      # interface outside
            TemperatureProfile(z=np.arange(40.0), T=np.ones(40),
                               interface_z=100.0, area=1.0)
        with self.assertRaises(ValueError):      # not increasing
            TemperatureProfile(z=np.arange(40.0)[::-1].copy(), T=np.ones(40),
                               interface_z=20.0, area=1.0)


class TestFluxFromReservoirs(unittest.TestCase):
    def test_balanced_reservoirs(self):
        f = FlagSet()
        q = flux_from_reservoirs(1.0e-12, 1.0e-12, 1.0e-9, 25e-18, f)
        self.assertAlmostEqual(q, 1.0e-12 / (1.0e-9 * 25e-18))
        self.assertEqual(f.codes(), set())

    def test_imbalance_is_flagged(self):
        f = FlagSet()
        flux_from_reservoirs(1.0e-12, 1.5e-12, 1.0e-9, 25e-18, f)
        self.assertIn(Code.NEMD_FLUX_INCONSISTENT, f.codes())

    def test_zero_duration_rejected(self):
        with self.assertRaises(ValueError):
            flux_from_reservoirs(1.0, 1.0, 0.0, 1.0, FlagSet())


# ---------------------------------------------------------------------------
# Geometry and deck
# ---------------------------------------------------------------------------

class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.g = NEMDGeometry()

    def test_layout_is_ordered_and_consistent(self):
        g = self.g
        self.assertLess(g.z_hot_start, g.z_bulk_a_start)
        self.assertLess(g.z_bulk_a_start, g.interface_z)
        self.assertLess(g.interface_z, g.z_bulk_b_end)
        self.assertLess(g.z_bulk_b_end, g.z_cold_end)
        self.assertLess(g.z_cold_end, g.total_length)

    def test_bulk_ranges_exclude_the_reservoirs(self):
        """The footgun this geometry exists to close."""
        g = self.g
        self.assertGreaterEqual(g.bulk_a[0], g.z_bulk_a_start)
        self.assertLessEqual(g.bulk_b[1], g.z_bulk_b_end)
        self.assertEqual(g.bulk_a[1], g.interface_z)
        self.assertEqual(g.bulk_b[0], g.interface_z)

    def test_atom_estimate_scales_with_volume(self):
        g1 = NEMDGeometry()
        g2 = NEMDGeometry(bulk_a_length=40e-9, bulk_b_length=40e-9)
        n1 = g1.estimate_atoms(5e28, 5e28)
        n2 = g2.estimate_atoms(5e28, 5e28)
        self.assertGreater(n2, n1)


class TestDeck(unittest.TestCase):
    def setUp(self):
        self.spec = NEMDSpec(
            geometry=NEMDGeometry(), element_a="Si", element_b="Ge",
            lattice_a=5.431e-10, lattice_b=5.658e-10, temperature=300.0,
        )

    def test_transport_direction_is_non_periodic(self):
        """Periodic z would let heat short-circuit around the box."""
        self.assertIn("boundary        p p f", build_input_deck(self.spec))

    def test_reservoirs_are_langevin_with_energy_tallied(self):
        deck = build_input_deck(self.spec)
        self.assertIn("langevin", deck)
        self.assertEqual(deck.count("tally yes"), 4)   # hot+cold, twice

    def test_hot_and_cold_setpoints_straddle_the_mean(self):
        deck = build_input_deck(self.spec)
        self.assertIn("320.00", deck)
        self.assertIn("280.00", deck)

    def test_bulk_runs_nve_not_thermostatted(self):
        deck = build_input_deck(self.spec)
        self.assertIn("fix             nve gmobile nve", deck)

    def test_walls_are_frozen(self):
        self.assertIn("setforce 0.0 0.0 0.0", build_input_deck(self.spec))

    def test_spot_pressure_block_only_when_requested(self):
        self.assertNotIn("npt", build_input_deck(self.spec))
        self.spec.spot_pressure_GPa = 7.6
        deck = build_input_deck(self.spec)
        self.assertIn("npt", deck)
        self.assertIn("76000.00", deck)       # 7.6 GPa in bar

    def test_profile_binning_matches_geometry(self):
        deck = build_input_deck(self.spec)
        self.assertIn(f"bin/1d z lower {self.spec.geometry.bin_width * 1e10:.4f}", deck)

    def test_build_mode_is_self_contained(self):
        """The deck must run without an externally supplied structure file."""
        deck = build_input_deck(self.spec)
        self.assertIn("create_box", deck)
        self.assertIn("create_atoms", deck)
        self.assertNotIn("read_data", deck)

    def test_read_data_mode_still_available(self):
        self.spec.structure_mode = "read_data"
        deck = build_input_deck(self.spec)
        self.assertIn("read_data       interface.data", deck)
        self.assertNotIn("create_atoms", deck)

    def test_unknown_structure_mode_rejected(self):
        self.spec.structure_mode = "telepathy"
        with self.assertRaises(ValueError):
            build_input_deck(self.spec)

    def test_periodic_cross_section_snaps_to_whole_unit_cells(self):
        """x and y are periodic; a partial cell leaves a seam at the image."""
        a = self.spec.lattice_used
        n = self.spec.snapped_cross_section / a
        self.assertAlmostEqual(n, round(n), places=9)

    def test_flux_area_follows_the_snapped_cross_section(self):
        """Using the requested area instead would scale h_K by the ratio."""
        self.assertNotAlmostEqual(self.spec.snapped_area,
                                  self.spec.geometry.area, places=20)
        deck = build_input_deck(self.spec)
        self.assertIn(f"area_m2 {self.spec.snapped_area:.6e}", deck)

    def test_lattice_strain_reported_for_mismatched_pair(self):
        sa, sb = self.spec.lattice_strain
        self.assertAlmostEqual(sa, 0.0, places=9)          # A sets the lattice
        self.assertLess(sb, -0.03)                          # Ge compressed ~4 %

    def test_strained_build_is_flagged(self):
        f = FlagSet()
        check_spec(self.spec, f)
        self.assertIn(Code.NEMD_LATTICE_STRAIN, f.codes())
        # strain and size limitation are different problems with different
        # remedies, so they must not share a code
        self.assertNotIn(Code.NEMD_SIZE_LIMITED, f.codes())

    def test_common_lattice_override(self):
        self.spec.common_lattice = 5.545e-10                # midpoint
        sa, sb = self.spec.lattice_strain
        self.assertGreater(sa, 0.0)
        self.assertLess(sb, 0.0)

    def test_timestep_converted_to_picoseconds(self):
        self.assertIn("timestep        0.001000", build_input_deck(self.spec))


class TestSpecChecks(unittest.TestCase):
    def test_short_slabs_flagged_as_size_limited(self):
        f = FlagSet()
        check_spec(NEMDSpec(geometry=NEMDGeometry(bulk_a_length=5e-9,
                                                  bulk_b_length=5e-9),
                            element_a="Si", element_b="Ge", lattice_a=5.4e-10,
                            lattice_b=5.6e-10, temperature=300.0), f)
        self.assertIn(Code.NEMD_SIZE_LIMITED, f.codes())

    def test_excessive_reservoir_split_flagged(self):
        f = FlagSet()
        check_spec(NEMDSpec(geometry=NEMDGeometry(), element_a="Si",
                            element_b="Ge", lattice_a=5.4e-10, lattice_b=5.6e-10,
                            temperature=300.0, delta_reservoir=200.0), f)
        self.assertIn(Code.NEMD_FLUX_NONLINEAR, f.codes())

    def test_reasonable_spec_raises_only_the_strain_flag(self):
        """A Si/Ge build cell legitimately carries ~4 % strain and says so."""
        f = FlagSet()
        check_spec(NEMDSpec(geometry=NEMDGeometry(), element_a="Si",
                            element_b="Ge", lattice_a=5.4e-10, lattice_b=5.6e-10,
                            temperature=300.0), f)
        self.assertEqual(f.codes(), {Code.NEMD_LATTICE_STRAIN})

    def test_lattice_matched_pair_is_entirely_clean(self):
        f = FlagSet()
        check_spec(NEMDSpec(geometry=NEMDGeometry(), element_a="Si",
                            element_b="Si", lattice_a=5.431e-10,
                            lattice_b=5.431e-10, temperature=300.0), f)
        self.assertEqual(f.codes(), set())


class TestParsers(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_profile_round_trip_and_extraction(self):
        """Write LAMMPS-format output, parse it, and recover the planted jump."""
        g = NEMDGeometry(wall_thickness=0, reservoir_thickness=4e-9,
                         bulk_a_length=20e-9, bulk_b_length=20e-9,
                         bin_width=0.5e-9)
        n = int(g.total_length / g.bin_width)
        z = (np.arange(n) + 0.5) * g.bin_width
        dT, q, k_a, k_b = 6.0, 1.5e9, 148.0, 60.0
        T = np.where(
            z < g.interface_z,
            310.0 + (q / k_a) * (g.interface_z - z),
            (310.0 - dT) - (q / k_b) * (z - g.interface_z),
        )
        path = self.tmp / "temperature_profile.dat"
        with open(path, "w") as fh:
            fh.write("# Chunk-averaged data\n# Timestep Number-of-chunks\n"
                     "# Chunk Coord1 Ncount v_kT\n")
            fh.write(f"2000000 {n} {n * 100}\n")
            for i, (zz, tt) in enumerate(zip(z, T), start=1):
                fh.write(f"{i} {zz * 1e10:.6f} 100 {tt:.6f}\n")

        f = FlagSet()
        prof = parse_profile(path, g, f)
        self.assertEqual(prof.z.size, n)
        self.assertAlmostEqual(prof.interface_z, g.interface_z)
        r = extract_kapitza(prof, q, f, bulk_a=g.bulk_a, bulk_b=g.bulk_b)
        self.assertAlmostEqual(r.delta_T.si, dT, places=4)
        # the file carries six decimals, so compare relatively not absolutely
        self.assertAlmostEqual(r.h_K.si / (q / dT), 1.0, places=6)

    def test_empty_bins_are_dropped_and_flagged(self):
        g = NEMDGeometry()
        n = 60
        z = (np.arange(n) + 0.5) * g.bin_width
        path = self.tmp / "p.dat"
        with open(path, "w") as fh:
            fh.write("# a\n# b\n# c\n")
            fh.write(f"100 {n} {n}\n")
            for i, zz in enumerate(z, start=1):
                cnt = 0 if i <= 3 else 50
                fh.write(f"{i} {zz * 1e10:.4f} {cnt} 300.0\n")
        f = FlagSet()
        prof = parse_profile(path, g, f)
        self.assertEqual(prof.z.size, n - 3)
        self.assertIn(Code.NEMD_FEW_BINS, f.codes())

    def test_only_the_last_frame_is_used(self):
        """Earlier frames are running averages that include the transient."""
        g = NEMDGeometry()
        n = g.n_bins
        z = (np.arange(n) + 0.5) * g.bin_width
        path = self.tmp / "p.dat"
        with open(path, "w") as fh:
            fh.write("# a\n# b\n# c\n")
            for frame, val in ((1000, 999.0), (2000, 300.0)):
                fh.write(f"{frame} {n} {n}\n")
                for i, zz in enumerate(z, start=1):
                    fh.write(f"{i} {zz * 1e10:.4f} 50 {val}\n")
        prof = parse_profile(path, g, FlagSet())
        self.assertTrue(np.allclose(prof.T, 300.0))

    def test_reservoir_energy_uses_the_difference(self):
        path = self.tmp / "e.dat"
        path.write_text("# time E_hot E_cold\n0.0 10.0 -10.0\n"
                        "1.0 60.0 -60.0\n2.0 110.0 -110.0\n")
        e_hot, e_cold, dur = parse_reservoir_energy(path)
        self.assertAlmostEqual(e_hot, 100.0 * 1.602176634e-19)
        self.assertAlmostEqual(e_cold, 100.0 * 1.602176634e-19)
        self.assertAlmostEqual(dur, 2.0e-12)

    def test_reservoir_energy_needs_two_samples(self):
        path = self.tmp / "e.dat"
        path.write_text("# h\n0.0 1.0 -1.0\n")
        with self.assertRaises(ValueError):
            parse_reservoir_energy(path)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestConvergence(unittest.TestCase):
    def test_converged_series_passes(self):
        f = FlagSet()
        v = check_series(ConvergenceSeries("length", [10, 20, 40, 80],
                                           [300e6, 330e6, 344e6, 346e6]), f)
        self.assertTrue(v.converged)
        self.assertNotIn(Code.NEMD_NOT_CONVERGED, f.codes())

    def test_drifting_series_is_flagged(self):
        f = FlagSet()
        v = check_series(ConvergenceSeries("length", [10, 20, 40],
                                           [200e6, 300e6, 420e6]), f)
        self.assertFalse(v.converged)
        self.assertIn(Code.NEMD_NOT_CONVERGED, f.codes())
        self.assertIn("rising", v.recommendation)

    def test_two_points_cannot_judge_convergence(self):
        f = FlagSet()
        v = check_series(ConvergenceSeries("flux", [1, 2], [300e6, 310e6]), f)
        self.assertFalse(v.converged)
        self.assertIn(Code.NEMD_NOT_CONVERGED, f.codes())

    def test_linear_response_passes(self):
        q = [5e8, 1e9, 2e9, 4e9]
        dT = [x / 250e6 for x in q]            # exactly h_K = 250 MW/m^2K
        f = FlagSet()
        res = check_flux_linearity(q, dT, f)
        self.assertTrue(res["linear"])
        self.assertAlmostEqual(res["h_K_from_slope"] / 250e6, 1.0, places=6)
        self.assertEqual(f.codes(), set())

    def test_offset_intercept_is_flagged(self):
        q = [5e8, 1e9, 2e9, 4e9]
        dT = [x / 250e6 + 3.0 for x in q]      # fixed offset, not interface
        f = FlagSet()
        res = check_flux_linearity(q, dT, f)
        self.assertFalse(res["linear"])
        self.assertIn(Code.NEMD_FLUX_NONLINEAR, f.codes())

    def test_nonlinear_response_is_flagged(self):
        q = [5e8, 1e9, 2e9, 4e9]
        dT = [(x / 250e6) * (1 + x / 4e9) for x in q]
        f = FlagSet()
        res = check_flux_linearity(q, dT, f)
        self.assertFalse(res["linear"])
        self.assertIn(Code.NEMD_FLUX_NONLINEAR, f.codes())

    def test_richardson_extrapolation_recovers_the_limit(self):
        limit = 400e6
        inv = [1 / 10e-9, 1 / 20e-9, 1 / 40e-9, 1 / 80e-9]
        h = [limit - 2e-3 * x for x in inv]
        self.assertAlmostEqual(richardson_extrapolate(inv, h) / limit, 1.0,
                               places=6)


# ---------------------------------------------------------------------------
# Compute layer
# ---------------------------------------------------------------------------

class TestLocalRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.r = LocalRunner()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, cmd, name="j"):
        h = self.r.submit(JobSpec(name=name, command=cmd, workdir=self.tmp / name))
        deadline = time.time() + 30
        while not self.r.poll(h).terminal and time.time() < deadline:
            time.sleep(0.1)
        return self.r.collect(h)

    def test_submit_is_non_blocking(self):
        """Nothing may block on a job; this is the architecture."""
        t0 = time.time()
        self.r.submit(JobSpec(name="slow", command="sleep 5",
                              workdir=self.tmp / "slow"))
        self.assertLess(time.time() - t0, 1.0)

    def test_success(self):
        res = self._run("echo hi", "ok")
        self.assertIs(res.state, JobState.COMPLETED)
        self.assertEqual(res.returncode, 0)
        self.assertTrue(res.ok)

    def test_explicit_exit_code_is_recorded(self):
        """`exit N` must not kill the wrapper before the sentinel is written."""
        res = self._run("exit 3", "rc3")
        self.assertIs(res.state, JobState.FAILED)
        self.assertEqual(res.returncode, 3)

    def test_missing_command_fails(self):
        res = self._run("no_such_binary_xyz", "miss")
        self.assertIs(res.state, JobState.FAILED)

    def test_killed_job_does_not_report_running_forever(self):
        """A zombie child answers os.kill(pid, 0); poll must not be fooled.

        SIGKILL cannot be trapped, so no sentinel is written and poll() has to
        decide from the process state alone. The /proc check is Linux-only;
        without the ps fallback this hangs on macOS and the BSDs until the
        deadline, which is how the bug was found.
        """
        t0 = time.time()
        res = self._run("kill -9 $$", "killed")
        elapsed = time.time() - t0
        self.assertLess(elapsed, 15.0,
                        "poll() never reached a terminal state -- a zombie is "
                        "being reported as RUNNING")
        self.assertTrue(res.state.terminal)
        self.assertIs(res.state, JobState.FAILED)

    def test_pid_alive_is_false_for_a_reaped_process(self):
        from openkapitza.compute.runner import _pid_alive
        import subprocess as sp
        p = sp.Popen(["/bin/sh", "-c", "exit 0"])
        p.wait()
        self.assertFalse(_pid_alive(p.pid))

    def test_pid_alive_is_true_for_a_live_process(self):
        from openkapitza.compute.runner import _pid_alive
        self.assertTrue(_pid_alive(os.getpid()))

    def test_handle_round_trips_through_disk(self):
        h = self.r.submit(JobSpec(name="rt", command="echo x",
                                  workdir=self.tmp / "rt"))
        loaded = JobHandle.load(self.tmp / "rt" / "job_handle.json")
        self.assertEqual(loaded.job_id, h.job_id)
        self.assertEqual(loaded.backend, "local")

    def test_stdout_is_captured(self):
        res = self._run("echo marker_abc", "cap")
        self.assertIn("marker_abc", res.stdout_path.read_text())


class TestSlurmRunner(unittest.TestCase):
    def test_script_contains_the_directives(self):
        spec = JobSpec(
            name="nemd", command="srun lmp -in nemd.in", workdir=Path("/scratch/x"),
            resources=Resources(nodes=4, tasks=32, walltime_hours=12.0,
                                partition="compute", account="thermal"),
            modules=["lammps/2023"],
        )
        s = SlurmRunner().script(spec)
        for expected in ["#SBATCH --nodes=4", "#SBATCH --ntasks-per-node=32",
                         "#SBATCH --time=12:00:00", "#SBATCH --partition=compute",
                         "#SBATCH --account=thermal", "module load lammps/2023",
                         "srun lmp -in nemd.in"]:
            self.assertIn(expected, s)

    def test_multiday_walltime_format(self):
        s = SlurmRunner().script(JobSpec(name="n", command="x",
                                         workdir=Path("/tmp/x"),
                                         resources=Resources(walltime_hours=50.0)))
        self.assertIn("#SBATCH --time=2-02:00:00", s)

    def test_unavailable_backend_is_not_silently_substituted(self):
        """Running a cluster campaign on a laptop by accident is worse than an error."""
        if SlurmRunner().available():
            self.skipTest("slurm is present here")
        with self.assertRaises(RuntimeError):
            select_runner("slurm")

    def test_select_falls_back_to_local(self):
        self.assertEqual(select_runner().name,
                         "slurm" if SlurmRunner().available() else "local")


class TestCostModel(unittest.TestCase):
    def test_cost_scales_with_atoms_and_steps(self):
        a = estimate_cost(10_000, 1_000_000, "sw")
        b = estimate_cost(20_000, 1_000_000, "sw")
        c = estimate_cost(10_000, 2_000_000, "sw")
        self.assertAlmostEqual(b.node_hours / a.node_hours, 2.0, places=6)
        self.assertAlmostEqual(c.node_hours / a.node_hours, 2.0, places=6)

    def test_mlip_is_orders_of_magnitude_more_expensive(self):
        classical = estimate_cost(40_000, 2_000_000, "sw").node_hours
        mlip = estimate_cost(40_000, 2_000_000, "mlip").node_hours
        self.assertGreater(mlip / classical, 100.0)

    def test_expensive_run_is_flagged(self):
        f = FlagSet()
        est = estimate_cost(2_000_000, 100_000_000, "tersoff", flags=f)
        self.assertGreater(est.node_hours, 500.0)
        self.assertIn(Code.COMPUTE_COST_HIGH, f.codes())

    def test_unknown_potential_falls_back_with_a_note(self):
        f = FlagSet()
        e = estimate_cost(1000, 1000, "unobtainium-net", flags=f)
        self.assertGreater(e.node_hours, 0)
        self.assertIn(Code.COMPUTE_COST_HIGH, f.codes())

    def test_big_run_on_local_backend_is_flagged(self):
        f = FlagSet()
        check_backend_fit(estimate_cost(500_000, 20_000_000, "sw"), "local", f)
        self.assertIn(Code.COMPUTE_BACKEND_UNAVAILABLE, f.codes())

    def test_resources_carry_a_walltime_safety_factor(self):
        e = estimate_cost(100_000, 5_000_000, "sw", cores=64)
        r = resources_for(e)
        self.assertGreater(r.walltime_hours, e.wall_hours)


if __name__ == "__main__":
    unittest.main(verbosity=2)
