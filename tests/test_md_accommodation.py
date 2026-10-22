"""Accommodation coefficients from gas-surface scattering MD (work package 5b).

Pure-Python tests pin the free-flight segmentation, the paired estimator and
its truncation guard, the well-depth calibration arithmetic, and the
section-11 cap. Engine tests run only when a LAMMPS binary is supplied:

    OPENKAPITZA_LMP=/path/to/lmp OPENKAPITZA_POTENTIALS=/path/to/potentials \\
        python3 -m unittest tests.test_md_accommodation

Known-answer validation (scripts/validate_accommodation.py, 600 atoms,
1500 ps, T_g 600 K), paired / unpaired:
    diffusive    0.9947 +/- 0.0046 / 0.9949 +/- 0.0046   (expect 1)
    specular     0.0000 exactly    / 0.0002 +/- 0.0114   (expect 0)
    maxwell 0.4  0.4015 +/- 0.0059 / 0.4023 +/- 0.0086   (expect 0.4)
    maxwell 0.8  0.8062 +/- 0.0056 / 0.8044 +/- 0.0061   (expect 0.8)
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from openkapitza.md import accommodation as acc
from openkapitza.md.accommodation import (
    KB_EV, AccommodationSpec, PairedEvents, accommodation_from_energies,
    analyse_well_scan, baule_goodman, build_accommodation_deck,
    build_well_scan_deck, free_flight_segments, lorentz_berthelot,
    paired_events, parse_surface, to_quantity,
)
from openkapitza.md.elastic import CubicSpec
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code
from openkapitza.schema.quantities import Confidence, Provenance

NI = CubicSpec("Ni", "fcc", 3.52, 58.6934, "eam", "Ni_u3.eam", None, "Ni")
Z_FREE, Z_TOP = 10.0, 50.0


def _trajectory(legs: list[tuple[int, float, float]]) -> tuple[np.ndarray, ...]:
    """One atom. Each leg is (frames, vz sign or 0 for 'in the well', energy)."""
    z, vz, ke = [], [], []
    for frames, sign, energy in legs:
        for k in range(frames):
            if sign == 0:
                z.append(Z_FREE - 3.0); vz.append(-1.0)
            elif sign < 0:
                z.append(Z_TOP - 2.0 - k); vz.append(-1.0)
            else:
                z.append(Z_FREE + 1.0 + k); vz.append(1.0)
            ke.append(energy)
    col = lambda a: np.asarray(a, dtype=float)[:, None]
    return col(z), col(vz), col(ke)


class Constants(unittest.TestCase):
    def test_lorentz_berthelot(self):
        eps, sigma = lorentz_berthelot("He", "Ni")
        self.assertAlmostEqual(eps, (10.22 * KB_EV * 0.5201) ** 0.5, places=10)
        self.assertAlmostEqual(sigma, 0.5 * (2.556 + 2.282), places=10)

    def test_baule_goodman_trend(self):
        # Heavier gas on the same solid accommodates more.
        self.assertLess(baule_goodman(4.0026, 58.69), baule_goodman(39.948, 58.69))

    def test_light_gas_gets_short_timestep(self):
        self.assertEqual(AccommodationSpec(NI, "He", metal_lj="Ni").dt_fs, 0.5)
        self.assertEqual(AccommodationSpec(NI, "Ar", metal_lj="Ni").dt_fs, 1.0)


class Segments(unittest.TestCase):
    def test_one_collision_pairs(self):
        z, vz, ke = _trajectory([(1, 0, 0.0), (6, -1, 0.05), (4, 0, 0.0),
                                 (6, +1, 0.03), (40, 0, 0.0)])
        down, up = free_flight_segments(z, vz, ke, Z_FREE, Z_TOP)
        self.assertEqual((down, up), ([0.05], [0.03]))
        p = paired_events(z, vz, ke, Z_FREE, Z_TOP, tail_fraction=0.1)
        self.assertEqual(p.e_in.tolist(), [0.05])
        self.assertEqual(p.e_out.tolist(), [0.03])
        self.assertEqual(p.n_incomplete, 0)

    def test_segment_at_frame_zero_dropped(self):
        # Its start is not observed, so it is not counted (selection rule).
        z, vz, ke = _trajectory([(6, -1, 0.05), (4, 0, 0.0), (6, +1, 0.03)])
        down, up = free_flight_segments(z, vz, ke, Z_FREE, Z_TOP)
        self.assertEqual((down, up), ([], [0.03]))

    def test_trapped_at_end_counts_incomplete(self):
        z, vz, ke = _trajectory([(1, 0, 0.0), (6, -1, 0.05), (60, 0, 0.0)])
        p = paired_events(z, vz, ke, Z_FREE, Z_TOP, tail_fraction=0.1)
        self.assertEqual((p.n_eligible, p.n_incomplete, p.e_in.size), (1, 1, 0))
        self.assertEqual(p.incomplete_fraction, 1.0)

    def test_incident_in_tail_not_counted(self):
        z, vz, ke = _trajectory([(60, 0, 0.0), (1, 0, 0.0), (6, -1, 0.05)])
        p = paired_events(z, vz, ke, Z_FREE, Z_TOP, tail_fraction=0.2)
        self.assertEqual(p.n_eligible, 0)


class Estimator(unittest.TestCase):
    def _synthetic(self, a, n=4000, noise=0.002, seed=3):
        rng = np.random.default_rng(seed)
        t_s, t_g = 300.0, 600.0
        e_s = 2 * KB_EV * t_s
        e_in = rng.gamma(2.0, KB_EV * t_g, n)           # flux-weighted Maxwellian
        e_out = e_in + a * (e_s - e_in) + rng.normal(0, noise, n)
        pairs = PairedEvents(e_in, e_out, n, 0)
        return e_in, e_out, pairs, t_s, t_g

    def test_paired_recovers_alpha_with_less_noise(self):
        e_in, e_out, pairs, t_s, t_g = self._synthetic(0.08)
        flags = FlagSet()
        r = accommodation_from_energies(list(e_in), list(e_out), t_s, t_g, flags,
                                        pairs=pairs)
        self.assertEqual(r.meta["estimator"], "paired")
        self.assertAlmostEqual(r.alpha, 0.08, delta=4 * r.stderr)
        self.assertLess(r.stderr, 0.3 * r.meta["stderr_unpaired"])
        self.assertNotIn("Paired alpha", " ".join(f.message for f in flags))

    def test_incomplete_pairs_flagged(self):
        e_in, e_out, _, t_s, t_g = self._synthetic(0.5)
        pairs = PairedEvents(e_in, e_out, e_in.size + 200, 200)
        flags = FlagSet()
        accommodation_from_energies(list(e_in), list(e_out), t_s, t_g, flags,
                                    pairs=pairs)
        self.assertTrue(any("no reflection" in f.message for f in flags))

    def test_disagreeing_estimators_flagged(self):
        e_in, e_out, pairs, t_s, t_g = self._synthetic(0.5)
        flags = FlagSet()
        # Unpaired streams that say alpha ~ 0 while the pairs say 0.5.
        accommodation_from_energies(list(e_in), list(e_in), t_s, t_g, flags,
                                    pairs=pairs)
        warn = [f for f in flags if "disagree" in f.message]
        self.assertEqual(len(warn), 1)


class Calibration(unittest.TestCase):
    def test_scan_deck_has_reference_and_grid(self):
        spec = AccommodationSpec(NI, "He", metal_lj="Ni")
        deck = build_well_scan_deck(spec, grid=2)
        self.assertIn("WELLREF", deck)
        self.assertEqual(deck.count('print           "WELLSCAN'), 2 * 2 * (
            int((2.5 * spec.interaction()[1] + 1.5) / 0.1) + 1))

    def test_analyse_finds_deepest_site(self):
        text = ("WELLREF -100.0 21.0\n"
                "WELLSCAN 0.0000 0.0000 2.0000 -100.010\n"
                "WELLSCAN 0.5000 0.0000 2.1000 -100.030\n"
                "WELLSCAN 0.5000 0.0000 3.0000 -100.005\n")
        s = analyse_well_scan(text, 0.02, 2.4)
        self.assertAlmostEqual(s.e_ads, 0.030, places=9)
        self.assertEqual((s.z_min, s.site), (2.1, (0.5, 0.0)))

    def test_unbound_scan_raises(self):
        with self.assertRaises(ValueError):
            analyse_well_scan("WELLREF -1.0 0\nWELLSCAN 0 0 2 -0.9\n", 0.01, 2.0)

    def test_calibration_is_linear_rescale(self):
        # E_ads is linear in eps at fixed sigma, so one scan fixes the scale.
        spec = AccommodationSpec(NI, "He", metal_lj="Ni")
        eps0, sigma = spec.interaction()
        fake = f"WELLREF 0.0 0.0\nWELLSCAN 0 0 2.0 {-6.0 * eps0:.10f}\n"
        saved = acc._run
        acc._run = lambda *a, **k: fake
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                Path(d, NI.potential).write_text("x")
                flags = FlagSet()
                cal, scan = acc.calibrate_well_depth(Path("lmp"), spec, Path(d), flags)
        finally:
            acc._run = saved
        self.assertTrue(cal.calibrated)
        self.assertAlmostEqual(cal.eps_gs, 6.0e-3 / 6.0, places=12)
        self.assertEqual(cal.sigma_gs, sigma)
        self.assertIn(Code.MD_ACCOMMODATION_UNCALIBRATED, flags.codes())

    def test_deck_measures_surface(self):
        deck = build_accommodation_deck(AccommodationSpec(NI, "He", metal_lj="Ni"))
        self.assertIn("ACCOM_SURFACE", deck)

    def test_parse_surface(self):
        t, n, sites = parse_surface("x\nACCOM_SURFACE t_surface 302.6 n_adsorbed "
                                    "3.8 n_sites 72\n")
        self.assertEqual((t, n, sites), (302.6, 3.8, 72))


class Sensitivity(unittest.TestCase):
    def test_scales_span_literature_range(self):
        lo, mid, hi = acc.target_range_scales("Ar")
        self.assertEqual(mid, 1.0)
        self.assertAlmostEqual(lo * 80.0, 60.0)
        self.assertAlmostEqual(hi * 80.0, 110.0)


class Cap(unittest.TestCase):
    def test_md_alpha_capped_medium(self):
        r = acc.AccommodationResult(0.1, 0.01, 0.1, 0.1, 0.1, 0.05, 1000, 1000,
                                    meta={"gas": "He", "solid": "Ni"})
        q = to_quantity(r, spread=(0.05, 0.2))
        self.assertEqual(q.provenance, Provenance.MD_COMPUTED)
        self.assertLessEqual(q.confidence, Confidence.MEDIUM)
        self.assertAlmostEqual(q.uncertainty, 0.075)


LMP = os.environ.get("OPENKAPITZA_LMP")
POT = os.environ.get("OPENKAPITZA_POTENTIALS")


@unittest.skipUnless(LMP and POT, "set OPENKAPITZA_LMP and OPENKAPITZA_POTENTIALS")
class Engine(unittest.TestCase):
    def test_calibrated_well_hits_target(self):
        flags = FlagSet()
        spec = AccommodationSpec(NI, "He", metal_lj="Ni")
        cal, _ = acc.calibrate_well_depth(Path(LMP), spec, Path(POT), flags)
        # Rescan with the calibrated well: the summed energy must hit 6 meV.
        cal2, scan2 = acc.calibrate_well_depth(Path(LMP), cal, Path(POT), FlagSet())
        self.assertAlmostEqual(scan2.e_ads * 1e3, 6.0, delta=0.06)
        self.assertAlmostEqual(cal2.eps_gs, cal.eps_gs, delta=0.01 * cal.eps_gs)


if __name__ == "__main__":
    unittest.main()
