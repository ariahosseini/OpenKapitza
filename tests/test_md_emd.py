"""Equilibrium (temperature-fluctuation) interface conductance (P1-8)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from openkapitza.md.emd_kapitza import (
    EMDSpec, KB, build_emd_deck, check, fit_tau, h_from_series,
)
from openkapitza.md.nemd import NEMDGeometry, NEMDSpec, validate_deck
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code, Severity


def _ou(tau_ps, n, dt_ps, sigma=8.0, seed=1):
    """Ornstein-Uhlenbeck dT(t) with a known relaxation time."""
    rng = np.random.default_rng(seed)
    a = math.exp(-dt_ps / tau_ps)
    x = np.empty(n); x[0] = 0.0
    noise = rng.normal(0, sigma * math.sqrt(1 - a * a), n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise[i]
    return x


class Analysis(unittest.TestCase):
    def test_recovers_a_known_relaxation_time_and_h(self):
        dt, tau = 0.01, 14.0                      # ps
        dT = _ou(tau, 1_000_000, dt)              # 10 ns = 714 tau
        t = np.arange(len(dT)) * dt
        n_a = n_b = 1000; area = 4.8e-18
        r = h_from_series(t, dT, n_a, n_b, area)
        self.assertAlmostEqual(r.tau / 1e-12, tau, delta=0.12 * tau)
        c_r = 3 * n_a * KB / 2
        self.assertAlmostEqual(r.h, c_r / (area * tau * 1e-12), delta=0.12 * r.h)
        self.assertIsNotNone(r.h_err)

    def test_biexponential_recovers_the_slow_interface_time(self):
        dt = 0.02
        slow = _ou(25.0, 400_000, dt, sigma=6.0, seed=3)          # interface, 8 ns
        fast = _ou(0.3, 400_000, dt, sigma=6.0, seed=4)           # partition jitter
        t = np.arange(len(slow)) * dt
        r = h_from_series(t, slow + fast, 600, 600, 5e-18, model="biexp")
        self.assertAlmostEqual(r.tau / 1e-12, 25.0, delta=0.15 * 25.0)
        self.assertLess(r.tau_fast / 1e-12, 2.0)
        self.assertAlmostEqual(r.slow_weight, 0.5, delta=0.1)

    def test_two_separated_timescales_leave_h_undetermined(self):
        dt = 0.02
        slow = _ou(25.0, 400_000, dt, sigma=6.0, seed=3)
        fast = _ou(0.3, 400_000, dt, sigma=6.0, seed=4)
        t = np.arange(len(slow)) * dt
        r = h_from_series(t, slow + fast, 600, 600, 5e-18, model="biexp")
        f = FlagSet(); check(r, f)
        flag = next(x for x in f.flags if x.code == Code.EMD_TWO_TIMESCALES)
        self.assertEqual(flag.severity, Severity.WARNING)
        self.assertGreater(r.h_fast, 10 * r.h)
        # a clean single process is not flagged
        one = _ou(14.0, 400_000, dt, seed=5)
        f = FlagSet(); check(h_from_series(t, one, 600, 600, 5e-18), f)
        self.assertNotIn(Code.EMD_TWO_TIMESCALES, f.codes())

    def test_short_series_is_flagged(self):
        dT = _ou(14.0, 100_000, 0.01); t = np.arange(len(dT)) * 0.01   # ~70 tau
        f = FlagSet(); check(h_from_series(t, dT, 1000, 1000, 4.8e-18), f)
        self.assertTrue(any("relaxation times long" in x.message for x in f.flags))

    def test_non_decaying_series_refused(self):
        with self.assertRaises(ValueError):
            fit_tau(np.arange(5.0), np.array([1.0, 1.0, 1.0, 1.0, 1.0]))

    def test_ratio_check_direction(self):
        dT = _ou(14.0, 100_000, 0.01); t = np.arange(len(dT)) * 0.01
        r = h_from_series(t, dT, 1000, 1000, 4.8e-18)
        f = FlagSet(); check(r, f, h_nemd=3 * r.h)
        self.assertEqual(next(x for x in f.flags if x.code == Code.EMD_RATIO).severity, Severity.INFO)
        f = FlagSet(); check(r, f, h_nemd=0.5 * r.h)          # NEMD below EMD: wrong
        self.assertEqual(next(x for x in f.flags if x.code == Code.EMD_RATIO).severity, Severity.WARNING)

    def test_long_slabs_flagged(self):
        dT = _ou(14.0, 100_000, 0.01); t = np.arange(len(dT)) * 0.01
        r = h_from_series(t, dT, 1000, 1000, 4.8e-18)
        f = FlagSet(); check(r, f, bulk=((1e-6, 100.0), (1e-6, 60.0)))
        self.assertIn(Code.NEMD_POOR_FIT, f.codes())


class Deck(unittest.TestCase):
    def test_same_cell_as_nemd_then_nve_fluctuations(self):
        s = NEMDSpec(geometry=NEMDGeometry(), element_a="Si", element_b="Ge",
                     lattice_a=5.431e-10, lattice_b=5.658e-10, temperature=300.0)
        deck = build_emd_deck(s, EMDSpec(production_ps=100.0))
        self.assertIn("change_box", deck)                       # same relaxation
        self.assertNotIn("langevin", deck)                      # no reservoirs
        self.assertIn("emd_temperatures.dat", deck)
        self.assertIn("run             100000", deck)
        self.assertEqual([p for p in validate_deck(deck) if "profile" not in p and "reservoir" not in p
                          and "langevin" not in p and "ave/chunk" not in p and "tally" not in p], [])


if __name__ == "__main__":
    unittest.main()
