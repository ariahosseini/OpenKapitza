"""Statistical calibration of the reservoir-power drift test (open issue P1-1).

The fixed-tolerance test it replaced reported every real run as not steady,
because its own noise (36 % on a 400 ps Si/Ge run) exceeded the 15 %
tolerance. These tests pin what the replacement must do, by Monte Carlo over
both noise models a cumulative tally can carry (random-walk increments, as
Langevin stochastic work is, and white noise on E):

- rarely call a truly stationary run "drifting" (false alarms);
- catch a real drift well above tolerance;
- say "unresolved", not "drifting", when noise hides the answer;
- resolve better on longer windows.
"""

from __future__ import annotations

import unittest

import numpy as np

from openkapitza.md.stationarity import Verdict, analyse_drift, window_powers


def _series(rng, n=4000, decay=0.0, noise=0.05, model="rw", tau=None):
    t = np.arange(n) * 0.1
    if tau:
        p = 1.0 + decay * np.exp(-t / tau)
    else:
        p = 1.0 - decay * (t / t[-1] - 0.5)
    e = np.concatenate([[0.0], np.cumsum(p[:-1] * 0.1)])
    if model == "rw":
        e = e + np.cumsum(rng.normal(0, noise, n))
    else:
        e = e + rng.normal(0, noise, n)
    return t, e


def _rates(rng, reps=200, **kw):
    v = [analyse_drift(*_series(rng, **kw)).verdict for _ in range(reps)]
    return {k: v.count(k) / reps for k in Verdict}


class Calibration(unittest.TestCase):
    def test_false_alarm_rate_is_low(self):
        rng = np.random.default_rng(11)
        for noise in (0.05, 0.2):
            r = _rates(rng, noise=noise)
            self.assertLessEqual(r[Verdict.DRIFTING], 0.03, noise)

    def test_large_drift_is_caught(self):
        rng = np.random.default_rng(12)
        self.assertGreaterEqual(_rates(rng, decay=0.6, noise=0.05)[Verdict.DRIFTING], 0.95)

    def test_decaying_transient_is_caught(self):
        # The physical case: power relaxing toward steady state inside the window.
        rng = np.random.default_rng(13)
        r = _rates(rng, decay=1.0, tau=40.0, noise=0.05)
        self.assertGreaterEqual(r[Verdict.DRIFTING], 0.95)

    def test_noise_gives_unresolved_not_drifting(self):
        rng = np.random.default_rng(14)
        r = _rates(rng, noise=0.2)
        self.assertGreaterEqual(r[Verdict.UNRESOLVED], 0.9)

    def test_quiet_stationary_is_certified(self):
        rng = np.random.default_rng(15)
        self.assertGreaterEqual(_rates(rng, noise=0.05)[Verdict.STATIONARY], 0.97)

    def test_resolution_improves_with_length(self):
        rng = np.random.default_rng(16)
        short = np.mean([analyse_drift(*_series(rng, n=2000, noise=0.1)).resolution
                         for _ in range(100)])
        long = np.mean([analyse_drift(*_series(rng, n=8000, noise=0.1)).resolution
                        for _ in range(100)])
        # Random-walk noise on a power estimate falls as 1/sqrt(T).
        self.assertAlmostEqual(short / long, 2.0, delta=0.4)


class Mechanics(unittest.TestCase):
    def test_window_powers_exact_on_linear_series(self):
        t = np.arange(100) * 0.5
        c, p = window_powers(t, 3.0 * t + 7.0, 5)
        np.testing.assert_allclose(p, 3.0)
        self.assertEqual(c.size, 5)

    def test_sign_of_power_does_not_matter(self):
        rng = np.random.default_rng(17)
        t, e = _series(rng, decay=0.6, noise=0.0)
        a, b = analyse_drift(t, e), analyse_drift(t, -e)
        self.assertEqual(a.verdict, b.verdict)
        self.assertAlmostEqual(a.drift, -b.drift if a.mean_power * b.mean_power < 0 else a.drift)

    def test_too_few_samples_raises(self):
        with self.assertRaises(ValueError):
            analyse_drift(np.arange(10.0), np.arange(10.0))


if __name__ == "__main__":
    unittest.main()


class BlockTrend(unittest.TestCase):
    """The per-frame temperature-jump trend: the steady-state test h_K needs."""

    def test_settling_jump_is_drifting(self):
        from openkapitza.md.stationarity import analyse_block_trend
        v = [30, 27, 24.5, 22.5, 21, 20, 19.4, 19.0]
        self.assertEqual(analyse_block_trend(v).verdict, Verdict.DRIFTING)

    def test_flat_quiet_jump_is_stationary(self):
        from openkapitza.md.stationarity import analyse_block_trend
        rng = np.random.default_rng(3)
        v = 20 + 0.2 * rng.standard_normal(8)
        self.assertEqual(analyse_block_trend(v).verdict, Verdict.STATIONARY)

    def test_four_noisy_frames_are_unresolved(self):
        # The real 400 ps Si/Ge frames: no significant trend with 4 blocks.
        from openkapitza.md.stationarity import analyse_block_trend
        r = analyse_block_trend([24.53, 29.13, 18.55, 18.55])
        self.assertEqual(r.verdict, Verdict.UNRESOLVED)

    def test_too_few_blocks_raise(self):
        from openkapitza.md.stationarity import analyse_block_trend
        with self.assertRaises(ValueError):
            analyse_block_trend([1.0, 2.0, 3.0])
