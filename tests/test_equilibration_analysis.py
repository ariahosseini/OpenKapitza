"""Tests for the equilibration-time analyser's judgement calls.

The drift metric itself is `check_flux_stationarity`'s and tested there. What
is new here, and what needs pinning, is the analyser's two judgements:

  - **The consecutive-window guard.** A single window dipping under tolerance
    is noise, not equilibration. On the real 60 ps run exactly one window
    (at 30 ps) showed 0.0 % / 4.7 % drift while every neighbour failed; without
    the guard the run would have been declared equilibrated there.
  - **The power-balance check.** Energy in must equal energy out at steady
    state. A persistent imbalance is a DIFFERENT failure from drift and will
    not go away by running longer. On that same run, drift alone passed the
    30 ps window while the imbalance there was 22 %, and the imbalance grew
    to 146 % by the end -- the signature of stored strain energy draining,
    which is what the relaxation step addresses.

The analyser returns 0 when it finds equilibration and 1 when it does not, so
the verdict is testable without parsing its output.
"""

from __future__ import annotations

import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.find_equilibration_time import analyse, windowed_drift

EV_PER_J = 1.0 / 1.602176634e-19


def _write(path: Path, times_ps, hot_eV, cold_eV) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# time[ps] E_hot[eV] E_cold[eV]\n")
        for t, h, c in zip(times_ps, hot_eV, cold_eV):
            fh.write(f"{t} {h} {c}\n")
    return path


def _run(path: Path, window_ps=20.0, tolerance=0.15) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = analyse(path.parent, window_ps, tolerance)
    return code, buf.getvalue()


class AnalyserFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _series(self, n, hot_rate, cold_rate):
        """Balanced, perfectly linear accumulators -> stationary."""
        times = [0.1 * i for i in range(n)]
        hot = [hot_rate * t for t in times]
        cold = [cold_rate * t for t in times]
        return _write(self.dir / "reservoir_energy.dat", times, hot, cold)


class TestStationaryRunIsRecognised(AnalyserFixture):
    def test_balanced_linear_series_is_equilibrated(self):
        code, out = _run(self._series(1200, -0.5, 0.5))
        self.assertEqual(code, 0, out)
        self.assertIn("EQUILIBRATED", out)


class TestPowerBalance(AnalyserFixture):
    def test_balanced_magnitudes_pass(self):
        code, _ = _run(self._series(1200, -0.5, 0.5))
        self.assertEqual(code, 0)

    def test_persistent_imbalance_fails_even_when_perfectly_steady(self):
        """Both accumulators are exactly linear, so drift is zero -- but the
        magnitudes differ by 3x. Drift alone would call this equilibrated."""
        code, out = _run(self._series(1200, -0.5, 1.5))
        self.assertEqual(code, 1)
        self.assertIn("never balances", out)
        self.assertIn("will NOT go away by", out)

    def test_imbalance_just_inside_tolerance_passes(self):
        # 10 % apart, under a 15 % tolerance.
        code, _ = _run(self._series(1200, -0.50, 0.55))
        self.assertEqual(code, 0)


class TestConsecutiveWindowGuard(AnalyserFixture):
    def test_one_quiet_window_is_not_equilibration(self):
        """Mostly drifting, with a single stationary stretch too short to count.

        Reproduces the real 60 ps run, where exactly one window looked clean.

        The rate grows EXPONENTIALLY outside the plateau, which keeps the
        relative drift roughly constant so those windows keep failing however
        far along they are. An oscillating rate does not work: averaged over a
        window its two halves agree, so it reads as stationary. Non-stationary
        means trending, not noisy.

        The plateau is 55 samples. Declaring equilibration needs 3 consecutive
        passing starts, i.e. window_n + 2*step = 74 samples of calm, so one
        plateau this size can produce at most one passing window.
        """
        times, hot, cold = [], [], []
        h = c = 0.0
        for i in range(600):
            rate = 0.5 if 250 <= i < 305 else 0.30 * math.exp(i * 0.01)
            h -= rate * 0.1
            c += rate * 0.1
            times.append(0.1 * i); hot.append(h); cold.append(c)
        path = _write(self.dir / "reservoir_energy.dat", times, hot, cold)
        code, out = _run(path, window_ps=5.0)
        self.assertEqual(code, 1, out)
        self.assertIn("NOT EQUILIBRATED", out)

    def test_sustained_stationarity_after_a_transient_is_found(self):
        """Transient first, then genuinely steady: must report a time."""
        times, hot, cold = [], [], []
        h = c = 0.0
        for i in range(1600):
            t = 0.1 * i
            # Rate decays to a constant with a 10 ps time constant.
            rate = 0.5 * (1.0 + 3.0 * math.exp(-t / 10.0))
            h -= rate * 0.1
            c += rate * 0.1
            times.append(t); hot.append(h); cold.append(c)
        path = _write(self.dir / "reservoir_energy.dat", times, hot, cold)
        code, out = _run(path, window_ps=20.0)
        self.assertEqual(code, 0, out)
        self.assertIn("EQUILIBRATED from about", out)


class TestGuards(AnalyserFixture):
    def test_too_few_samples_refuses_rather_than_guessing(self):
        path = _write(self.dir / "reservoir_energy.dat",
                      [0.1 * i for i in range(10)],
                      [-0.05 * i for i in range(10)],
                      [0.05 * i for i in range(10)])
        code, out = _run(path)
        self.assertEqual(code, 2)
        self.assertIn("longer run", out)

    def test_missing_file_is_reported(self):
        code, out = _run(self.dir / "reservoir_energy.dat")
        self.assertEqual(code, 2)
        self.assertIn("no reservoir_energy.dat", out)

    def test_zero_power_does_not_divide_by_zero(self):
        code, _ = _run(self._series(1200, 0.0, 0.0))
        self.assertIn(code, (1, 2))


class TestWindowedDrift(unittest.TestCase):
    def test_linear_series_has_zero_drift(self):
        t = [1e-12 * i for i in range(200)]
        e = [-0.5 * x for x in t]
        drift, power = windowed_drift(t, e, 0, 200)
        self.assertLess(drift, 1e-9)
        self.assertAlmostEqual(power, -0.5, places=6)

    def test_curving_series_has_nonzero_drift(self):
        t = [1e-12 * i for i in range(200)]
        e = [-0.5 * x - 1e10 * x * x for x in t]
        drift, _ = windowed_drift(t, e, 0, 200)
        self.assertGreater(drift, 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
