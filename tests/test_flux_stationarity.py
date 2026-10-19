"""Was the heat flow stationary while the flux was measured?

WHY THIS EXISTS. Two runs of the same cell, differing only in duration,
returned h_K = 1348 (60 ps) and 17453 MW/m^2K (20 ps) -- a factor of 13 --
while raising the *identical* set of eleven flags between them. Neither the
extraction nor `check_run_health` distinguished them. `NEMD_UNEQUILIBRATED`
already existed but never fired, because it compares mean temperature against
the setpoint: the reservoirs sit at their setpoints long before the thermal
profile between them stops developing.

So an unconverged run could return a plausible-looking number with no flag
saying the number was meaningless. `check_flux_stationarity` closes that by
testing what actually must hold at steady state -- constant dE/dt in each
reservoir -- measured as the drift between the first and second half of the
averaging window.

Measured on the two real runs: hot/cold drift 19 %/97 % at 60 ps and
76 %/169 % at 20 ps. It flags both, which is correct; neither was stationary.
"""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

from openkapitza.md import check_flux_stationarity, parse_reservoir_series
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code

EV_PER_J = 1.0 / 1.602176634e-19


def _write_series(path: Path, times_ps, hot_eV, cold_eV) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# time[ps] E_hot[eV] E_cold[eV]\n")
        for t, h, c in zip(times_ps, hot_eV, cold_eV):
            fh.write(f"{t} {h} {c}\n")
    return path


class StationarityFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        random.seed(3)

    def tearDown(self):
        self._tmp.cleanup()

    def _stationary(self, n=200, rate=-0.5, noise=0.0):
        """Constant dE/dt: cumulative energy linear in time."""
        times = [0.1 * i for i in range(n)]
        hot = [rate * t + random.gauss(0, noise) for t in times]
        cold = [-rate * t + random.gauss(0, noise) for t in times]
        return _write_series(self.tmp / "stationary.dat", times, hot, cold)

    def _drifting(self, n=200, tau=5.0):
        """Rate still decaying: cumulative energy curves over the window."""
        times = [0.1 * i for i in range(n)]
        # E(t) = -A*(1 - exp(-t/tau)) has a rate that falls toward zero.
        hot = [-20.0 * (1.0 - math.exp(-t / tau)) for t in times]
        cold = [20.0 * (1.0 - math.exp(-t / tau)) for t in times]
        return _write_series(self.tmp / "drifting.dat", times, hot, cold)


class TestStationarityDetection(StationarityFixture):
    def test_stationary_series_raises_nothing(self):
        flags = FlagSet()
        drifts = check_flux_stationarity(self._stationary(), flags)
        self.assertEqual(flags.codes(), set())
        for side, value in drifts.items():
            self.assertLess(value, 0.15, side)

    def test_stationary_with_noise_still_passes(self):
        """Real data is noisy; the check must not cry wolf on noise alone."""
        flags = FlagSet()
        check_flux_stationarity(self._stationary(noise=0.05), flags)
        self.assertEqual(flags.codes(), set())

    def test_drifting_series_is_flagged(self):
        flags = FlagSet()
        drifts = check_flux_stationarity(self._drifting(), flags)
        self.assertIn(Code.NEMD_UNEQUILIBRATED, flags.codes())
        self.assertGreater(max(drifts.values()), 0.15)

    def test_flag_message_names_the_measured_drift(self):
        """A flag that does not carry its number is not actionable."""
        flags = FlagSet()
        check_flux_stationarity(self._drifting(), flags)
        flag = next(f for f in flags.flags
                    if f.code == Code.NEMD_UNEQUILIBRATED)
        self.assertRegex(flag.message, r"\d+\s*%")
        self.assertIn("steady state", flag.message)

    def test_too_few_samples_is_flagged_not_silently_passed(self):
        """The worst outcome is a check that quietly does nothing."""
        flags = FlagSet()
        path = _write_series(self.tmp / "short.dat", [0.0, 0.1, 0.2],
                             [0.0, -0.5, -1.0], [0.0, 0.5, 1.0])
        drifts = check_flux_stationarity(path, flags)
        self.assertEqual(drifts, {})
        self.assertIn(Code.NEMD_UNEQUILIBRATED, flags.codes())
        self.assertIn("too few", flags.flags[0].message.lower())

    def test_tolerance_is_honoured(self):
        flags_strict = FlagSet()
        check_flux_stationarity(self._stationary(noise=0.05), flags_strict,
                                tolerance=1e-9)
        self.assertIn(Code.NEMD_UNEQUILIBRATED, flags_strict.codes())

        flags_loose = FlagSet()
        check_flux_stationarity(self._drifting(), flags_loose, tolerance=10.0)
        self.assertEqual(flags_loose.codes(), set())


class TestSeriesParser(StationarityFixture):
    def test_units_are_converted(self):
        """ps -> s and eV -> J, matching parse_reservoir_energy."""
        path = _write_series(self.tmp / "u.dat", [0.0, 1.0], [0.0, 1.0], [0.0, -1.0])
        t, hot, cold = parse_reservoir_series(path)
        self.assertAlmostEqual(t[1], 1e-12, places=18)
        self.assertAlmostEqual(hot[1], 1.602176634e-19, places=28)
        self.assertAlmostEqual(cold[1], -1.602176634e-19, places=28)

    def test_comments_and_short_lines_are_skipped(self):
        path = self.tmp / "messy.dat"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# header\n\n0 0 0\n1 -1\nnot a number here\n2 -2 2\n")
        t, hot, cold = parse_reservoir_series(path)
        self.assertEqual(len(t), 2)
        self.assertEqual(len(hot), 2)
        self.assertEqual(len(cold), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
