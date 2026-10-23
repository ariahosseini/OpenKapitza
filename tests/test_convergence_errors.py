"""Convergence verdicts that know the error on h_K (open issue P1-3).

Same lesson as the stationarity test (P1-1): a fixed tolerance on a noisy
difference cannot tell "still moving" from "noisy". With errors supplied,
check_series certifies convergence only when change + 2 sigma fits inside the
tolerance, warns only on a significant change above it, and otherwise says
the comparison is statistics-limited.
"""

from __future__ import annotations

import unittest

from openkapitza.md.convergence import ConvergenceSeries, check_series
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code, Severity


def _verdict(h, err=None, tol=0.05):
    flags = FlagSet()
    v = check_series(ConvergenceSeries("length", [10, 20, 40], h, "nm", err), flags,
                     tolerance=tol)
    return v, flags


class ErrorAwareVerdicts(unittest.TestCase):
    def test_without_errors_behaviour_is_unchanged(self):
        v, _ = _verdict([300e6, 330e6, 335e6])
        self.assertTrue(v.converged)

    def test_precise_and_flat_is_converged(self):
        v, flags = _verdict([300e6, 330e6, 335e6], [3e6, 3e6, 3e6])
        self.assertTrue(v.converged)
        self.assertEqual(len(flags), 0)

    def test_noisy_flat_is_unresolved_not_failed(self):
        # 1.5 % change, but 10 % errors: cannot certify 5 %.
        v, flags = _verdict([300e6, 330e6, 335e6], [33e6, 33e6, 33e6])
        self.assertFalse(v.converged)
        self.assertTrue(v.detail["unresolved"])
        sev = [f.severity for f in flags if f.code == Code.NEMD_NOT_CONVERGED]
        self.assertEqual(sev, [Severity.CAUTION])

    def test_significant_rise_is_a_warning(self):
        v, flags = _verdict([300e6, 330e6, 400e6], [3e6, 3e6, 3e6])
        self.assertFalse(v.converged)
        self.assertFalse(v.detail["unresolved"])
        sev = [f.severity for f in flags if f.code == Code.NEMD_NOT_CONVERGED]
        self.assertEqual(sev, [Severity.WARNING])

    def test_errors_are_sorted_with_values(self):
        s = ConvergenceSeries("length", [40, 10, 20], [3, 1, 2], "nm", [0.3, 0.1, 0.2])
        self.assertEqual(s.h_K_err, [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
