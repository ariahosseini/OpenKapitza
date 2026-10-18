"""Invariant tests for the Gaussian gap integral.

`gap_conductance_integral` is an opt-in alternative to the lumped
`h_g = k_g/(Y + M)`. These tests pin the properties that make it correct,
independent of any measured data. Three of them exist because the first two
invariants tried during development were WRONG while the code was right, and
the wrong invariants are easy to reach for again:

  - Plain Jensen (`I >= lumped`) does NOT hold. The Gaussian is deliberately
    left unnormalised over t > 0 so the t <= 0 contact area stays out of the
    gap path, and at small Y/sigma that shrinking gas-covered area fraction
    legitimately beats the convexity gain.
  - Jensen against `k_g/(Y + M)` after dividing by the area fraction does NOT
    hold either. Truncating at t > 0 removes the low tail, so the truncated
    mean E[t | t > 0] EXCEEDS Y and the bound must use that larger mean.

The true invariant is Jensen on the normalised measure against the truncated
mean, which holds exactly and is tight in the limit.
"""

from __future__ import annotations

import math
import random
import unittest

from openkapitza.gap import gas_state, gap_conductance, gap_conductance_integral
from openkapitza.schema import FlagSet, Quantity, load_gas

T_REF = Quantity.parse(440.0, "K")
ACC = Quantity.parse(0.9, "-")

_phi = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
_Phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))

GASES = ("He", "Ar", "N2")
PRESSURES = (1.0, 10.0, 100.0, 1000.0, 10132.5, 101325.0)


def _pair(gas, p_pa, Y_um, sigma_um, n_quad=2000):
    state = gas_state(load_gas(gas, p_pa), T_REF)
    Y = Quantity.parse(Y_um, "um")
    sigma = Quantity.parse(sigma_um, "um")
    lumped = gap_conductance(state, Y, ACC, ACC, FlagSet()).h_g.si
    res = gap_conductance_integral(
        state, Y, sigma, ACC, ACC, FlagSet(), n_quad=n_quad
    )
    return state, lumped, res


class TestGapIntegral(unittest.TestCase):
    def test_jensen_on_normalised_measure_against_truncated_mean(self):
        """The one bound that actually holds: I/Phi >= k_g/(tbar + M)."""
        worst = math.inf
        for p_pa in PRESSURES:
            for gas in GASES:
                for sigma_um in (0.1, 0.5, 1.0, 3.0, 10.0):
                    for ys in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
                        _, _, res = _pair(gas, p_pa, ys * sigma_um, sigma_um)
                        sigma_si = sigma_um * 1e-6
                        tbar = ys * sigma_si + sigma_si * _phi(ys) / _Phi(ys)
                        bound = res.state.k_g.si / (tbar + res.M.si)
                        worst = min(worst, (res.h_g.si / _Phi(ys)) / bound)
        self.assertGreaterEqual(worst, 0.999, f"Jensen violated, min {worst}")

    def test_narrow_roughness_recovers_the_lumped_form(self):
        """sigma -> 0 is a delta at Y, so the integral must collapse to k_g/(Y+M)."""
        prev = math.inf
        for sigma_um in (1.0, 0.3, 0.1, 0.03, 0.01):
            _, lumped, res = _pair("N2", 101325.0, 5.0, sigma_um)
            err = abs(res.h_g.si / lumped - 1.0)
            self.assertLess(err, prev)          # monotone approach
            prev = err
        self.assertLess(err, 1e-5)

    def test_matches_independent_monte_carlo(self):
        """Quadrature vs direct sampling of t ~ N(Y, sigma), keeping t > 0."""
        random.seed(11)
        for gas, p_pa, Y_um, sigma_um in (
            ("N2", 101325.0, 5.0, 1.5),
            ("He", 133.3, 3.0, 1.0),
            ("Ar", 1333.2, 8.0, 3.0),
        ):
            state, _, res = _pair(gas, p_pa, Y_um, sigma_um)
            Y_si, sigma_si, M = Y_um * 1e-6, sigma_um * 1e-6, res.M.si
            n, acc = 400_000, 0.0
            for _ in range(n):
                t = random.gauss(Y_si, sigma_si)
                if t > 0.0:
                    acc += state.k_g.si / (t + M)
            self.assertAlmostEqual(res.h_g.si / (acc / n), 1.0, delta=0.01)

    def test_quadrature_is_converged(self):
        ref = _pair("N2", 101325.0, 5.0, 1.5, n_quad=16000)[2].h_g.si
        for n_quad in (250, 500, 1000, 2000):
            got = _pair("N2", 101325.0, 5.0, 1.5, n_quad=n_quad)[2].h_g.si
            self.assertAlmostEqual(got / ref, 1.0, delta=1e-6)

    def test_odd_quadrature_count_is_accepted(self):
        """Simpson needs an even count; an odd request must be bumped, not rejected."""
        _, _, res = _pair("Ar", 1000.0, 4.0, 1.2, n_quad=999)
        self.assertGreater(res.h_g.si, 0.0)

    def test_rejects_nonpositive_roughness(self):
        state = gas_state(load_gas("N2", 101325.0), T_REF)
        with self.assertRaises(ValueError):
            gap_conductance_integral(
                state, Quantity.parse(5.0, "um"), Quantity.parse(0.0, "um"),
                ACC, ACC, FlagSet(),
            )

    def test_deficit_of_lumped_form_grows_toward_the_continuum_limit(self):
        """The reason this function exists.

        The lumped form's shortfall is largest when M is small against Y,
        because there the 1/t convexity is not softened by the M offset. This
        is the trend seen against the digitised Song & Yovanovich figures.
        """
        ratios = []
        for p_pa in (100.0, 1000.0, 10132.5, 101325.0):
            _, lumped, res = _pair("N2", p_pa, 5.0, 1.43)
            ratios.append(res.h_g.si / lumped)
        self.assertEqual(ratios, sorted(ratios), f"not monotone: {ratios}")
        self.assertGreater(ratios[-1], ratios[0])

    def test_reported_note_records_the_enhancement(self):
        _, lumped, res = _pair("N2", 101325.0, 5.0, 1.43)
        self.assertIn("Gaussian gap integral", res.h_g.note)
        self.assertIn("lumped", res.h_g.note)

    def test_lumped_form_is_still_the_default(self):
        """Guard against someone quietly swapping the default.

        The project specifies h_g = k_g/(Y + M), and the evidence for the
        integral so far rests on DIGITISED figure data with a documented
        extraction bias. Changing the default needs exact tabulated data.
        """
        state = gas_state(load_gas("N2", 101325.0), T_REF)
        Y = Quantity.parse(5.0, "um")
        res = gap_conductance(state, Y, ACC, ACC, FlagSet())
        self.assertAlmostEqual(
            res.h_g.si, state.k_g.si / (Y.si + res.M.si), delta=1e-9
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
