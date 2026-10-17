"""Physics-invariant tests for the OpenKapitza continuum core.

These check properties the code must have regardless of what any particular
experiment measured: scaling exponents, limiting behaviour, conservation of
the network topology, and whether the caveat flags actually fire. They do not
depend on published data, so they stay valid while the material database is
still unverified.

Comparison against measured joint conductance lives in tests/test_benchmarks.py
and is skipped until a dataset is supplied.

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest

from openkapitza import build_joint, run_continuum
from openkapitza.contact import (
    CMY_PRESSURE_MAX,
    CMY_PRESSURE_MIN,
    Regime,
    combined_roughness,
    combined_slope,
    contact_conductance_plastic,
    harmonic_conductivity,
    mean_plane_separation,
)
from openkapitza.gap import gas_state, gap_conductance, vacuum_gap
from openkapitza.md import debye_heat_capacity, dmm_conductance, user_kapitza
from openkapitza.network import combine
from openkapitza.schema import (
    Code,
    Confidence,
    FlagSet,
    Provenance,
    Quantity,
    Severity,
    load_gas,
    load_material,
    make_derived,
)
from openkapitza.schema.units import K_B, from_si, to_si


def Q(v, u, **kw):
    return Quantity.parse(v, u, **kw)


# ---------------------------------------------------------------------------
# Schema layer
# ---------------------------------------------------------------------------

class TestUnits(unittest.TestCase):
    def test_round_trip(self):
        for value, unit in [(1.2, "um"), (2.0, "MPa"), (300.0, "K"),
                            (101.325, "kPa"), (5.0, "GPa"), (2.33, "g/cm3")]:
            self.assertAlmostEqual(from_si(to_si(value, unit), unit), value, places=9)

    def test_celsius_offset(self):
        self.assertAlmostEqual(to_si(25.0, "C"), 298.15, places=9)
        self.assertAlmostEqual(from_si(298.15, "C"), 25.0, places=9)

    def test_unknown_unit_rejected(self):
        with self.assertRaises(Exception):
            Quantity(si=1.0, unit="furlongs")


class TestQuantity(unittest.TestCase):
    def test_derived_confidence_capped_by_inputs(self):
        lo = Quantity(si=1.0, unit="-", confidence=Confidence.LOW)
        hi = Quantity(si=1.0, unit="-", confidence=Confidence.HIGH)
        d = make_derived(2.0, unit="-", inputs=[lo, hi])
        self.assertEqual(d.confidence, Confidence.LOW)

    def test_explicit_confidence_is_a_cap_not_an_override(self):
        """A model may only lower confidence, never raise it."""
        lo = Quantity(si=1.0, unit="-", confidence=Confidence.LOW)
        d = make_derived(2.0, unit="-", inputs=[lo], confidence=Confidence.HIGH)
        self.assertEqual(d.confidence, Confidence.LOW)

    def test_relative_uncertainty(self):
        q = Q(2.0, "MPa", rel_uncertainty=0.1)
        self.assertAlmostEqual(q.rel_uncertainty, 0.1, places=12)


class TestFlags(unittest.TestCase):
    def test_duplicates_are_deduped(self):
        f = FlagSet()
        for _ in range(3):
            f.add("X", Severity.INFO, "same message")
        self.assertEqual(len(f), 1)

    def test_distinct_messages_kept(self):
        f = FlagSet()
        f.add("X", Severity.INFO, "a")
        f.add("X", Severity.INFO, "b")
        self.assertEqual(len(f), 2)

    def test_worst_severity(self):
        f = FlagSet()
        f.add("A", Severity.INFO, "a")
        f.add("B", Severity.CRITICAL, "b")
        self.assertEqual(f.worst, Severity.CRITICAL)


# ---------------------------------------------------------------------------
# Contact mechanics
# ---------------------------------------------------------------------------

class TestCombiningRules(unittest.TestCase):
    def test_roughness_combines_in_quadrature(self):
        s = combined_roughness(Q(3.0, "um"), Q(4.0, "um"))
        self.assertAlmostEqual(s.to("um"), 5.0, places=9)

    def test_slope_combines_in_quadrature(self):
        m = combined_slope(Q(0.03, "-"), Q(0.04, "-"))
        self.assertAlmostEqual(m.si, 0.05, places=12)

    def test_harmonic_mean_of_equal_conductivities(self):
        k = harmonic_conductivity(Q(50.0, "W/mK"), Q(50.0, "W/mK"))
        self.assertAlmostEqual(k.si, 50.0, places=9)

    def test_harmonic_mean_dominated_by_poorer_conductor(self):
        """k_s is bounded above by twice the smaller conductivity."""
        k = harmonic_conductivity(Q(1.0, "W/mK"), Q(1000.0, "W/mK"))
        self.assertLess(k.si, 2.0)
        self.assertGreater(k.si, 1.0)


class TestCMYScaling(unittest.TestCase):
    """h_c must follow the published power laws exactly."""

    def _h(self, p_over_hc):
        f = FlagSet()
        return contact_conductance_plastic(
            Q(16.2, "W/mK"), Q(1.7, "um"), Q(0.11, "-"),
            Q(p_over_hc, "-"), f,
        ).h_c.si

    def test_plastic_exponent_is_0p95(self):
        h1, h2 = self._h(1e-3), self._h(1e-2)
        measured = math.log(h2 / h1) / math.log(10.0)
        self.assertAlmostEqual(measured, 0.95, places=9)

    def test_linear_in_slope_over_roughness(self):
        f = FlagSet()
        base = contact_conductance_plastic(
            Q(16.2, "W/mK"), Q(1.7, "um"), Q(0.11, "-"), Q(1e-3, "-"), f
        ).h_c.si
        doubled = contact_conductance_plastic(
            Q(16.2, "W/mK"), Q(1.7, "um"), Q(0.22, "-"), Q(1e-3, "-"), f
        ).h_c.si
        self.assertAlmostEqual(doubled / base, 2.0, places=9)

    def test_area_fraction_equals_dimensionless_pressure(self):
        f = FlagSet()
        r = contact_conductance_plastic(
            Q(16.2, "W/mK"), Q(1.7, "um"), Q(0.11, "-"), Q(4e-3, "-"), f
        )
        self.assertAlmostEqual(r.xi.si, 4e-3, places=12)

    def test_validity_window_flag_fires_below_range(self):
        f = FlagSet()
        contact_conductance_plastic(
            Q(16.2, "W/mK"), Q(1.7, "um"), Q(0.11, "-"),
            Q(CMY_PRESSURE_MIN / 10, "-"), f,
        )
        self.assertIn(Code.CMY_PRESSURE_RANGE, f.codes())

    def test_validity_window_flag_fires_above_range(self):
        f = FlagSet()
        contact_conductance_plastic(
            Q(16.2, "W/mK"), Q(1.7, "um"), Q(0.11, "-"),
            Q(CMY_PRESSURE_MAX * 2, "-"), f,
        )
        self.assertIn(Code.CMY_PRESSURE_RANGE, f.codes())

    def test_no_flag_inside_range(self):
        f = FlagSet()
        contact_conductance_plastic(
            Q(16.2, "W/mK"), Q(1.7, "um"), Q(0.11, "-"), Q(1e-3, "-"), f
        )
        self.assertNotIn(Code.CMY_PRESSURE_RANGE, f.codes())

    def test_saturated_contact_is_critical(self):
        f = FlagSet()
        r = contact_conductance_plastic(
            Q(16.2, "W/mK"), Q(1.7, "um"), Q(0.11, "-"), Q(1.5, "-"), f
        )
        self.assertIn(Code.CONTACT_SATURATED, f.codes())
        self.assertEqual(f.worst, Severity.CRITICAL)
        self.assertLessEqual(r.xi.si, 1.0)


class TestMeanPlaneSeparation(unittest.TestCase):
    def test_stays_near_1p5_sigma(self):
        """The exponent is small, so Y/sigma varies only weakly."""
        f = FlagSet()
        sig = Q(1.0, "um")
        ratios = [
            mean_plane_separation(sig, Q(p, "-"), f).si / sig.si
            for p in (1e-5, 1e-3, 1e-2)
        ]
        for r in ratios:
            self.assertGreater(r, 1.5)
            self.assertLess(r, 5.0)
        self.assertGreater(ratios[0], ratios[-1])   # decreasing with pressure


class TestRegimeSelection(unittest.TestCase):
    def test_smooth_hard_pair_is_elastic(self):
        j = build_joint("Si", "Ge", sigma_a_um=0.05, sigma_b_um=0.05,
                        slope_a=0.01, slope_b=0.01, pressure_MPa=5.0,
                        temperature_K=300.0)
        r = run_continuum(j, with_uncertainty=False)
        self.assertIs(r.regime.regime, Regime.ELASTIC)
        self.assertIn("elastic", r.branches)
        self.assertNotIn("plastic", r.branches)

    def test_rough_soft_pair_is_plastic(self):
        j = build_joint("SS304", "SS304", sigma_a_um=1.2, sigma_b_um=1.2,
                        slope_a=0.08, slope_b=0.08, pressure_MPa=2.0,
                        temperature_K=300.0)
        r = run_continuum(j, with_uncertainty=False)
        self.assertIs(r.regime.regime, Regime.PLASTIC)

    def test_override_is_respected(self):
        j = build_joint("Si", "Ge", sigma_a_um=0.05, sigma_b_um=0.05,
                        slope_a=0.01, slope_b=0.01, pressure_MPa=5.0,
                        temperature_K=300.0, regime_override="plastic")
        r = run_continuum(j, with_uncertainty=False)
        self.assertTrue(r.regime.forced)
        self.assertIn("plastic", r.branches)


# ---------------------------------------------------------------------------
# Gas gap
# ---------------------------------------------------------------------------

class TestGasProperties(unittest.TestCase):
    def test_air_mean_free_path_at_stp(self):
        """~67 nm for air at 300 K, 1 atm is a standard textbook figure.

        This is the strongest independent check available on the gas layer:
        it exercises the Sutherland viscosity fit, the molar mass and the
        kinetic-theory relation together.
        """
        g = load_gas("air", 101325.0)
        st = gas_state(g, Q(300.0, "K"))
        self.assertAlmostEqual(st.mfp.to("nm"), 67.0, delta=3.0)

    def test_air_conductivity_at_300K(self):
        g = load_gas("air", 101325.0)
        st = gas_state(g, Q(300.0, "K"))
        self.assertAlmostEqual(st.k_g.si, 0.0262, delta=0.0015)

    def test_mean_free_path_scales_inversely_with_pressure(self):
        g1 = load_gas("air", 101325.0)
        g2 = load_gas("air", 10132.5)
        T = Q(300.0, "K")
        r = gas_state(g2, T).mfp.si / gas_state(g1, T).mfp.si
        self.assertAlmostEqual(r, 10.0, places=6)

    def test_helium_has_longer_path_than_argon(self):
        T = Q(300.0, "K")
        he = gas_state(load_gas("He", 101325.0), T).mfp.si
        ar = gas_state(load_gas("Ar", 101325.0), T).mfp.si
        self.assertGreater(he, ar)


class TestGapConductance(unittest.TestCase):
    def test_vacuum_is_exactly_zero(self):
        self.assertEqual(vacuum_gap().si, 0.0)

    def test_continuum_limit_recovers_kg_over_Y(self):
        """At high gas pressure M -> 0 and h_g -> k_g/Y."""
        f = FlagSet()
        g = load_gas("air", 1e7)          # 100 bar: mfp is tiny
        st = gas_state(g, Q(300.0, "K"))
        Y = Q(5.0, "um")
        r = gap_conductance(st, Y, Q(0.9, "-"), Q(0.9, "-"), f)
        self.assertAlmostEqual(r.h_g.si, st.k_g.si / Y.si, delta=0.01 * r.h_g.si)

    def test_poor_accommodation_reduces_conductance(self):
        f = FlagSet()
        st = gas_state(load_gas("He", 101325.0), Q(300.0, "K"))
        Y = Q(5.0, "um")
        good = gap_conductance(st, Y, Q(0.9, "-"), Q(0.9, "-"), f).h_g.si
        poor = gap_conductance(st, Y, Q(0.05, "-"), Q(0.05, "-"), f).h_g.si
        self.assertLess(poor, good)

    def test_rarefied_flag_fires_at_low_pressure(self):
        f = FlagSet()
        st = gas_state(load_gas("He", 100.0), Q(300.0, "K"))
        gap_conductance(st, Q(5.0, "um"), Q(0.3, "-"), Q(0.3, "-"), f)
        self.assertIn(Code.GAP_RAREFIED, f.codes())
        self.assertIn(Code.GAP_CONTINUUM_BREAKDOWN, f.codes())

    def test_accommodation_out_of_range_rejected(self):
        f = FlagSet()
        st = gas_state(load_gas("air", 101325.0), Q(300.0, "K"))
        with self.assertRaises(ValueError):
            gap_conductance(st, Q(5.0, "um"), Q(0.0, "-"), Q(0.5, "-"), f)
        with self.assertRaises(ValueError):
            gap_conductance(st, Q(5.0, "um"), Q(1.5, "-"), Q(0.5, "-"), f)


# ---------------------------------------------------------------------------
# Atomistic placeholder
# ---------------------------------------------------------------------------

class TestDebyeHeatCapacity(unittest.TestCase):
    def test_dulong_petit_limit(self):
        """C -> 3 n k_B well above the Debye temperature."""
        n = 5.0e28
        C = debye_heat_capacity(Q(300.0, "K"), Quantity(si=n, unit="-"),
                                Q(6000.0, "K"))
        self.assertAlmostEqual(C.si / (3.0 * n * K_B), 1.0, delta=0.01)

    def test_low_temperature_t_cubed(self):
        """C ~ T^3 well below the Debye temperature."""
        n = Quantity(si=5.0e28, unit="-")
        theta = Q(600.0, "K")
        c1 = debye_heat_capacity(theta, n, Q(10.0, "K")).si
        c2 = debye_heat_capacity(theta, n, Q(20.0, "K")).si
        self.assertAlmostEqual(c2 / c1, 8.0, delta=0.3)


class TestDMM(unittest.TestCase):
    def test_identical_materials_give_half_transmission(self):
        """A diffuse interface between identical solids transmits exactly 1/2.

        This is the one exact result DMM has, and it is a sharp check on the
        velocity-sum weighting.
        """
        f = FlagSet()
        si = load_material("Si")
        est = dmm_conductance(si, si, Q(300.0, "K"), f)
        self.assertAlmostEqual(est.transmission.si, 0.5, places=12)

    def test_transmission_pair_sums_to_one(self):
        f = FlagSet()
        a, b = load_material("Si"), load_material("Ge")
        ab = dmm_conductance(a, b, Q(300.0, "K"), f).transmission.si
        ba = dmm_conductance(b, a, Q(300.0, "K"), f).transmission.si
        self.assertAlmostEqual(ab + ba, 1.0, places=12)

    def test_placeholder_is_flagged_and_untrusted(self):
        f = FlagSet()
        si = load_material("Si")
        est = dmm_conductance(si, si, Q(300.0, "K"), f)
        self.assertIn(Code.KAPITZA_PLACEHOLDER, f.codes())
        self.assertEqual(est.h_K.provenance, Provenance.PLACEHOLDER)
        self.assertLessEqual(est.h_K.confidence, Confidence.LOW)

    def test_quantum_flag_fires_below_half_debye(self):
        f = FlagSet()
        si = load_material("Si")       # theta_D = 645 K
        dmm_conductance(si, si, Q(100.0, "K"), f)
        self.assertIn(Code.QUANTUM_EFFECTS, f.codes())

    def test_quantum_flag_silent_above_debye(self):
        f = FlagSet()
        si = load_material("Si")
        dmm_conductance(si, si, Q(900.0, "K"), f)
        self.assertNotIn(Code.QUANTUM_EFFECTS, f.codes())


# ---------------------------------------------------------------------------
# The resistance network -- the corrections this package exists for
# ---------------------------------------------------------------------------

class TestAreaScaling(unittest.TestCase):
    """The central correction: h_K is per REAL area, h_c per APPARENT area."""

    def _net(self, xi=1e-3, h_K=300e6, h_c=2000.0, h_g=0.0):
        f = FlagSet()
        contact = contact_conductance_plastic(
            Q(16.2, "W/mK"), Q(1.7, "um"), Q(0.11, "-"), Q(xi, "-"), f
        )
        # force h_c to the requested value so the algebra is isolated
        contact.h_c = contact.h_c.with_(si=h_c)
        kap = user_kapitza(h_K)
        return combine(
            contact, kap, None, Q(2.0, "GPa"), f,
            gap_conductance_override=Quantity(si=h_g, unit="W/m2K"),
        ), f

    def test_kapitza_resistance_is_divided_by_area_fraction(self):
        net, _ = self._net(xi=1e-3, h_K=300e6)
        self.assertAlmostEqual(net.R_kapitza.si, net.R_kapitza_raw.si / 1e-3, places=15)

    def test_area_scaling_matters_by_orders_of_magnitude(self):
        """The naive sum understates the Kapitza term by 1/xi."""
        net, _ = self._net(xi=1e-3, h_K=300e6)
        self.assertAlmostEqual(net.R_kapitza.si / net.R_kapitza_raw.si, 1000.0, places=6)

    def test_series_resistances_add(self):
        net, _ = self._net()
        self.assertAlmostEqual(
            1.0 / net.h_solid.si,
            net.R_constriction.si + net.R_kapitza.si,
            places=15,
        )

    def test_parallel_conductances_add(self):
        net, _ = self._net(h_g=5000.0)
        self.assertAlmostEqual(
            net.h_joint.si, net.h_solid.si + net.h_gap.si, places=9
        )

    def test_perfect_interface_reduces_to_contact_plus_gap(self):
        """h_K -> infinity must recover h_c + h_g exactly."""
        net, _ = self._net(h_K=1e30, h_c=2000.0, h_g=5000.0)
        self.assertAlmostEqual(net.h_joint.si, 7000.0, delta=1.0)

    def test_vacuum_joint_equals_solid_path(self):
        net, _ = self._net(h_g=0.0)
        self.assertAlmostEqual(net.h_joint.si, net.h_solid.si, places=12)

    def test_spot_pressure_is_hardness_not_apparent_pressure(self):
        """NEMD must be run at the spot stress, which is of order H_c."""
        net, _ = self._net()
        self.assertAlmostEqual(net.spot_pressure.to("GPa"), 2.0, places=9)

    def test_zero_area_fraction_rejected(self):
        with self.assertRaises(ValueError):
            self._net(xi=0.0)


class TestSensitivityGate(unittest.TestCase):
    def test_negligible_interface_is_flagged(self):
        """A soft rough joint is constriction-limited; MD cannot help."""
        j = build_joint("SS304", "SS304", sigma_a_um=1.2, sigma_b_um=1.2,
                        slope_a=0.08, slope_b=0.08, pressure_MPa=2.0,
                        temperature_K=300.0)
        r = run_continuum(j, with_uncertainty=False)
        self.assertIn(Code.KAPITZA_NEGLIGIBLE, r.flags.codes())
        self.assertLess(r.primary.sensitivity["relative_span"], 0.05)

    def test_dominant_interface_is_flagged(self):
        """A near-perfect contact with a poor interface is interface-limited."""
        f = FlagSet()
        contact = contact_conductance_plastic(
            Q(150.0, "W/mK"), Q(0.01, "um"), Q(0.005, "-"), Q(1e-2, "-"), f
        )
        kap = user_kapitza(5e6)          # a poor interface, 5 MW/m^2K
        combine(contact, kap, None, Q(10.0, "GPa"), f,
                gap_conductance_override=Quantity(si=0.0, unit="W/m2K"))
        self.assertIn(Code.KAPITZA_DOMINANT, f.codes())

    def test_sensitivity_bracket_contains_nominal(self):
        net, _ = TestAreaScaling()._net(h_K=300e6, h_g=1000.0)
        s = net.sensitivity
        self.assertLessEqual(s["h_joint_low"], s["h_joint_nominal"])
        self.assertGreaterEqual(s["h_joint_high"], s["h_joint_nominal"])
        self.assertGreaterEqual(s["h_joint_perfect_interface"], s["h_joint_high"])


class TestMonteCarlo(unittest.TestCase):
    def test_band_brackets_the_nominal_value(self):
        j = build_joint("SS304", "SS304", sigma_a_um=1.2, sigma_b_um=1.2,
                        slope_a=0.08, slope_b=0.08, pressure_MPa=2.0,
                        temperature_K=300.0, gas="air", gas_pressure_kPa=101.325)
        r = run_continuum(j, n_samples=300)
        mc = r.uncertainty["h_joint"]
        self.assertLess(mc.p05, mc.nominal)
        self.assertGreater(mc.p95, mc.nominal)
        self.assertEqual(mc.n_failed, 0)

    def test_reproducible(self):
        def once():
            j = build_joint("SS304", "SS304", sigma_a_um=1.2, sigma_b_um=1.2,
                            slope_a=0.08, slope_b=0.08, pressure_MPa=2.0,
                            temperature_K=300.0)
            return run_continuum(j, n_samples=200).uncertainty["h_joint"].median
        self.assertEqual(once(), once())


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

class TestInputHandling(unittest.TestCase):
    def test_missing_slope_triggers_correlation_and_flag(self):
        j = build_joint("SS304", "SS304", sigma_a_um=1.2, sigma_b_um=1.2,
                        pressure_MPa=2.0, temperature_K=300.0)
        r = run_continuum(j, with_uncertainty=False)
        self.assertIn(Code.CMY_SLOPE_ESTIMATED, r.flags.codes())
        self.assertGreater(r.m_eff.si, 0.0)

    def test_gas_without_pressure_is_rejected(self):
        with self.assertRaises(ValueError):
            build_joint("SS304", "SS304", sigma_a_um=1.2, sigma_b_um=1.2,
                        pressure_MPa=2.0, temperature_K=300.0, gas="air")

    def test_vacuum_has_no_gap_result(self):
        j = build_joint("SS304", "SS304", sigma_a_um=1.2, sigma_b_um=1.2,
                        slope_a=0.08, slope_b=0.08, pressure_MPa=2.0,
                        temperature_K=300.0)
        r = run_continuum(j, with_uncertainty=False)
        self.assertIsNone(r.gap)
        self.assertEqual(r.primary.h_gap.si, 0.0)

    def test_gas_increases_joint_conductance(self):
        kw = dict(sigma_a_um=1.2, sigma_b_um=1.2, slope_a=0.08, slope_b=0.08,
                  pressure_MPa=2.0, temperature_K=300.0)
        vac = run_continuum(build_joint("SS304", "SS304", **kw),
                            with_uncertainty=False).primary.h_joint.si
        air = run_continuum(build_joint("SS304", "SS304", gas="air",
                                        gas_pressure_kPa=101.325, **kw),
                            with_uncertainty=False).primary.h_joint.si
        self.assertGreater(air, vac)

    def test_unverified_material_is_flagged(self):
        f = FlagSet()
        load_material("Si", f)
        self.assertIn(Code.MATERIAL_UNVERIFIED, f.codes())

    def test_unknown_material_raises(self):
        with self.assertRaises(KeyError):
            load_material("unobtainium")


class TestMonotonicity(unittest.TestCase):
    """Directional checks: does the answer move the way physics says it should?"""

    def _h(self, **over):
        kw = dict(sigma_a_um=1.2, sigma_b_um=1.2, slope_a=0.08, slope_b=0.08,
                  pressure_MPa=2.0, temperature_K=300.0)
        kw.update(over)
        return run_continuum(build_joint("SS304", "SS304", **kw),
                             with_uncertainty=False).primary.h_joint.si

    def test_higher_pressure_increases_conductance(self):
        self.assertGreater(self._h(pressure_MPa=10.0), self._h(pressure_MPa=1.0))

    def test_rougher_surface_decreases_conductance(self):
        self.assertLess(self._h(sigma_a_um=5.0, sigma_b_um=5.0), self._h())

    def test_helium_beats_air(self):
        """Helium conducts better despite worse accommodation."""
        he = self._h(gas="He", gas_pressure_kPa=101.325)
        air = self._h(gas="air", gas_pressure_kPa=101.325)
        self.assertGreater(he, air)

    def test_lower_gas_pressure_decreases_conductance(self):
        hi = self._h(gas="air", gas_pressure_kPa=101.325)
        lo = self._h(gas="air", gas_pressure_kPa=0.1)
        self.assertGreater(hi, lo)


if __name__ == "__main__":
    unittest.main(verbosity=2)
