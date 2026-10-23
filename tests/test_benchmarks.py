"""Validation against measured joint conductance.

Data: Bahrami (2004) PhD thesis, Appendix A.2, originally Hegazy (1985).
207 points, Ni 200 and SS 304, conforming rough flats **in a vacuum**, 0.41
to 8.89 MPa. See `benchmarks/SOURCES.md`.

Vacuum means the gap path is absent, so this isolates the contact path:
h_joint reduces to h_c in series with the (negligible) area-scaled Kapitza
term. The gap model is therefore NOT exercised here and remains unvalidated
against measurement.

Material properties come from the CSV, not from `materials.yaml`. The tests
ran at 110-145 C, where stainless conductivity is about 19 W/m.K against the
16.2 at 300 K in the database; using the room-temperature value would put a
~19 % error straight into h_c and the suite would be grading the database
rather than the correlation.

Agreement is reported as the geometric mean ratio and the spread separately,
because they fail differently: a systematic bias points at H_c or the slope,
scatter points at the correlation.
"""

from __future__ import annotations

import csv
import math
import statistics
import unittest
from pathlib import Path

from openkapitza.contact import (
    contact_conductance_plastic,
    contact_hardness,
    harmonic_conductivity,
)
from openkapitza.schema import FlagSet, Provenance, Quantity, load_material

DATA = Path(__file__).parent / "benchmarks" / "joint_conductance.csv"

# Realistic acceptance for an empirical contact correlation against
# experiment. The published figure for the extended CMY correlation on these
# same materials is 3.95-6.43 % RMS; this package implements the plainer
# 1982 recorrelation, so it is held to a looser standard.
FACTOR_TOLERANCE = 2.0
REQUIRED_WITHIN = 0.80
MAX_GEOMETRIC_BIAS = 1.5


def _f(row: dict, key: str) -> float | None:
    v = (row.get(key) or "").strip()
    return float(v) if v else None


def load_cases() -> list[dict]:
    if not DATA.exists():
        return []
    with open(DATA, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("case_id")]


def predict(row: dict) -> float:
    """Predicted contact conductance for one row, in W/(m^2 K).

    Built from the correlation directly rather than through `run_continuum`,
    so the per-row properties from the source are used instead of the
    database, and so the vacuum case is not routed through the gap model at
    all.
    """
    flags = FlagSet()

    sigma_eff = Quantity.parse(_f(row, "sigma_eff_um"), "um",
                               provenance=Provenance.USER_MEASURED)
    m_eff = Quantity.parse(_f(row, "slope_eff"), "-",
                           provenance=Provenance.USER_MEASURED)
    pressure = Quantity.parse(_f(row, "pressure_MPa"), "MPa",
                              provenance=Provenance.USER_MEASURED)

    k_s_override = _f(row, "k_s_W_mK")
    if k_s_override is not None:
        k_s = Quantity.parse(k_s_override, "W/mK",
                             provenance=Provenance.USER_MEASURED)
    else:
        mat = load_material(row["material_a"])
        k_s = harmonic_conductivity(mat.k, mat.k)

    mat_a = load_material(row["material_a"])
    mat_b = load_material(row["material_b"])
    c1 = _f(row, "c1_GPa")
    c2 = _f(row, "c2")
    if c1 is not None and c2 is not None:
        for mat in (mat_a, mat_b):
            mat.c1 = Quantity.parse(c1, "GPa", provenance=Provenance.LITERATURE)
            mat.c2 = Quantity.parse(c2, "-", provenance=Provenance.LITERATURE)

    hardness = contact_hardness(mat_a, mat_b, pressure, sigma_eff, m_eff, flags)
    contact = contact_conductance_plastic(
        k_s, sigma_eff, m_eff, hardness.P_over_Hc, flags
    )
    return contact.h_c.si


def _slope(x: list[float], y: list[float]) -> float:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = sum((a - mx) ** 2 for a in x)
    return num / den


class TestBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        if not cls.cases:
            raise unittest.SkipTest(
                f"no benchmark rows in {DATA}; see benchmarks/SOURCES.md"
            )
        cls.ratios = []
        cls.failures = []
        for row in cls.cases:
            predicted = predict(row)
            measured = _f(row, "h_measured_W_m2K")
            ratio = predicted / measured
            cls.ratios.append(ratio)
            if not (1 / FACTOR_TOLERANCE <= ratio <= FACTOR_TOLERANCE):
                cls.failures.append(
                    f"  {row['case_id']:14s} P={_f(row, 'pressure_MPa'):6.3f} MPa  "
                    f"predicted {predicted:9.1f}  measured {measured:9.1f}  "
                    f"ratio {ratio:5.2f}"
                )

    def test_light_load_flag_marks_the_under_predicted_points(self):
        """P0-7: the CMY_LIGHT_LOAD flag must sit where CMY reads low."""
        from openkapitza.contact import light_load_check
        from openkapitza.schema.flags import Code
        flagged, clear = [], []
        for row, ratio in zip(self.cases, self.ratios):
            flags = FlagSet()
            mat_a = load_material(row["material_a"]); mat_b = load_material(row["material_b"])
            for mat in (mat_a, mat_b):
                mat.c1 = Quantity.parse(_f(row, "c1_GPa"), "GPa", provenance=Provenance.LITERATURE)
                mat.c2 = Quantity.parse(_f(row, "c2"), "-", provenance=Provenance.LITERATURE)
            hard = contact_hardness(
                mat_a, mat_b,
                Quantity.parse(_f(row, "pressure_MPa"), "MPa", provenance=Provenance.USER_MEASURED),
                Quantity.parse(_f(row, "sigma_eff_um"), "um", provenance=Provenance.USER_MEASURED),
                Quantity.parse(_f(row, "slope_eff"), "-", provenance=Provenance.USER_MEASURED),
                FlagSet())
            light_load_check(hard.P_over_Hc, flags)
            (flagged if Code.CMY_LIGHT_LOAD in flags.codes() else clear).append(1 / ratio)
        gm = lambda v: math.exp(statistics.fmean(math.log(x) for x in v))
        self.assertGreater(len(flagged), 20)
        self.assertGreater(gm(flagged), 1.05)      # measured above predicted
        self.assertLess(gm(clear), 1.0)

    @classmethod
    def tearDownClass(cls):
        logs = math.log
        gm = math.exp(statistics.fmean([logs(r) for r in cls.ratios]))
        spread = (
            math.exp(statistics.pstdev([logs(r) for r in cls.ratios]))
            if len(cls.ratios) > 1 else 1.0
        )
        within = sum(
            1 for r in cls.ratios
            if 1 / FACTOR_TOLERANCE <= r <= FACTOR_TOLERANCE
        ) / len(cls.ratios)
        print(
            f"\n[benchmark] {len(cls.ratios)} points | geometric mean ratio "
            f"{gm:.3f} | log-spread factor {spread:.2f} | "
            f"{within * 100:.0f} % within {FACTOR_TOLERANCE:g}x"
        )
        if cls.failures:
            print(f"[benchmark] {len(cls.failures)} outside the band:")
            print("\n".join(cls.failures[:12]))
            if len(cls.failures) > 12:
                print(f"  ... and {len(cls.failures) - 12} more")

    def test_most_points_within_a_factor_of_two(self):
        within = sum(
            1 for r in self.ratios
            if 1 / FACTOR_TOLERANCE <= r <= FACTOR_TOLERANCE
        ) / len(self.ratios)
        self.assertGreaterEqual(
            within, REQUIRED_WITHIN,
            f"only {within * 100:.0f} % of {len(self.ratios)} points within a "
            f"factor of {FACTOR_TOLERANCE:g}",
        )

    def test_no_large_systematic_bias(self):
        """Bias and scatter fail differently and are checked separately.

        A consistent over- or under-prediction points at H_c or the asperity
        slope; scatter points at the correlation itself.
        """
        gm = math.exp(statistics.fmean([math.log(r) for r in self.ratios]))
        self.assertLess(
            abs(math.log(gm)), math.log(MAX_GEOMETRIC_BIAS),
            f"systematic bias: geometric mean ratio {gm:.3f}",
        )

    def test_pressure_exponent_matches_within_each_specimen(self):
        """The exponent is a stronger test than the magnitude.

        Magnitude absorbs any error in H_c; the slope d(ln h)/d(ln P) does
        not, so it tests the correlation's form rather than its inputs.
        """
        groups: dict[str, list[tuple[float, float, float]]] = {}
        for row in self.cases:
            key = row["case_id"].rsplit("-", 1)[0]
            groups.setdefault(key, []).append(
                (_f(row, "pressure_MPa"), predict(row),
                 _f(row, "h_measured_W_m2K"))
            )
        checked = 0
        for key, pts in sorted(groups.items()):
            if len(pts) < 3:
                continue
            pts.sort()
            lp = [math.log(p) for p, _, _ in pts]
            lpred = [math.log(v) for _, v, _ in pts]
            lmeas = [math.log(v) for _, _, v in pts]
            sp, sm = _slope(lp, lpred), _slope(lp, lmeas)
            checked += 1
            self.assertAlmostEqual(
                sp, sm, delta=0.25,
                msg=(f"{key}: predicted d(ln h)/d(ln P) = {sp:.3f} vs "
                     f"measured {sm:.3f}"),
            )
        self.assertGreater(checked, 0, "no specimen group had 3+ points")

    def test_every_row_is_inside_the_cmy_validity_window(self):
        """A row outside the fitted range would be testing extrapolation."""
        from openkapitza.contact import CMY_PRESSURE_MAX, CMY_PRESSURE_MIN
        outside = []
        for row in self.cases:
            flags = FlagSet()
            mat_a = load_material(row["material_a"])
            mat_b = load_material(row["material_b"])
            mat_a.c1 = mat_b.c1 = Quantity.parse(_f(row, "c1_GPa"), "GPa")
            mat_a.c2 = mat_b.c2 = Quantity.parse(_f(row, "c2"), "-")
            ratio = contact_hardness(
                mat_a, mat_b,
                Quantity.parse(_f(row, "pressure_MPa"), "MPa"),
                Quantity.parse(_f(row, "sigma_eff_um"), "um"),
                Quantity.parse(_f(row, "slope_eff"), "-"),
                flags,
            ).P_over_Hc.si
            if not (CMY_PRESSURE_MIN <= ratio <= CMY_PRESSURE_MAX):
                outside.append((row["case_id"], ratio))
        self.assertEqual(
            outside, [],
            f"{len(outside)} rows fall outside the CMY validity window",
        )


class TestMeasuredPowerExponents(unittest.TestCase):
    """Check the correlation's exponents against measured ones.

    Song & Yovanovich (1988), *Relative Contact Pressure: Dependence on
    Surface Roughness and Vickers Microhardness*, J. Thermophysics and Heat
    Transfer 2(1), 43-47, fit power laws to Hegazy's data:

        Table 6, h_c proportional to P^n1:
            Ni 200 0.95, SS 304 0.97, Zr-4 0.94, Zr-2.5Nb 0.95
        Table 8, h_c proportional to (sigma/m)^n2:
            Ni 200 -0.76, SS 304 -0.77, Zr-4 -0.76, Zr-2.5Nb -0.75

    These test the SHAPE of the correlation rather than its magnitude, which
    makes them independent of any error in H_c or in the material data. They
    need no new measurements -- only the published exponents.
    """

    MEASURED_N1 = {"Ni200": 0.95, "SS304": 0.97}
    MEASURED_N2 = {"Ni200": -0.76, "SS304": -0.77}

    def _h_c(self, material, pressure_MPa, sigma_um, slope, k_s=19.0):
        from openkapitza.contact import (contact_conductance_plastic,
                                         contact_hardness)
        flags = FlagSet()
        mat_a = load_material(material)
        mat_b = load_material(material)
        for m in (mat_a, mat_b):
            m.c1 = Quantity.parse(6.3, "GPa")
            m.c2 = Quantity.parse(-0.26, "-")
        sigma = Quantity.parse(sigma_um, "um")
        m_eff = Quantity.parse(slope, "-")
        P = Quantity.parse(pressure_MPa, "MPa")
        hardness = contact_hardness(mat_a, mat_b, P, sigma, m_eff, flags)
        return contact_conductance_plastic(
            Quantity.parse(k_s, "W/mK"), sigma, m_eff,
            hardness.P_over_Hc, flags,
        ).h_c.si

    def test_pressure_exponent_matches_measurement(self):
        for material, n1 in self.MEASURED_N1.items():
            pressures = [0.5, 1.0, 2.0, 4.0, 8.0]
            hs = [self._h_c(material, P, 2.71, 0.116) for P in pressures]
            slope = _slope([math.log(p) for p in pressures],
                           [math.log(h) for h in hs])
            self.assertAlmostEqual(
                slope, n1, delta=0.05,
                msg=f"{material}: d(ln h_c)/d(ln P) = {slope:.3f} vs "
                    f"measured {n1}",
            )

    def test_roughness_slope_exponent_matches_measurement(self):
        """h_c against sigma/m at fixed pressure.

        The exponent is not simply -1 from the (m/sigma) prefactor: H_c
        itself depends on sigma/m through the Vickers correlation, and the
        two contributions combine to about -0.75.
        """
        for material, n2 in self.MEASURED_N2.items():
            # vary sigma at fixed slope, so sigma/m spans a wide range
            sigmas = [0.5, 1.5, 4.0, 8.0, 14.0]
            ratio = [s / 0.15 for s in sigmas]
            hs = [self._h_c(material, 2.0, s, 0.15) for s in sigmas]
            slope = _slope([math.log(r) for r in ratio],
                           [math.log(h) for h in hs])
            self.assertAlmostEqual(
                slope, n2, delta=0.05,
                msg=f"{material}: d(ln h_c)/d(ln(sigma/m)) = {slope:.3f} vs "
                    f"measured {n2}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
