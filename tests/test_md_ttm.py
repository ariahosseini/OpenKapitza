"""Two-temperature NEMD deck for metal/non-metal interfaces (P1-14)."""

from __future__ import annotations

import math
import os
import re
import tempfile
import time
import unittest
from pathlib import Path

from openkapitza.md.nemd import (
    NEMDGeometry, NEMDSpec, build_input_deck, check_spec, headroom, ttm_properties,
    validate_deck,
)
from openkapitza.md.ttm import (
    COOLING_LENGTHS_EXCLUDED, KB, TTMSpec, cooling_length, fit_exclusion,
    friction_gamma_p, grid, number_density, read_electron_profile,
    sommerfeld_volumetric, to_metal_units, compare_runs,
)
from openkapitza.schema import FlagSet
from openkapitza.schema.flags import Code, Severity


def _spec(**kw) -> NEMDSpec:
    g = NEMDGeometry(wall_thickness=0.8e-9, reservoir_thickness=1.0e-9,
                     bulk_a_length=4e-9, bulk_b_length=30e-9, cross_section=2.2e-9,
                     bin_width=0.4e-9)
    base = dict(geometry=g, element_a="Si", element_b="Al", lattice_a=5.431e-10,
                lattice_b=4.0495e-10, temperature=300.0, pair_style="meam",
                pair_file="library.meam", coeff_args="library.meam AlS SiS AlSiMgCuFe.meam SiS AlS",
                lattice_style_a="diamond", lattice_style_b="fcc", ttm=TTMSpec("Al"))
    base.update(kw)
    return NEMDSpec(**base)


class Units(unittest.TestCase):
    def test_metal_unit_conversions(self):
        c, k = to_metal_units(1.602176634e11, 1602.176634)
        self.assertAlmostEqual(c, 1.0, places=9)        # eV/(K A^3)
        self.assertAlmostEqual(k, 1.0, places=9)        # eV/(ps A K)

    def test_friction_reproduces_the_coupling(self):
        """g = 3 n k_B gamma / m: invert the per-atom friction."""
        g, m, n = 2.45e17, 26.9815, number_density("Al")
        gp = friction_gamma_p(g, m, n)                  # g/mol/ps
        gamma_si = gp / (1e3 * 6.02214076e23 * 1e-12)
        self.assertAlmostEqual(3 * n * KB * gamma_si / (m * 1.66053906660e-27) / g, 1.0, places=9)

    def test_sommerfeld_matches_independent_cu_value(self):
        # Wang, Ruan & Roy 2012 use 96.8 J/m^3K^2 for Cu.
        self.assertAlmostEqual(sommerfeld_volumetric("Cu") / 96.8, 1.0, delta=0.02)
        self.assertAlmostEqual(sommerfeld_volumetric("Al"), 135.0, delta=1.0)


class Grid(unittest.TestCase):
    def test_electrons_only_in_the_metal(self):
        s = _spec()
        text = ttm_properties(s)
        rows = [ln.split() for ln in text.splitlines() if ln and not ln.startswith("#")]
        nz, metal = grid(s, s.ttm, float(text.split("(box ")[1].split(" A")[0]))
        self.assertEqual(len(rows), nz)
        for r, is_metal in zip(rows, metal):
            c, k, gp = map(float, r[3:6])
            self.assertGreater(c, 0.0)                  # update divides by C_vol
            if is_metal:
                self.assertGreater(k, 0.0); self.assertGreater(gp, 0.0)
            else:
                self.assertEqual(k, 0.0); self.assertEqual(gp, 0.0)
        # metal cells are one contiguous block above the interface
        idx = [i for i, m in enumerate(metal) if m]
        self.assertEqual(idx, list(range(idx[0], idx[-1] + 1)))
        box = float(text.split("(box ")[1].split(" A")[0])
        centre_first = (idx[0] + 0.5) * box / nz * 1e-10
        centre_last = (idx[-1] + 0.5) * box / nz * 1e-10
        self.assertGreater(centre_first, s.geometry.interface_z)    # no electrons in Si
        self.assertLess(centre_last, s.geometry.z_cold_end)          # none in the top wall

    def test_read_data_needs_the_box_height(self):
        s = _spec(structure_mode="read_data")
        with self.assertRaises(ValueError):
            build_input_deck(s)


class Deck(unittest.TestCase):
    def test_periodic_z_with_guarded_vacuum_and_fix_after_equilibration(self):
        deck = build_input_deck(_spec())
        self.assertIn("boundary        p p p", deck)
        self.assertEqual(validate_deck(deck), [])
        i_eq = deck.index("unfix           equil")
        i_ttm = deck.index("fix             twotemp")
        i_nve = deck.index("fix             nve gmobile nve")
        i_relax = deck.index("change_box")
        self.assertLess(i_relax, i_eq); self.assertLess(i_eq, i_ttm); self.assertLess(i_ttm, i_nve)
        self.assertRegex(deck, r'if "\$\(v_zgap\) < 12\.0000" then .* "quit 1"')
        self.assertIn("group           gmetal intersect gmetal_all gmobile", deck)
        # nothing after the fix changes the box
        self.assertNotRegex(deck[i_ttm:], r"(?m)^(fix\s+\S+\s+\S+\s+npt|change_box|fix\s+\S+\s+\S+\s+deform)")

    def test_mean_electron_temperature_is_averaged_in_production(self):
        s = _spec()
        deck = build_input_deck(s)
        i_reset = deck.rindex("reset_timestep  0")
        i_avg = deck.index("fix             te_avg all ave/time")
        self.assertLess(i_reset, i_avg)          # after the clock restarts, like the profile
        m = re.search(r"variable\s+te_metal equal \(f_twotemp\[1\]-300\.00\*(\S+)\*(\d+)\*lx\*ly\*lz/(\d+)\)/\((\S+)\*(\d+)\*", deck)
        n_free, nz, n_metal = int(m.group(2)), int(m.group(3)), int(m.group(5))
        self.assertEqual(n_free + n_metal, nz)
        # energy of all cells at uniform T maps back to T
        c = float(m.group(1)); V = 7.0 * 8.0 * 233.0
        e_total = 340.0 * c * n_metal * V / nz + 300.0 * c * n_free * V / nz
        self.assertAlmostEqual((e_total - 300 * c * n_free * V / nz) / (c * n_metal * V / nz), 340.0)

    def test_headroom_carries_the_vacuum_guard(self):
        with_ttm, without = _spec(), _spec(ttm=None)
        self.assertAlmostEqual(headroom(with_ttm) - headroom(without), 12e-10, places=15)

    def test_non_ttm_decks_unchanged_and_periodic_z_needs_the_guard(self):
        deck = build_input_deck(_spec(ttm=None))
        self.assertIn("boundary        p p f", deck)
        self.assertNotIn("ttm/thermal", deck)
        bad = deck.replace("boundary        p p f", "boundary        p p p")
        self.assertTrue(any("z boundary" in p for p in validate_deck(bad)))

    def test_audit_passes_a_ttm_deck(self):
        import importlib.util
        path = Path(__file__).resolve().parent.parent / "scripts" / "audit_nemd.py"
        sp = importlib.util.spec_from_file_location("audit_nemd", path)
        mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
        self.assertEqual(mod.audit_one(_spec(), "ttm"), [])

    def test_needs_relaxation(self):
        with self.assertRaises(ValueError):
            build_input_deck(_spec(relax_structure=False))


class CampaignRecord(unittest.TestCase):
    def test_round_trip_keeps_ttm_and_recent_fields(self):
        from openkapitza.md.campaign import _spec_record, spec_from_record
        import json
        s = _spec(spot_pressure_GPa=1.0, spot_pressure_mode="strain", couple_xy=False,
                  ttm=TTMSpec("Al", exclusion_lc=2.0))
        back = spec_from_record(json.loads(json.dumps(_spec_record(s))))
        self.assertEqual(back.ttm, s.ttm)
        self.assertEqual((back.spot_pressure_mode, back.couple_xy, back.types_a, back.types_b),
                         (s.spot_pressure_mode, s.couple_xy, s.types_a, s.types_b))
        self.assertEqual(build_input_deck(back), build_input_deck(s))


class Caveats(unittest.TestCase):
    def test_cross_interface_channel_always_flagged(self):
        f = FlagSet(); check_spec(_spec(), f)
        flag = next(x for x in f.flags if x.code == Code.NEMD_PHONON_ONLY_METAL)
        self.assertIn("NOT modelled", flag.message)

    def test_short_metal_and_reduced_exclusion_are_warned(self):
        s = _spec(geometry=NEMDGeometry(wall_thickness=0.8e-9, reservoir_thickness=1.0e-9,
                                        bulk_a_length=4e-9, bulk_b_length=8e-9,
                                        cross_section=2.2e-9, bin_width=0.4e-9),
                  ttm=TTMSpec("Al", exclusion_lc=1.0))
        f = FlagSet(); check_spec(s, f)
        msgs = [x.message for x in f.flags if x.code == Code.NEMD_FEW_BINS]
        self.assertTrue(any("cooling lengths" in m for m in msgs))
        self.assertTrue(any("Metal-side fit starts 1" in m for m in msgs))

    def test_thin_electron_cells_are_warned(self):
        f = FlagSet(); check_spec(_spec(), f)            # 2.2 nm wide, 1 nm cells
        self.assertTrue(any("atoms per electron cell" in x.message for x in f.flags))
        f = FlagSet(); check_spec(_spec(ttm=TTMSpec("Al", cell_ang=80.0)), f)
        self.assertTrue(any("longer than the cooling length" in x.message for x in f.flags))

    def test_fit_exclusion_is_three_cooling_lengths(self):
        lc = cooling_length(2.45e17, 10.67)
        self.assertAlmostEqual(fit_exclusion(TTMSpec("Al")), COOLING_LENGTHS_EXCLUDED * lc)
        self.assertAlmostEqual(lc * 1e9, 6.6, delta=0.1)

    def test_unsourced_metal_refused(self):
        with self.assertRaises(ValueError):
            TTMSpec("Unobtainium").resolved()


class Compare(unittest.TestCase):
    def test_series_inversion_and_unresolved_case(self):
        f = FlagSet()
        c = compare_runs(460e6, 5e6, 347e6, 5e6, f)
        self.assertAlmostEqual(1 / c.h_ep, 1 / 347e6 - 1 / 460e6)
        self.assertAlmostEqual(c.reduction, 1 - 347 / 460)
        f = FlagSet()
        c = compare_runs(460e6, 100e6, 430e6, 100e6, f)
        self.assertIsNone(c.h_ep)
        self.assertIn(Code.NEMD_ERROR_UNDERSTATED, f.codes())


class ElectronProfile(unittest.TestCase):
    def test_reads_the_latest_snapshots_by_write_time(self):
        with tempfile.TemporaryDirectory() as d:
            for k, (step, T) in enumerate([(5000, 999.0), (1000, 300.0), (2000, 310.0)]):
                p = Path(d) / f"T_e.out.{step}"
                p.write_text("# header\n" + "".join(f"1 1 {iz} {T + iz}\n" for iz in (2, 1, 3)))
                os.utime(p, ns=(k * 10**9, k * 10**9))     # written in this order
            z, T, n = read_electron_profile(d, 30.0, frames=2)
            self.assertEqual(n, 2)
            self.assertEqual(list(T), [306.0, 307.0, 308.0])   # mean of 300+iz, 310+iz
            self.assertAlmostEqual(z[0], 5.0)


if __name__ == "__main__":
    unittest.main()
