"""Phonon (VDOS) gate (md/phonon_dos.py, KB 37)."""

from __future__ import annotations

import os
import unittest

import numpy as np

from openkapitza.md.elastic import CubicSpec
from openkapitza.md.phonon_dos import VDOSSpec, build_vdos_deck, gate, peaks, vdos

SI = CubicSpec("Si(D)", "diamond", 5.431, 28.0855, "tersoff", "SiCGe.tersoff", "Si(D)", "Si")


class Spectrum(unittest.TestCase):
    def synthetic(self, freqs_thz, amps, sample_fs=4.0, n=4000, atoms=8):
        t = np.arange(n) * sample_fs * 1e-15
        rng = np.random.default_rng(1)
        v = np.zeros((n, atoms, 3))
        for f0, a in zip(freqs_thz, amps):
            ph = rng.uniform(0, 2 * np.pi, (atoms, 3))
            v += a * np.sin(2 * np.pi * f0 * 1e12 * t[:, None, None] + ph)
        return vdos(v, sample_fs)

    def test_peaks_found_where_the_modes_are(self):
        f, g = self.synthetic([4.5, 6.8, 15.5], [1.0, 0.6, 1.2])
        pk = peaks(f, g)
        self.assertAlmostEqual(pk["ta"], 4.5, delta=0.15)
        self.assertAlmostEqual(pk["optical"], 15.5, delta=0.15)
        self.assertAlmostEqual(np.trapezoid(g, f) if hasattr(np, "trapezoid")
                               else np.trapz(g, f), 1.0, places=6)

    def test_gate(self):
        self.assertTrue(gate({"optical": 15.4, "ta": 4.6, "f_max": 16}, "Si").passes)
        r = gate({"optical": 16.8, "ta": 4.6, "f_max": 17.7}, "Si")      # SW, measured
        self.assertFalse(r.passes)
        self.assertAlmostEqual(r.errors["optical"], 16.8 / 15.6 - 1)

    def test_deck(self):
        d = build_vdos_deck(VDOSSpec(SI))
        self.assertIn("dump            v all custom 4 velocities.dump id vx vy vz", d)
        self.assertIn("pair_coeff      * * SiCGe.tersoff Si(D)", d)
        self.assertLess(d.index("minimize"), d.index("velocity"))


@unittest.skipUnless(os.environ.get("OPENKAPITZA_LMP") and os.environ.get("OPENKAPITZA_POTENTIALS"),
                     "set OPENKAPITZA_LMP and OPENKAPITZA_POTENTIALS")
class WithLammps(unittest.TestCase):
    def test_tersoff_si_passes(self):
        from openkapitza.md.phonon_dos import run_vdos
        f, g = run_vdos(os.environ["OPENKAPITZA_LMP"], VDOSSpec(SI),
                        os.environ["OPENKAPITZA_POTENTIALS"])
        self.assertTrue(gate(peaks(f, g), "Si").passes)


if __name__ == "__main__":
    unittest.main()
