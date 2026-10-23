# Built structures

Kept so expensive builds are not repeated. Each is labelled with how it was
made; the flags it raised are part of its description, not a footnote.

| File | What | How | Validation |
|---|---|---|---|
| `aSiO2_munetoh_1e13Ks_27.16A.data` | a-SiO₂, 3072 atoms, 27.16 × 27.16 × 61.1 Å (lateral = 5 Si cells of the potential's a0 5.4321 Å) | `SiO.tersoff` (Munetoh 2007), β-cristobalite start, melt 6000 K, quench **10¹³ K/s**, zero-pressure anneal | **Fails the gate**: 77 % fourfold Si (15 % five-, 7 % threefold), Si–O 1.65 Å, O–O 2.71 Å; density 2.27 g/cm³ and angles pass. See open issue P2-2. |
| `Si100_aSiO2_strong_4nm.data` | Si(100) / a-SiO₂ NEMD cell, 4 nm each side, types 1 = crystal Si, 2 = oxide Si, 3 = O; box includes piston headroom | `build_si_sio2.py --bonding strong`: bridging-O capped Si surface, 1.6 Å gap, 2000 K oxide-side anneal | Interface **strong**: 15.9 Si(c)–O bonds/nm² (117 % of abrupt full oxidation); Si⁺¹/⁺²/⁺³ 6/33/15 % within 4 Å. Glass as above. Si side fails the elastic gate (P2-4). |
| `Si100_aSiO2_strong_4nm.nemd.in` | Matching NEMD deck (read-in mode, piston relaxation, 500 ps production at 0.5 fs) | `build_si_sio2.py` | — |

Run the NEMD deck on the cluster from a directory holding the `.data` file
renamed to `interface.data` and `SiO.tersoff`.
