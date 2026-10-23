# Benchmark data

`joint_conductance.csv` holds **207 measured points**: Ni 200 and SS 304,
conforming rough flats in a vacuum, 0.41–8.89 MPa, from Bahrami (2004) PhD
thesis Appendix A.2 (data originally Hegazy 1985). `scripts/build_benchmark_csv.py`
regenerates it and carries the transcription, so the numbers are auditable.

## Result

| | |
|---|---|
| points | 207 (115 Ni 200, 92 SS 304) |
| geometric mean ratio | **1.030** |
| RMS difference | **12.7 %** |
| within a factor of 2 | **100 %** |
| worst single point | 0.62× |

Per specimen, across roughness from 0.48 to 13.94 µm — a factor of 29 — with
no trend in the ratio:

| specimen | σ (µm) | m | n | gm ratio | RMS % |
|---|---|---|---|---|---|
| PNI0102 | 0.90 | 0.110 | 23 | 0.994 | 11.5 |
| PNI0304 | 3.43 | 0.190 | 23 | 1.019 | 11.7 |
| PNI0506 | 4.24 | 0.188 | 23 | 1.105 | 14.7 |
| PNI0708 | 9.53 | 0.228 | 23 | 0.986 | 10.8 |
| PNI0910 | 13.94 | 0.233 | 23 | 1.048 | 9.9 |
| PSS0102 | 0.48 | 0.072 | 23 | 0.988 | 7.0 |
| PSS0304 | 2.71 | 0.116 | 23 | 1.144 | 19.5 |
| PSS0506 | 5.88 | 0.146 | 23 | 0.994 | 12.9 |
| PSS0708 | 10.95 | 0.190 | 23 | 1.003 | 12.2 |

Ni 200 and SS 304 both land at gm = 1.030 independently, which is better
evidence than either alone: two metals differing fourfold in conductivity
agreeing to the same factor is hard to get by accident.

For scale, the published figure for the **extended** CMY correlation on these
same materials is 3.95–6.43 % RMS. This package implements the plainer 1982
recorrelation, so 12.7 % is the expected neighbourhood rather than a shortfall.

**One specimen is in the elastoplastic transition.** PSS0102 has ψ = 2.04,
inside the ambiguous band, so the regime selector reports both branches. The
measurement prefers plastic (gm 0.988, RMS 7.0 %) over elastic (gm 0.906,
RMS 10.5 %) — but the elastic branch is not absurd there, which is what the
band is supposed to mean.

## Extended set — `joint_conductance_ext.csv` (23 Oct 2026)

174 points (158 usable) from the same thesis appendix, built by
`scripts/build_benchmark_ext_csv.py` and tested in `tests/test_benchmarks_ext.py`:

| set | table | pairs | pred/meas (gm) |
|---|---|---|---|
| Antonetti | A.40–A.43 | Ni 200 ×4 | 0.92–1.02 |
| Milanez et al. | A.63 | SS 304 bead-blasted/lapped, from 16 kPa | 0.96 |
| McWaid | A.64–A.65 | SS 304 ×2 (c₂ = 0) | 1.07–1.12 |
| Nho | A.72–A.73 | Ni 200 ground/lapped | 0.83–0.87 |
| Nho | A.68–A.69 | Al 6061-T6 ground/lapped | 0.87–0.98 |
| Nho | A.66, A.67, A.70, A.71 | Al 6061-T6 / Ni 200 | 0.92–1.01 |

All 158: 0.950, RMS 13.3 %, 100 % within 2×. Al-containing: 0.951, 11.1 %.
Rows with R_exp printed too coarsely (0.005/R_exp > 10 %) are excluded.
Our CMY matches Bahrami's own model column within 3 % on every same-material
set, so the transcription and area are confirmed independently of the data.
Zircaloy-4, Zr-2.5Nb and silver-coated Ni are left out (not packaging, or a
coated contact). Nho's ground surfaces are anisotropic.

## Measured interface conductance — `kapitza_measured.csv`

Al(80 nm)/Si(100) by TDTR at room temperature, eight surface treatments:
Hopkins, Phinney, Serrano & Beechem, PRB 82, 085307 (2010), Table I.
158–193 MW/m²K with the native oxide stripped, 115–144 with it. The Debye DMM
here (321) is 1.66× the best value. Tested in `tests/test_kapitza_measured.py`.
This is the first measured target for the Kapitza term (the area-scaling
bridge itself is still untested, below).

## Grease TIMs — `tim_greases.csv`

Five commercial greases, ASTM D5470 at 139 kPa, BLT 29–266 µm: DeVoto et al.,
NREL/CP-5400-68014 (ITherm 2017), Tables 1–3. R = BLT/k + R_c with
k 0.7–3.8 W/mK, R_c 4.2–28.3 mm²K/W. Tested in `tests/test_tim_greases.py`.

## What this does NOT validate

- **The gap model.** Every point in `joint_conductance.csv` is in a vacuum, so
  `h_g` is zero throughout and nothing in `openkapitza/gap` is exercised by
  the headline statistic. A separate, **digitised** dataset now gives an
  exploratory test — see "Digitised gap data" below. That test is good enough
  to show the model has roughly the right magnitude and to expose a trend; it
  is **not** a validation, and it is deliberately excluded from the 207-point
  result above.
- **The area-scaling bridge.** The Kapitza term is negligible at these
  conductances, so `1/(ξ·h_K)` contributes nothing measurable and the central
  Phase-0 correction is untested against experiment. (The Al/Si TDTR set tests
  h_K itself, not its scaling into a rough joint.)
- **Air.** The extended set adds Al 6061-T6, but every joint is in a vacuum.
- **The elastic branch.** Eight of nine specimens are plastic and the ninth
  is ambiguous. Mikic's correlation has no clean test here.
- **`materials.yaml`.** Properties come from the CSV, per row, as the source
  reported them. The tests ran at 110–145 °C where stainless conductivity is
  ~19 W/m·K against the 16.2 at 300 K in the database; using the database
  value would put ~19 % into `h_c` and the suite would be grading the
  database rather than the correlation.

## Gap-model inputs, checked against measurement

From Song & Yovanovich (1988) Part II, `mhtl88-7.pdf`, Table 1 and the gas
conductivity correlations:

| quantity | agreement |
|---|---|
| mean free path, He / Ar / N₂ at 288 K, 760 torr | within **0.3–1.0 %** of Kennard |
| k_g over 27–300 °C, three gases | within **1–4 %** |
| γ, Pr | agree; N₂ Prandtl corrected 0.72 → 0.69 |

The paper also states the gap model in the same form this package uses,
`M = ((2−TAC₁)/TAC₁ + (2−TAC₂)/TAC₂)·β·Λ`, with `β = (2γ/(γ+1))·(1/Pr)`.

**Accommodation coefficients, measured.** Song & Yovanovich (1989),
`mhtl89-4.pdf`, Table 2 gives least-square estimates on the same bead-blasted
specimens:

| gas | SS 304 | Ni 200 |
|---|---|---|
| He | 0.67, 0.58 | 0.61, 0.60 |
| Ar | 0.96, 0.88 | 0.93, 0.87 |
| N₂ | 0.87, 0.79 | 0.81, 0.80 |

The previous helium default in `gases.yaml` was 0.30, from clean-surface
literature. Table 3 of that paper confirms the published spread is real
(0.327 for Zircaloy-2 up to 0.65 for silver) — but bead-blasted, air-exposed
engineering surfaces sit near the top of it. Since α = (2−a₁)/a₁ + (2−a₂)/a₂,
0.30 → 0.61 changes M by a factor of 1.9.

## Measured power exponents

Song & Yovanovich (1988), `mhtl88-8.pdf`, Tables 6 and 8 fit power laws to
Hegazy's data. These test the correlation's **shape**, independent of any
error in H_c or the material properties:

| | measured | this package |
|---|---|---|
| h_c ∝ P^n₁, Ni 200 | 0.95 | ✓ within 0.05 |
| h_c ∝ P^n₁, SS 304 | 0.97 | ✓ within 0.05 |
| h_c ∝ (σ/m)^n₂, Ni 200 | −0.76 | ✓ within 0.05 |
| h_c ∝ (σ/m)^n₂, SS 304 | −0.77 | ✓ within 0.05 |

`mhtl88-8.pdf` Eq. (16) also settles, in print, where the constant 1.62 sits:

    P/H_c = [ P / ( H_b·(1.62σ/(m·d_b))^c₀ ) ]^(1/(1+0.071c₀))

Inside the power. This confirms the implementation and the earlier numerical
verification against the implicit system; a secondary transcription that put
it outside was wrong.

## Digitised gap data — exploratory only

`gap_conductance_digitised.csv`, 113 points, produced by
`scripts/digitize_gap_figures.py` from Song & Yovanovich (1989),
`mhtl89-4.pdf` p. 11, Figures 3–6: joint conductance against gas pressure for
four conforming rough specimen pairs (2 × SS 304, 2 × Ni 200) in He, Ar and
N₂. Kept in a **separate file**, tagged `digitised`, and excluded from the
headline statistic.

**Why separate.** Digitising a figure introduces error that belongs to the
extraction script, not to the source. These numbers are good enough to test
whether the gap model has the right shape and magnitude. They are not good
enough to quote as measurements.

### Extraction quality, honestly

| Check | Result |
|---|---|
| Upper gas pressure recovered vs published | within 1–3.5 % — x-calibration sound |
| Upper `h_j` recovered vs published | within 1–5 % — y-calibration sound |
| **Lower gas pressure recovered vs published** | **+25 to +45 % HIGH — biased** |
| Vacuum markers detected | 0 of 4 plots |

The lower-bound bias is the one that matters. The extractor systematically
loses the dense, overlapping low-pressure marker cluster, which is exactly
the rarefied regime where the gap model is hardest and most interesting.
Recovered coverage is 12.5–712 torr against a published 9.4–711. Marker
classification by gas was not independently verified. Nothing here should be
read as covering the rarefied limit properly.

### What the comparison shows

`h_g` is compared as `h_j(digitised) − h_c(published vacuum)`, which avoids
needing `k_s` at the test temperature — the subtraction removes the contact
path using the source's own number rather than ours. Note this subtraction is
ill-conditioned at low gas pressure, where `h_g` is a small difference of two
larger numbers, compounding the extraction bias at that end.

Geometric mean of model/measured, over 113 points:

| | lumped `k_g/(Y+M)` | Gaussian gap integral |
|---|---|---|
| all points | 0.885 | 0.930 |
| `M/Y > 2` (rarefied) | 1.053 | 1.056 |
| `M/Y` 0.5–2 | 1.009 | 1.033 |
| `M/Y` 0.1–0.5 | 0.879 | 0.934 |
| `M/Y < 0.1` (continuum) | **0.679** | 0.746 |
| log–log slope vs `P_g` | −0.0495 | −0.0251 |

99 % of points fall within 2× either way.

### The finding

The agreement is not uniform. It is good in the rarefied limit and degrades
monotonically toward the continuum limit, reaching a ~32 % shortfall at
`M/Y < 0.1`. That direction is diagnostic: as `M → 0` the rarefaction term
drops out and `h_g → k_g/Y`, so a continuum-limit shortfall **cannot** be
blamed on `M` or on the accommodation coefficients. It points instead at the
uniform-gap assumption built into the lumped form.

Local gap thickness is not uniform — it is Gaussian about `Y` with spread
`σ` — and local conductance goes as `1/(t+M)`, which is convex. Averaging
over the distribution therefore exceeds the value at the mean thickness, and
the gain is largest when `M` is small and the convexity is not softened by
the `M` offset. `gap_conductance_integral` implements that average and closes
about a third of the continuum gap while halving the pressure trend.

**It is not the default, and should not become the default on this evidence.**
The project specifies `h_g = k_g/(Y + M)`; the correction is motivated by
digitised data with a documented bias at one end. `tests/test_gap_integral.py`
pins the integral's invariants and includes a guard against quietly swapping
the default. Settling this needs exact tabulated `h_j(P_g)` values.

The residual ~25 % at `M/Y < 0.1` is unexplained. Candidates, untested: the
effective gap thickness differing from `Y` as computed here (our `Y` = 5.41 µm
against their effective gap `R_p` = 5.55 µm, 2.5 % apart, so probably not
this); gas conductivity at the test temperature; or extraction error in `h_j`
at the high-pressure end.

## Available but not yet entered

`mhtl88-7.pdf` Table 2 gives four light-load vacuum points at pressures
*below* the Hegazy range, from a different apparatus:

| Exp | material | σ (µm) | m | P (MPa) | T_c (°C) | h_c (W/m²K) |
|---|---|---|---|---|---|---|
| 1 | SS 304 | 1.53 | 0.090 | 0.60 ± 0.02 | 172 ± 4 | 452 ± 25 |
| 2 | SS 304 | 4.83 | 0.128 | 0.47 ± 0.02 | 168 ± 4 | 241 ± 3 |
| 3 | Ni 200 | 2.32 | 0.126 | 0.52 ± 0.02 | 170 ± 3 | 1130 ± 30 |
| 4 | Ni 200 | 11.8 | 0.206 | 0.38 ± 0.01 | 172 ± 4 | 725 ± 30 |

(Slopes from Bahrami 2004 Table 6, which indexes the same specimens.)

**Not entered, because k_s is not reported at these temperatures.** The
Hegazy rows carry a measured k_s; these would need it extrapolated from
125–145 °C out to ~170 °C, and for Ni 200 the conductivity is not monotonic
there. Entering an estimated k_s would put an unquantified error into a
benchmark whose purpose is to quantify error. Worth adding if the value can
be sourced.

Note also a transcription ambiguity: Exp 3's roughness is 2.32 µm in Table 2
and Bahrami's Table 6, but 2.23 µm in the Figure 5 caption of the same paper.

The gap-conductance measurements themselves are plotted (Figs 1–6), not
tabulated, so validating h_g against measurement still needs those figures
digitised.

## Still wanted

Gas-filled data on characterised specimens, which would exercise the gap
path. Song (1988) is the dataset — SS 304 and Ni 200 with N₂, Ar and He from
10⁻⁵ to 650 torr — and its experimental chapter is openly hosted as
`mhtl88-7.pdf` on the Waterloo MHTL site, but it is a scan with no text
layer.

## What was found in the literature search

### Confirmations

**The gap model's form is externally confirmed.** Bahrami, Yovanovich & Culham,
*Thermal joint resistance of conforming rough surfaces with gas-filled gaps*
(2004) uses `q_g = k_g(T₁−T₂)/(d + M)` with `M` built from the two thermal
accommodation coefficients, the specific-heat ratio, the Prandtl number and the
molecular mean free path, and takes the effective separation `d` as the mean
plane separation `Y` over the apparent area. That is what `openkapitza/gap`
implements.

**Published Song–Yovanovich coefficients now in `materials.yaml`.** From
Bahrami, Yovanovich & Culham, *Micro and Macro Hardness Measurements,
Correlations, and Contact Models*, Table 3 (fitted values):

| Material | c₁ | c₂ |
|---|---|---|
| SS 304 | 6.271 GPa | −0.229 |
| Ni 200 | 6.304 GPa | −0.264 |
| Zircaloy-4 | 5.677 GPa | −0.278 |
| Zr-Nb | 5.884 GPa | −0.267 |

Zr-4 and Zr-Nb are **not** in `materials.yaml` — their other bulk properties
have not been entered. Add them with the rest of the entry, not coefficients
alone.

**An accuracy target.** *Four Decades of Research on Thermal Contact, Gap, and
Joint Resistance in Microelectronics* reports the extended CMY correlation
agreeing with vacuum data for Ni 200, SS 304 and two zirconium alloys at
**3.95–6.43 % RMS**. A reimplementation that lands near that band against the
same data is behaving; one that does not has a problem.

**The formula question is settled.** Two sources, as extracted from PDF,
disagreed over whether the constant 1.62 multiplies c₁ outside the power or
sits inside it. Solving the implicit system both sources also give shows the
form used here (inside) tracks it to within 1 %, and the other reading is 45 %
low. `TestSongYovanovichAgainstTheImplicitSystem` locks this down so the
question does not have to be reopened. Do not "correct" it from a transcription.

## The primary datasets

Citations verified verbatim from the reference list of Bahrami, Yovanovich &
Culham, *J. Thermophysics and Heat Transfer* **18**(3), 2004
(`mhtl04-1.pdf`). Both are PhD dissertations from the Department of
Mechanical Engineering, University of Waterloo, Waterloo, ON, Canada:

- **Hegazy, A. A. (1985)**, *Thermal Joint Conductance of Conforming Rough
  Surfaces: Effect of Surface Micro-Hardness Variation*. SS 304, N₂ and
  helium, vacuum and 40–574 torr. About **160 points**.
- **Song, S. (1988)**, *Analytical and Experimental Study of Heat Transfer
  Through Gas Layers of Contact Interfaces*. SS 304 and Ni 200, N₂, argon and
  helium, gas pressures 10⁻⁵ to 650 torr. More than **350 points**.

### A better route than the theses

Song's experimental chapter was published separately, and the Microelectronics
Heat Transfer Laboratory hosts the PDFs openly:

- Song, S., Yovanovich, M. M. & Goodman, F. (1988), *Thermal Gap Conductance
  of Conforming Surfaces in Contact, Part I: Theory*, 10th Symposium on
  Thermophysical Properties, Gaithersburg MD.
- **Song, S. & Yovanovich, M. M. (1988), *Thermal Gap Conductance of
  Conforming Surfaces in Contact, Part II: Experimental Results*, same
  symposium** — hosted as `mhtl88-7.pdf` under
  `https://www.mhtlab.uwaterloo.ca/pdf_papers/`.

Part II is the data paper and is freely downloadable. **It is a scanned
image with no text layer**, so automated extraction returns nothing; the
tables have to be read from the page images. There is also a later journal
version, *Thermal Gap Conductance of Conforming Surfaces in Contact*, ASME
J. Heat Transfer **115**(3), 533 (1993), which may carry the same tables with
a text layer.

A related conference paper, Song & Yovanovich (1989), *Contact Interface Gas
Heat Transfer: A Method of Measuring Thermal Accommodation Coefficient*, 9th
Annual International Electronics Packaging Conference, Vol. 2, pp. 925–936, is
the right source for the accommodation coefficients the gap model currently
defaults — the weakest numbers in `gases.yaml`.

### Specimen conditions, available now

Hegazy (1985), SS 304, nominal contact pressure 0.459–8.769 MPa:

| Test | Gas | σ (µm) | m |
|---|---|---|---|
| T1 | N₂ | 5.65 | 0.153 |
| T2 | N₂, He | 5.61 | 0.151 |
| T3 | N₂, He | 6.29 | 0.195 |
| T4 | N₂, He | 4.02 | 0.168 |

Song (1988):

| Test | Solid | Gas | P (MPa) | σ (µm) | m |
|---|---|---|---|---|---|
| T1 | SS 304 | N₂, Ar, He | 0.595–0.615 | 1.53 | 0.090 |
| T2 | SS 304 | N₂, Ar, He | 0.467–0.491 | 4.83 | 0.128 |
| T3 | Ni 200 | N₂, Ar, He | 0.511–0.530 | 2.32 | 0.126 |
| T4 | Ni 200 | N₂, Ar, He | 0.371–0.389 | 11.8 | 0.206 |
| T5 | SS 304 | N₂, He | 0.403–7.739 | 6.45 | 0.132 |
| T6 | SS 304 | N₂, He | 0.526–8.713 | 2.09 | 0.904 **(suspect)** |
| T7 | Ni 200 | N₂, He | 0.367–6.550 | 11.8 | 0.206 |

**T6's slope of 0.904 is almost certainly a typo.** The same paper gives the
overall slope range as 0.08–0.205, and 0.904 is four times the largest value in
its own table — most likely 0.09. Do not enter T6 without checking the
dissertation; `h_c` is linear in `m`, so the error would pass straight into the
comparison.

## What is needed to fill the CSV

Any one of these closes it:

1. **The two dissertations.** Both are University of Waterloo theses and should
   be reachable through an institutional library or UWSpace. They contain the
   tabulated `h_j` versus `P` data directly.
2. **Digitised figures** from the 2004 paper (Figs 12–14). These are
   nondimensionalised, so the scaling has to be undone before the values are
   comparable — workable, but a second source of error.
3. **Your own measurements**, which would be better than either: they come with
   the surface characterisation already attached and no transcription step.

Enter values in the column schema in `README.md` in this directory, one row per
(specimen, pressure, gas) point, with the source recorded per row.

## What a pass looks like

`test_benchmarks.py` reports the geometric-mean ratio and the spread
separately, because they fail differently. A systematic bias points at `H_c` or
the asperity slope; scatter points at the correlation. The suite requires 80 %
of points within a factor of two and a geometric mean within 1.5× — deliberately
loose for a first pass. Tighten toward the published 3.95–6.43 % RMS once the
data is in and the inputs are right.
