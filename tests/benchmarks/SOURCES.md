# Where the benchmark data lives

`joint_conductance.csv` is still empty. This records what was found in an open
literature search, what it does and does not give, and what is still needed.

## What the search established

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
