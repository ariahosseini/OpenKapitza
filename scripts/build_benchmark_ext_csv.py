"""Build the EXTENDED benchmark CSV from Bahrami's thesis, Appendix A.2.

Companion to build_benchmark_csv.py (the 207-point Hegazy set, which stays the
headline benchmark). This adds the other conforming, vacuum data sets in the
same appendix that matter for packaging metals:

  Antonetti (A.40-A.43)   Ni 200 / Ni 200, four roughnesses, b_L = 14.3 mm
  Milanez et al. (A.63)   SS 304 bead-blasted / lapped, LIGHT loads from 16 kPa
                          (the S1 specimen of Milanez, Yovanovich & Mantelli 2004)
  McWaid (A.64, A.65)     SS 304 / SS 304, c2 = 0 (Hv constant)
  Nho (A.66-A.73)         Ni 200, Al 6061-T6, and Al 6061-T6 / Ni 200 (ground vs
                          lapped, both orders): the only Al data in the appendix

Nho's ground surfaces are ANISOTROPIC (his thesis title: "... Anisotropic
Ground/Lapped Interfaces"); CMY assumes isotropic roughness, so his sets test
the model slightly outside its assumptions.

Left out: Zircaloy-4 and Zr-2.5Nb (nuclear alloys) and Antonetti's silver-
coated Ni (a coated contact, not a bare CMY joint).

Per-pair inputs (E', COMBINED sigma/m, c1/c2, k_s, b_L, T_c) are Table A.39;
per-point F [N], R_exp [K/W] and Bahrami's model R_j [K/W] are the tables
named below. As for the Hegazy set: A = pi b_L^2, P = F/A, h = 1/(R_exp A).

PRECISION. R_exp is printed to two decimals, so a point at 0.02 K/W carries
+/-25 % from rounding alone (Nho's Al data reach 0.01 K/W). Each row records
rounding_rel = 0.005/R_exp; the tests use only rows with rounding_rel <= 0.10.

The Nho Al/Ni rows: Table A.39 lists the four dissimilar pairs by surface
(G = ground, L = lapped) in the same order as the table titles, so A.70 is
"Ni200-G, Al6061T6-L" (k_s 104.3, 168 C), A.67 "Al6061T6-L, Ni200-G" (102.7,
210 C), A.66 "Al6061T6-G, Ni200-L" (108.1, 135 C), A.71 "Ni200-L, Al6061T6-G"
(108.8, 125 C).

    python3 scripts/build_benchmark_ext_csv.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "benchmarks" / "joint_conductance_ext.csv"
THESIS = "Bahrami 2004 PhD thesis, Table A.39 + A.{t}"
ORIGIN = {"A": "data orig. Antonetti 1983", "M": "data orig. Milanez, Yovanovich & Mantelli 2003",
          "MW": "data orig. McWaid 1990", "N": "data orig. Nho 1990"}

# key -> (material_a, material_b, sigma um, m, c1 GPa, c2, k_s, b_L mm, T_c C, E' GPa, table)
INFO = {
    "A_P3435": ("Ni200", "Ni200", 8.48, 0.344, 6.3, -0.26, 67.1, 14.3, 110, 112.09, "40"),
    "A_P2627": ("Ni200", "Ni200", 1.23, 0.139, 6.3, -0.26, 64.5, 14.3, 150, 112.09, "41"),
    "A_P1011": ("Ni200", "Ni200", 4.27, 0.237, 6.3, -0.26, 67.7, 14.3, 100, 112.09, "42"),
    "A_P0809": ("Ni200", "Ni200", 4.29, 0.240, 6.3, -0.26, 67.3, 14.3, 108, 112.09, "43"),
    "M_T1": ("SS304", "SS304", 0.72, 0.041, 6.27, -0.23, 18.8, 12.5, 39, 113.74, "63"),
    "MW_SC1SC2": ("SS304", "SS304", 1.44, 0.089, 4.5, 0.0, 16.0, 12.7, 52, 113.74, "64"),
    "MW_SM1SM2": ("SS304", "SS304", 1.34, 0.105, 4.8, 0.0, 16.0, 12.7, 52, 113.74, "65"),
    "N_Ni_GL": ("Ni200", "Ni200", 0.87, 0.050, 4.6, -0.21, 68.9, 12.5, 195, 112.08, "72"),
    "N_Ni_LG": ("Ni200", "Ni200", 0.87, 0.050, 4.6, -0.21, 69.4, 12.5, 185, 112.08, "73"),
    "N_Al_GL": ("Al6061", "Al6061", 0.86, 0.058, 0.9, -0.006, 211.4, 12.5, 223, 39.11, "68"),
    "N_Al_LG": ("Al6061", "Al6061", 0.86, 0.058, 0.9, -0.006, 211.5, 12.5, 227, 39.11, "69"),
    "N_NiG_AlL": ("Ni200", "Al6061", 0.90, 0.048, 1.1, -0.008, 104.3, 12.5, 168, 56.23, "70"),
    "N_AlL_NiG": ("Al6061", "Ni200", 0.90, 0.048, 1.1, -0.008, 102.7, 12.5, 210, 56.23, "67"),
    "N_AlG_NiL": ("Al6061", "Ni200", 1.20, 0.057, 1.03, -0.001, 108.1, 12.5, 135, 56.23, "66"),
    "N_NiL_AlG": ("Ni200", "Al6061", 1.20, 0.057, 1.03, -0.001, 108.8, 12.5, 125, 56.23, "71"),
}

# key -> [(F newtons, R_exp K/W, R_j Bahrami model K/W), ...]
# parsed from the thesis text and checked row by row against the printed
# (R_j - R_exp)/R_j column (all agree within the 2-decimal rounding of R).
DATA = {
    "A_P3435": [   # Table A.40
        (366.01, 1.54, 1.55), (625.62, 0.93, 0.91), (1019.83, 0.53, 0.56), (1333.28, 0.42, 0.43),
        (1742.88, 0.32, 0.33), (2200.55, 0.28, 0.26),
    ],
    "A_P2627": [   # Table A.41
        (398.70, 0.59, 0.69), (628.18, 0.37, 0.44), (1040.98, 0.24, 0.27), (1349.31, 0.18, 0.21),
        (1818.52, 0.15, 0.15), (2249.91, 0.12, 0.12),
    ],
    "A_P1011": [   # Table A.42
        (328.19, 1.25, 1.36), (737.15, 0.62, 0.61), (933.94, 0.47, 0.48), (1171.11, 0.38, 0.38),
        (1378.15, 0.36, 0.32), (2060.82, 0.21, 0.22),
    ],
    "A_P0809": [   # Table A.43
        (447.42, 0.89, 1), (765.35, 0.57, 0.58), (999.32, 0.4, 0.45), (1233.93, 0.36, 0.36),
        (1570.45, 0.3, 0.28), (1852.49, 0.24, 0.24), (2330.68, 0.18, 0.19),
    ],
    "M_T1": [   # Table A.63
        (7.74, 145.41, 221.4), (22.87, 60.29, 74.94), (42.75, 29.98, 40.09), (90.06, 16.57, 19.03),
        (180.64, 9.33, 9.49), (365.04, 5.33, 4.69), (594.56, 3.51, 2.88), (977.63, 1.78, 1.75),
        (1027.12, 1.46, 1.67), (1205.77, 1.27, 1.42), (1358.66, 1.04, 1.26),
    ],
    "MW_SC1SC2": [   # Table A.64
        (262.98, 7.02, 9.75), (724.59, 3.34, 3.54), (1216.10, 2.28, 2.11), (1702.54, 1.7, 1.51),
        (2031.90, 1.41, 1.26), (2432.20, 1.2, 1.05), (2943.97, 1, 0.87), (3394.94, 0.89, 0.75),
        (3987.79, 0.77, 0.64),
    ],
    "MW_SM1SM2": [   # Table A.65
        (280.72, 5.64, 7.72), (729.66, 2.79, 2.97), (1216.10, 1.92, 1.78), (1631.60, 1.48, 1.33),
        (2062.30, 1.23, 1.05), (2452.46, 1.09, 0.88), (2873.03, 0.91, 0.75), (3339.20, 0.82, 0.65),
        (3997.92, 0.71, 0.54),
    ],
    "N_Ni_GL": [   # Table A.72
        (309.25, 0.69, 1.17), (343.61, 0.64, 1.06), (446.70, 0.55, 0.81), (535.05, 0.52, 0.68),
        (579.23, 0.49, 0.63), (672.50, 0.44, 0.54), (785.40, 0.39, 0.46), (957.20, 0.36, 0.38),
        (1040.65, 0.32, 0.35), (1237.00, 0.28, 0.29), (1477.53, 0.25, 0.25), (1703.33, 0.21, 0.21),
        (2027.31, 0.18, 0.18), (2326.74, 0.16, 0.16), (2793.07, 0.13, 0.13), (3180.86, 0.11, 0.11),
        (3568.65, 0.1, 0.1),
    ],
    "N_Ni_LG": [   # Table A.73
        (304.34, 0.69, 1.18), (377.97, 0.6, 0.95), (461.42, 0.55, 0.78), (613.59, 0.45, 0.59),
        (657.77, 0.42, 0.55), (775.58, 0.39, 0.46), (913.03, 0.33, 0.39), (1178.10, 0.28, 0.31),
        (1340.09, 0.25, 0.27), (1462.80, 0.22, 0.25), (1629.70, 0.2, 0.22), (2027.31, 0.16, 0.18),
        (2302.20, 0.14, 0.16), (2788.16, 0.12, 0.13), (2930.52, 0.11, 0.12),
    ],
    "N_Al_GL": [   # Table A.68
        (191.44, 0.14, 0.18), (269.98, 0.11, 0.13), (309.25, 0.1, 0.11), (338.70, 0.09, 0.1),
        (436.88, 0.07, 0.08), (530.14, 0.06, 0.07), (647.95, 0.05, 0.05), (760.85, 0.04, 0.05),
        (883.57, 0.04, 0.04), (1251.73, 0.03, 0.03), (1295.91, 0.02, 0.03), (1482.44, 0.02, 0.02),
        (1801.51, 0.02, 0.02), (1978.22, 0.02, 0.02),
    ],
    "N_Al_LG": [   # Table A.69
        (176.71, 0.17, 0.2), (225.80, 0.14, 0.16), (274.89, 0.12, 0.13), (338.70, 0.1, 0.1),
        (412.33, 0.08, 0.09), (544.87, 0.07, 0.06), (643.04, 0.06, 0.05), (731.40, 0.05, 0.05),
        (878.66, 0.04, 0.04), (1021.02, 0.04, 0.03), (1227.18, 0.03, 0.03), (1467.71, 0.03, 0.02),
        (1668.97, 0.02, 0.02), (2002.77, 0.02, 0.02), (2316.92, 0.02, 0.02), (2719.44, 0.01, 0.01),
    ],
    "N_NiG_AlL": [   # Table A.70
        (215.98, 0.39, 0.5), (245.44, 0.39, 0.44), (269.98, 0.36, 0.4), (323.98, 0.4, 0.33),
        (387.79, 0.31, 0.28), (436.88, 0.28, 0.25), (525.24, 0.23, 0.21), (598.87, 0.2, 0.18),
        (687.22, 0.18, 0.16), (805.03, 0.15, 0.13), (1011.20, 0.12, 0.11), (1217.37, 0.1, 0.09),
        (1418.63, 0.07, 0.08), (1580.61, 0.06, 0.07), (1988.04, 0.04, 0.05),
    ],
    "N_AlL_NiG": [   # Table A.67
        (220.89, 0.43, 0.41), (255.25, 0.35, 0.35), (309.25, 0.32, 0.29), (338.70, 0.28, 0.27),
        (382.88, 0.26, 0.24), (427.06, 0.23, 0.21), (530.14, 0.18, 0.17), (657.77, 0.15, 0.14),
        (736.31, 0.14, 0.12), (937.57, 0.12, 0.1), (1035.74, 0.11, 0.09), (1212.46, 0.09, 0.07),
        (1408.81, 0.08, 0.06), (1654.24, 0.06, 0.05), (2012.58, 0.06, 0.04), (2307.11, 0.05, 0.04),
    ],
    "N_AlG_NiL": [   # Table A.66
        (269.98, 0.3, 0.4), (348.52, 0.27, 0.31), (402.52, 0.24, 0.27), (495.78, 0.22, 0.22),
        (574.32, 0.19, 0.19), (662.68, 0.17, 0.16), (770.67, 0.16, 0.14), (878.66, 0.15, 0.12),
        (1079.92, 0.12, 0.1), (1207.55, 0.1, 0.09), (1452.99, 0.1, 0.07), (1673.88, 0.08, 0.07),
        (1929.13, 0.07, 0.06),
    ],
    "N_NiL_AlG": [   # Table A.71
        (235.62, 0.4, 0.45), (284.71, 0.33, 0.37), (319.07, 0.3, 0.33), (373.06, 0.26, 0.28),
        (436.88, 0.23, 0.24), (535.05, 0.19, 0.2), (589.05, 0.18, 0.18), (711.77, 0.16, 0.15),
        (854.12, 0.12, 0.12), (952.30, 0.12, 0.11), (1109.37, 0.1, 0.1), (1354.81, 0.08, 0.08),
        (1605.16, 0.07, 0.07), (1909.50, 0.06, 0.06),
    ],
}

COLUMNS = ["case_id", "dataset", "material_a", "material_b", "sigma_eff_um", "slope_eff",
           "pressure_MPa", "temperature_K", "gas", "gas_pressure_kPa", "h_measured_W_m2K",
           "k_s_W_mK", "c1_GPa", "c2", "E_eff_GPa", "R_exp_K_W", "rounding_rel",
           "h_model_bahrami_W_m2K", "source"]


def rows():
    for key, pts in DATA.items():
        ma, mb, s, m, c1, c2, ks, bl, tc, ee, tab = INFO[key]
        area = math.pi * (bl * 1e-3) ** 2
        src = f"{THESIS.format(t=tab)} ({ORIGIN[key.split('_')[0]]})"
        for i, (force, r_exp, r_j) in enumerate(pts, 1):
            yield {"case_id": f"{key}-{i:02d}", "dataset": key, "material_a": ma, "material_b": mb,
                   "sigma_eff_um": s, "slope_eff": m, "pressure_MPa": f"{force / area / 1e6:.5f}",
                   "temperature_K": f"{tc + 273.15:.2f}", "gas": "", "gas_pressure_kPa": "",
                   "h_measured_W_m2K": f"{1 / (r_exp * area):.2f}", "k_s_W_mK": ks, "c1_GPa": c1,
                   "c2": c2, "E_eff_GPa": ee, "R_exp_K_W": r_exp,
                   "rounding_rel": f"{0.005 / r_exp:.4f}",
                   "h_model_bahrami_W_m2K": f"{1 / (r_j * area):.2f}", "source": src}


def main() -> None:
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        n = 0
        for r in rows():
            w.writerow(r)
            n += 1
    print(f"wrote {n} rows to {OUT}")


if __name__ == "__main__":
    main()
