"""Build the benchmark CSV from Bahrami's thesis, Appendix A.2.

Source
------
Bahrami, M. (2004), *Modeling of Thermal Joint Resistance for Sphere-Flat
Contacts in a Vacuum*, PhD thesis, Dept. of Mechanical Engineering,
University of Waterloo. Appendix A.2, "Conforming Contacts".

  - Table A.39 gives, per specimen pair: effective modulus E', the COMBINED
    roughness and slope written as "sigma/m", the Vickers coefficients
    "c1/c2", the harmonic-mean conductivity k_s at the test temperature, the
    specimen radius b_L, and the mean joint temperature T_c.
  - Tables A.46-A.54 give, per data point: applied force F in newtons and
    measured joint resistance R_exp in K/W.

All tests are **conforming rough flats in a vacuum**, so the gap path is
absent and the comparison isolates the contact path. The thesis notes that
the macro thermal resistance is zero for these specimens, i.e. they are flat
enough that surface curvature contributes nothing.

Conversions
-----------
b_L is the specimen RADIUS. Confirmed against the independently published
pressure range for the Hegazy tests, 0.459-8.769 MPa: with b_L as the radius
the forces here span 0.41-8.89 MPa, with b_L as the diameter they would span
1.6-35.5 MPa.

    A = pi * b_L^2
    P = F / A
    h_measured = 1 / (R_exp * A)

sigma and m in Table A.39 are already the COMBINED pair values. The builder
therefore emits sigma_eff and slope_eff directly rather than per-surface
numbers, and the loader divides by sqrt(2) when it needs per-surface inputs.

Why the per-row property overrides
----------------------------------
k_s, c1, c2 and E' are written into each row from the thesis table rather
than looked up. These tests run at 110-145 C, where stainless conductivity is
roughly 19 W/m.K against the 16.2 at 300 K in `materials.yaml`; using the
room-temperature value would put a ~19 % error straight into h_c and the
benchmark would be measuring the database, not the correlation. Testing the
correlation against the same inputs the source used is the honest separation.

Note the thesis uses c1 = 6.3 GPa, c2 = -0.26 for BOTH Ni 200 and SS 304,
rounded. The per-material fits in `materials.yaml` (6.271/-0.229 and
6.304/-0.264) come from a later paper. Rows carry the thesis values so the
comparison is like for like.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "benchmarks" / "joint_conductance.csv"

SOURCE = "Bahrami 2004 PhD thesis, Table A.39 + A.46-A.54 (data orig. Hegazy 1985)"

# key -> (material, sigma_eff um, slope_eff, c1 GPa, c2, k_s W/mK, b_L mm, T_c C)
SPECIMENS = {
    "PNI0102": ("Ni200",  0.90, 0.110, 6.3, -0.26, 75.3, 12.5, 120),
    "PNI0304": ("Ni200",  3.43, 0.190, 6.3, -0.26, 76.0, 12.5, 115),
    "PNI0506": ("Ni200",  4.24, 0.188, 6.3, -0.26, 75.9, 12.5, 110),
    "PNI0708": ("Ni200",  9.53, 0.228, 6.3, -0.26, 75.7, 12.5, 115),
    "PNI0910": ("Ni200", 13.94, 0.233, 6.3, -0.26, 75.8, 12.5, 115),
    "PSS0102": ("SS304",  0.48, 0.072, 6.3, -0.26, 19.2, 12.5, 140),
    "PSS0304": ("SS304",  2.71, 0.116, 6.3, -0.26, 19.1, 12.5, 145),
    "PSS0506": ("SS304",  5.88, 0.146, 6.3, -0.26, 18.9, 12.5, 130),
    "PSS0708": ("SS304", 10.95, 0.190, 6.3, -0.26, 18.9, 12.5, 125),
}

# key -> [(F newtons, R_exp K/W), ...]   transcribed from Tables A.46-A.54
DATA: dict[str, list[tuple[float, float]]] = {
    "PNI0102": [
        (242.98, 0.64), (288.63, 0.57), (360.30, 0.48), (422.64, 0.43),
        (494.80, 0.39), (556.65, 0.36), (665.13, 0.31), (772.14, 0.28),
        (885.05, 0.25), (991.57, 0.23), (1104.96, 0.21), (1314.07, 0.18),
        (1540.36, 0.16), (1765.18, 0.15), (2005.22, 0.12), (2199.61, 0.11),
        (2413.14, 0.10), (2644.83, 0.09), (2848.05, 0.09), (3092.01, 0.08),
        (3522.51, 0.07), (3955.46, 0.06), (4334.42, 0.06),
    ],
    "PNI0304": [
        (234.15, 1.16), (290.11, 1.06), (359.32, 0.90), (424.61, 0.80),
        (489.89, 0.73), (559.60, 0.66), (668.08, 0.58), (802.09, 0.51),
        (890.45, 0.47), (990.58, 0.43), (1109.37, 0.39), (1342.05, 0.32),
        (1546.25, 0.28), (1760.27, 0.25), (1997.37, 0.22), (2206.48, 0.19),
        (2419.52, 0.18), (2639.43, 0.17), (2865.23, 0.16), (3094.96, 0.15),
        (3559.82, 0.13), (3951.53, 0.12), (4319.69, 0.11),
    ],
    "PNI0506": [
        (242.00, 1.56), (287.65, 1.43), (354.90, 1.21), (427.06, 1.06),
        (494.80, 0.95), (555.18, 0.88), (662.68, 0.76), (775.09, 0.68),
        (887.50, 0.62), (996.96, 0.55), (1151.59, 0.47), (1322.41, 0.42),
        (1533.49, 0.36), (1764.20, 0.32), (1973.80, 0.28), (2203.53, 0.25),
        (2424.92, 0.23), (2632.07, 0.21), (2862.78, 0.20), (3084.65, 0.18),
        (3517.11, 0.16), (3963.32, 0.14), (4361.91, 0.14),
    ],
    "PNI0708": [
        (226.29, 2.05), (292.56, 1.90), (356.87, 1.69), (426.08, 1.51),
        (490.87, 1.37), (559.11, 1.24), (671.52, 1.07), (780.98, 0.95),
        (896.83, 0.85), (997.95, 0.79), (1103.48, 0.70), (1318.00, 0.59),
        (1540.36, 0.50), (1766.16, 0.45), (1989.51, 0.41), (2225.62, 0.36),
        (2419.52, 0.34), (2645.32, 0.31), (2864.74, 0.29), (3079.25, 0.26),
        (3516.62, 0.24), (3953.99, 0.21), (4309.38, 0.19),
    ],
    "PNI0910": [
        (232.18, 3.04), (290.60, 2.90), (356.87, 2.49), (421.66, 2.17),
        (468.78, 1.98), (556.16, 1.79), (682.31, 1.50), (780.49, 1.35),
        (894.86, 1.18), (996.96, 1.08), (1106.92, 0.96), (1333.21, 0.81),
        (1534.96, 0.70), (1763.22, 0.62), (1997.37, 0.54), (2213.35, 0.49),
        (2428.35, 0.46), (2656.61, 0.41), (2854.92, 0.39), (3082.69, 0.36),
        (3526.44, 0.31), (3967.73, 0.28), (4315.76, 0.26),
    ],
    "PSS0102": [
        (218.93, 2.89), (290.60, 2.14), (358.34, 1.74), (425.10, 1.50),
        (494.80, 1.32), (546.34, 1.20), (668.57, 1.04), (781.47, 0.92),
        (912.53, 0.82), (987.64, 0.78), (1104.47, 0.71), (1320.94, 0.60),
        (1545.27, 0.51), (1762.24, 0.46), (1988.04, 0.40), (2205.50, 0.36),
        (2420.50, 0.33), (2636.97, 0.31), (2818.11, 0.26), (3066.98, 0.26),
        (3511.22, 0.23), (3953.01, 0.20), (4329.51, 0.18),
    ],
    "PSS0304": [
        (211.57, 6.79), (293.05, 5.67), (362.26, 4.93), (423.62, 4.43),
        (494.80, 3.97), (552.23, 3.66), (664.15, 3.19), (774.11, 2.84),
        (878.17, 2.63), (989.60, 2.33), (1100.54, 2.14), (1314.07, 1.82),
        (1547.73, 1.54), (1769.11, 1.37), (1984.11, 1.19), (2200.10, 1.09),
        (2419.52, 1.00), (2638.45, 0.91), (2874.07, 0.84), (3080.23, 0.79),
        (3527.91, 0.70), (3983.44, 0.61), (4331.47, 0.56),
    ],
    "PSS0506": [
        (200.77, 8.45), (296.00, 7.28), (356.87, 6.45), (425.10, 5.74),
        (494.80, 5.17), (562.05, 4.73), (667.10, 4.12), (775.58, 3.68),
        (888.48, 3.30), (1014.15, 2.95), (1101.03, 2.76), (1334.69, 2.31),
        (1533.00, 2.04), (1752.91, 1.79), (1989.02, 1.60), (2226.60, 1.45),
        (2424.43, 1.35), (2632.56, 1.24), (2890.27, 1.14), (3086.12, 1.09),
        (3521.04, 0.96), (3968.22, 0.86), (4329.02, 0.80),
    ],
    "PSS0708": [
        (218.44, 11.01), (286.67, 9.94), (354.41, 8.56), (414.30, 7.43),
        (491.86, 6.84), (557.14, 6.33), (693.11, 5.52), (774.11, 4.87),
        (879.16, 4.39), (1008.25, 4.01), (1098.08, 3.64), (1311.61, 3.08),
        (1538.40, 2.66), (1787.76, 2.30), (1991.97, 2.12), (2200.59, 1.92),
        (2415.59, 1.77), (2632.56, 1.63), (2867.19, 1.50), (3080.23, 1.42),
        (3525.46, 1.25), (3967.73, 1.11), (4307.42, 1.04),
    ],
}

COLUMNS = [
    "case_id", "material_a", "material_b", "sigma_eff_um", "slope_eff",
    "pressure_MPa", "temperature_K", "gas", "gas_pressure_kPa",
    "h_measured_W_m2K", "k_s_W_mK", "c1_GPa", "c2", "E_eff_GPa", "source",
]


def main() -> int:
    rows = []
    for key, points in DATA.items():
        material, sigma, slope, c1, c2, k_s, b_L_mm, T_c = SPECIMENS[key]
        area = math.pi * (b_L_mm * 1e-3) ** 2
        for i, (force_N, R_exp) in enumerate(points, start=1):
            rows.append({
                "case_id": f"{key}-{i:02d}",
                "material_a": material,
                "material_b": material,
                "sigma_eff_um": f"{sigma:g}",
                "slope_eff": f"{slope:g}",
                "pressure_MPa": f"{force_N / area / 1e6:.5f}",
                "temperature_K": f"{T_c + 273.15:.2f}",
                "gas": "",                       # vacuum
                "gas_pressure_kPa": "",
                "h_measured_W_m2K": f"{1.0 / (R_exp * area):.2f}",
                "k_s_W_mK": f"{k_s:g}",
                "c1_GPa": f"{c1:g}",
                "c2": f"{c2:g}",
                "E_eff_GPa": "112.08" if material == "Ni200" else "113.74",
                "source": SOURCE,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT}")
    by_mat: dict[str, int] = {}
    for r in rows:
        by_mat[r["material_a"]] = by_mat.get(r["material_a"], 0) + 1
    for mat, n in sorted(by_mat.items()):
        print(f"  {mat}: {n} points")
    ps = [float(r["pressure_MPa"]) for r in rows]
    hs = [float(r["h_measured_W_m2K"]) for r in rows]
    print(f"  pressure {min(ps):.3f} - {max(ps):.3f} MPa")
    print(f"  h_measured {min(hs):.0f} - {max(hs):.0f} W/m^2K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
