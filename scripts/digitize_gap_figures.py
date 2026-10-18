"""Digitise the joint-conductance figures of Song & Yovanovich (1989).

Source
------
Song, S. & Yovanovich, M. M. (1989), *Contact Interface Gas Heat Transfer: A
Method of Measuring Thermal Accommodation Coefficient*, 9th Annual
International Electronics Packaging Conference, Vol. 2, pp. 925-936,
Figures 3-6 (`mhtl89-4.pdf`, page 11).

Each figure plots joint conductance h_j against gas pressure P_g on log-log
axes for one specimen pair, with separate markers for helium (triangle),
argon (circle) and nitrogen (square), plus a filled circle for the vacuum
measurement. These are the only form in which the gas-filled measurements
were published; the paper tabulates ranges, not points.

WHY THIS IS NOT THE SAME AS THE TABULATED DATA
-----------------------------------------------
Digitising a figure introduces error that belongs to this script, not to the
source. The output is therefore written to a SEPARATE file from the exact
tabulated benchmark, tagged as digitised, and excluded from the headline
agreement statistic. It is good enough to test whether the gap model has the
right shape and magnitude; it is not good enough to quote as a measurement.

The extraction is self-validating. The paper's Table 2 independently states,
for each experiment, the vacuum conductance h_c, the gas pressure range and
the h_g range. The script recovers those from the pixels and compares. If
the calibration were wrong, the recovered vacuum point would not land on the
published h_c.

Method
------
1. Rasterise the page at high DPI.
2. Locate each subplot's axis frame as the longest straight dark runs.
3. Calibrate: the frame corners are exactly the decade limits printed on the
   axes, so pixel -> data is a log-linear map.
4. Find marker blobs inside the frame, reject anything the wrong size, and
   classify by polygon vertex count: 3 triangle, 4 square, otherwise circle.
   A blob that is mostly ink is the filled vacuum marker.
5. Map centroids to data coordinates.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Fraction-of-width band occupied by the legend. Gas measurements begin near
# 9.4 torr (0.32 of the width) and the vacuum marker sits on the left axis
# (~0.0), so nothing real lies between.
LEGEND_X_LO, LEGEND_X_HI = 0.035, 0.30

# Axis limits printed on all four figures.
X_MIN, X_MAX = 1.0, 1000.0           # gas pressure, torr
Y_MIN, Y_MAX = 100.0, 100000.0       # joint conductance, W/m^2 K

# Published cross-checks, Table 2 of mhtl88-7 (same specimens, same tests).
# (specimen, sigma um, slope, P MPa, T_c C, h_c W/m2K, Pg lo, Pg hi)
EXPECTED = {
    1: ("SS304", 1.53, 0.090, 0.60, 172, 452.0, 9.4, 710.9),
    2: ("SS304", 4.83, 0.128, 0.47, 168, 241.0, 9.5, 665.0),
    3: ("Ni200", 2.32, 0.126, 0.52, 170, 1130.0, 9.6, 697.7),
    4: ("Ni200", 11.80, 0.206, 0.38, 172, 725.0, 9.4, 699.7),
}


def rasterise(pdf: Path, page: int, dpi: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "page"
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page),
         "-png", str(pdf), str(stem)],
        check=True, capture_output=True,
    )
    hits = sorted(out_dir.glob("page*.png"))
    if not hits:
        raise RuntimeError("pdftoppm produced no image")
    return hits[0]


def find_frames(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Locate the four subplot axis frames as (x0, y0, x1, y1) in pixels.

    Morphological line detection was tried first and failed: the scan has
    enough skew that a tall vertical structuring element does not fit inside
    the frame lines, and the vertical ink vanished. Projection profiles are
    insensitive to that, because a line a few pixels off vertical still puts
    most of its ink in one column band.

    The page margins are blanked first; the scan carries a solid black bar
    along one edge that otherwise dominates every row profile.
    """
    b = binary.copy()
    h, w = b.shape
    b[:, :60] = 0
    b[:, -60:] = 0
    b[:40, :] = 0
    b[-120:, :] = 0

    rows = (b.sum(axis=1) / 255.0) / w
    cols = (b.sum(axis=0) / 255.0) / h

    def bands(signal: np.ndarray, threshold: float, gap: int = 25):
        idx = np.where(signal > threshold)[0]
        if idx.size == 0:
            return []
        out, start, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - prev > gap:
                out.append(int((start + prev) // 2))
                start = i
            prev = i
        out.append(int((start + prev) // 2))
        return out

    ys = sorted(bands(rows, 0.25))
    xs = sorted(bands(cols, 0.18))
    if len(xs) < 4 or len(ys) < 4:
        return []

    def pair_up(positions: list[int]) -> list[tuple[int, int]]:
        """Keep consecutive pairs that span a full subplot.

        Figure captions also register as strong rows, so simply taking the
        first four candidates mixes a caption into the bottom frame -- which
        is exactly what happened, putting the lower subplots' calibration out
        by two orders of magnitude. The subplots are all the same size, so
        the real frame pairs are the ones whose span is near the largest;
        caption-to-frame gaps are roughly a quarter of that.
        """
        spans = [
            (positions[i + 1] - positions[i], positions[i], positions[i + 1])
            for i in range(len(positions) - 1)
        ]
        if not spans:
            return []
        widest = max(s[0] for s in spans)
        keep, used = [], set()
        for span, a, b in spans:
            if span < 0.75 * widest:
                continue
            if a in used or b in used:
                continue
            keep.append((a, b))
            used.update((a, b))
        return keep

    y_pairs = pair_up(ys)
    x_pairs = pair_up(xs)
    if len(y_pairs) < 2 or len(x_pairs) < 2:
        return []

    frames = []
    for top, bottom in y_pairs[:2]:
        for left, right in x_pairs[:2]:
            frames.append((left, top, right, bottom))
    return frames


def classify(contour) -> str:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.045 * peri, True)
    v = len(approx)
    if v == 3:
        return "He"
    if v == 4:
        return "N2"
    return "Ar"


def extract(image: np.ndarray, frame: tuple[int, int, int, int]) -> list[dict]:
    x0, y0, x1, y1 = frame
    pad = 6
    roi = image[y0 + pad:y1 - pad, x0 + pad:x1 - pad]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

    # Close the dotted gridlines' gaps without eroding markers.
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    areas = [cv2.contourArea(c) for c in contours]
    if not areas:
        return []
    typical = np.median([a for a in areas if a > 50] or [1.0])

    width = (x1 - pad) - (x0 + pad)
    height = (y1 - pad) - (y0 + pad)
    lx = math.log10(X_MAX) - math.log10(X_MIN)
    ly = math.log10(Y_MAX) - math.log10(Y_MIN)

    points = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < 0.25 * typical or a > 6.0 * typical:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 8 or bh < 8 or bw > 0.12 * width or bh > 0.12 * height:
            continue
        if not (0.55 < bw / bh < 1.8):          # markers are near-square
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        # The legend occupies a band where no measurement can lie: gas data
        # starts near 9.4 torr and the vacuum marker sits on the left axis.
        if LEGEND_X_LO < cx / width < LEGEND_X_HI:
            continue
        mask = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        fill = cv2.countNonZero(cv2.bitwise_and(binary, mask)) / max(a, 1.0)
        kind = "Vac" if fill > 0.80 else classify(c)
        points.append({
            "gas": kind,
            "P_torr": 10 ** (math.log10(X_MIN) + (cx / width) * lx),
            "h_j": 10 ** (math.log10(Y_MAX) - (cy / height) * ly),
            "fill": round(fill, 3),
        })
    return points


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--page", type=int, default=11)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--work", default="/tmp/okdigitize")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    work = Path(args.work)
    png = rasterise(Path(args.pdf), args.page, args.dpi, work)
    image = cv2.imread(str(png))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

    frames = find_frames(binary)
    print(f"rasterised {png.name} at {args.dpi} dpi, {image.shape[1]}x{image.shape[0]}")
    print(f"found {len(frames)} axis frames")
    if len(frames) != 4:
        print("expected 4 subplots; calibration cannot be trusted. Stopping.")
        return 1

    rows = []
    for idx, frame in enumerate(frames, start=1):
        pts = extract(image, frame)
        spec, sigma, slope, P_MPa, T_c, h_c_pub, pg_lo, pg_hi = EXPECTED[idx]
        gases = [p for p in pts if p["gas"] != "Vac"]
        vac = [p for p in pts if p["gas"] == "Vac"]

        print(f"\nExp #{idx} ({spec}) — {len(pts)} markers "
              f"({len(gases)} gas, {len(vac)} vacuum)")
        if vac:
            got = float(np.median([v['h_j'] for v in vac]))
            err = (got - h_c_pub) / h_c_pub * 100
            print(f"  vacuum h_c  recovered {got:8.1f}  published {h_c_pub:8.1f}"
                  f"   {err:+6.1f} %   <- calibration check")
        if gases:
            lo = min(p["P_torr"] for p in gases)
            hi = max(p["P_torr"] for p in gases)
            print(f"  P_g range   recovered {lo:6.1f}-{hi:6.1f} torr  "
                  f"published {pg_lo:.1f}-{pg_hi:.1f}")
            for g in ("He", "Ar", "N2"):
                n = sum(1 for p in gases if p["gas"] == g)
                print(f"    {g:3s}: {n:3d} points")

        for p in gases:
            rows.append({
                "case_id": f"SY89-E{idx}-{p['gas']}-{len(rows):03d}",
                "experiment": idx,
                "material": spec,
                "gas": p["gas"],
                "sigma_eff_um": sigma,
                "slope_eff": slope,
                "pressure_MPa": P_MPa,
                "temperature_K": round(T_c + 273.15, 2),
                "gas_pressure_torr": round(p["P_torr"], 2),
                "h_joint_W_m2K": round(p["h_j"], 1),
                "h_c_vacuum_W_m2K": h_c_pub,
                "source": "Song & Yovanovich 1989 Fig 3-6, DIGITISED",
            })

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent
        / "tests" / "benchmarks" / "gap_conductance_digitised.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {len(rows)} digitised points to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
