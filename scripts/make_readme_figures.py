"""Regenerate the README figures (docs/figures/*.png, light and dark).

    python3 scripts/make_readme_figures.py

The continuum figure is computed live with the package. The MD figures read
docs/figures/data/md_results.json and nonbonded_sweep.json, small summaries of
runs on the author's M3 Pro (KB 17-19, 26); the run folders themselves are not in the repository.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / "docs" / "figures"
DATA = json.loads((OUT / "data" / "md_results.json").read_text())

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], band="#cde2fb"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"], band="#184f95"),
}


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["ink2"])
    ax.tick_params(colors=t["ink2"], labelsize=9)
    ax.yaxis.label.set_color(t["ink2"])
    ax.xaxis.label.set_color(t["ink2"])
    ax.title.set_color(t["ink"])
    ax.grid(True, axis="y", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)


def plain_log(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fig_new(t, *size):
    f = plt.figure(figsize=size, dpi=150)
    f.patch.set_facecolor(t["surface"])
    return f


def save(f, name, mode):
    f.savefig(OUT / f"{name}-{mode}.png", facecolor=f.get_facecolor(), bbox_inches="tight")
    plt.close(f)


# ---- 1. continuum: a joint against contact pressure, three gaps -------------------
def joint_curves():
    from openkapitza import build_joint, run_continuum
    P = np.logspace(-1, 1, 13)
    out = {}
    for label, kw in (("vacuum", dict(gas=None)),
                      ("air, 1 atm", dict(gas="air", gas_pressure_kPa=101.325)),
                      ("He, 1 atm", dict(gas="He", gas_pressure_kPa=101.325))):
        h = []
        for p in P:
            j = build_joint("SS304", "SS304", sigma_a_um=1.2, sigma_b_um=1.2, slope_a=0.08,
                            slope_b=0.08, pressure_MPa=float(p), temperature_K=300.0, **kw)
            h.append(run_continuum(j).primary.h_joint.si)
        out[label] = np.array(h)
    return P, out


def fig_joint(t, mode, P, curves):
    f = fig_new(t, 6.4, 3.6)
    ax = f.add_subplot(111)
    style(ax, t)
    for (label, h), c in zip(curves.items(), t["series"]):
        ax.plot(P, h / 1e3, color=c, lw=2, marker="o", ms=4)
        ax.annotate(label, (P[-1], h[-1] / 1e3), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=t["ink"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("apparent contact pressure (MPa)")
    ax.set_ylabel("joint conductance (kW/m²K)")
    ax.set_title("SS 304 / SS 304, σ = 1.2 µm, m = 0.08, 300 K", fontsize=10, loc="left", color=t["ink"])
    ax.grid(True, axis="x", color=t["grid"], lw=0.8)
    ax.set_xlim(P[0] * 0.9, P[-1] * 2.3)
    plain_log(ax, "x"); plain_log(ax, "y")
    save(f, "joint_pressure", mode)


# ---- 2. MD: Si/Ge convergence, one panel per axis ------------------------------
AXES = [
    ("cross-section (nm)", [("r01", 2.17), ("r02", 3.26), ("r00", 4.34), ("r03", 5.43), ("r04", 6.52)]),
    ("length per side (nm)", [("r05", 8), ("r00", 16), ("r06", 32), ("r07", 64)]),
    ("production (ps)", [("r08", 250), ("r00", 500), ("r09", 1000)]),
    ("ΔT drive (K)", [("r10", 20), ("r00", 40), ("r11", 60)]),
    ("reservoir (nm)", [("r12", 1), ("r00", 2), ("r13", 4)]),
]


def fig_convergence(t, mode):
    camp = {k.split("_")[0]: v for k, v in DATA["campaign"].items()}
    base = camp["r00"]
    f = fig_new(t, 10.5, 2.9)
    axs = f.subplots(1, len(AXES), sharey=True)
    for ax, (label, pts) in zip(axs, AXES):
        style(ax, t)
        ax.axhspan(base["h"] - base["err"], base["h"] + base["err"], color=t["band"], alpha=0.5,
                   lw=0)
        xs = np.arange(len(pts))
        for x, (rung, _) in zip(xs, pts):
            r = camp[rung]
            ok = not r["warn"] or r["warn"] == ["NEMD_RELAX_UNSETTLED"]
            ax.errorbar(x, r["h"], yerr=r["err"], fmt="o", ms=6, lw=1.5, capsize=0,
                        color=t["series"][0], mfc=t["series"][0] if ok else t["surface"],
                        mew=1.5)
        ax.set_xticks(xs, [f"{v:g}" for _, v in pts])
        ax.set_xlabel(label, fontsize=9)
        ax.set_xlim(-0.6, len(pts) - 0.4)
    axs[0].set_ylabel("h_K (MW/m²K)")
    axs[0].set_ylim(380, 840)
    axs[1].annotate("not yet steady", (3, camp["r07"]["h"]), xytext=(-8, 14),
                    textcoords="offset points", ha="right", fontsize=8, color=t["ink2"])
    f.suptitle("Si/Ge interface conductance (Tersoff NEMD, 300 K): one axis varied per panel; "
               "band = base cell 483 ± 20", x=0.01, ha="left", fontsize=10, color=t["ink"])
    f.tight_layout()
    save(f, "sige_convergence", mode)


# ---- 3. MD: a temperature profile ---------------------------------------------
def fig_profile(t, mode):
    p = DATA["profile_r00"]
    z, T = np.array(p["z_nm"]), np.array(p["T_K"])
    f = fig_new(t, 6.4, 3.2)
    ax = f.add_subplot(111)
    style(ax, t)
    for lo, hi in (p["hot_nm"], p["cold_nm"]):
        ax.axvspan(lo, hi, color=t["grid"], lw=0)
    ax.axvline(p["interface_nm"], color=t["ink2"], lw=1, ls=(0, (3, 3)))
    ax.plot(z, T, color=t["series"][0], lw=2, marker="o", ms=3)
    ax.text(np.mean(p["hot_nm"]), T.max() + 1.5, "hot\nreservoir", ha="center", va="bottom",
            fontsize=8, color=t["ink2"])
    ax.text(np.mean(p["cold_nm"]), T.max() + 1.5, "cold\nreservoir", ha="center", va="bottom",
            fontsize=8, color=t["ink2"])
    ax.text(p["interface_nm"] + 0.4, T.min() + 2, "Si | Ge interface:\nthe jump is ΔT_K",
            fontsize=8, color=t["ink2"])
    ax.set_xlabel("position along the heat flow (nm)")
    ax.set_ylabel("temperature (K)")
    ax.set_ylim(T.min() - 4, T.max() + 9)
    ax.set_title("Si/Ge NEMD temperature profile, averaged over 500 ps", fontsize=10, loc="left", color=t["ink"])
    save(f, "nemd_profile", mode)


# ---- 4. MD: interface types (the non-bonded sweep, KB 26) -------------------------
SWEEP = json.loads((OUT / "data" / "nonbonded_sweep.json").read_text())
LOADS = (0, 0.5, 1, 2)


def fig_interfaces(t, mode):
    f = fig_new(t, 10.0, 3.6)
    a1, a2 = f.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in (a1, a2):
        style(ax, t)
        b = SWEEP["bonded_ref"]
        ax.axhspan(b["h"] - b["err"], b["h"] + b["err"], color=t["band"], alpha=0.6, lw=0)
    for ax, x in ((a1, 0.02), (a2, -0.4)):
        ax.text(x, SWEEP["bonded_ref"]["h"] * 1.12, "bonded interface, 493 (Langevin)",
                fontsize=8, color=t["ink2"])
    # left: van der Waals strength x load
    for (scale, name), c, mk in zip(((0.25, "LJ ×¼"), (1, "LJ ×1"), (4, "LJ ×4")),
                                    t["series"], ("o", "s", "D")):
        r = [SWEEP[f"lj{scale:g}_p{p:g}"] for p in LOADS]
        ax = a1
        ax.errorbar(LOADS, [x["h"] for x in r], yerr=[x["err"] for x in r], color=c, marker=mk,
                    ms=6, lw=2, capsize=0)
        ax.annotate(name, (LOADS[-1], r[-1]["h"]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9, color=t["ink"])
    lv = SWEEP["lj1_p1_langevin"]
    a1.errorbar([1.06], [lv["h"]], yerr=[lv["err"]], fmt="s", ms=6, mfc=t["surface"],
                color=t["series"][1], mew=1.5, capsize=0)
    a1.annotate("LJ ×1, Langevin", (1.06, lv["h"]), xytext=(7, 4), textcoords="offset points",
                fontsize=8, color=t["ink2"])
    a1.set_xticks(LOADS, [f"{p:g}" for p in LOADS])
    a1.set_xlim(-0.15, 2.45)
    a1.set_xlabel("contact pressure (GPa)")
    a1.set_ylabel("h_K (MW/m²K)")
    a1.set_title("Pressed, not bonded: cross-interface LJ strength × load",
                 fontsize=10, loc="left", color=t["ink"])
    # right: a gas layer in the contact at 1 GPa
    cats = [("bare", "lj1_p1", None), ("Ar", "Ar_c1_p1", "Ar_c0.5_p1"),
            ("N₂", "N2_c1_p1", "N2_c0.5_p1"), ("He", "He_c1_p1", "He_c0.5_p1")]
    for dx, idx, mk, name, filled in ((-0.1, 1, "o", "full monolayer", True),
                                     (0.1, 2, "o", "half coverage", False)):
        xs, ys, es = [], [], []
        for i, (_, full, half) in enumerate(cats):
            k = full if filled else half
            if k is None:
                continue
            xs.append(i + (dx if half else 0)); ys.append(SWEEP[k]["h"]); es.append(SWEEP[k]["err"])
        a2.errorbar(xs, ys, yerr=es, fmt=mk, ms=7, color=t["series"][0], mew=1.5, capsize=0,
                    mfc=t["series"][0] if filled else t["surface"], label=name)
    a2.set_xticks(range(len(cats)), [c[0] for c in cats])
    a2.set_xlim(-0.5, len(cats) - 0.5)
    a2.set_xlabel("what fills the contact (LJ ×1, 1 GPa)")
    a2.set_title("A physisorbed gas layer", fontsize=10, loc="left", color=t["ink"])
    leg = a2.legend(frameon=False, fontsize=8, loc="lower right")
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    a1.set_yscale("log")
    a1.set_ylim(3, 800)
    plain_log(a1)
    f.text(0.01, -0.04, "Si/Ge, 4.3 nm cross-section, 8 nm per side, 300 K; imposed flux (HEX) "
           "unless marked. Relative trends hold; absolute HEX values may read low (KB 26).",
           fontsize=8, color=t["ink2"])
    f.tight_layout()
    save(f, "interface_types", mode)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    P, curves = joint_curves()
    for mode, t in THEMES.items():
        fig_joint(t, mode, P, curves)
        fig_convergence(t, mode)
        fig_profile(t, mode)
        fig_interfaces(t, mode)
    for label, h in curves.items():
        print(f"{label:12} h_joint {h[0]:.0f} -> {h[-1]:.0f} W/m^2K over {P[0]:g}-{P[-1]:g} MPa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
