"""Regenerate the README figures (docs/figures/*.png, light and dark).

    python3 scripts/make_readme_figures.py

The continuum figures are computed live with the package. The MD figures read
docs/figures/data/md_results.json, nonbonded_sweep.json and cell_profiles.json,
small summaries of runs on the author's M3 Pro (KB 17-19, 26), and the atom
pictures read thin slabs of the relaxed cells in docs/figures/data/structures/
(written by scripts/extract_structure_slices.py from a job root); the run folders themselves are not in the repository.
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


# ---- 1b. continuum: the parallel paths, and the gas path against gas pressure ------
def fig_paths(t, mode):
    from openkapitza import build_joint, run_continuum
    kw = dict(sigma_a_um=1.2, sigma_b_um=1.2, slope_a=0.08, slope_b=0.08, temperature_K=300.0)
    P = np.logspace(-1, 1, 13)
    rs = [run_continuum(build_joint("SS304", "SS304", pressure_MPa=float(p), gas="air",
                                    gas_pressure_kPa=101.325, **kw)).primary for p in P]
    f = fig_new(t, 10.5, 3.5)
    a1, a2 = f.subplots(1, 2)
    for ax in (a1, a2):
        style(ax, t)
        ax.grid(True, axis="x", color=t["grid"], lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
    series = (("joint", [r.h_joint.si for r in rs]),
              ("contact spots", [r.h_solid.si for r in rs]),
              ("air in the gaps", [r.h_gap.si for r in rs]))
    where = {"joint": (12, 6), "contact spots": (4, 10), "air in the gaps": (2, -12)}
    for (label, h), c, mk in zip(series, t["series"], ("o", "s", "D")):
        a1.plot(P, np.array(h) / 1e3, color=c, lw=2, marker=mk, ms=4)
        i, dy = where[label]
        a1.annotate(label, (P[i], h[i] / 1e3), xytext=(6 if label == "joint" else 4, dy),
                    textcoords="offset points", ha="left", va="center", fontsize=9,
                    color=t["ink"])
    a1.set_xlim(P[0] * 0.9, P[-1] * 2.5)
    a1.set_xlabel("apparent contact pressure (MPa)")
    a1.set_ylabel("conductance (kW/m²K)")
    a1.set_title("SS 304 in air: two paths in parallel", fontsize=10, loc="left",
                 color=t["ink"])
    plain_log(a1, "x"); plain_log(a1, "y")
    pg = np.logspace(-2, 3, 16)
    for (gas, label), c, mk in zip((("He", "He"), ("air", "air"), ("Ar", "Ar")),
                                   t["series"], ("o", "s", "D")):
        h = [run_continuum(build_joint("SS304", "SS304", pressure_MPa=1.0, gas=gas,
                                       gas_pressure_kPa=float(g), **kw)).primary.h_gap.si
             for g in pg]
        a2.plot(pg, np.array(h) / 1e3, color=c, lw=2, marker=mk, ms=4)
        a2.annotate(label, (pg[-1], h[-1] / 1e3), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=t["ink"])
    a2.axvline(101.325, color=t["ink2"], lw=1, ls=(0, (3, 3)))
    a2.text(101.325 * 0.8, a2.get_ylim()[0] * 1.6, "1 atm", ha="right", fontsize=8,
            color=t["ink2"])
    a2.set_xlim(pg[0] * 0.8, pg[-1] * 4)
    a2.set_xlabel("gas pressure (kPa)")
    a2.set_ylabel("gap conductance h_g (kW/m²K)")
    a2.set_title("The gas path: rarefied below ~1 kPa, saturated at 1 atm", fontsize=10,
                 loc="left", color=t["ink"])
    plain_log(a2, "x"); plain_log(a2, "y")
    f.tight_layout()
    save(f, "continuum_paths", mode)


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


# ---- 5. atoms: what the MD cells look like --------------------------------------
STRUCT = OUT / "data" / "structures"
# Jmol-style element colours (the usual convention for atom pictures), drawn
# radii in A: covalent for the solids, about 0.4 sigma for the gases.
ATOM = {"Si": ("#d8b48c", 1.05), "Si(ox)": ("#d8b48c", 1.05), "Ge": ("#5f9ea0", 1.12),
        "O": ("#e8443a", 0.72), "Ar": ("#7fc8de", 1.36), "N2": ("#4a6fe3", 1.30),
        "He": ("#c9f0f2", 1.00)}


def load_slab(name):
    d = np.load(STRUCT / f"{name}.npz")
    els = dict(zip(d["types"].tolist(), d["elements"].tolist()))
    return d["atoms"], d["box"], d["slab"], els


def shade(hex_color, f):
    """Mix toward black (f < 0) or white (f > 0)."""
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(hex_color))
    return tuple(c + (1 - c) * f) if f > 0 else tuple(c * (1 + f))


def draw_atoms(ax, name, zwin, *, horizontal=False, edge="#00000055"):
    """Depth-sorted, shaded spheres of one slab; z inside zwin only."""
    from matplotlib.collections import EllipseCollection
    atoms, box, slab, els = load_slab(name)
    a = atoms[(atoms[:, 3] >= zwin[0]) & (atoms[:, 3] <= zwin[1])]
    a = a[np.argsort(a[:, 2])]                       # far (small y) first
    depth = (a[:, 2] - slab[0]) / (slab[1] - slab[0])    # 0 far, 1 near
    xs, zs = a[:, 1] - box[0][0], a[:, 3]
    xy = np.c_[zs, xs] if horizontal else np.c_[xs, zs]
    cols, hi, d = [], [], []
    for t, dep in zip(a[:, 0], depth):
        c, r = ATOM[els[int(t)]]
        cols.append(shade(c, -0.35 * (1 - dep)))
        hi.append(shade(c, 0.55))
        d.append(2 * r)
    d = np.asarray(d)
    ax.add_collection(EllipseCollection(d, d, np.zeros_like(d), units="xy", offsets=xy,
                                        offset_transform=ax.transData, facecolors=cols,
                                        edgecolors=edge, linewidths=0.4))
    off = np.array([-0.22, 0.22]) * d[:, None]
    ax.add_collection(EllipseCollection(0.35 * d, 0.35 * d, np.zeros_like(d), units="xy",
                                        offsets=xy + off, offset_transform=ax.transData,
                                        facecolors=hi, edgecolors="none", alpha=0.75))
    ax.set_aspect("equal")
    return els


def fig_atoms(t, mode):
    panels = [("sige_bonded", "Bonded Si/Ge", None),
              ("sige_vdw_1GPa", "van der Waals, 1 GPa", None),
              ("sige_ar_layer", "Ar monolayer", None),
              ("sige_n2_half", "N₂, half layer", None),
              ("si_asio2", "Si / amorphous SiO₂", None)]
    f = fig_new(t, 10.5, 3.7)
    axs = f.subplots(1, len(panels))
    seen = {}
    for ax, (name, title, _) in zip(axs, panels):
        atoms, box, slab, els = load_slab(name)
        ta = [k for k, e in els.items() if e == "Si"][0]
        z_if = atoms[atoms[:, 0] == ta, 3].max()
        if name == "si_asio2":
            z_if = 58.5
        win = (z_if - 14, z_if + 14)
        for k, e in draw_atoms(ax, name, win).items():
            seen.setdefault(e, ATOM[e][0])
        width = box[0][1] - box[0][0]
        ax.set_xlim(0, min(width, 27.0))
        ax.set_ylim(*win)
        ax.set_facecolor(t["surface"])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=9, color=t["ink"])
    axs[0].annotate("", xy=(-2.5, axs[0].get_ylim()[1] - 3), xytext=(-2.5, axs[0].get_ylim()[0] + 3),
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="->", color=t["ink2"], lw=1.2))
    axs[0].text(-4.0, np.mean(axs[0].get_ylim()), "heat flow", rotation=90, ha="center",
                va="center", fontsize=8, color=t["ink2"])
    from matplotlib.lines import Line2D
    labels = {"Si": "Si", "Ge": "Ge", "O": "O", "Ar": "Ar", "N2": "N₂ (one sphere)"}
    handles = [Line2D([], [], ls="", marker="o", ms=8, mfc=seen[e], mec="#00000055", label=l)
               for e, l in labels.items() if e in seen]
    leg = f.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                   fontsize=8, bbox_to_anchor=(0.5, -0.02))
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    f.suptitle("The interface region of five MD cells after relaxation, 300 K "
               "(2.8 nm tall, a 0.54 nm slab seen side-on; hot side below)",
               x=0.01, ha="left", fontsize=10, color=t["ink"])
    f.tight_layout(rect=(0, 0.04, 1, 1))
    save(f, "interface_atoms", mode)


def fig_cell(t, mode):
    """The whole NEMD column (Ar-layer cell) above three steady profiles."""
    prof = json.loads((OUT / "data" / "cell_profiles.json").read_text())
    atoms, box, slab, els = load_slab("sige_ar_layer")
    top = float(atoms[:, 3].max())
    z_if = float(atoms[atoms[:, 0] == 3, 3].mean())
    f = fig_new(t, 10.5, 4.3)
    gs = f.add_gridspec(2, 1, height_ratios=[1.0, 2.2], hspace=0.08)
    a1 = f.add_subplot(gs[0])
    a2 = f.add_subplot(gs[1], sharex=a1)
    draw_atoms(a1, "sige_ar_layer", (-1, top + 2), horizontal=True, edge="#00000033")
    a1.set_ylim(0, 22)
    a1.set_xlim(-2, top + 2)
    a1.set_aspect("equal", adjustable="datalim")
    a1.axis("off")
    regions = [(0, 10, "wall"), (10, 30, "hot\nreservoir"), (30, z_if, "Si"),
               (z_if, top - 30, "Ge"), (top - 30, top - 10, "cold\nreservoir"),
               (top - 10, top + 1.5, "wall")]
    for lo, hi, name in regions:
        a1.plot([lo + 0.6, hi - 0.6], [24.5, 24.5], color=t["ink2"], lw=1, clip_on=False)
        a1.text((lo + hi) / 2, 26, name, ha="center", va="bottom", fontsize=8,
                color=t["ink2"], clip_on=False)
    a1.annotate("Ar layer", (z_if, 11), xytext=(z_if + 22, 30), fontsize=8, color=t["ink"],
                arrowprops=dict(arrowstyle="-", color=t["ink2"], lw=0.8), annotation_clip=False)
    style(a2, t)
    names = {"bonded": "bonded", "vdw_1GPa": "van der Waals, 1 GPa",
             "ar_layer": "Ar layer, 1 GPa"}
    for (k, label), c, mk in zip(names.items(), t["series"], ("o", "s", "D")):
        p = prof[k]
        z, T = np.array(p["z_ang"]), np.array(p["T_K"])
        mob = (z > 30) & (z < z.max() - 15)
        y = (T - T[mob].mean()) / (p["q_W_m2"] / 1e9)
        a2.plot(z, y, color=c, lw=1.8, marker=mk, ms=3.5,
                label=f"{label}: h_K = {p['h_MW']:.0f} MW/m²K")
    for lo, hi, _ in regions[1:5:3]:
        a2.axvspan(lo, hi, color=t["grid"], lw=0, alpha=0.6)
    a2.axvline(z_if, color=t["ink2"], lw=1, ls=(0, (3, 3)))
    a2.set_xlabel("position along the heat flow (Å)")
    a2.set_ylabel("(T − mean) / q  (m²K/GW)")
    a2.text(z_if + 3, -5, "the step is 1/h_K", fontsize=8, color=t["ink2"])
    leg = a2.legend(frameon=False, fontsize=8, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    f.suptitle("One NEMD cell end to end (Si / Ar layer / Ge, 21,000 atoms; a 0.54 nm slab "
               "of 2.2 nm width drawn), and the steady profiles of three cells divided by "
               "their heat flux", x=0.01, y=1.03, ha="left", fontsize=10, color=t["ink"])
    save(f, "nemd_cell", mode)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    P, curves = joint_curves()
    for mode, t in THEMES.items():
        fig_joint(t, mode, P, curves)
        fig_paths(t, mode)
        fig_convergence(t, mode)
        fig_interfaces(t, mode)
        fig_atoms(t, mode)
        fig_cell(t, mode)
    for label, h in curves.items():
        print(f"{label:12} h_joint {h[0]:.0f} -> {h[-1]:.0f} W/m^2K over {P[0]:g}-{P[-1]:g} MPa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
