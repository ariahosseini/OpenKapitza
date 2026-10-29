"""Is a regression surrogate for interface conductance worth adding? (P1-15, KB §15)

Fits a Gaussian process (and, for comparison with the paper, boosted trees) to
the NIMS training table (Wu et al., Sci. Data 7, 36 (2020), Zenodo
10.5281/zenodo.3564173, CC BY 4.0) and scores it three ways:

  random 5-fold      what the papers report: the same pairs in train and test
  leave-pair-out     every row of one material pair held out together: the case
                     OpenKapitza would face with a new joint
  vs DMM             on the rows whose two materials are in our database, the
                     leave-pair-out GP against this package's Debye DMM

Scores are in log space (h spans 1-1000 MW/m^2K): RMS of log10(pred/meas) and
the share of predictions within 1.5x and 2x of the measurement.

    python3 scripts/evaluate_itr_surrogate.py training_dataset_for_ITR_prediction.xlsx
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

META = ("id-t", "Interface", "Film", "substrate", "interlayer1", "interlayer2", "itr")


def load(path: Path):
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["training data"]
    rows = [r for r in ws.iter_rows(values_only=True)]
    head = [str(h) for h in rows[0]]
    data = [r for r in rows[1:] if r and r[head.index("itr")] not in (None, "")]
    feats = [h for h in head if h not in META]
    X = np.array([[float(r[head.index(f)] or 0.0) for f in feats] for r in data])
    itr = np.array([float(r[head.index("itr")]) for r in data])            # m^2K/GW
    h = 1e3 / itr                                                          # MW/m^2K
    film = [str(r[head.index("Film")]).strip() for r in data]
    sub = [str(r[head.index("substrate")]).strip() for r in data]
    layer = np.array([int(r[head.index("interlayer")] or 0) for r in data])
    T = np.array([float(r[head.index("T")]) for r in data])
    pairs = ["/".join(sorted((a, b))) for a, b in zip(film, sub)]
    return X, h, feats, film, sub, layer, T, pairs


def scores(pred: np.ndarray, meas: np.ndarray) -> dict:
    r = np.log10(pred / meas)
    return {"n": int(len(r)), "rms_log10": float(np.sqrt(np.mean(r ** 2))),
            "median_factor": float(10 ** np.median(np.abs(r))),
            "within_1.5x": float(np.mean(np.abs(r) <= np.log10(1.5))),
            "within_2x": float(np.mean(np.abs(r) <= np.log10(2.0)))}


def models():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    gp = lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
        ConstantKernel(1.0) * Matern(length_scale=np.ones(1) * 3.0, nu=2.5) + WhiteKernel(0.05),
        normalize_y=True, n_restarts_optimizer=1, random_state=0))
    gb = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                           subsample=0.8, random_state=0)
    return {"gaussian process": gp, "boosted trees": gb}


def cross_val(make, X, y, folds) -> np.ndarray:
    pred = np.empty_like(y)
    for tr, te in folds:
        m = make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def dmm_predictions(film, sub, T, idx):
    """This package's Debye DMM for rows whose materials are in the database."""
    from openkapitza.md.dmm import dmm_conductance
    from openkapitza.schema import FlagSet
    from openkapitza.schema.db import available_materials, load_material
    from openkapitza.schema.quantities import Quantity
    have = set(available_materials())
    out, keep = [], []
    cache = {}
    for i in idx:
        a, b = film[i], sub[i]
        if a not in have or b not in have or a == b:
            continue                 # same material (bonded wafers): DMM does not describe it
        key = (a, b, round(T[i]))
        if key not in cache:
            ma, mb = load_material(a, FlagSet()), load_material(b, FlagSet())
            cache[key] = dmm_conductance(ma, mb, Quantity(T[i], "K"), FlagSet()).h_K.si / 1e6
        out.append(cache[key])
        keep.append(i)
    return np.array(out), np.array(keep, dtype=int)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx", type=Path)
    a = ap.parse_args(argv)
    from sklearn.model_selection import GroupKFold, KFold
    X, h, feats, film, sub, layer, T, pairs = load(a.xlsx)
    y = np.log10(h)
    groups = np.array(pairs)
    n_pairs = len(set(pairs))
    print(f"{len(y)} rows, {n_pairs} material pairs, {len(feats)} descriptors; "
          f"h {h.min():.1f}-{h.max():.0f} MW/m²K\n")
    rand = list(KFold(5, shuffle=True, random_state=0).split(X))
    lopo = list(GroupKFold(n_splits=min(n_pairs, 40)).split(X, y, groups))
    res = {}
    for name, make in models().items():
        for scheme, folds in (("random 5-fold", rand), ("leave-pair-out", lopo)):
            p = 10 ** cross_val(make, X, y, folds)
            res[(name, scheme)] = p
            s = scores(p, h)
            print(f"{name:17} {scheme:15} rms log10 {s['rms_log10']:.2f}  median x{s['median_factor']:.2f}"
                  f"  within 1.5x {s['within_1.5x']:.0%}  within 2x {s['within_2x']:.0%}")
    # constant baseline: the training median, leave-pair-out
    base = np.empty_like(h)
    for tr, te in lopo:
        base[te] = np.median(h[tr])
    s = scores(base, h)
    print(f"{'median baseline':17} {'leave-pair-out':15} rms log10 {s['rms_log10']:.2f}  "
          f"median x{s['median_factor']:.2f}  within 1.5x {s['within_1.5x']:.0%}  "
          f"within 2x {s['within_2x']:.0%}")

    print("\nOn rows our database can compute (both materials known):")
    dmm, idx = dmm_predictions(film, sub, T, range(len(h)))
    if len(idx) == 0:
        print("  none")
        return 0
    covered = sorted({pairs[i] for i in idx})
    print(f"  {len(idx)} rows, pairs: {', '.join(covered)}")
    for label, sel in (("all", np.ones(len(idx), bool)), ("no interlayer", layer[idx] == 0)):
        if not sel.any():
            continue
        ii = idx[sel]
        sd = scores(dmm[sel], h[ii])
        sg = scores(res[("gaussian process", "leave-pair-out")][ii], h[ii])
        sb = scores(base[ii], h[ii])
        for nm, s_ in (("DMM (this package)", sd), ("GP leave-pair-out", sg),
                       ("median baseline", sb)):
            print(f"  [{label:13}] {nm:20} n={s_['n']:3d} rms log10 {s_['rms_log10']:.2f}  "
                  f"median x{s_['median_factor']:.2f}  within 2x {s_['within_2x']:.0%}")
        # bias: DMM is an ideal-interface estimate; how far above the measurements?
        print(f"  [{label:13}] DMM / measured: median {np.median(dmm[sel] / h[ii]):.2f}, "
              f"range {np.min(dmm[sel] / h[ii]):.2f}-{np.max(dmm[sel] / h[ii]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
