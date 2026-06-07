"""
Stage 7 helper -- build outputs/stage5/higgs_features.npz with the pivot
betweenness feature.

compute_higgs_features() (stage5_bridge.py) computes the betweenness feature via
igraph .betweenness(cutoff=5) over all 456k nodes, which is intractable on this
graph. The methodology chapter specifies k=1000 pivot sampling for the Higgs
betweenness feature, so this script loads the precomputed pivot betweenness
on disk instead.

This script reproduces the 5 fast features exactly as stage5_bridge does
(degree, neighbor_degree_avg, eigenvector, depth_from_source, pagerank) and uses
the already-computed pivot betweenness on disk for the betweenness column. It
writes the same key set the retrain script (stage7_retrain_inference.py) and
stage5 inference read.

Guard: the build aborts if it exceeds 60s (it normally takes ~7s); there is no
betweenness computation here to loop on.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse.linalg import eigsh

from . import config
from .sirf import GraphData, draw_fixed_seeds
from .stage5_bridge import _pagerank, _depth_from_seeds

OUT = config.OUTPUTS_DIR / "stage5" / "higgs_features.npz"
PIVOT_BTW = config.OUTPUTS_DIR / "verification" / "higgs_betweenness_pivot1000.npz"
GUARD_S = 60.0


def _guard(t0, step):
    el = time.time() - t0
    if el > GUARD_S:
        raise RuntimeError(f"build exceeded {GUARD_S}s at '{step}' ({el:.0f}s) -- abort")
    print(f"[feat] {step} ok [{el:.0f}s]", flush=True)


def main():
    if not PIVOT_BTW.exists():
        raise FileNotFoundError(
            f"pivot betweenness missing at {PIVOT_BTW}; "
            f"run verify_task2_higgs_pivot_betweenness first.")

    g = GraphData.load()
    n = g.n
    t0 = time.time()

    degree = g.degree.astype(np.float64)
    _guard(t0, "degree")

    nda = np.zeros(n)
    safe = degree > 0
    np.divide(g.csr.dot(degree), degree, out=nda, where=safe)   # mean neighbor degree
    _guard(t0, "neighbor_degree_avg")

    vals, vecs = eigsh(g.csr.astype(np.float64), k=1, which="LA")
    eig = np.abs(vecs[:, 0])
    _guard(t0, "eigenvector")

    pr = _pagerank(g.csr)
    _guard(t0, "pagerank")

    seeds = draw_fixed_seeds(n, config.K_SEEDS, config.SEED_REGIME_RANDOM)
    depth = _depth_from_seeds(g.csr, seeds)
    _guard(t0, "depth_from_source")

    # betweenness feature: k=1000 pivot sampling (per methodology), loaded from disk
    btw = np.load(PIVOT_BTW)["betweenness"].astype(np.float64)
    if btw.shape[0] != n:
        raise ValueError(f"pivot betweenness len {btw.shape[0]} != n {n}")
    _guard(t0, "betweenness (pivot, loaded)")

    feats = {"degree": degree, "betweenness": btw, "eigenvector": eig,
             "neighbor_degree_avg": nda, "depth_from_source": depth,
             "pagerank": pr}
    np.savez_compressed(OUT, **feats)
    print(f"[feat] saved {OUT} keys={list(feats.keys())} "
          f"[total {time.time() - t0:.0f}s]", flush=True)
    # sanity
    for k, v in feats.items():
        print(f"[feat]   {k:<22} shape={v.shape} "
              f"min={np.nanmin(v):.3g} max={np.nanmax(v):.3g}")


if __name__ == "__main__":
    main()
