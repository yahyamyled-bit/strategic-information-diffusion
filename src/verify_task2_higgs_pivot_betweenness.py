"""
Verification Task 2: Pivot-sampled (Brandes-Pich) betweenness on the Higgs graph.

Chapter sec 2.2.3 pre-registers k=1000 pivot sampling. Brandes-Pich estimates
betweenness for ALL nodes by running single-source Brandes from k randomly
chosen pivot sources. In python-igraph 0.11+, this is `sources=` (which
restricts the BFS source set); `vertices=` is OUTPUT restriction (target
restriction), which is NOT pivot sampling and would still require full
Brandes complexity from all sources.

Runs g.betweenness(sources=<1000 pivots>, cutoff=None), wall-clocks the call,
saves the full per-node betweenness vector (length n = 456,626) to
outputs/verification/higgs_betweenness_pivot1000.npz.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "verification"
OUT.mkdir(parents=True, exist_ok=True)

ADJ_PATH = PROJECT_ROOT / "data" / "processed" / "higgs_adjacency.npz"
NODE_ARR_PATH = PROJECT_ROOT / "data" / "processed" / "higgs_node_arrays.npz"

PIVOT_SAMPLE_SIZE = 1000
SEED = 20260612


def main():
    t_start_total = time.time()

    csr = sp.load_npz(ADJ_PATH)
    n = csr.shape[0]
    print(f"[task2] loaded adjacency: n={n}, nnz={csr.nnz}")

    import igraph as ig
    # Build igraph undirected graph from upper-triangle of CSR (symmetric)
    coo = csr.tocoo()
    ut = coo.row < coo.col
    edges = list(zip(coo.row[ut].tolist(), coo.col[ut].tolist()))
    t_build = time.time()
    g = ig.Graph(n=n, edges=edges, directed=False)
    print(f"[task2] built igraph: {n} verts, {g.ecount()} edges, build={time.time()-t_build:.1f}s")

    # Pivot sample
    rng = np.random.default_rng(SEED)
    pivots = rng.choice(n, size=PIVOT_SAMPLE_SIZE, replace=False)
    pivots_list = pivots.astype(int).tolist()

    # WALL-CLOCK the betweenness call only
    # NOTE: `sources=` restricts the BFS source set to the pivots (Brandes-Pich
    # estimator). `vertices=None` keeps the output a full-length betweenness
    # vector over all n nodes. Using `vertices=` instead would be target
    # restriction, not pivot sampling, and require full-Brandes complexity.
    print(f"[task2] starting g.betweenness(sources={PIVOT_SAMPLE_SIZE} pivots, cutoff=None)...")
    t_btw_start = time.time()
    btw = g.betweenness(sources=pivots_list, cutoff=None)
    t_btw = time.time() - t_btw_start
    print(f"[task2] betweenness wall-clock: {t_btw:.2f} s ({t_btw/60:.2f} min)")

    btw_arr = np.array(btw, dtype=np.float64)
    print(f"[task2] btw vector length = {btw_arr.size}, "
          f"min={btw_arr.min()}, max={btw_arr.max()}, mean={btw_arr.mean():.4g}, "
          f"std={btw_arr.std():.4g}, nonzero_frac={(btw_arr>0).mean():.4f}")

    # Load degree for sanity-check correlation
    arrs = np.load(NODE_ARR_PATH)
    degree = arrs["degree"].astype(np.float64)
    assert degree.size == btw_arr.size, "degree / betweenness length mismatch"
    rho_deg = float(spearmanr(btw_arr, degree).correlation)
    print(f"[task2] Spearman rho(btw_pivot, degree) = {rho_deg:.4f}")

    # Save NPZ
    out_npz = OUT / "higgs_betweenness_pivot1000.npz"
    np.savez_compressed(out_npz,
                        betweenness=btw_arr,
                        pivots=pivots.astype(np.int64),
                        seed=np.int64(SEED))
    print(f"[task2] saved {out_npz}")

    # Provenance JSON
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    meta = {
        "task": "Task 2 -- Higgs pivot-sampled betweenness wall-clock",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "script": "src/verify_task2_higgs_pivot_betweenness.py",
        "command": " ".join(sys.argv),
        "git_sha": git_sha,
        "adjacency_path": str(ADJ_PATH),
        "n_nodes": int(n),
        "n_edges_undirected": int(g.ecount()),
        "pivot_sample_size": PIVOT_SAMPLE_SIZE,
        "pivot_seed": SEED,
        "cutoff": None,
        "wall_seconds_betweenness_call": round(t_btw, 4),
        "wall_seconds_total_script": round(time.time() - t_start_total, 4),
        "btw_stats": {
            "len": int(btw_arr.size),
            "min": float(btw_arr.min()),
            "max": float(btw_arr.max()),
            "mean": float(btw_arr.mean()),
            "std": float(btw_arr.std()),
            "nonzero_count": int((btw_arr > 0).sum()),
            "nonzero_fraction": float((btw_arr > 0).mean()),
        },
        "spearman_rho_btw_pivot_vs_degree": rho_deg,
        "npz_path": str(out_npz),
    }
    fp = OUT / "higgs_pivot_betweenness_meta.json"
    with open(fp, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[task2] wrote {fp}")
    return meta


if __name__ == "__main__":
    main()
