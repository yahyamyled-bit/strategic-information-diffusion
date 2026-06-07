"""
Verification Task 3: GNN OOD diagnostic on Higgs (and UPFD test-set comparison).

Constructs Higgs's 5-feature matrix (degree, betweenness_pivot, eigenvector,
neighbor_degree_avg, depth_from_source), standardizes it with the same training
statistics used in stage5_bridge.py (feature_stats.json), runs forward inference
under each trained model (GCN, GraphSAGE, GAT). Reports per-model min/max/mean/
std + 10-bin histograms on Higgs and on the UPFD held-out test split.

Also computes Spearman rho between each model's Higgs scores and (a) Higgs
degree and (b) the pivot-sampled betweenness from Task 2.

Thread count capped for parallel execution with other workers.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["TORCH_NUM_THREADS"] = "2"

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

STAGE5 = PROJECT_ROOT / "outputs" / "stage5"
MODELS_DIR = STAGE5 / "models"

ADJ_PATH = PROJECT_ROOT / "data" / "processed" / "higgs_adjacency.npz"
NODE_ARR_PATH = PROJECT_ROOT / "data" / "processed" / "higgs_node_arrays.npz"
BTW_NPZ = OUT / "higgs_betweenness_pivot1000.npz"

FEATURE_NAMES = ["degree", "betweenness", "eigenvector",
                 "neighbor_degree_avg", "depth_from_source"]


def build_higgs_features(csr, degree, pivot_btw):
    """Return dict with the 5 features used in stage5_bridge.compute_higgs_features."""
    import torch  # unused here but keep import for symmetry
    n = csr.shape[0]
    t0 = time.time()

    # eigenvector via eigsh on the symmetric adjacency
    from scipy.sparse.linalg import eigsh
    vals, vecs = eigsh(csr.astype(np.float64), k=1, which="LA")
    eig = np.abs(vecs[:, 0])
    print(f"[task3] eigenvector ok [{time.time()-t0:.1f}s]")

    # NDA = (A @ degree) / degree
    deg64 = degree.astype(np.float64)
    nda = np.zeros(n)
    safe = deg64 > 0
    np.divide(csr.dot(deg64), deg64, out=nda, where=safe)
    print(f"[task3] NDA ok [{time.time()-t0:.1f}s]")

    # depth from same seed set as bridge code
    from src.sirf import draw_fixed_seeds
    from src import config as _cfg
    seeds = draw_fixed_seeds(n, _cfg.K_SEEDS, _cfg.SEED_REGIME_RANDOM)

    def _depth_from_seeds(csr_local, seeds_local, bound=_cfg.BFS_DEPTH_BOUND):
        depth = np.full(n, bound + 1, dtype=np.float64)
        depth[seeds_local] = 0
        indptr, indices = csr_local.indptr, csr_local.indices
        cur = np.array(seeds_local, dtype=np.int64)
        for d in range(1, bound + 1):
            if cur.size == 0:
                break
            nbrs = np.concatenate([indices[indptr[u]:indptr[u + 1]] for u in cur])
            nbrs = np.unique(nbrs)
            new = nbrs[depth[nbrs] > bound]
            depth[new] = d
            cur = new
        return depth

    depth = _depth_from_seeds(csr, seeds)
    print(f"[task3] depth-from-seeds ok [{time.time()-t0:.1f}s]")

    feats = {
        "degree": deg64,
        "betweenness": pivot_btw.astype(np.float64),
        "eigenvector": eig.astype(np.float64),
        "neighbor_degree_avg": nda.astype(np.float64),
        "depth_from_source": depth.astype(np.float64),
    }
    return feats


def histogram_10(arr):
    h, e = np.histogram(arr, bins=10)
    return {"counts": h.tolist(), "edges": e.tolist()}


def stats_dict(arr):
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "n": int(arr.size),
    }


def main():
    import torch
    torch.set_num_threads(2)

    t0 = time.time()

    # ----- Load training-split stats and models -----
    with open(STAGE5 / "feature_stats.json") as fh:
        stats = json.load(fh)
    mean = np.array(stats["mean"], dtype=np.float64)
    std = np.array(stats["std"], dtype=np.float64)
    print(f"[task3] stats mean={mean}, std={std}")

    # ----- Higgs adjacency + degree -----
    csr = sp.load_npz(ADJ_PATH)
    arrs = np.load(NODE_ARR_PATH)
    degree_higgs = arrs["degree"]
    n_higgs = csr.shape[0]
    print(f"[task3] loaded Higgs n={n_higgs}")

    # ----- Pivot betweenness from Task 2 -----
    btw_npz = np.load(BTW_NPZ)
    pivot_btw = btw_npz["betweenness"]
    assert pivot_btw.size == n_higgs, f"btw size {pivot_btw.size} != n_higgs {n_higgs}"
    print(f"[task3] loaded pivot betweenness (Task 2): min={pivot_btw.min()}, max={pivot_btw.max()}")

    # ----- Compute remaining Higgs features -----
    feats = build_higgs_features(csr, degree_higgs, pivot_btw)

    X = np.column_stack([feats[f] for f in FEATURE_NAMES])
    print(f"[task3] X shape = {X.shape}")
    Xs = (X - mean) / std
    print(f"[task3] standardized: min={Xs.min(axis=0)}, max={Xs.max(axis=0)}")

    # Build edge_index for Higgs (both directions, symmetric)
    coo = csr.tocoo()
    edge_index_higgs = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    Xs_t = torch.tensor(Xs, dtype=torch.float32)

    # ----- Build and load models -----
    from src.stage5_gnn import _build_models
    models = _build_models()

    # ----- Higgs inference -----
    higgs_scores = {}
    for name in ("GCN", "GraphSAGE", "GAT"):
        m = models[name]
        m.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", weights_only=True))
        m.eval()
        with torch.no_grad():
            s = m(Xs_t, edge_index_higgs).numpy()
        higgs_scores[name] = s
        print(f"[task3] Higgs {name}: min={s.min():.4f} max={s.max():.4f} mean={s.mean():.4f} std={s.std():.4f}")

    # ----- UPFD test split inference (re-using the existing training pipeline) -----
    from src.stage5_gnn import build_upfd_dataset, apply_stats, _to_pyg
    print("[task3] building UPFD politifact dataset for held-out test...")
    trees, split_tag = build_upfd_dataset("politifact", fake_only=True)
    test_pyg = _to_pyg(trees, split_tag, stats, "test")
    print(f"[task3] UPFD test #trees = {len(test_pyg)}, total nodes = {sum(int(d.x.shape[0]) for d in test_pyg)}")

    upfd_scores = {n: [] for n in ("GCN", "GraphSAGE", "GAT")}
    for name in ("GCN", "GraphSAGE", "GAT"):
        m = models[name]
        # already loaded above
        m.eval()
        with torch.no_grad():
            preds = [m(d.x, d.edge_index).numpy() for d in test_pyg]
        s = np.concatenate(preds)
        upfd_scores[name] = s
        print(f"[task3] UPFD-test {name}: min={s.min():.4f} max={s.max():.4f} mean={s.mean():.4f} std={s.std():.4f}")

    # ----- Spearman: Higgs scores vs degree, vs pivot betweenness -----
    spearman_table = {}
    for name in ("GCN", "GraphSAGE", "GAT"):
        s = higgs_scores[name]
        r_d = float(spearmanr(s, degree_higgs.astype(np.float64)).correlation)
        r_b = float(spearmanr(s, pivot_btw.astype(np.float64)).correlation)
        spearman_table[name] = {"vs_degree": r_d, "vs_pivot_betweenness": r_b}
        print(f"[task3] Spearman {name}: vs_deg={r_d:.4f}  vs_pivot_btw={r_b:.4f}")

    # ----- Build report dict -----
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    report = {
        "task": "Task 3 -- GNN OOD diagnostic on Higgs",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "script": "src/verify_task3_gnn_ood.py",
        "command": " ".join(sys.argv),
        "git_sha": git_sha,
        "feature_order": FEATURE_NAMES,
        "training_stats_used": {"mean": stats["mean"], "std": stats["std"]},
        "higgs": {
            "n": int(n_higgs),
            "betweenness_source": "pivot1000 from Task 2",
            "standardized_feature_minmax": {
                FEATURE_NAMES[i]: {"min": float(Xs[:, i].min()), "max": float(Xs[:, i].max())}
                for i in range(5)
            },
            "scores": {
                name: {**stats_dict(higgs_scores[name]),
                       "hist_10": histogram_10(higgs_scores[name])}
                for name in ("GCN", "GraphSAGE", "GAT")
            },
            "spearman_vs_baselines": spearman_table,
        },
        "upfd_test": {
            "num_trees": len(test_pyg),
            "num_nodes_total": int(sum(int(d.x.shape[0]) for d in test_pyg)),
            "scores": {
                name: {**stats_dict(upfd_scores[name]),
                       "hist_10": histogram_10(upfd_scores[name])}
                for name in ("GCN", "GraphSAGE", "GAT")
            },
        },
        "wall_seconds_total": round(time.time() - t0, 2),
    }
    fp = OUT / "gnn_ood_diagnostic.json"
    with open(fp, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[task3] wrote {fp}")

    # Save full per-node GNN score vectors for downstream use (Stage 6 GNN targeting)
    scores_npz = OUT / "higgs_gnn_scores.npz"
    np.savez_compressed(
        scores_npz,
        gcn=higgs_scores["GCN"].astype(np.float32),
        graphsage=higgs_scores["GraphSAGE"].astype(np.float32),
        gat=higgs_scores["GAT"].astype(np.float32),
    )
    print(f"[task3] wrote {scores_npz} (full Higgs node scores for downstream use)")

    return report


if __name__ == "__main__":
    main()
