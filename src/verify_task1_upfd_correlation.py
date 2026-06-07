"""
Verification Task 1: Spearman correlation of s(v) with centrality on UPFD trees.

Computes s(v) = |descendants(v)| / (N - 1) and four centralities (degree,
betweenness, eigenvector, NDA) per node, pooled across all train+val+test
trees of UPFD PolitiFact (feature: profile). Reports Spearman rho between
s and each centrality.

This is a one-shot verification; outputs JSON to outputs/verification/.
"""

from __future__ import annotations

import os
# Cap threads to leave headroom when run alongside other heavy workers.
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
import networkx as nx
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "verification"
OUT.mkdir(parents=True, exist_ok=True)


def cascade_contribution(g: nx.Graph, root: int, num_nodes: int) -> np.ndarray:
    """s(v) = |descendants(v)| / (N-1) in the tree rooted at `root`."""
    if num_nodes <= 1:
        return np.zeros(num_nodes)
    subtree = np.ones(num_nodes)
    order, parent = [], {root: -1}
    stack = [root]
    seen = {root}
    while stack:
        x = stack.pop()
        order.append(x)
        for nb in g.neighbors(x):
            if nb not in seen:
                seen.add(nb)
                parent[nb] = x
                stack.append(nb)
    for x in reversed(order):
        p = parent[x]
        if p >= 0:
            subtree[p] += subtree[x]
    descendants = subtree - 1.0
    return descendants / (num_nodes - 1)


def per_tree_features(edge_index: np.ndarray, num_nodes: int, root: int = 0):
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    ei = np.asarray(edge_index)
    g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))

    deg = np.array([d for _, d in sorted(g.degree())], dtype=np.float64)
    if num_nodes >= 3:
        btw = nx.betweenness_centrality(g, normalized=True)
        btw = np.array([btw[i] for i in range(num_nodes)], dtype=np.float64)
    else:
        btw = np.zeros(num_nodes, dtype=np.float64)
    try:
        eig = nx.eigenvector_centrality_numpy(g, max_iter=500)
        eig = np.array([eig[i] for i in range(num_nodes)], dtype=np.float64)
    except Exception:
        eig = np.zeros(num_nodes, dtype=np.float64)
    nda = nx.average_neighbor_degree(g)
    nda = np.array([nda[i] for i in range(num_nodes)], dtype=np.float64)

    s = cascade_contribution(g, root, num_nodes)
    return s, deg, btw, eig, nda


def main(fake_only: bool):
    from torch_geometric.datasets import UPFD

    t0 = time.time()
    s_all, deg_all, btw_all, eig_all, nda_all = [], [], [], [], []
    split_counts = {}
    tree_count = 0
    skipped_small = 0

    for split in ("train", "val", "test"):
        ds = UPFD(str(PROJECT_ROOT / "data" / "upfd"), "politifact", "profile", split)
        n_kept = 0
        for d in ds:
            if fake_only and int(d.y) != 1:
                continue
            if d.num_nodes < 2:
                skipped_small += 1
                continue
            ei = d.edge_index.numpy()
            s, deg, btw, eig, nda = per_tree_features(ei, int(d.num_nodes))
            s_all.append(s); deg_all.append(deg); btw_all.append(btw)
            eig_all.append(eig); nda_all.append(nda)
            n_kept += 1
        split_counts[split] = n_kept
        tree_count += n_kept
        print(f"[task1] split={split} trees={n_kept} elapsed={time.time()-t0:.1f}s")

    s = np.concatenate(s_all)
    deg = np.concatenate(deg_all)
    btw = np.concatenate(btw_all)
    eig = np.concatenate(eig_all)
    nda = np.concatenate(nda_all)
    n_total = s.size
    n_zero = int((s == 0).sum())
    frac_zero = float(n_zero) / float(n_total)

    rho_deg = float(spearmanr(s, deg).correlation)
    rho_btw = float(spearmanr(s, btw).correlation)
    rho_eig = float(spearmanr(s, eig).correlation)
    rho_nda = float(spearmanr(s, nda).correlation)

    # Also report on fake-only-vs-all and basic descriptives
    s_mean = float(s.mean()); s_std = float(s.std())
    print(f"[task1] N={n_total} | rho(s,deg)={rho_deg:.4f} | rho(s,btw)={rho_btw:.4f} | "
          f"rho(s,eig)={rho_eig:.4f} | rho(s,nda)={rho_nda:.4f} | frac_zero={frac_zero:.4f}")

    # git provenance
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    out = {
        "task": "Task 1 -- UPFD PolitiFact: rho(s, centrality)",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "script": "src/verify_task1_upfd_correlation.py",
        "command": " ".join(sys.argv),
        "git_sha": git_sha,
        "dataset": {"corpus": "politifact", "feature": "profile",
                    "root": str(PROJECT_ROOT / "data" / "upfd"),
                    "fake_only": bool(fake_only),
                    "splits_used": ["train", "val", "test"]},
        "tree_counts_per_split": split_counts,
        "tree_count_total": tree_count,
        "trees_skipped_num_nodes_lt_2": skipped_small,
        "pooled_node_count": int(n_total),
        "pooled_zero_label_count": int(n_zero),
        "pooled_fraction_leaves_s_eq_0": float(frac_zero),
        "s_mean": s_mean, "s_std": s_std,
        "spearman": {
            "s_vs_degree": round(rho_deg, 6),
            "s_vs_betweenness": round(rho_btw, 6),
            "s_vs_eigenvector": round(rho_eig, 6),
            "s_vs_neighbor_degree_avg": round(rho_nda, 6),
        },
        "spearman_4dp": {
            "s_vs_degree": round(rho_deg, 4),
            "s_vs_betweenness": round(rho_btw, 4),
            "s_vs_eigenvector": round(rho_eig, 4),
            "s_vs_neighbor_degree_avg": round(rho_nda, 4),
        },
        "wall_seconds": round(time.time() - t0, 2),
    }
    suffix = "fake_only" if fake_only else "all_labels"
    fp = OUT / f"upfd_label_correlation_{suffix}.json"
    with open(fp, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[task1] wrote {fp}")
    return out


if __name__ == "__main__":
    # Training used fake_only=True; we run both so the result covers fake-only
    # and full-label cases.
    main(fake_only=True)
    main(fake_only=False)
