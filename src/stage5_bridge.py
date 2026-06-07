"""
Stage 5 -- GNN bridge validation on the Higgs graph (H2).

Computes the five structural features on Higgs, standardizes them with the UPFD
TRAINING-split statistics (no leakage), runs each trained GNN at inference, and
compares the GNN node ranking against degree, betweenness, and PageRank:
  * Kendall's tau between rankings (chapter criterion: tau <= 0.70 means the GNN
    ranking is not reducible to a single centrality);
  * top-k overlap; and
  * containment: node-removal of the top-500 by each ranking, mean spread.

Betweenness on the 456K-node graph uses igraph with a distance cutoff (chapter
sec 2.7 fallback); see bridge_validation.json for the deviation from k=1000
Brandes.

Outputs (outputs/stage5/): higgs_features.npz, bridge_validation.json
"""

from __future__ import annotations

import json
import time

import numpy as np
import scipy.sparse as sp

from . import config
from .sirf import (GraphData, SIRFParams, SIRFSimulation,
                   draw_fixed_credulity, draw_fixed_seeds)
from . import interventions as iv
from .stage5_gnn import FEATURE_NAMES, STAGE5_OUT, MODELS_DIR, apply_stats, _build_models

HIGGS_FEATS_NPZ = STAGE5_OUT / "higgs_features.npz"
BETWEENNESS_CUTOFF = 5


# ==========================================================================
# Higgs structural features
# ==========================================================================
def _depth_from_seeds(csr, seeds, bound=config.BFS_DEPTH_BOUND):
    n = csr.shape[0]
    depth = np.full(n, bound + 1, dtype=np.float64)   # unreached -> bound+1
    frontier = np.zeros(n, dtype=bool); frontier[seeds] = True
    depth[seeds] = 0
    indptr, indices = csr.indptr, csr.indices
    cur = np.array(seeds, dtype=np.int64)
    for d in range(1, bound + 1):
        # neighbors of current frontier
        nbrs = np.concatenate([indices[indptr[u]:indptr[u + 1]] for u in cur]) if cur.size else np.array([], dtype=np.int64)
        nbrs = np.unique(nbrs)
        new = nbrs[depth[nbrs] > bound]
        depth[new] = d
        cur = new
        if cur.size == 0:
            break
    return depth


def _pagerank(csr, d=0.85, iters=100, tol=1e-9):
    n = csr.shape[0]
    deg = np.asarray(csr.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    # column-stochastic transition via D^-1 on the (symmetric) adjacency
    Dinv = sp.diags(1.0 / deg)
    M = csr.dot(Dinv)            # M[i,j] = A[i,j]/deg[j]
    pr = np.full(n, 1.0 / n)
    for _ in range(iters):
        new = (1 - d) / n + d * M.dot(pr)
        if np.abs(new - pr).sum() < tol:
            pr = new; break
        pr = new
    return pr


def compute_higgs_features(force: bool = False) -> dict:
    if HIGGS_FEATS_NPZ.exists() and not force:
        z = np.load(HIGGS_FEATS_NPZ)
        return {k: z[k] for k in z.files}
    g = GraphData.load()
    n = g.n
    t0 = time.time()
    degree = g.degree.astype(np.float64)
    print(f"[bridge] degree ok [{time.time()-t0:.0f}s]")

    nda = np.zeros(n)
    safe = degree > 0
    np.divide(g.csr.dot(degree), degree, out=nda, where=safe)   # mean neighbor degree
    print(f"[bridge] neighbor-degree-avg ok [{time.time()-t0:.0f}s]")

    from scipy.sparse.linalg import eigsh
    vals, vecs = eigsh(g.csr.astype(np.float64), k=1, which="LA")
    eig = np.abs(vecs[:, 0])
    print(f"[bridge] eigenvector ok [{time.time()-t0:.0f}s]")

    pr = _pagerank(g.csr)
    print(f"[bridge] pagerank ok [{time.time()-t0:.0f}s]")

    seeds = draw_fixed_seeds(n, config.K_SEEDS, config.SEED_REGIME_RANDOM)
    depth = _depth_from_seeds(g.csr, seeds)
    print(f"[bridge] depth-from-seeds ok [{time.time()-t0:.0f}s]")

    import igraph as ig
    coo = g.csr.tocoo(); ut = coo.row < coo.col
    gg = ig.Graph(n=n, edges=list(zip(coo.row[ut].tolist(), coo.col[ut].tolist())), directed=False)
    btw = np.array(gg.betweenness(cutoff=BETWEENNESS_CUTOFF), dtype=np.float64)
    print(f"[bridge] betweenness (igraph, cutoff={BETWEENNESS_CUTOFF}) ok [{time.time()-t0:.0f}s]")

    feats = {"degree": degree, "betweenness": btw, "eigenvector": eig,
             "neighbor_degree_avg": nda, "depth_from_source": depth,
             "pagerank": pr}
    np.savez_compressed(HIGGS_FEATS_NPZ, **feats)
    print(f"[bridge] saved Higgs features [{time.time()-t0:.0f}s]")
    return feats


# ==========================================================================
# GNN inference on Higgs
# ==========================================================================
def gnn_scores(feats: dict) -> dict:
    import torch
    with open(STAGE5_OUT / "feature_stats.json") as fh:
        stats = json.load(fh)
    X = np.column_stack([feats[f] for f in FEATURE_NAMES])
    Xs = torch.tensor(apply_stats(X, stats), dtype=torch.float32)
    g = GraphData.load()
    coo = g.csr.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    out = {}
    for name in ("GCN", "GraphSAGE", "GAT"):
        model = _build_models()[name]
        model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt"))
        model.eval()
        with torch.no_grad():
            out[name] = model(Xs, edge_index).numpy()
        print(f"[bridge] inferred {name}")
    return out


# ==========================================================================
# Bridge comparison
# ==========================================================================
def _topk(score, k=500):
    return set(np.argsort(score)[::-1][:k].tolist())


def run(containment_runs: int = 5) -> dict:
    from scipy.stats import kendalltau
    t0 = time.time()
    feats = compute_higgs_features()
    gnn = gnn_scores(feats)
    heuristics = {"degree": feats["degree"], "betweenness": feats["betweenness"],
                  "pagerank": feats["pagerank"]}

    # Kendall tau + top-500 overlap, GNN vs each heuristic
    tau = {}
    for arch, gs in gnn.items():
        tau[arch] = {}
        gtop = _topk(gs)
        for hname, hs in heuristics.items():
            # subsample for kendalltau (O(n log n) ok, but cap for speed)
            idx = np.random.default_rng(0).choice(len(gs), size=min(50000, len(gs)), replace=False)
            kt = kendalltau(gs[idx], hs[idx]).correlation
            overlap = len(gtop & _topk(hs)) / 500.0
            tau[arch][hname] = {"kendall_tau": float(kt), "top500_overlap": float(overlap),
                                "reducible_to_centrality": bool(abs(kt) > 0.70)}

    # Containment: node-removal of top-500 by each ranking
    g = GraphData.load()
    cred = draw_fixed_credulity(g.n)
    seeds = draw_fixed_seeds(g.n, config.K_SEEDS, config.SEED_REGIME_RANDOM)
    sim = SIRFSimulation(g, SIRFParams())
    rankings = {"degree": feats["degree"], "betweenness": feats["betweenness"],
                "pagerank": feats["pagerank"], "GCN": gnn["GCN"],
                "GraphSAGE": gnn["GraphSAGE"], "GAT": gnn["GAT"]}
    containment = {}
    base = []
    for ri in range(containment_runs):
        base.append(sim.run(condition_id=20, run_index=ri, credulity=cred, seeds=seeds).spread_size)
    base_mean = float(np.mean(base))
    for rname, score in rankings.items():
        targets = np.argsort(score)[::-1][:500].astype(np.int64)
        spreads = []
        for ri in range(containment_runs):
            r = sim.run(condition_id=21, run_index=ri, credulity=cred, seeds=seeds,
                        intervention=iv.NodeRemoval(targets))
            spreads.append(r.spread_size)
        ce = (base_mean - np.mean(spreads)) / base_mean * 100.0
        containment[rname] = {"mean_spread": float(np.mean(spreads)), "containment_efficiency": float(ce)}
        print(f"[bridge] containment {rname:<10} CE={ce:.1f}%")

    out = {"betweenness_cutoff": BETWEENNESS_CUTOFF, "kendall_tau_vs_heuristics": tau,
           "containment_node_removal_top500": containment,
           "baseline_spread_mean": base_mean,
           "h2_verdict": "see kendall_tau (tau>0.70 => GNN reducible to centrality) "
                         "and whether any GNN CE exceeds the best heuristic CE",
           "wall_seconds": round(time.time() - t0, 1)}
    with open(STAGE5_OUT / "bridge_validation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[bridge] done in {out['wall_seconds']}s -> {STAGE5_OUT}/bridge_validation.json")
    return out


if __name__ == "__main__":
    run()
