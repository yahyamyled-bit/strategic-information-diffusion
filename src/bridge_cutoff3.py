"""
Parallel cutoff=3 bridge validation run.

Runs alongside the cutoff=5 bridge process; igraph is single-threaded. Writes to
distinct filenames so there is no conflict with the cutoff=5 process:
  outputs/stage5/higgs_features_cutoff3.npz
  outputs/stage5/bridge_validation_cutoff3.json

Uses the chapter's igraph fallback (sec 2.7) with a tighter cutoff for tractability.
Confirmatory; H2 already negative (s(v) ~ betweenness, rho=0.999 on UPFD).
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.stats import kendalltau

from . import config
from . import interventions as iv
from .sirf import (GraphData, SIRFParams, SIRFSimulation,
                   draw_fixed_credulity, draw_fixed_seeds)
from .stage5_gnn import FEATURE_NAMES, STAGE5_OUT, MODELS_DIR, apply_stats, _build_models
from .stage5_bridge import _depth_from_seeds, _pagerank

CUTOFF = 3
OUT_FEATS = STAGE5_OUT / "higgs_features_cutoff3.npz"
OUT_JSON = STAGE5_OUT / "bridge_validation_cutoff3.json"


def compute_features() -> dict:
    g = GraphData.load(); n = g.n
    t0 = time.time()

    degree = g.degree.astype(np.float64)
    print(f"[c3] degree ok [{time.time()-t0:.0f}s]", flush=True)

    nda = np.zeros(n); safe = degree > 0
    np.divide(g.csr.dot(degree), degree, out=nda, where=safe)
    print(f"[c3] neighbor-degree-avg ok [{time.time()-t0:.0f}s]", flush=True)

    _, vecs = eigsh(g.csr.astype(np.float64), k=1, which="LA")
    eig = np.abs(vecs[:, 0])
    print(f"[c3] eigenvector ok [{time.time()-t0:.0f}s]", flush=True)

    pr = _pagerank(g.csr)
    print(f"[c3] pagerank ok [{time.time()-t0:.0f}s]", flush=True)

    seeds = draw_fixed_seeds(n, config.K_SEEDS, config.SEED_REGIME_RANDOM)
    depth = _depth_from_seeds(g.csr, seeds)
    print(f"[c3] depth-from-seeds ok [{time.time()-t0:.0f}s]", flush=True)

    import igraph as ig
    coo = g.csr.tocoo(); ut = coo.row < coo.col
    gg = ig.Graph(n=n, edges=list(zip(coo.row[ut].tolist(), coo.col[ut].tolist())),
                  directed=False)
    print(f"[c3] igraph built [{time.time()-t0:.0f}s]", flush=True)

    btw = np.array(gg.betweenness(cutoff=CUTOFF), dtype=np.float64)
    print(f"[c3] betweenness cutoff={CUTOFF} ok [{time.time()-t0:.0f}s]", flush=True)

    feats = {"degree": degree, "betweenness": btw, "eigenvector": eig,
             "neighbor_degree_avg": nda, "depth_from_source": depth, "pagerank": pr}
    np.savez_compressed(OUT_FEATS, **feats)
    print(f"[c3] saved {OUT_FEATS.name} [{time.time()-t0:.0f}s]", flush=True)
    return feats


def gnn_scores_local(feats: dict) -> dict:
    """GNN inference on Higgs (own copy to keep this script self-contained)."""
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
        m = _build_models()[name]
        m.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt"))
        m.eval()
        with torch.no_grad():
            out[name] = m(Xs, edge_index).numpy()
        print(f"[c3] inferred {name}", flush=True)
    return out


def _topk(score, k=500):
    return set(np.argsort(score)[::-1][:k].tolist())


def run(containment_runs: int = 5) -> dict:
    t0 = time.time()
    feats = compute_features()
    gnn = gnn_scores_local(feats)
    heuristics = {"degree": feats["degree"], "betweenness": feats["betweenness"],
                  "pagerank": feats["pagerank"]}

    tau = {}
    for arch, gs in gnn.items():
        tau[arch] = {}
        gtop = _topk(gs)
        for hname, hs in heuristics.items():
            idx = np.random.default_rng(0).choice(len(gs), size=min(50000, len(gs)),
                                                  replace=False)
            kt = kendalltau(gs[idx], hs[idx]).correlation
            overlap = len(gtop & _topk(hs)) / 500.0
            tau[arch][hname] = {"kendall_tau": float(kt),
                                "top500_overlap": float(overlap),
                                "reducible_to_centrality": bool(abs(kt) > 0.70)}
            print(f"[c3] tau {arch} vs {hname}: {kt:+.3f}, top500 overlap {overlap:.2f}",
                  flush=True)

    g = GraphData.load()
    cred = draw_fixed_credulity(g.n)
    seeds = draw_fixed_seeds(g.n, config.K_SEEDS, config.SEED_REGIME_RANDOM)
    sim = SIRFSimulation(g, SIRFParams())
    rankings = {"degree": feats["degree"], "betweenness": feats["betweenness"],
                "pagerank": feats["pagerank"], "GCN": gnn["GCN"],
                "GraphSAGE": gnn["GraphSAGE"], "GAT": gnn["GAT"]}

    base = [sim.run(condition_id=30, run_index=ri, credulity=cred, seeds=seeds).spread_size
            for ri in range(containment_runs)]
    base_mean = float(np.mean(base))
    print(f"[c3] baseline mean spread {base_mean:.0f}", flush=True)

    containment = {}
    for rname, score in rankings.items():
        targets = np.argsort(score)[::-1][:500].astype(np.int64)
        spreads = [sim.run(condition_id=31, run_index=ri, credulity=cred, seeds=seeds,
                           intervention=iv.NodeRemoval(targets)).spread_size
                   for ri in range(containment_runs)]
        ce = (base_mean - np.mean(spreads)) / base_mean * 100.0
        containment[rname] = {"mean_spread": float(np.mean(spreads)),
                              "containment_efficiency": float(ce)}
        print(f"[c3] containment {rname:<10} CE={ce:.1f}%", flush=True)

    out = {"betweenness_cutoff": CUTOFF, "kendall_tau_vs_heuristics": tau,
           "containment_node_removal_top500": containment,
           "baseline_spread_mean": base_mean,
           "h2_verdict_note": "Confirmatory; H2 already negative (s(v)~betweenness rho=0.999 on UPFD trees). See bridge_validation_cutoff3 numbers.",
           "wall_seconds": round(time.time() - t0, 1)}
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[c3] done in {out['wall_seconds']}s -> {OUT_JSON.name}", flush=True)
    return out


if __name__ == "__main__":
    run()
