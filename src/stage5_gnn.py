"""
Stage 5 -- GNN training and bridge validation.

Trains GCN, GraphSAGE, and GAT to predict each node's cascade-contribution score
s(v) = |descendants(v)| / (N - 1) from five structural features (chapter sec
2.2.3):

  1. degree
  2. betweenness centrality
  3. eigenvector centrality
  4. neighbor-degree average
  5. depth-from-source  (tree: hops from root; Higgs: BFS from nearest seed, <=6)

Clustering coefficient and k-core are excluded (degenerate on trees).
Training is on the UPFD PolitiFact FAKE subset only (per the calibrate/validate
split; GossipCop is reserved for validation). Features are
standardized with the TRAINING-split statistics; the same statistics are reused
at Higgs inference (no leakage).

Training defaults (not fixed by the methodology
chapter): MSE loss, Adam, 2 layers, 64 hidden, early stopping on a held-out split.

This module covers training and the UPFD/Twitter side. Higgs feature computation,
inference, and the bridge validation (Kendall tau vs degree/betweenness/PageRank
+ containment comparison) live in stage5_bridge.py because the Higgs features
(esp. approximate betweenness) are a separate heavy computation.

Outputs (outputs/stage5/):
  feature_stats.json   -- training-split mean/std used for standardization
  gnn_metrics.json     -- per-architecture train/val/test loss + rank correlation
  models/<arch>.pt     -- trained weights
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import networkx as nx

from . import config

STAGE5_OUT = config.OUTPUTS_DIR / "stage5"
MODELS_DIR = STAGE5_OUT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = ["degree", "betweenness", "eigenvector",
                 "neighbor_degree_avg", "depth_from_source"]


# ==========================================================================
# UPFD tree -> 5 structural features + cascade-contribution labels
# ==========================================================================
def _tree_to_nx(edge_index: np.ndarray, num_nodes: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    ei = np.asarray(edge_index)
    g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    return g


def _depth_from_root(g: nx.Graph, root: int, num_nodes: int) -> np.ndarray:
    depth = np.full(num_nodes, 0.0)
    for node, d in nx.single_source_shortest_path_length(g, root).items():
        depth[node] = d
    return depth


def _cascade_contribution(g: nx.Graph, root: int, num_nodes: int) -> np.ndarray:
    """s(v) = |descendants(v)| / (N-1) in the tree rooted at `root`."""
    if num_nodes <= 1:
        return np.zeros(num_nodes)
    # subtree size via DFS post-order on the tree (undirected, rooted at root)
    subtree = np.ones(num_nodes)
    order, parent = [], {root: -1}
    stack = [root]
    seen = {root}
    while stack:
        x = stack.pop()
        order.append(x)
        for nb in g.neighbors(x):
            if nb not in seen:
                seen.add(nb); parent[nb] = x; stack.append(nb)
    for x in reversed(order):
        p = parent[x]
        if p >= 0:
            subtree[p] += subtree[x]
    descendants = subtree - 1.0          # exclude the node itself
    return descendants / (num_nodes - 1)


def tree_features_and_labels(edge_index: np.ndarray, num_nodes: int,
                             root: int = 0):
    """Return (X [num_nodes,5], y [num_nodes]) for one UPFD tree."""
    g = _tree_to_nx(edge_index, num_nodes)
    deg = np.array([d for _, d in sorted(g.degree())], dtype=np.float64)
    if num_nodes >= 3:
        btw = nx.betweenness_centrality(g, normalized=True)
        btw = np.array([btw[i] for i in range(num_nodes)])
    else:
        btw = np.zeros(num_nodes)
    try:
        eig = nx.eigenvector_centrality_numpy(g, max_iter=500)
        eig = np.array([eig[i] for i in range(num_nodes)])
    except Exception:
        eig = np.zeros(num_nodes)
    avgnd = nx.average_neighbor_degree(g)
    avgnd = np.array([avgnd[i] for i in range(num_nodes)])
    depth = _depth_from_root(g, root, num_nodes)
    X = np.column_stack([deg, btw, eig, avgnd, depth])
    y = _cascade_contribution(g, root, num_nodes)
    return X.astype(np.float32), y.astype(np.float32)


def build_upfd_dataset(corpus: str = "politifact", fake_only: bool = True):
    """Build per-tree (X, y, edge_index) for the UPFD corpus, plus split tags."""
    from torch_geometric.datasets import UPFD
    trees, split_tag = [], []
    t0 = time.time()
    for split in ("train", "val", "test"):
        ds = UPFD("data/upfd", corpus, "profile", split)
        for d in ds:
            if fake_only and int(d.y) != 1:
                continue
            if d.num_nodes < 2:
                continue
            X, y = tree_features_and_labels(d.edge_index.numpy(), d.num_nodes)
            trees.append((X, y, d.edge_index))
            split_tag.append(split)
    print(f"[stage5] built {len(trees)} {corpus} fake trees "
          f"[{time.time()-t0:.0f}s]")
    return trees, np.array(split_tag)


# ==========================================================================
# Standardization (fit on training split, reuse everywhere -- no leakage)
# ==========================================================================
def fit_feature_stats(trees, split_tag) -> dict:
    Xtr = np.vstack([X for (X, _, _), s in zip(trees, split_tag) if s == "train"])
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist(), "features": FEATURE_NAMES}


def apply_stats(X: np.ndarray, stats: dict) -> np.ndarray:
    return (X - np.array(stats["mean"])) / np.array(stats["std"])


# ==========================================================================
# GNN architectures (node-level regression)
# ==========================================================================
def _build_models(in_dim=5, hidden=64):
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv

    class GNN(nn.Module):
        def __init__(self, conv_cls, **kw):
            super().__init__()
            self.c1 = conv_cls(in_dim, hidden, **kw)
            self.c2 = conv_cls(hidden, hidden, **kw)
            self.head = nn.Linear(hidden, 1)
            self.act = nn.ReLU()

        def forward(self, x, edge_index):
            h = self.act(self.c1(x, edge_index))
            h = self.act(self.c2(h, edge_index))
            return self.head(h).squeeze(-1)

    return {
        "GCN": GNN(GCNConv),
        "GraphSAGE": GNN(SAGEConv),
        "GAT": GNN(GATConv, heads=1),
    }


def train_one(model, train_data, val_data, epochs=300, lr=1e-2, patience=30):
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()
    best_val, best_state, wait = float("inf"), None, 0
    rng = np.random.default_rng(config.MASTER_SEED)
    for ep in range(epochs):
        model.train()
        # mini-batch SGD: one optimizer step per tree (many more updates than a
        # single full-batch step per epoch, which collapsed to predicting ~0 on
        # the zero-inflated labels).
        for i in rng.permutation(len(train_data)):
            d = train_data[i]
            opt.zero_grad()
            loss = loss_fn(model(d.x, d.edge_index), d.y)
            loss.backward()
            opt.step()
        # validation
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(loss_fn(model(d.x, d.edge_index), d.y))
                                for d in val_data]))
        if vl < best_val - 1e-5:
            best_val, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def _to_pyg(trees, split_tag, stats, which):
    import torch
    from torch_geometric.data import Data
    out = []
    for (X, y, ei), s in zip(trees, split_tag, strict=False):
        if s != which:
            continue
        out.append(Data(x=torch.tensor(apply_stats(X, stats), dtype=torch.float32),
                        y=torch.tensor(y, dtype=torch.float32),
                        edge_index=ei))
    return out


def run_training(corpus: str = "politifact"):
    import torch
    from scipy.stats import spearmanr
    trees, split_tag = build_upfd_dataset(corpus, fake_only=True)
    stats = fit_feature_stats(trees, split_tag)
    with open(STAGE5_OUT / "feature_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    train_d = _to_pyg(trees, split_tag, stats, "train")
    val_d = _to_pyg(trees, split_tag, stats, "val")
    test_d = _to_pyg(trees, split_tag, stats, "test")
    print(f"[stage5] trees: train={len(train_d)} val={len(val_d)} test={len(test_d)}")

    metrics = {}
    for name, model in _build_models().items():
        t0 = time.time()
        best_val = train_one(model, train_d, val_d)
        # test loss + rank correlation between predicted and true s(v)
        model.eval()
        with torch.no_grad():
            preds = np.concatenate([model(d.x, d.edge_index).numpy() for d in test_d])
            trues = np.concatenate([d.y.numpy() for d in test_d])
        mse = float(np.mean((preds - trues) ** 2))
        rho = float(spearmanr(preds, trues).correlation)
        torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
        metrics[name] = {"val_mse": best_val, "test_mse": mse,
                         "test_spearman": rho, "train_seconds": round(time.time()-t0, 1)}
        print(f"[stage5] {name}: val_mse={best_val:.4f} test_mse={mse:.4f} "
              f"spearman={rho:.3f} [{metrics[name]['train_seconds']}s]")

    with open(STAGE5_OUT / "gnn_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[stage5] saved models + metrics -> {STAGE5_OUT}")
    return metrics


if __name__ == "__main__":
    run_training()
