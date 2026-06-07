"""
MLP baseline for the H2 / s(v) prediction task.

Trains a 2-layer MLP (no message passing) on the same 5 node-level features the
GNNs use, on the same UPFD-PolitiFact-fake training corpus. Tests whether graph
convolution adds anything beyond a plain MLP on the same features.

Also reports Spearman of raw degree and betweenness vs true s(v) on the test
set as a no-model baseline.

Outputs (outputs/stage5/): mlp_baseline.json
"""

from __future__ import annotations

import json
import time
import numpy as np

from .stage5_gnn import (build_upfd_dataset, fit_feature_stats, apply_stats,
                         STAGE5_OUT)
from . import config


def _to_arrays(trees, split_tag, stats, which):
    """Concatenate node features and labels for a given split, after standardisation."""
    Xs, ys = [], []
    for (X, y, _), s in zip(trees, split_tag, strict=False):
        if s != which:
            continue
        Xs.append(apply_stats(X, stats))
        ys.append(y)
    return np.vstack(Xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def _to_arrays_with_raw_centrality(trees, split_tag, which):
    """Get raw (un-standardised) feature values for the named split, so we can
    correlate raw degree / raw betweenness against true s(v) on the same nodes."""
    Xs, ys = [], []
    for (X, y, _), s in zip(trees, split_tag, strict=False):
        if s != which:
            continue
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs).astype(np.float64), np.concatenate(ys).astype(np.float64)


def train_mlp(corpus: str = "politifact", hidden: int = 64,
              epochs: int = 300, lr: float = 1e-2, patience: int = 30,
              seed: int = config.MASTER_SEED):
    import torch
    import torch.nn as nn
    from scipy.stats import spearmanr

    t0 = time.time()
    print("[mlp] building UPFD dataset (fake, politifact)...")
    trees, split_tag = build_upfd_dataset(corpus, fake_only=True)
    stats = fit_feature_stats(trees, split_tag)

    Xtr, ytr = _to_arrays(trees, split_tag, stats, "train")
    Xva, yva = _to_arrays(trees, split_tag, stats, "val")
    Xte, yte = _to_arrays(trees, split_tag, stats, "test")
    print(f"[mlp] train={Xtr.shape}  val={Xva.shape}  test={Xte.shape}")

    Xte_raw, yte_raw = _to_arrays_with_raw_centrality(trees, split_tag, "test")
    raw_deg = Xte_raw[:, 0]
    raw_btw = Xte_raw[:, 1]
    rho_deg = float(spearmanr(raw_deg, yte_raw).correlation)
    rho_btw = float(spearmanr(raw_btw, yte_raw).correlation)
    print(f"[mlp] baseline (no model) test Spearman: degree={rho_deg:.4f}  betweenness={rho_btw:.4f}")

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(5, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva); yva_t = torch.tensor(yva)
    Xte_t = torch.tensor(Xte); yte_t = torch.tensor(yte)

    best_val, best_state, wait = float("inf"), None, 0
    batch_size = 1024
    n_train = Xtr.shape[0]
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr_t[idx]; yb = ytr_t[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xva_t).squeeze(-1), yva_t))
        if vl < best_val - 1e-5:
            best_val, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(Xte_t).squeeze(-1).numpy()
    test_mse = float(np.mean((preds - yte) ** 2))
    rho_mlp = float(spearmanr(preds, yte).correlation)
    print(f"[mlp] MLP test MSE={test_mse:.6f}  Spearman={rho_mlp:.4f}")

    out = {
        "task": "MLP baseline for s(v) prediction (no message passing)",
        "corpus": corpus,
        "fake_only": True,
        "features": ["degree", "betweenness", "eigenvector",
                     "neighbor_degree_avg", "depth_from_source"],
        "n_train_nodes": int(Xtr.shape[0]),
        "n_val_nodes": int(Xva.shape[0]),
        "n_test_nodes": int(Xte.shape[0]),
        "hidden": hidden,
        "best_val_mse": float(best_val),
        "test_mse": test_mse,
        "mlp_test_spearman": rho_mlp,
        "raw_baseline_test_spearman": {
            "degree": rho_deg,
            "betweenness": rho_btw,
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(STAGE5_OUT / "mlp_baseline.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[mlp] saved {STAGE5_OUT / 'mlp_baseline.json'}")
    return out


if __name__ == "__main__":
    train_mlp()
