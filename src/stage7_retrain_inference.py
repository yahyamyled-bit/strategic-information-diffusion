"""
Stage 7 -- Retrain the GNNs on the Higgs inference graph.

Trains GCN, GraphSAGE, GAT (and an MLP baseline) to predict the corrected label
s_higgs (realized-diffusion cascade contribution, built by
src/stage7_inference_label.py) directly on the Higgs follower graph, using the
SAME five structural features and the SAME architectures as Stage 5 -- only the
label and the training substrate change. This isolates the H2 question (does
message passing beat centrality?) from the degenerate UPFD-tree label that made
the original answer negative by construction.

Reuses from Stage 5 (unchanged):
  * _build_models()  -- src/stage5_gnn.py (GCN / GraphSAGE / GAT, 2 conv + head)
  * apply_stats()    -- src/stage5_gnn.py (standardize with given mean/std)
  * FEATURE_NAMES    -- src/stage5_gnn.py (feature order)
  * the MLP architecture of src/train_mlp_baseline.py (Linear 5->64->64->1),
    trained here on the Higgs node features / s_higgs label / same masks as the
    GNNs (no message passing, no edge_index) for an apples-to-apples baseline.
  * features         -- outputs/stage5/higgs_features.npz

Split: node-level 70/15/15 train/val/test, rng = default_rng(MASTER_SEED).
Standardization stats are fit on the TRAIN mask only (new stats, NOT the old UPFD
feature_stats.json) -- no leakage.

Determinism: torch.manual_seed(config.MASTER_SEED) before building any model.

Outputs (outputs/stage7/):
  models/{GCN,GraphSAGE,GAT}.pt
  gnn_metrics_inference.json   -- per model test_mse, r2, test_spearman,
                                  precision_at_500; + MLP; + degree/pagerank
                                  baseline Spearman on the test mask
  higgs_gnn_scores_inference.npz -- full-node predictions, keys gcn/graphsage/gat
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.stats import spearmanr

from . import config
from .sirf import GraphData
from .stage5_gnn import FEATURE_NAMES, apply_stats, _build_models

STAGE7_OUT = config.OUTPUTS_DIR / "stage7"
MODELS_DIR = STAGE7_OUT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NPY = STAGE7_OUT / "s_higgs_label.npy"
FEATS_NPZ = config.OUTPUTS_DIR / "stage5" / "higgs_features.npz"
METRICS_JSON = STAGE7_OUT / "gnn_metrics_inference.json"
SCORES_NPZ = STAGE7_OUT / "higgs_gnn_scores_inference.npz"

# score keys read by the Stage 6 experiment
SCORE_KEY = {"GCN": "gcn", "GraphSAGE": "graphsage", "GAT": "gat"}

MAX_EPOCHS = 200
PATIENCE = 25
LR = 1e-2
WEIGHT_DECAY = 5e-4
EPOCH_BUDGET_S = 30.0   # slow-epoch warning threshold (warning only, no cap)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data():
    feats = np.load(FEATS_NPZ)
    X = np.column_stack([feats[f] for f in FEATURE_NAMES]).astype(np.float64)
    y = np.load(LABEL_NPY).astype(np.float64)
    degree = feats["degree"].astype(np.float64)
    pagerank = feats["pagerank"].astype(np.float64)
    assert X.shape[0] == y.shape[0], (X.shape, y.shape)
    return X, y, degree, pagerank


def make_masks(n: int):
    rng = np.random.default_rng(config.MASTER_SEED)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    tr = np.zeros(n, dtype=bool); va = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
    tr[perm[:n_tr]] = True
    va[perm[n_tr:n_tr + n_va]] = True
    te[perm[n_tr + n_va:]] = True
    return tr, va, te


def fit_train_stats(X: np.ndarray, train_mask: np.ndarray) -> dict:
    Xtr = X[train_mask]
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist(), "features": FEATURE_NAMES}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def precision_at_k(pred: np.ndarray, true: np.ndarray, k: int = 500) -> float:
    """Global precision@k: |top-k pred intersect top-k true| / k over ALL nodes."""
    top_pred = set(np.argpartition(pred, -k)[-k:].tolist())
    top_true = set(np.argpartition(true, -k)[-k:].tolist())
    return len(top_pred & top_true) / float(k)


def r2_score(pred_test: np.ndarray, y_test: np.ndarray) -> float:
    var = float(np.var(y_test))
    mse = float(np.mean((pred_test - y_test) ** 2))
    return 1.0 - mse / var if var > 0 else float("nan")


# --------------------------------------------------------------------------
# GNN full-graph training
# --------------------------------------------------------------------------
def train_gnn(name, model, x, edge_index, y, train_mask, val_mask):
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()
    best_val, best_state, wait = float("inf"), None, 0
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        te0 = time.time()
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss = loss_fn(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vout = model(x, edge_index)
            vl = float(loss_fn(vout[val_mask], y[val_mask]))
        ep_s = time.time() - te0
        if ep < 5 or (ep + 1) % 5 == 0:
            print(f"[stage7:{name}] epoch {ep + 1:>3} "
                  f"train_mse={float(loss):.3e} val_mse={vl:.3e} "
                  f"[{ep_s:.1f}s/ep, {time.time() - t0:.0f}s]")
            if ep == 0 and ep_s > EPOCH_BUDGET_S:
                print(f"[stage7:{name}] WARNING epoch {ep_s:.1f}s > {EPOCH_BUDGET_S}s "
                      f"budget -- full-graph training is slow on this CPU; "
                      f"consider a mini-batch NeighborLoader fallback.")
        if vl < best_val - 1e-9:
            best_val, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[stage7:{name}] early stop at epoch {ep + 1} "
                      f"(best val_mse={best_val:.3e})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


# --------------------------------------------------------------------------
# MLP baseline (no message passing)  -- arch mirrors train_mlp_baseline.py
# --------------------------------------------------------------------------
def train_mlp(x, y, train_mask, val_mask, hidden=64):
    import torch
    import torch.nn as nn
    torch.manual_seed(config.MASTER_SEED)
    model = nn.Sequential(
        nn.Linear(5, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    xtr, ytr = x[train_mask], y[train_mask]
    xva, yva = x[val_mask], y[val_mask]
    n_train = int(train_mask.sum())
    batch = 1024
    rng = np.random.default_rng(config.MASTER_SEED)
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = rng.permutation(n_train)
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]).squeeze(-1), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(xva).squeeze(-1), yva))
        if vl < best_val - 1e-9:
            best_val, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        full = model(x).squeeze(-1).cpu().numpy()
    return full, best_val


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    import torch

    X, y, degree, pagerank = load_data()
    n = X.shape[0]
    print(f"[stage7] n={n}  features={FEATURE_NAMES}")

    train_mask, val_mask, test_mask = make_masks(n)
    print(f"[stage7] split: train={int(train_mask.sum())} "
          f"val={int(val_mask.sum())} test={int(test_mask.sum())}")

    stats = fit_train_stats(X, train_mask)
    Xs = apply_stats(X, stats).astype(np.float32)

    g = GraphData.load()
    coo = g.csr.tocoo()
    edge_index_np = np.vstack([coo.row, coo.col])
    print(f"[stage7] edges (directed, both ways) = {edge_index_np.shape[1]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[stage7] device = cuda ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print("[stage7] device = cpu (CUDA not available)", flush=True)
    x_t = torch.tensor(Xs, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)
    tr_t = torch.tensor(train_mask).to(device); va_t = torch.tensor(val_mask).to(device)

    y_test = y[test_mask]
    var_test = float(np.var(y_test))

    metrics = {
        "label": "s_higgs (realized-diffusion cascade contribution)",
        "n_nodes": int(n),
        "split": {"train": int(train_mask.sum()), "val": int(val_mask.sum()),
                  "test": int(test_mask.sum())},
        "y_test_var": var_test,
        "models": {},
    }
    scores = {}

    # --- GNNs ---
    torch.manual_seed(config.MASTER_SEED)   # before building models (determinism)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.MASTER_SEED)
    models = _build_models()
    for name in ("GCN", "GraphSAGE", "GAT"):
        model = models[name].to(device)
        t0 = time.time()
        best_val = train_gnn(name, model, x_t, edge_index, y_t, tr_t, va_t)
        model.eval()
        with torch.no_grad():
            pred_all = model(x_t, edge_index).cpu().numpy()
        pred_test = pred_all[test_mask]
        m = {
            "val_mse": best_val,
            "test_mse": float(np.mean((pred_test - y_test) ** 2)),
            "r2": r2_score(pred_test, y_test),
            "test_spearman": float(spearmanr(pred_test, y_test).correlation),
            "precision_at_500": precision_at_k(pred_all, y, 500),
            "train_seconds": round(time.time() - t0, 1),
        }
        metrics["models"][name] = m
        scores[SCORE_KEY[name]] = pred_all.astype(np.float32)
        torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
        print(f"[stage7] {name}: test_mse={m['test_mse']:.3e} r2={m['r2']:.3f} "
              f"spearman={m['test_spearman']:.3f} P@500={m['precision_at_500']:.3f} "
              f"[{m['train_seconds']}s]", flush=True)
        # incremental save so a later model crashing cannot erase completed work
        with open(METRICS_JSON, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        np.savez_compressed(SCORES_NPZ, **scores)
        del model, pred_all
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- MLP baseline (no edge_index) ---
    t0 = time.time()
    mlp_pred_all, mlp_val = train_mlp(x_t, y_t, tr_t, va_t)
    mlp_pred_test = mlp_pred_all[test_mask]
    metrics["models"]["MLP"] = {
        "val_mse": float(mlp_val),
        "test_mse": float(np.mean((mlp_pred_test - y_test) ** 2)),
        "r2": r2_score(mlp_pred_test, y_test),
        "test_spearman": float(spearmanr(mlp_pred_test, y_test).correlation),
        "precision_at_500": precision_at_k(mlp_pred_all, y, 500),
        "train_seconds": round(time.time() - t0, 1),
    }
    mm = metrics["models"]["MLP"]
    print(f"[stage7] MLP: test_mse={mm['test_mse']:.3e} r2={mm['r2']:.3f} "
          f"spearman={mm['test_spearman']:.3f} P@500={mm['precision_at_500']:.3f}")

    # --- Heuristic baselines on the test mask (do-nothing comparators) ---
    metrics["baselines_test_spearman"] = {
        "degree": float(spearmanr(degree[test_mask], y_test).correlation),
        "pagerank": float(spearmanr(pagerank[test_mask], y_test).correlation),
    }
    metrics["baselines_precision_at_500"] = {
        "degree": precision_at_k(degree, y, 500),
        "pagerank": precision_at_k(pagerank, y, 500),
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    np.savez_compressed(SCORES_NPZ, **scores)
    print(f"[stage7] saved {METRICS_JSON}")
    print(f"[stage7] saved {SCORES_NPZ}  keys={list(scores.keys())}")

    # --- H2 read-off ---
    bl = metrics["baselines_test_spearman"]
    print("=" * 66)
    print("  H2 read-off  (does any GNN beat degree AND pagerank?)")
    print(f"    baseline Spearman: degree={bl['degree']:+.3f} pagerank={bl['pagerank']:+.3f}")
    for name in ("GCN", "GraphSAGE", "GAT", "MLP"):
        m = metrics["models"][name]
        beats = (m["test_spearman"] > bl["degree"]) and (m["test_spearman"] > bl["pagerank"])
        print(f"    {name:<10} spearman={m['test_spearman']:+.3f} "
              f"P@500={m['precision_at_500']:.3f}  beats both centralities: {beats}")
    print("=" * 66)
    return metrics


if __name__ == "__main__":
    main()
