"""
Stage 1 -- Network setup.

Loads the Higgs Twitter follower edge list (SNAP), builds the undirected graph,
computes the per-node `influence` attribute (min-max normalized log-degree), and
saves fast reusable artifacts for the simulation:

  * data/processed/higgs_adjacency.npz    -- symmetric binary CSR adjacency
  * data/processed/higgs_node_arrays.npz  -- degree, influence, original node ids
  * outputs/stage1/graph_stats.json       -- verification stats

The CSR adjacency is the canonical substrate for the vectorized SIRF loop
(Stage 2). A NetworkX view is available on demand via `build_networkx()` for the
stages that need NetworkX-native operations (Louvain, centralities).

Per the methodology: undirected follower graph, credulity/threshold/state are
per-run (assigned in Stage 2), influence is a static structural attribute.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config
from .utils import write_json


def load_directed_edges() -> np.ndarray:
    """Parse the gzipped follower edge list into an (E, 2) int64 array."""
    if not config.HIGGS_SOCIAL_GZ.exists():
        raise FileNotFoundError(
            f"Missing {config.HIGGS_SOCIAL_GZ}. Download from {config.HIGGS_SOCIAL_URL}"
        )
    df = pd.read_csv(
        config.HIGGS_SOCIAL_GZ,
        sep=" ",
        header=None,
        names=["u", "v"],
        dtype=np.int64,
        engine="c",
    )
    return df.to_numpy()


def build_adjacency(edges: np.ndarray):
    """Build a symmetric, binary, self-loop-free CSR adjacency.

    Returns (csr, degree, original_ids) where row/col index i corresponds to
    original node id original_ids[i]. Node ids are remapped to a contiguous
    0..N-1 range; for the Higgs graph this is effectively id-1 but the remap is
    explicit to handle gaps in node-id space.
    """
    u = edges[:, 0]
    v = edges[:, 1]

    original_ids = np.unique(np.concatenate([u, v]))
    n = original_ids.size

    # Map original id -> contiguous index via searchsorted (original_ids sorted).
    ui = np.searchsorted(original_ids, u).astype(np.int32)
    vi = np.searchsorted(original_ids, v).astype(np.int32)

    # Drop self-loops if any.
    keep = ui != vi
    ui, vi = ui[keep], vi[keep]

    # Symmetrize: add both directions, then dedupe via CSR.
    rows = np.concatenate([ui, vi])
    cols = np.concatenate([vi, ui])
    data = np.ones(rows.size, dtype=np.int8)

    coo = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    csr = coo.tocsr()
    csr.sum_duplicates()
    csr.data[:] = 1  # binarize (mutual follows collapse to a single edge)
    csr.eliminate_zeros()

    degree = np.asarray(csr.sum(axis=1)).ravel().astype(np.int64)
    return csr, degree, original_ids


def compute_influence(degree: np.ndarray) -> np.ndarray:
    """influence_v = (log d_v - log d_min) / (log d_max - log d_min), in [0, 1].

    Guard clauses from chapter sec 2.2.3: isolated nodes (d_v = 0) are excluded
    from the normalization and assigned 0; if d_min == d_max (degenerate degree
    distribution) every non-isolated node is assigned 0.5. Neither fires on the
    Higgs graph (0 isolated nodes, d_min=1 != d_max), so the output is unchanged
    there; the guards are present so the function matches the spec on any graph.
    """
    d = degree.astype(np.float64)
    non_isolated = d > 0
    influence = np.zeros_like(d)
    if not non_isolated.any():
        return influence
    d_min = max(float(d[non_isolated].min()), 1.0)
    d_max = float(d.max())
    log_min, log_max = np.log(d_min), np.log(d_max)
    denom = log_max - log_min
    if denom <= 0.0:
        influence[non_isolated] = 0.5
        return influence
    with np.errstate(divide="ignore"):
        log_d = np.log(np.clip(d, 1.0, None))
    norm = (log_d - log_min) / denom
    influence[non_isolated] = np.clip(norm[non_isolated], 0.0, 1.0)
    return influence


def build_networkx(csr: sp.csr_matrix | None = None):
    """Construct the undirected NetworkX graph on demand (memory-heavy).

    Used by stages that require NetworkX-native ops. Not needed for the Stage 2
    hot loop, which runs on the CSR adjacency.
    """
    import networkx as nx

    if csr is None:
        csr = sp.load_npz(config.GRAPH_NPZ)
    return nx.from_scipy_sparse_array(csr, edge_attribute=None)


def run(verify_with_networkx: bool = False) -> dict:
    t0 = time.time()
    print(f"[stage1] loading edges from {config.HIGGS_SOCIAL_GZ.name} ...")
    edges = load_directed_edges()
    n_directed = edges.shape[0]
    print(f"[stage1] {n_directed:,} directed edges parsed in {time.time()-t0:.1f}s")

    csr, degree, original_ids = build_adjacency(edges)
    n_nodes = csr.shape[0]
    n_undirected = int(csr.nnz // 2)
    influence = compute_influence(degree)

    # Persist artifacts.
    sp.save_npz(config.GRAPH_NPZ, csr)
    np.savez_compressed(
        config.NODE_ARRAYS_NPZ,
        degree=degree,
        influence=influence,
        original_ids=original_ids,
    )

    stats = {
        "n_nodes": n_nodes,
        "n_directed_edges_raw": n_directed,
        "n_undirected_edges_unique": n_undirected,
        "expected_nodes": config.EXPECTED_NODES,
        "expected_directed_edges": config.EXPECTED_DIRECTED_EDGES,
        "nodes_match": n_nodes == config.EXPECTED_NODES,
        "directed_edges_match": n_directed == config.EXPECTED_DIRECTED_EDGES,
        "degree_min": int(degree.min()),
        "degree_max": int(degree.max()),
        "degree_mean": float(degree.mean()),
        "degree_median": float(np.median(degree)),
        "n_isolated_nodes": int((degree == 0).sum()),
        "influence_min": float(influence.min()),
        "influence_max": float(influence.max()),
        "influence_mean": float(influence.mean()),
        "id_min": int(original_ids.min()),
        "id_max": int(original_ids.max()),
        "ids_contiguous": bool(original_ids.size == (original_ids.max() - original_ids.min() + 1)),
        "build_seconds": round(time.time() - t0, 1),
    }

    if verify_with_networkx:
        g = build_networkx(csr)
        stats["networkx_nodes"] = g.number_of_nodes()
        stats["networkx_edges"] = g.number_of_edges()

    write_json(config.GRAPH_STATS_JSON, stats)

    print(f"[stage1] nodes={n_nodes:,} (expected {config.EXPECTED_NODES:,}, "
          f"match={stats['nodes_match']})")
    print(f"[stage1] directed edges={n_directed:,} (expected "
          f"{config.EXPECTED_DIRECTED_EDGES:,}, match={stats['directed_edges_match']})")
    print(f"[stage1] unique undirected edges={n_undirected:,}")
    print(f"[stage1] degree: min={stats['degree_min']}, max={stats['degree_max']:,}, "
          f"mean={stats['degree_mean']:.1f}, median={stats['degree_median']:.0f}")
    print(f"[stage1] influence in [{stats['influence_min']:.3f}, "
          f"{stats['influence_max']:.3f}], mean={stats['influence_mean']:.3f}")
    print(f"[stage1] saved adjacency -> {config.GRAPH_NPZ.name}, "
          f"arrays -> {config.NODE_ARRAYS_NPZ.name}, stats -> {config.GRAPH_STATS_JSON.name}")
    print(f"[stage1] done in {stats['build_seconds']}s")
    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 1: build Higgs follower graph.")
    p.add_argument("--verify-networkx", action="store_true",
                   help="Also build the NetworkX graph to cross-check counts (memory-heavy).")
    args = p.parse_args()
    run(verify_with_networkx=args.verify_networkx)
