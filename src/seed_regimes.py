"""
Seed-selection regimes for the SIRF experiments.

Two regimes per chapter sec 2.4.1:
  * SEED_REGIME_RANDOM (primary): uniform random sample of k Higgs nodes.
    Implemented in sirf.draw_fixed_seeds.
  * SEED_REGIME_STRUCTURAL (secondary, H3): sample k seeds weighted by structural
    similarity to UPFD root nodes -- Higgs nodes whose (degree, neighbor-degree
    average) fall within one standard deviation of the UPFD-root means receive
    an elevated multiplicative weight.

The chapter (sec 2.4.1) defines a fallback: if UPFD/Higgs feature-space overlap
is too thin, the sensitivity analysis is dropped and reported as a limitation.
The overlap is checked before sampling and OverlapTooThinError is raised if it
fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config

# Multiplicative boost for Higgs nodes within 1 SD of the UPFD-root profile on
# BOTH the degree and neighbor-degree-average dimensions. The chapter says these
# nodes "receive elevated sampling probability"; we encode that as a boost factor.
STRUCTURAL_BOOST = 100.0

# Minimum number of Higgs nodes that must lie within 1 SD on both axes before we
# accept the structural regime; below this the fallback fires.
MIN_OVERLAP = 1000


class OverlapTooThinError(RuntimeError):
    """Raised if UPFD-root / Higgs feature overlap is too thin (chapter fallback)."""


# ==========================================================================
# UPFD root-node structural profile
# ==========================================================================
def upfd_root_profile(corpus: str = "politifact",
                      upfd_root: str = "data/upfd") -> dict:
    """Mean + SD of (degree, neighbor-degree-avg) of UPFD root nodes (node 0 in
    each tree). Uses all UPFD trees from the named corpus across train/val/test."""
    from torch_geometric.datasets import UPFD
    import networkx as nx

    degrees, ndas = [], []
    for split in ("train", "val", "test"):
        ds = UPFD(upfd_root, corpus, "profile", split)
        for data in ds:
            if data.num_nodes < 2:
                continue
            ei = data.edge_index.numpy()
            g = nx.Graph()
            g.add_nodes_from(range(data.num_nodes))
            g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
            root_deg = g.degree(0)
            if root_deg == 0:
                continue
            nbr_degs = [g.degree(u) for u in g.neighbors(0)]
            degrees.append(root_deg)
            ndas.append(float(np.mean(nbr_degs)))

    return {
        "n_roots": len(degrees),
        "mean_degree": float(np.mean(degrees)),
        "std_degree": float(np.std(degrees, ddof=1)),
        "mean_nda": float(np.mean(ndas)),
        "std_nda": float(np.std(ndas, ddof=1)),
        "median_degree": float(np.median(degrees)),
        "median_nda": float(np.median(ndas)),
    }


# ==========================================================================
# Structural-likelihood seed sampling on Higgs
# ==========================================================================
def draw_structural_seeds(graph, k: int = config.K_SEEDS,
                          master_seed: int = config.MASTER_SEED,
                          boost: float = STRUCTURAL_BOOST,
                          profile: dict | None = None) -> np.ndarray:
    """Sample k Higgs node indices weighted by structural similarity to UPFD roots.

    The score is a binary boost: a node receives `boost`x base weight if its
    (degree, neighbor-degree-avg) lies within 1 SD of the UPFD-root means on
    BOTH dimensions; otherwise the base weight (1.0). Sampling without replacement.

    Drawn ONCE per seed regime (held-fixed across all runs of the regime), keyed
    deterministically on the master seed -- the SEED_REGIME_STRUCTURAL counterpart
    to sirf.draw_fixed_seeds for the random regime.
    """
    if profile is None:
        profile = upfd_root_profile()

    n = graph.n
    degree = graph.degree.astype(np.float64)
    nda = np.zeros(n, dtype=np.float64)
    safe = degree > 0
    np.divide(graph.csr.dot(degree), degree, out=nda, where=safe)

    deg_in = np.abs(degree - profile["mean_degree"]) <= profile["std_degree"]
    nda_in = np.abs(nda - profile["mean_nda"]) <= profile["std_nda"]
    in_band = deg_in & nda_in
    n_overlap = int(in_band.sum())

    if n_overlap < MIN_OVERLAP:
        raise OverlapTooThinError(
            f"Only {n_overlap} Higgs nodes lie within 1 SD of the UPFD-root profile "
            f"on both axes (threshold {MIN_OVERLAP}). Per chapter sec 2.4.1, the H3 "
            f"sensitivity analysis is dropped and the limitation is reported.")

    weights = np.ones(n, dtype=np.float64)
    weights[in_band] *= boost
    weights /= weights.sum()

    rng = np.random.default_rng(
        np.random.SeedSequence(entropy=[master_seed, 8003,
                                        config.SEED_REGIME_STRUCTURAL]))
    seeds = rng.choice(n, size=k, replace=False, p=weights).astype(np.int64)
    return seeds


def describe_overlap(graph, profile: dict | None = None) -> dict:
    """Diagnostic: how many Higgs nodes lie in the UPFD-root 1-SD band?"""
    if profile is None:
        profile = upfd_root_profile()
    n = graph.n
    degree = graph.degree.astype(np.float64)
    nda = np.zeros(n, dtype=np.float64)
    safe = degree > 0
    np.divide(graph.csr.dot(degree), degree, out=nda, where=safe)
    deg_in = np.abs(degree - profile["mean_degree"]) <= profile["std_degree"]
    nda_in = np.abs(nda - profile["mean_nda"]) <= profile["std_nda"]
    return {
        "profile": profile,
        "higgs_n": int(n),
        "in_band_degree_only": int(deg_in.sum()),
        "in_band_nda_only": int(nda_in.sum()),
        "in_band_both": int((deg_in & nda_in).sum()),
        "in_band_both_pct": float((deg_in & nda_in).mean() * 100),
    }
