"""
Cascade-structure metrics: depth, breadth, structural virality.

Shared by Stage 3 (operating-point selection by structural fit to PolitiFact)
and Stage 4 (validation against held-out GossipCop + Twitter15/16). The three
metrics are the chapter's (sec 2.6):

  * depth              -- longest root-to-leaf path in the cascade tree.
  * breadth            -- maximum number of nodes at any single depth level.
  * structural virality-- mean pairwise distance between nodes in the cascade
                          (Goel et al. 2016), computed exactly on the tree via
                          the edge-contribution identity:
                              sum_pairwise_dist = sum_e s_e * (N - s_e)
                          where s_e = #nodes in the subtree below edge e, so
                          virality = sum_pairwise_dist / C(N, 2).

The simulation produces a network diffusion, not a tree, so we reconstruct a
cascade tree from infection generations. Reconstruction convention: each
infected non-seed node's parent is the prior-infected neighbor with the highest
influence, the largest single contributor to its peer-influence term (ties
broken by recency). This mirrors real cascades, where most reshares attach
directly to a high-reach source (broadcast shape) rather than forming long
chains. With k seeds the result is a forest of <= k trees; for a single per-run
number, depth/breadth are taken over the whole forest and structural virality is
the size-weighted mean over the seed-trees.
"""

from __future__ import annotations

import ast
import glob
import os

import numpy as np
import scipy.sparse as sp


# ==========================================================================
# Core: metrics from a parent forest
# ==========================================================================
def _forest_metrics(parent: np.ndarray, node_idx: np.ndarray,
                    gen_order: np.ndarray) -> dict:
    """Depth, breadth, structural virality for a forest given parent pointers.

    parent[v] = parent node index, or -1 for a root / node not in the cascade.
    node_idx  = the nodes that belong to the cascade (roots + infected).
    gen_order = node_idx sorted so a parent always precedes its children
                (e.g., by infection generation), so depths fill in one pass.
    """
    n_total = parent.shape[0]
    depth = np.full(n_total, -1, dtype=np.int64)
    root_of = np.full(n_total, -1, dtype=np.int64)

    for v in gen_order:
        p = parent[v]
        if p < 0:
            depth[v] = 0
            root_of[v] = v
        else:
            depth[v] = depth[p] + 1
            root_of[v] = root_of[p]

    d = depth[node_idx]
    cascade_depth = int(d.max()) if d.size else 0
    # breadth: max nodes at any single depth level (across the forest). NOTE this
    # is an absolute count and therefore scales with cascade size, so it is not
    # directly comparable between tiny real trees and huge simulated cascades
    # (same scale issue as raw cascade size). We also report breadth_frac =
    # breadth / n_nodes, a scale-robust shape measure, for cross-scale KS tests.
    breadth = int(np.bincount(d).max()) if d.size else 0
    breadth_frac = float(breadth / node_idx.size) if node_idx.size else 0.0

    # structural virality per tree via the edge-contribution identity
    # subtree size below each node = 1 + sum of children's subtree sizes;
    # process nodes in reverse generation order so children precede parents.
    subtree = np.ones(n_total, dtype=np.int64)
    for v in gen_order[::-1]:
        p = parent[v]
        if p >= 0:
            subtree[p] += subtree[v]

    # tree size per root
    roots = node_idx[parent[node_idx] < 0]
    tree_size = {int(r): int(subtree[r]) for r in roots}

    # sum over non-root edges of s_e*(N_tree - s_e), grouped by tree
    sum_pair = {int(r): 0 for r in roots}
    for v in node_idx:
        if parent[v] < 0:
            continue
        r = int(root_of[v])
        s = int(subtree[v])
        sum_pair[r] += s * (tree_size[r] - s)

    # size-weighted mean structural virality across trees with >= 2 nodes
    num, den = 0.0, 0.0
    for r, N in tree_size.items():
        if N >= 2:
            vir = sum_pair[r] / (N * (N - 1) / 2.0)
            num += vir * N
            den += N
    structural_virality = float(num / den) if den > 0 else 0.0

    return {"depth": cascade_depth, "breadth": breadth,
            "breadth_frac": breadth_frac,
            "structural_virality": structural_virality,
            "n_nodes": int(node_idx.size), "n_trees": int(roots.size)}


# ==========================================================================
# Simulated cascade (Higgs) -> structure
# ==========================================================================
def reconstruct_parents(infection_step: np.ndarray, csr: sp.csr_matrix,
                        influence: np.ndarray) -> np.ndarray:
    """Parent = the prior-infected neighbor that most plausibly caused the share.

    Under the utility model a node's peer-influence term is the influence-weighted
    sum over its infected neighbors, so the single largest contributor is the
    highest-influence prior-infected neighbor. That node is assigned as the parent
    (ties broken by recency). This mirrors real cascades, where most reshares
    attach directly to a high-reach source (broadcast shape) rather than forming
    long chains.

    Vectorized over the time-ordered infected edges. Returns parent[v] (-1 for
    seeds and never-infected nodes).
    """
    n = infection_step.shape[0]
    parent = np.full(n, -1, dtype=np.int64)
    coo = csr.tocoo()
    u, v = coo.row, coo.col
    su, sv = infection_step[u], infection_step[v]
    # candidate parent-edges u->v: both infected, u strictly earlier than v,
    # and v is not a seed (sv > 0).
    cand = (su >= 0) & (sv > 0) & (su < sv)
    u, v, su = u[cand], v[cand], su[cand]
    if u.size == 0:
        return parent
    # per child v, pick max influence among prior-infected neighbors, tie max su.
    order = np.lexsort((su, influence[u], v))   # primary v, then influence, then su
    v_sorted = v[order]
    last = np.ones(v_sorted.size, dtype=bool)
    last[:-1] = v_sorted[1:] != v_sorted[:-1]
    sel = order[last]
    parent[v[sel]] = u[sel]
    return parent


def simulated_cascade_structure(infection_step: np.ndarray, csr: sp.csr_matrix,
                                influence: np.ndarray) -> dict:
    infected = np.flatnonzero(infection_step >= 0)
    if infected.size <= 1:
        return {"depth": 0, "breadth": int(infected.size),
                "structural_virality": 0.0, "n_nodes": int(infected.size),
                "n_trees": int(infected.size)}
    parent = reconstruct_parents(infection_step, csr, influence)
    gen_order = infected[np.argsort(infection_step[infected], kind="stable")]
    return _forest_metrics(parent, infected, gen_order)


# ==========================================================================
# Real trees -> structure
# ==========================================================================
def _metrics_from_edge_list(edges: list[tuple[int, int]], n_nodes: int,
                            root: int = 0) -> dict:
    """Single rooted tree given (parent, child) edges with integer node ids.

    Robust to noisy real-tree files: builds child adjacency, then BFS from the
    root so the traversal order is a valid parent-before-child order and any
    nodes not reachable from the root (data fragments) are dropped.
    """
    from collections import deque
    parent = np.full(n_nodes, -1, dtype=np.int64)
    children: dict[int, list[int]] = {}
    for p, c in edges:
        if 0 <= p < n_nodes and 0 <= c < n_nodes and c != root and parent[c] < 0:
            parent[c] = p
            children.setdefault(p, []).append(c)
    order = [root]
    seen = np.zeros(n_nodes, dtype=bool)
    seen[root] = True
    dq = deque([root])
    while dq:
        x = dq.popleft()
        for ch in children.get(x, ()):
            if not seen[ch]:
                seen[ch] = True
                order.append(ch)
                dq.append(ch)
    node_idx = np.array(order, dtype=np.int64)
    # nodes reachable from root keep their parent; drop any others implicitly.
    parent_clean = np.full(n_nodes, -1, dtype=np.int64)
    parent_clean[node_idx] = parent[node_idx]
    parent_clean[root] = -1
    return _forest_metrics(parent_clean, node_idx, node_idx)


def upfd_tree_structure(edge_index: np.ndarray, num_nodes: int) -> dict:
    """UPFD PyG tree: edge_index (2, E), rooted at node 0."""
    ei = np.asarray(edge_index)
    # UPFD edges are directed root->...; orient parent->child as (src, dst).
    edges = [(int(s), int(d)) for s, d in zip(ei[0], ei[1])]
    return _metrics_from_edge_list(edges, num_nodes, root=0)


def parse_twitter_tree(path: str) -> dict:
    """Twitter15/16 tree file -> structure. Nodes keyed by tweet id; ROOT virtual."""
    edges_raw = []
    nodes = {}
    def nid(key):
        if key not in nodes:
            nodes[key] = len(nodes)
        return nodes[key]
    root = nid("ROOT")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "->" not in line:
                continue
            left, right = line.split("->")
            try:
                p = ast.literal_eval(left.strip())
                c = ast.literal_eval(right.strip())
            except (ValueError, SyntaxError):
                continue
            pkey = p[1] if p[0] != "ROOT" else "ROOT"   # tweet id, or ROOT
            ckey = c[1]
            edges_raw.append((nid(pkey), nid(ckey)))
    n = len(nodes)
    return _metrics_from_edge_list(edges_raw, n, root=root)


def corpus_structure_distribution(loader_iter) -> dict:
    """Aggregate per-tree metrics over a corpus into arrays for KS testing."""
    depth, breadth, bfrac, vir = [], [], [], []
    for m in loader_iter:
        if m["n_nodes"] >= 2:
            depth.append(m["depth"]); breadth.append(m["breadth"])
            bfrac.append(m["breadth_frac"]); vir.append(m["structural_virality"])
    return {"depth": np.array(depth), "breadth": np.array(breadth),
            "breadth_frac": np.array(bfrac),
            "structural_virality": np.array(vir), "n_trees": len(depth)}


def twitter_corpus_iter(which: str = "twitter15", base: str = "data/twitter1516"):
    for path in glob.glob(os.path.join(base, which, "tree", "*.txt")):
        yield parse_twitter_tree(path)
