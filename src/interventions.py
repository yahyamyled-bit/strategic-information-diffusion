"""
Stage 6 -- the five intervention strategies.

Flagging semantics:
  * Reactive interventions (visibility reduction, fact-checking, edge rewiring)
    flag a target I->F the moment it becomes an active sharer ("flag-on-
    infection"), reducing its onward transmission from full rate to eta.
    (Preemptively flagging S hubs at eta>0 increases spread; see Stage 6
    artifact test.)
  * Node removal flags all targets -> F preemptively at t=1 with eta=0 (deplatform
    up front; equivalent whether the hub is S or I).
  * Early detection flags Susceptible targets -> F preemptively at t_detect with
    eta=0.50.

Targeting ranking is supplied by the runner (degree centrality for H1, since H2
found the GNN reduces to centrality). Effective F-transmission factors per
chapter Eq. 2.5: removal 0; visibility {0.50,0.70,0.90}; fact-check/rewiring 0.10
(default); early detection 0.50.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import config

COMMUNITIES_NPY = config.PROCESSED_DIR / "higgs_communities.npy"


# ==========================================================================
# Base (reactive / flag-on-infection)
# ==========================================================================
class Intervention:
    name = "base"

    def __init__(self, targets, tau_int: int = 1, eta_target: float = config.ETA_DEFAULT):
        self.targets = np.asarray(targets, dtype=np.int64)
        self.tau_int = int(tau_int)
        self.eta_target = float(eta_target)
        self._flagged = None          # which targets have been flagged so far
        self.A_override = None         # set by topology-changing interventions
        self.deg_override = None

    def on_init(self, sim, state, eta_eff, rng):
        self._flagged = np.zeros(self.targets.size, dtype=bool)

    def credibility_adjustment(self, sim, credibility, t):
        return credibility

    def apply(self, sim, state, eta_eff, t, rng):
        """Reactive: flag a target I->F once it becomes an active sharer."""
        newly = (state[self.targets] == config.I) & ~self._flagged
        if not newly.any():
            return False
        nodes = self.targets[newly]
        state[nodes] = config.F
        eta_eff[nodes] = self.eta_target
        self._flagged[newly] = True
        return True

    def affected_first_step(self, sim) -> int:
        return int(self.targets.size)


# ==========================================================================
# 1. Node removal (preemptive, eta = 0)
# ==========================================================================
class NodeRemoval(Intervention):
    name = "node_removal"

    def __init__(self, targets):
        super().__init__(targets, tau_int=1, eta_target=0.0)
        self._done = False

    def apply(self, sim, state, eta_eff, t, rng):
        if self._done:
            return False
        state[self.targets] = config.F      # deplatform up front (S or I -> F)
        eta_eff[self.targets] = 0.0
        self._done = True
        return True


# ==========================================================================
# 2. Visibility reduction (reactive, eta_vis)
# ==========================================================================
class VisibilityReduction(Intervention):
    name = "visibility_reduction"

    def __init__(self, targets, eta_vis: float = 0.70):
        super().__init__(targets, tau_int=1, eta_target=eta_vis)


# ==========================================================================
# 3. Fact-checking labels (reactive; +lambda on downstream S after latency)
# ==========================================================================
class FactChecking(Intervention):
    name = "fact_checking"

    def __init__(self, targets, lam: float = 0.25, latency: int = 7,
                 eta_target: float = None):
        # Accept an explicit eta_target (used by --eta robustness sweeps);
        # fall back to config.ETA_DEFAULT when caller doesn't override.
        if eta_target is None:
            eta_target = config.ETA_DEFAULT
        super().__init__(targets, tau_int=1, eta_target=eta_target)
        self.lam = float(lam)
        self.latency = int(latency)
        self._exposed = None

    def on_init(self, sim, state, eta_eff, rng):
        super().on_init(sim, state, eta_eff, rng)
        ind = np.zeros(sim.g.n, dtype=np.float64)
        ind[self.targets] = 1.0
        self._exposed = sim.g.csr.dot(ind) > 0.0   # S agents adjacent to labeled content

    def credibility_adjustment(self, sim, credibility, t):
        if self._exposed is None or t < self.tau_int + self.latency:
            return credibility
        cred = np.array(credibility, dtype=np.float64, copy=True)
        cred[self._exposed] += self.lam            # +lambda BEFORE the (1-c_v) factor
        return cred

    def affected_first_step(self, sim) -> int:
        return int(self.targets.size + (self._exposed.sum() if self._exposed is not None else 0))


# ==========================================================================
# 4. Edge rewiring (preemptive topology change; reactive eta flagging)
# ==========================================================================
class EdgeRewiring(Intervention):
    name = "edge_rewiring"

    def __init__(self, targets, phi: float = 0.10, eta_target: float = None):
        # Accept explicit eta_target for --eta robustness sweeps;
        # fall back to config.ETA_DEFAULT when caller doesn't override.
        if eta_target is None:
            eta_target = config.ETA_DEFAULT
        super().__init__(targets, tau_int=1, eta_target=eta_target)
        self.phi = float(phi)
        self._n_affected = 0

    def on_init(self, sim, state, eta_eff, rng):
        super().on_init(sim, state, eta_eff, rng)
        comm = load_or_compute_communities(sim.g)
        self.A_override, self.deg_override, self._n_affected = rewire(
            sim.g.csr, self.targets, comm, self.phi, rng)

    def affected_first_step(self, sim) -> int:
        # chapter sec 2.5 approximation: nodes whose immediate-neighbor set changes
        return int(self._n_affected)


# ==========================================================================
# 5. Early detection (preemptive on S nodes at t_detect, eta = 0.50)
# ==========================================================================
class EarlyDetection(Intervention):
    name = "early_detection"

    def __init__(self, targets, t_detect: int = 10, eta_target: float = 0.50):
        # eta_target overridable for the early-detection artifact test in
        # src/exp_early_eta_sweep.py. Default 0.50 reproduces the matched-
        # intensity main run.
        super().__init__(targets, tau_int=int(t_detect), eta_target=eta_target)
        self._done = False

    def apply(self, sim, state, eta_eff, t, rng):
        if self._done:
            return False
        tgt = self.targets[state[self.targets] == config.S]   # only those still S
        state[tgt] = config.F
        eta_eff[tgt] = self.eta_target
        self._done = True
        return True


# ==========================================================================
# Community detection + rewiring helpers
# ==========================================================================
def load_or_compute_communities(graph) -> np.ndarray:
    if COMMUNITIES_NPY.exists():
        return np.load(COMMUNITIES_NPY)
    import igraph as ig
    coo = graph.csr.tocoo()
    ut = coo.row < coo.col                  # upper triangle => unique undirected edges
    g = ig.Graph(n=graph.n, edges=list(zip(coo.row[ut].tolist(), coo.col[ut].tolist())),
                 directed=False)
    comm = np.array(g.community_multilevel().membership, dtype=np.int32)   # Louvain
    np.save(COMMUNITIES_NPY, comm)
    return comm


def rewire(csr: sp.csr_matrix, targets: np.ndarray, comm: np.ndarray,
           phi: float, rng) -> tuple:
    """Rewire phi of within-community edges incident on target nodes to random
    cross-community nodes. Returns (A_override, deg_override, n_affected)."""
    n = csr.shape[0]
    coo = csr.tocoo()
    ut = coo.row < coo.col
    eu, ev = coo.row[ut].copy(), coo.col[ut].copy()       # unique undirected edges
    target_set = np.zeros(n, dtype=bool)
    target_set[targets] = True

    incident = target_set[eu] | target_set[ev]
    same_comm = comm[eu] == comm[ev]
    cand = np.flatnonzero(incident & same_comm)
    n_rewire = int(phi * cand.size)
    affected = set()
    if n_rewire > 0:
        sel = rng.choice(cand, size=n_rewire, replace=False)
        for e in sel:
            a = eu[e] if target_set[eu[e]] else ev[e]     # the target endpoint
            old = ev[e] if a == eu[e] else eu[e]
            w = int(rng.integers(0, n))
            for _ in range(8):
                if comm[w] != comm[a] and w != a:
                    break
                w = int(rng.integers(0, n))
            eu[e], ev[e] = (a, w) if a < w else (w, a)    # a keeps the edge; other -> w
            affected.update((int(a), int(old), int(w)))

    data = np.ones(eu.size, dtype=np.int8)
    A = sp.coo_matrix((np.concatenate([data, data]),
                       (np.concatenate([eu, ev]), np.concatenate([ev, eu]))),
                      shape=(n, n)).tocsr()
    A.sum_duplicates(); A.data[:] = 1; A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
    return A, deg, len(affected)
