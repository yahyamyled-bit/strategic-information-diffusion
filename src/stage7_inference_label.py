"""
Stage 7 -- GNN training label on the Higgs inference graph.

The original GNN label s(v) = |descendants| / (N-1) was computed on UPFD
propagation *trees*, where subtree size is nearly identical to centrality
(rho ~ 0.996 degree / 0.999 betweenness). On a tree the label is degenerate, so
the GNN had nothing to learn that the centrality heuristics did not already
encode -- H2 was negative by construction.

This script redefines the label as realized-diffusion *cascade contribution*
measured on the Higgs inference graph's own SIRF cascades:

  build_label()        -- average per-node cascade contribution over 100 baseline
                          SIRF runs (calibrated operating point, same held-fixed
                          credulity / seed set as Stage 4).
  decorrelation_gate() -- report rho(s_higgs, degree) and rho(s_higgs,
                          betweenness); the label is only useful if these fall
                          well below the ~0.999 of the tree label.

Determinism: every per-run cascade is deterministic in (condition_id=0,
run_index), credulity and seeds are fixed draws, so re-running reproduces
s_higgs bit-for-bit. No model init here, so no torch seed is needed.

Per-node label for a single run (mirrors src/structure.py._forest_metrics):
  reconstruct the cascade forest from infection generations, then
      label(v) = descendants(v) / (tree_size(root_of[v]) - 1)
  i.e. the fraction of v's seed-tree that lies downstream of v. The seed/root
  gets 1.0 (it owns its whole cascade); a leaf gets 0.0; nodes not in this
  run's cascade get 0.0. Averaged over runs -> s_higgs in [0, 1].

Outputs (outputs/stage7/):
  s_higgs_label.npy        -- the label, length 456,626
  label_decorrelation.json -- {rho_deg, rho_btw, frac_zero, mean, var, n_runs}
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import spearmanr

from . import config, structure
from .sirf import (GraphData, SIRFParams, SIRFSimulation,
                   draw_fixed_credulity, draw_fixed_seeds)
from .utils import write_json

STAGE7_OUT = config.OUTPUTS_DIR / "stage7"
STAGE7_OUT.mkdir(parents=True, exist_ok=True)
LABEL_NPY = STAGE7_OUT / "s_higgs_label.npy"
DECORR_JSON = STAGE7_OUT / "label_decorrelation.json"
BETWEENNESS_NPZ = config.OUTPUTS_DIR / "verification" / "higgs_betweenness_pivot1000.npz"

N_RUNS = 100


# --------------------------------------------------------------------------
# Per-node cascade contribution for one run
# --------------------------------------------------------------------------
def per_node_descendant_fraction(infection_step: np.ndarray,
                                 csr, influence: np.ndarray) -> np.ndarray:
    """label(v) = descendants(v) / (tree_size(root_of[v]) - 1), 0 outside cascade.

    Mirrors the subtree / root_of bookkeeping in structure._forest_metrics, but
    returns the per-node downstream fraction instead of the aggregate forest
    metrics. Uses the same parent reconstruction (highest-influence prior-infected
    neighbor) as Stage 3/4, so the label is consistent with the cascade trees the
    thesis already reports.
    """
    n = infection_step.shape[0]
    out = np.zeros(n, dtype=np.float64)
    infected = np.flatnonzero(infection_step >= 0)
    if infected.size <= 1:
        return out

    parent = structure.reconstruct_parents(infection_step, csr, influence)
    # generation order: a parent always precedes its children
    gen_order = infected[np.argsort(infection_step[infected], kind="stable")]

    # subtree sizes (children precede parents -> reverse generation order).
    # subtree[v] counts v plus everything downstream of v; for a root it is the
    # whole tree size.
    subtree = np.ones(n, dtype=np.int64)
    for v in gen_order[::-1]:
        p = parent[v]
        if p >= 0:
            subtree[p] += subtree[v]

    # root_of (parents precede children -> forward generation order).
    root_of = np.full(n, -1, dtype=np.int64)
    for v in gen_order:
        p = parent[v]
        root_of[v] = v if p < 0 else root_of[p]

    desc = subtree[infected] - 1                      # descendants(v)
    denom = subtree[root_of[infected]] - 1            # tree_size(root) - 1
    valid = denom > 0                                 # drop singleton trees (0/0)
    out[infected[valid]] = desc[valid] / denom[valid]
    return out


# --------------------------------------------------------------------------
# build the label
# --------------------------------------------------------------------------
def build_label(n_runs: int = N_RUNS) -> np.ndarray:
    g = GraphData.load()
    cred = draw_fixed_credulity(g.n)
    seeds = draw_fixed_seeds(g.n, config.K_SEEDS, config.SEED_REGIME_RANDOM)
    sim = SIRFSimulation(g, SIRFParams())  # calibrated operating point

    acc = np.zeros(g.n, dtype=np.float64)
    t0 = time.time()
    for ri in range(n_runs):
        r = sim.run(condition_id=config.BASELINE_CONDITION_ID, run_index=ri,
                    credulity=cred, seeds=seeds, track_structure=True)
        acc += per_node_descendant_fraction(r.infection_step, g.csr, g.influence)
        if (ri + 1) % 10 == 0 or ri == 0:
            print(f"[stage7] run {ri + 1:>3}/{n_runs}: "
                  f"spread={r.spread_size / g.n:.1%}  "
                  f"[{time.time() - t0:.0f}s]")

    s_higgs = acc / n_runs
    np.save(LABEL_NPY, s_higgs)
    print(f"[stage7] saved {LABEL_NPY}  (len {s_higgs.size})")
    return s_higgs


# --------------------------------------------------------------------------
# decorrelation check
# --------------------------------------------------------------------------
def decorrelation_gate(s_higgs: np.ndarray) -> dict:
    g = GraphData.load()
    deg = g.degree.astype(np.float64)
    btw = np.load(BETWEENNESS_NPZ)["betweenness"].astype(np.float64)

    rho_deg = float(spearmanr(s_higgs, deg).correlation)
    rho_btw = float(spearmanr(s_higgs, btw).correlation)
    frac_zero = float(np.mean(s_higgs == 0.0))

    result = {
        "rho_deg": rho_deg,
        "rho_btw": rho_btw,
        "frac_zero": frac_zero,
        "mean": float(s_higgs.mean()),
        "var": float(s_higgs.var()),
        "n_runs": N_RUNS,
        "n_nodes": int(s_higgs.size),
    }
    write_json(DECORR_JSON, result)
    print(f"[stage7] saved {DECORR_JSON}")

    # pass = both rho well below ~0.9 (the tree label sits at ~0.999)
    passed = (abs(rho_deg) < 0.9) and (abs(rho_btw) < 0.9)
    verdict = "decorrelated" if passed else "still centrality-bound"
    print("=" * 66)
    print("  decorrelation check")
    print(f"    rho(s_higgs, degree)      = {rho_deg:+.4f}")
    print(f"    rho(s_higgs, betweenness) = {rho_btw:+.4f}")
    print(f"    frac_zero = {frac_zero:.4f}   mean = {result['mean']:.6f}   "
          f"var = {result['var']:.3e}")
    print(f"    old UPFD-tree label for reference: rho_deg=0.9963  rho_btw=0.9992")
    print(f"  --> {verdict}")
    print("=" * 66)
    return result


def main() -> None:
    s_higgs = build_label()
    decorrelation_gate(s_higgs)


if __name__ == "__main__":
    main()
