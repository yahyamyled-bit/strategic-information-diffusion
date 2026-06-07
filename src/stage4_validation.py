"""
Stage 4 -- Cascade-structure validation.

Compares the structure of simulated baseline cascades (at the calibrated
operating point) against held-out real cascades on the chapter's three
dimensions (sec 2.6): cascade depth, breadth, and structural virality, via the
two-sample Kolmogorov-Smirnov test. A dimension "passes" at p > 0.05.

Per the chapter's data hygiene, PolitiFact was the calibration corpus and is
excluded here; validation uses GossipCop (fake) + Twitter15/16.

Two deviations from the chapter's literal text, documented in the methodology:
  * breadth is compared as a fraction (breadth / cascade size), because raw
    breadth is an absolute count that scales with cascade size and is therefore
    not comparable between tiny real trees and large simulated cascades (the same
    scale issue that broke the calibration size-criterion). Depth and structural
    virality are scale-invariant and compared directly.
  * the chapter expects a structural discrepancy to be remediable by tuning the
    recovery rate rho or the threshold; we found it is not (a model-class gap:
    multi-hop follower-graph contagion runs deeper than single-article broadcast
    trees, stable across rho in {0.10, 0.20, 0.40}). The depth/virality
    discrepancy is reported as a stated limitation per the chapter's discrepancy
    clause.

Outputs (outputs/stage4/):
  real_distributions.npz  -- cached per-tree structure for the real corpora
  validation.json         -- KS results + medians + pass/fail + discrepancy note
  validation.png          -- distribution overlays
"""

from __future__ import annotations

import time

import numpy as np
from scipy import stats

from . import config, structure
from .sirf import (GraphData, SIRFParams, SIRFSimulation,
                   draw_fixed_credulity, draw_fixed_seeds)
from .utils import write_json

STAGE4_OUT = config.OUTPUTS_DIR / "stage4"
STAGE4_OUT.mkdir(parents=True, exist_ok=True)
REAL_NPZ = STAGE4_OUT / "real_distributions.npz"

METRICS = ["depth", "breadth_frac", "structural_virality"]


# ==========================================================================
# Real corpora (held out): GossipCop-fake + Twitter15/16
# ==========================================================================
def _gossipcop_fake_iter():
    from torch_geometric.datasets import UPFD
    for split in ("train", "val", "test"):
        ds = UPFD("data/upfd", "gossipcop", "profile", split)
        for d in ds:
            if int(d.y) == 1:  # fake only
                yield structure.upfd_tree_structure(d.edge_index.numpy(), d.num_nodes)


def _twitter_all_iter():
    yield from structure.twitter_corpus_iter("twitter15")
    yield from structure.twitter_corpus_iter("twitter16")


def compute_real_distributions(force: bool = False) -> dict:
    if REAL_NPZ.exists() and not force:
        z = np.load(REAL_NPZ)
        return {k: z[k] for k in z.files}
    print("[stage4] computing real-corpus structure distributions ...")
    out = {}
    for name, it in (("gossipcop", _gossipcop_fake_iter()), ("twitter", _twitter_all_iter())):
        dist = structure.corpus_structure_distribution(it)
        for m in METRICS:
            out[f"{name}__{m}"] = dist[m]
        print(f"[stage4]   {name}: {dist['n_trees']} trees")
    np.savez_compressed(REAL_NPZ, **out)
    return out


# ==========================================================================
# Simulated cascade structure distribution at the calibrated operating point
# ==========================================================================
def compute_sim_distribution(n_runs: int) -> dict:
    g = GraphData.load()
    cred = draw_fixed_credulity(g.n)
    seeds = draw_fixed_seeds(g.n, config.K_SEEDS, config.SEED_REGIME_RANDOM)
    sim = SIRFSimulation(g, SIRFParams())  # uses calibrated config weights
    t0 = time.time()
    rows = {m: [] for m in METRICS}
    rows["spread_frac"] = []
    for ri in range(n_runs):
        r = sim.run(condition_id=config.BASELINE_CONDITION_ID, run_index=ri,
                    credulity=cred, seeds=seeds, track_structure=True)
        s = structure.simulated_cascade_structure(r.infection_step, g.csr, g.influence)
        for m in METRICS:
            rows[m].append(s[m])
        rows["spread_frac"].append(r.spread_size / g.n)
        if (ri + 1) % 10 == 0 or ri == 0:
            print(f"[stage4]  sim run {ri+1:>3}/{n_runs}: spread={r.spread_size/g.n:.1%} "
                  f"depth={s['depth']} vir={s['structural_virality']:.2f} "
                  f"[{time.time()-t0:.0f}s]")
    return {m: np.array(v) for m, v in rows.items()}


# ==========================================================================
# KS comparison
# ==========================================================================
def _ks(sim: np.ndarray, real: np.ndarray) -> dict:
    sim = sim[np.isfinite(sim)]
    real = real[np.isfinite(real)]
    if sim.size < 2 or real.size < 2:
        return {"ks_stat": None, "p_value": None, "pass": None,
                "sim_median": float(np.median(sim)) if sim.size else None,
                "real_median": float(np.median(real)) if real.size else None}
    ks = stats.ks_2samp(sim, real)
    return {"ks_stat": float(ks.statistic), "p_value": float(ks.pvalue),
            "pass": bool(ks.pvalue > 0.05),
            "sim_median": float(np.median(sim)), "real_median": float(np.median(real))}


def run(n_runs: int = 100) -> dict:
    t0 = time.time()
    real = compute_real_distributions()
    sim = compute_sim_distribution(n_runs)

    results = {"n_sim_runs": n_runs,
               "operating_point": {"alpha": config.ALPHA, "beta": config.BETA,
                                   "gamma": config.GAMMA},
               "sim_spread_frac_median": float(np.median(sim["spread_frac"])),
               "comparisons": {}}
    for corpus in ("gossipcop", "twitter"):
        results["comparisons"][corpus] = {}
        for m in METRICS:
            results["comparisons"][corpus][m] = _ks(sim[m], real[f"{corpus}__{m}"])

    # Overall pass = all metrics non-significant on both corpora (chapter rule).
    passes = [results["comparisons"][c][m]["pass"]
              for c in ("gossipcop", "twitter") for m in METRICS]
    results["overall_pass"] = all(p for p in passes if p is not None)
    results["discrepancy_note"] = (
        "Depth and structural virality of simulated cascades exceed those of real "
        "single-article retweet trees (a multi-hop follower-graph contagion vs "
        "broadcast-tree model-class gap, stable across recovery rates). Reported "
        "as a stated limitation per chapter sec 2.6.")
    results["wall_seconds"] = round(time.time() - t0, 1)
    write_json(STAGE4_OUT / "validation.json", results)
    _make_plots(sim, real)

    print("\n[stage4] KS results (pass = p>0.05):")
    for c in ("gossipcop", "twitter"):
        for m in METRICS:
            r = results["comparisons"][c][m]
            print(f"  {c:<9} {m:<20} sim_med={r['sim_median']:.2f} "
                  f"real_med={r['real_median']:.2f} "
                  f"KS={r['ks_stat']:.3f} p={r['p_value']:.1e} "
                  f"{'PASS' if r['pass'] else 'fail'}")
    print(f"[stage4] overall pass: {results['overall_pass']}")
    print(f"[stage4] done in {results['wall_seconds']}s -> outputs/stage4/")
    return results


def _make_plots(sim, real):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = {"depth": "Cascade depth", "breadth_frac": "Breadth / size",
              "structural_virality": "Structural virality"}
    for ax, m in zip(axes, METRICS):
        ax.hist(real[f"gossipcop__{m}"], bins=30, density=True, alpha=0.5,
                label="GossipCop (real)", color="tab:blue")
        ax.hist(real[f"twitter__{m}"], bins=30, density=True, alpha=0.4,
                label="Twitter15/16 (real)", color="tab:green")
        ax.hist(sim[m], bins=15, density=True, alpha=0.6,
                label="Simulated", color="tab:red")
        ax.set(title=titles[m], xlabel=m, ylabel="density")
        ax.legend(fontsize=8)
    fig.suptitle("Stage 4: simulated vs real cascade structure")
    fig.tight_layout()
    fig.savefig(STAGE4_OUT / "validation.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage 4: cascade-structure validation.")
    p.add_argument("--runs", type=int, default=100)
    args = p.parse_args()
    run(n_runs=args.runs)
