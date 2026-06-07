"""
Stage 6 -- Intervention experiments (matched-intensity / H1 core).

Compares the six matched-intensity conditions (baseline + the five strategies at
reference intensity, k_target = 500) on the four dependent variables, with the
chapter's statistical battery. Targeting = degree centrality (H2 found the GNN
reduces to centrality; the GNN-vs-heuristic test is the separate bridge validation).

Reference parameterization (chapter Table 2.interventions, mid intensity):
  node removal (eta=0); visibility (eta_vis=0.70); fact-check (lambda=0.25, 7-step
  latency); edge rewiring (phi=0.10); early detection (eta=0.50, t_detect=10).

Dependent variables (chapter sec 2.6):
  spread size; spread speed (steps to 90% of final spread); peak infection;
  containment efficiency CE = (S_baseline - S_intervention)/S_baseline * 100
  (negative values retained, not clipped).

Statistics: one-way ANOVA across the 6 conditions; 15 pairwise independent-samples
t-tests with Bonferroni (alpha = 0.05/15 = 0.0033); Kruskal-Wallis robustness
check; bootstrap 95% CIs (10k resamples). H3 seed-sensitivity, dose-response,
F-duration and eta robustness are separate runs.

Outputs (outputs/stage6/): matched_intensity.json, matched_intensity.png
"""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np
from scipy import stats as sps

from . import config
from . import interventions as iv
from .sirf import (GraphData, SIRFParams, SIRFSimulation,
                   draw_fixed_credulity, draw_fixed_seeds)
from .stats import bootstrap_ci
from .utils import write_json

STAGE6_OUT = config.OUTPUTS_DIR / "stage6"
STAGE6_OUT.mkdir(parents=True, exist_ok=True)
VERIF_OUT = config.OUTPUTS_DIR / "verification"

K_TARGET = 500
BONFERRONI_ALPHA = 0.05 / 15

# condition_id assignments (drive the per-run RNG; distinct per condition)
COND_IDS = {"baseline": 10, "node_removal": 11, "visibility_reduction": 12,
            "fact_checking": 13, "edge_rewiring": 14, "early_detection": 15}

# Valid targeting methods (chapter sec 2.6.3: GNN + degree, betweenness, PageRank)
TARGETING_METHODS = ("degree", "betweenness", "pagerank",
                     "gnn_gcn", "gnn_sage", "gnn_gat")


def degree_targets(graph: GraphData, k: int = K_TARGET) -> np.ndarray:
    """Top-k nodes by degree centrality (kept for back-compat with H1 main run)."""
    return np.argsort(graph.degree)[::-1][:k].astype(np.int64)


def _load_pivot_betweenness() -> np.ndarray:
    p = VERIF_OUT / "higgs_betweenness_pivot1000.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"Pivot betweenness not found at {p}. "
            f"Run verify_task2_higgs_pivot_betweenness first.")
    return np.load(p)["betweenness"].astype(np.float64)


def _load_or_compute_pagerank(graph: GraphData) -> np.ndarray:
    cache = VERIF_OUT / "higgs_pagerank.npz"
    if cache.exists():
        return np.load(cache)["pagerank"].astype(np.float64)
    from .stage5_bridge import _pagerank
    VERIF_OUT.mkdir(parents=True, exist_ok=True)
    pr = _pagerank(graph.csr)
    np.savez_compressed(cache, pagerank=pr.astype(np.float32))
    return pr.astype(np.float64)


def _load_gnn_scores(arch: str) -> np.ndarray:
    p = VERIF_OUT / "higgs_gnn_scores.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"GNN scores not found at {p}. Run verify_task3_gnn_ood first.")
    z = np.load(p)
    key = {"gnn_gcn": "gcn", "gnn_sage": "graphsage", "gnn_gat": "gat"}[arch]
    return z[key].astype(np.float64)


def get_targets(graph: GraphData, k: int, method: str) -> np.ndarray:
    """Return top-k node indices according to the named targeting method.

    For non-degree methods, supporting data must exist on disk (Task 2 / Task 3
    verification outputs, or the PageRank cache)."""
    if method == "degree":
        score = graph.degree.astype(np.float64)
    elif method == "betweenness":
        score = _load_pivot_betweenness()
    elif method == "pagerank":
        score = _load_or_compute_pagerank(graph)
    elif method in ("gnn_gcn", "gnn_sage", "gnn_gat"):
        score = _load_gnn_scores(method)
    else:
        raise ValueError(
            f"unknown targeting method: {method!r}. "
            f"Valid: {TARGETING_METHODS}")
    if score.size != graph.n:
        raise ValueError(
            f"targeting score length {score.size} != graph.n {graph.n}")
    return np.argsort(score)[::-1][:k].astype(np.int64)


def make_factories(targets, eta_default: float = config.ETA_DEFAULT,
                   early_eta: float = 0.50):
    """condition name -> callable producing a fresh intervention (or None).

    `eta_default` propagates to interventions that use the default F-transmission
    factor (FactChecking, EdgeRewiring). Node removal (eta=0) and visibility (eta_vis)
    ignore this argument because they hardcode their own eta per chapter
    Table 2.interventions. early_eta defaults to 0.50 (chapter pre-registration);
    src/exp_early_eta_sweep.py overrides it for the artifact test.
    """
    return {
        "baseline": lambda: None,
        "node_removal": lambda: iv.NodeRemoval(targets),
        "visibility_reduction": lambda: iv.VisibilityReduction(targets, eta_vis=0.70),
        "fact_checking": lambda: iv.FactChecking(targets, lam=0.25, latency=7,
                                                 eta_target=eta_default),
        "edge_rewiring": lambda: iv.EdgeRewiring(targets, phi=0.10,
                                                 eta_target=eta_default),
        "early_detection": lambda: iv.EarlyDetection(targets, t_detect=10,
                                                      eta_target=early_eta),
    }


def run_condition(sim, name, factory, n_runs, credulity, seeds,
                  f_duration: int = None):
    cid = COND_IDS[name]
    spread, speed, peak, dur, first = [], [], [], [], []
    for ri in range(n_runs):
        intervention = factory()
        r = sim.run(condition_id=cid, run_index=ri, credulity=credulity,
                    seeds=seeds, intervention=intervention, f_duration=f_duration)
        spread.append(r.spread_size); speed.append(r.spread_speed)
        peak.append(r.peak_infection); dur.append(r.duration)
        first.append(r.intervention_first_step)
    return {"spread_size": np.array(spread), "spread_speed": np.array(speed),
            "peak_infection": np.array(peak), "duration": np.array(dur)}


def _agg(v):
    ci = bootstrap_ci(v)
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
            "median": float(np.median(v)), "min": float(np.min(v)),
            "max": float(np.max(v)), "ci95_low": ci["ci_low"], "ci95_high": ci["ci_high"]}


def run(n_runs: int = 50, alpha: float = None, beta: float = None,
        gamma: float = None, eta: float = None, f_duration: int = None,
        seed_regime: str = "random", suffix: str = "",
        targeting: str = "degree", k_target: int = K_TARGET,
        early_eta: float = 0.50) -> dict:
    t0 = time.time()
    g = GraphData.load()
    credulity = draw_fixed_credulity(g.n)

    # Seed selection: random uniform (primary) or structural-likelihood (H3, secondary).
    if seed_regime == "structural":
        from .seed_regimes import draw_structural_seeds
        seeds = draw_structural_seeds(g, config.K_SEEDS)
    else:
        seeds = draw_fixed_seeds(g.n, config.K_SEEDS, config.SEED_REGIME_RANDOM)

    targets = get_targets(g, k_target, targeting)
    params = SIRFParams()
    if alpha is not None: params.alpha = alpha
    if beta is not None:  params.beta = beta
    if gamma is not None: params.gamma = gamma
    if eta is not None:   params.eta_default = eta
    sim = SIRFSimulation(g, params)
    factories = make_factories(targets, eta_default=params.eta_default,
                                early_eta=early_eta)

    print(f"[stage6{suffix}] matched-intensity: 6 conditions x {n_runs} runs, k={k_target}, "
          f"targeting={targeting}, op-point a={params.alpha} b={params.beta} g={params.gamma}, "
          f"eta={params.eta_default}, f_duration={f_duration}, seed_regime={seed_regime}",
          flush=True)

    results = {}
    for name, fac in factories.items():
        tc = time.time()
        results[name] = run_condition(sim, name, fac, n_runs, credulity, seeds,
                                      f_duration=f_duration)
        print(f"[stage6]  {name:<22} spread={results[name]['spread_size'].mean():,.0f} "
              f"peak={results[name]['peak_infection'].mean():,.0f} "
              f"[{time.time()-tc:.0f}s]")

    base_mean = float(results["baseline"]["spread_size"].mean())
    # containment efficiency per run (negative retained)
    ce = {}
    for name in factories:
        ce[name] = (base_mean - results[name]["spread_size"]) / base_mean * 100.0

    # affected-first-step counts
    affected = {}
    for name, fac in factories.items():
        if name == "baseline":
            continue
        iobj = fac()
        iobj.on_init(sim, np.full(g.n, config.S, dtype=np.int8),
                     np.full(g.n, config.ETA_DEFAULT), np.random.default_rng(0))
        affected[name] = iobj.affected_first_step(sim)

    # ---- statistics on containment efficiency ----
    inter_names = [n for n in factories if n != "baseline"]
    groups_all = [ce[n] for n in factories]                    # 6 groups (baseline ~0)
    anova = sps.f_oneway(*groups_all)
    kw = sps.kruskal(*groups_all)
    pairwise = {}
    for a, b in combinations(list(factories), 2):
        tt = sps.ttest_ind(ce[a], ce[b], equal_var=False)
        pairwise[f"{a}__vs__{b}"] = {
            "t": float(tt.statistic), "p": float(tt.pvalue),
            "significant_bonferroni": bool(tt.pvalue < BONFERRONI_ALPHA)}

    out = {
        "n_runs": n_runs, "k_target": k_target, "targeting": targeting,
        "operating_point": {"alpha": params.alpha, "beta": params.beta, "gamma": params.gamma},
        "eta_default": params.eta_default,
        "f_duration": f_duration,
        "seed_regime": seed_regime,
        "suffix": suffix,
        "baseline_spread_mean": base_mean,
        "affected_first_step": affected,
        "dependent_variables": {
            name: {dv: _agg(results[name][dv]) for dv in
                   ("spread_size", "spread_speed", "peak_infection", "duration")}
            for name in factories},
        "containment_efficiency": {name: _agg(ce[name]) for name in factories},
        "anova": {"F": float(anova.statistic), "p": float(anova.pvalue)},
        "kruskal_wallis": {"H": float(kw.statistic), "p": float(kw.pvalue)},
        "pairwise_ttests_bonferroni_alpha": BONFERRONI_ALPHA,
        "pairwise_ttests": pairwise,
        "wall_seconds": round(time.time() - t0, 1),
    }
    write_json(STAGE6_OUT / f"matched_intensity{suffix}.json", out)
    _make_plots(ce, results, factories, suffix=suffix)

    print(f"\n[stage6] containment efficiency (mean %, 95% CI):")
    for name in inter_names:
        a = out["containment_efficiency"][name]
        print(f"  {name:<22} CE={a['mean']:6.1f}%  CI[{a['ci95_low']:.1f},{a['ci95_high']:.1f}]  "
              f"affected={affected[name]:,}")
    print(f"[stage6] ANOVA F={out['anova']['F']:.2f} p={out['anova']['p']:.2e} | "
          f"KW H={out['kruskal_wallis']['H']:.2f} p={out['kruskal_wallis']['p']:.2e}")
    print(f"[stage6] done in {out['wall_seconds']}s -> outputs/stage6/")
    return out


def _make_plots(ce, results, factories, suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in factories if n != "baseline"]
    fig, ax = plt.subplots(figsize=(9, 5))
    means = [ce[n].mean() for n in names]
    errs = [ce[n].std(ddof=1) for n in names]
    ax.bar(range(len(names)), means, yerr=errs, capsize=4, color="teal")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set(ylabel="Containment efficiency (%)",
           title="Stage 6: matched-intensity intervention comparison (k=500)")
    fig.tight_layout()
    fig.savefig(STAGE6_OUT / f"matched_intensity{suffix}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage 6: matched-intensity comparison.")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--alpha", type=float, default=None,
                   help="Override calibrated alpha (for parameter-robustness runs).")
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--eta", type=float, default=None,
                   help="Override eta_default (eta-robustness check).")
    p.add_argument("--f-duration", dest="f_duration", type=int, default=None,
                   help="F->S after K steps (F-state duration sensitivity).")
    p.add_argument("--seed-regime", dest="seed_regime", type=str, default="random",
                   choices=["random", "structural"],
                   help="random = primary uniform; structural = H3 secondary regime.")
    p.add_argument("--suffix", type=str, default="",
                   help="Filename suffix, e.g. _alt1, to avoid overwriting the main result.")
    p.add_argument("--targeting", type=str, default="degree",
                   choices=list(TARGETING_METHODS),
                   help="Centrality used to pick top-k targets (chapter 2.6.3 H2 comparators).")
    p.add_argument("--k-target", dest="k_target", type=int, default=K_TARGET,
                   help="Number of top-k hub targets (dose-response sweep).")
    args = p.parse_args()
    run(n_runs=args.runs, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        eta=args.eta, f_duration=args.f_duration, seed_regime=args.seed_regime,
        suffix=args.suffix, targeting=args.targeting, k_target=args.k_target)
