"""
Stage 2 -- Baseline simulation.

Runs the no-intervention SIRF model across many runs (random-uniform seed
regime, k=10) and establishes the baseline dependent-variable distributions that
Stage 6 measures containment efficiency against.

Outputs (outputs/stage2/):
  * baseline_runs.csv        -- one row per run: spread_size, speed, peak, ...
  * baseline_summary.json    -- aggregate stats + bootstrap 95% CIs
  * baseline_timeseries.npz  -- padded per-run S/I/R/new_I/cumulative curves
  * baseline_curves.png      -- infection curves + mean
  * baseline_distributions.png -- DV histograms
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config
from .sirf import (GraphData, RunResult, SIRFParams, SIRFSimulation,
                   draw_fixed_credulity, draw_fixed_seeds)
from .stats import bootstrap_ci
from .utils import write_json


def _pad(arrays: list[np.ndarray], length: int, fill_last: bool) -> np.ndarray:
    out = np.zeros((len(arrays), length), dtype=np.float64)
    for i, a in enumerate(arrays):
        out[i, : a.size] = a
        if fill_last and a.size < length and a.size > 0:
            out[i, a.size:] = a[-1]
    return out


def run(n_runs: int = config.BASELINE_N_RUNS,
        seed_regime: int = config.SEED_REGIME_RANDOM) -> dict:
    t0 = time.time()
    g = GraphData.load()
    params = SIRFParams()
    sim = SIRFSimulation(g, params)

    # Held-fixed factors (Design A): credulity drawn once for the whole
    # experiment, the seed set once per regime; both passed into every run.
    credulity = draw_fixed_credulity(g.n)
    seeds = draw_fixed_seeds(g.n, params.k_seeds, seed_regime)
    print(f"[stage2] graph n={g.n:,}, running {n_runs} baseline runs "
          f"(alpha={params.alpha}, beta={params.beta}, gamma={params.gamma}); "
          f"fixed credulity (mean={credulity.mean():.3f}) and {params.k_seeds} fixed seeds ...")

    results: list[RunResult] = []
    for ri in range(n_runs):
        r = sim.run(condition_id=config.BASELINE_CONDITION_ID,
                    run_index=ri, seed_regime=seed_regime,
                    credulity=credulity, seeds=seeds)
        results.append(r)
        if (ri + 1) % 5 == 0 or ri == 0:
            print(f"[stage2]  run {ri+1:>3}/{n_runs}: spread={r.spread_size:,} "
                  f"({r.spread_size/g.n:.1%}) peak={r.peak_infection:,}@{r.peak_step} "
                  f"speed90={r.spread_speed} dur={r.duration} "
                  f"[{time.time()-t0:.0f}s]")

    # --- Per-run summary table ---
    df = pd.DataFrame([r.summary() for r in results])
    df.to_csv(config.STAGE2_OUT / "baseline_runs.csv", index=False)

    # --- Aggregate stats with bootstrap CIs on the four DVs ---
    def agg(col):
        v = df[col].to_numpy()
        ci = bootstrap_ci(v)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "min": int(v.min()), "max": int(v.max()),
                "median": float(np.median(v)),
                "ci95_low": ci["ci_low"], "ci95_high": ci["ci_high"]}

    summary = {
        "n_runs": n_runs,
        "seed_regime": seed_regime,
        "n_nodes": g.n,
        "params": {"alpha": params.alpha, "beta": params.beta,
                   "gamma": params.gamma, "rho": params.rho,
                   "t_max": params.t_max, "k_seeds": params.k_seeds,
                   "theta_beta": [params.theta_a, params.theta_b],
                   "note": "alpha/beta/gamma are PROVISIONAL (Stage 3 calibrates)"},
        "spread_size": agg("spread_size"),
        "spread_speed": agg("spread_speed"),
        "peak_infection": agg("peak_infection"),
        "duration": agg("duration"),
        "spread_size_fraction_mean": float(df["spread_size"].mean() / g.n),
        "seeds": results[0].seeds.tolist(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    write_json(config.STAGE2_OUT / "baseline_summary.json", summary)

    # --- Per-run time series (padded) ---
    max_dur = max(r.duration for r in results)
    np.savez_compressed(
        config.STAGE2_OUT / "baseline_timeseries.npz",
        counts_I=_pad([r.counts_I for r in results], max_dur, fill_last=False),
        new_I=_pad([r.new_I for r in results], max_dur, fill_last=False),
        cumulative=_pad([r.cumulative_infected for r in results], max_dur, fill_last=True),
        durations=np.array([r.duration for r in results]),
    )

    _make_plots(results, g.n, max_dur)

    print(f"[stage2] spread_size: mean={summary['spread_size']['mean']:,.0f} "
          f"({summary['spread_size_fraction_mean']:.1%}) "
          f"95% CI [{summary['spread_size']['ci95_low']:,.0f}, "
          f"{summary['spread_size']['ci95_high']:,.0f}]")
    print(f"[stage2] peak={summary['peak_infection']['mean']:,.0f}, "
          f"speed90={summary['spread_speed']['mean']:.1f}, "
          f"duration={summary['duration']['mean']:.1f}")
    print(f"[stage2] done in {summary['wall_seconds']}s -> outputs/stage2/")
    return summary


def _make_plots(results, n_nodes, max_dur):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")

    # Infection curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    I_curves = _pad([r.counts_I for r in results], max_dur, fill_last=False)
    cum_curves = _pad([r.cumulative_infected for r in results], max_dur, fill_last=True)
    x = np.arange(1, max_dur + 1)
    for c in I_curves:
        axes[0].plot(x, c, color="tab:red", alpha=0.15, lw=0.8)
    axes[0].plot(x, I_curves.mean(axis=0), color="black", lw=2, label="mean")
    axes[0].set(title="Active sharers (I) over time", xlabel="step",
                ylabel="nodes in state I")
    axes[0].legend()
    for c in cum_curves:
        axes[1].plot(x, c, color="tab:blue", alpha=0.15, lw=0.8)
    axes[1].plot(x, cum_curves.mean(axis=0), color="black", lw=2, label="mean")
    axes[1].set(title="Cumulative ever-infected", xlabel="step",
                ylabel="nodes ever in I")
    axes[1].legend()
    fig.suptitle(f"Baseline SIRF cascades (N={len(results)} runs, "
                 f"{n_nodes:,}-node Higgs graph)")
    fig.tight_layout()
    fig.savefig(config.STAGE2_OUT / "baseline_curves.png", dpi=130)
    plt.close(fig)

    # DV distributions
    df = pd.DataFrame([r.summary() for r in results])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(
        axes,
        ["spread_size", "peak_infection", "spread_speed"],
        ["Spread size (total ever-I)", "Peak infection", "Spread speed (steps to 90%)"],
    ):
        sns.histplot(df[col], ax=ax, kde=True, color="tab:purple")
        ax.set(title=title, xlabel="")
    fig.suptitle("Baseline dependent-variable distributions")
    fig.tight_layout()
    fig.savefig(config.STAGE2_OUT / "baseline_distributions.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 2: baseline SIRF simulation.")
    p.add_argument("--runs", type=int, default=config.BASELINE_N_RUNS)
    args = p.parse_args()
    run(n_runs=args.runs)
