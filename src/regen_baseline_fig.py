"""
Baseline infection-curve figure at the calibrated operating point.

Stage 2 runs before Stage 3 calibration, so outputs/stage2/baseline_curves.png
uses provisional parameters. This re-runs the no-intervention baseline at the
calibrated operating point (config.ALPHA/BETA/GAMMA) and renders the same
two-panel figure.

Output (outputs/figures/):
  * baseline_curves_oppoint.png  -- infection curves + mean

Run:  python -m src.regen_baseline_fig
"""
from __future__ import annotations

import sys
import time
import numpy as np

from . import config
from .sirf import (GraphData, RunResult, SIRFParams, SIRFSimulation,
                   draw_fixed_credulity, draw_fixed_seeds)

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
# Stage-6 baseline condition id, so per-run randomness matches the H1 baseline.
BASELINE_CID = 10
OUT_PNG = config.OUTPUTS_DIR / "figures" / "baseline_curves_oppoint.png"


def _pad(arrays, length, fill_last):
    out = np.zeros((len(arrays), length), dtype=np.float64)
    for i, a in enumerate(arrays):
        out[i, : a.size] = a
        if fill_last and a.size < length and a.size > 0:
            out[i, a.size:] = a[-1]
    return out


def main():
    t0 = time.time()
    g = GraphData.load()
    params = SIRFParams()  # defaults to the config operating point
    sim = SIRFSimulation(g, params)
    credulity = draw_fixed_credulity(g.n)
    seeds = draw_fixed_seeds(g.n, params.k_seeds, config.SEED_REGIME_RANDOM)
    print(f"[regen] n={g.n:,}  {N_RUNS} baseline runs at operating point "
          f"(alpha={params.alpha}, beta={params.beta}, gamma={params.gamma})")

    results: list[RunResult] = []
    for ri in range(N_RUNS):
        r = sim.run(condition_id=BASELINE_CID, run_index=ri,
                    seed_regime=config.SEED_REGIME_RANDOM,
                    credulity=credulity, seeds=seeds)
        results.append(r)
        if (ri + 1) % 10 == 0 or ri == 0:
            print(f"[regen] run {ri+1:>3}/{N_RUNS}: spread={r.spread_size:,} "
                  f"({r.spread_size/g.n:.1%}) peak={r.peak_infection:,} "
                  f"[{time.time()-t0:.0f}s]")

    spreads = np.array([r.spread_size for r in results])
    peaks = np.array([r.peak_infection for r in results])
    print(f"[regen] spread mean={spreads.mean():,.0f} ({spreads.mean()/g.n:.1%}), "
          f"peak mean={peaks.mean():,.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")

    max_dur = max(r.duration for r in results)
    I_curves = _pad([r.counts_I for r in results], max_dur, fill_last=False)
    cum_curves = _pad([r.cumulative_infected for r in results], max_dur, fill_last=True)
    x = np.arange(1, max_dur + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
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
    fig.suptitle(f"Baseline SIRF trajectory (operating point "
                 f"$\\alpha$=0.30, $\\beta$=0.80, $\\gamma$=0.36; "
                 f"N={len(results)} runs, {g.n:,}-node Higgs graph)")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"[regen] wrote {OUT_PNG}  in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
