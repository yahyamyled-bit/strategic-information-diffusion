"""
Recovery-probability (rho) sweep for the cascade-structure metrics.

Re-runs the no-intervention baseline cascade-structure measurement over a set of
recovery probabilities, reusing Stage 4's structure path
(structure.simulated_cascade_structure on track_structure runs) and varying only
rho. Records per-rho median depth / breadth_frac / structural_virality /
spread_frac; does not touch validation.json.

Output (outputs/stage4/):
  * rho_sweep.json

Run:  python -m src.rho_sweep_validation            # rhos {0.10,0.20,0.40}, n=100
      python -m src.rho_sweep_validation 0.10,0.20,0.40 50   # custom rhos, n
"""
from __future__ import annotations

import sys
import time
import numpy as np

from . import config, structure
from .sirf import (GraphData, SIRFParams, SIRFSimulation,
                   draw_fixed_credulity, draw_fixed_seeds)
from .stage4_validation import METRICS
from .utils import write_json

RHOS = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.10, 0.20, 0.40]
N_RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
OUT = config.OUTPUTS_DIR / "stage4" / "rho_sweep.json"


def main():
    g = GraphData.load()
    cred = draw_fixed_credulity(g.n)
    seeds = draw_fixed_seeds(g.n, config.K_SEEDS, config.SEED_REGIME_RANDOM)
    t0 = time.time()
    out = {"n_runs": N_RUNS,
           "operating_point": {"alpha": config.ALPHA, "beta": config.BETA, "gamma": config.GAMMA},
           "seed_regime": "random", "by_rho": {}}
    for rho in RHOS:
        sim = SIRFSimulation(g, SIRFParams(rho=rho))
        rows = {m: [] for m in METRICS}
        rows["spread_frac"] = []
        for ri in range(N_RUNS):
            r = sim.run(condition_id=config.BASELINE_CONDITION_ID, run_index=ri,
                        credulity=cred, seeds=seeds, track_structure=True)
            s = structure.simulated_cascade_structure(r.infection_step, g.csr, g.influence)
            for m in METRICS:
                rows[m].append(s[m])
            rows["spread_frac"].append(r.spread_size / g.n)
        med = {m: float(np.median(np.array(rows[m])[np.isfinite(rows[m])])) for m in METRICS}
        med["spread_frac"] = float(np.median(rows["spread_frac"]))
        out["by_rho"][f"{rho:.2f}"] = {"median": med, "n": N_RUNS}
        print(f"[rho-sweep] rho={rho:.2f}: depth_med={med['depth']:.1f} "
              f"vir_med={med['structural_virality']:.2f} "
              f"breadthfrac_med={med['breadth_frac']:.2f} "
              f"spread={med['spread_frac']:.1%} [{time.time()-t0:.0f}s]")
    out["wall_seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT, out)
    print(f"[rho-sweep] wrote {OUT} in {out['wall_seconds']}s")


if __name__ == "__main__":
    main()
