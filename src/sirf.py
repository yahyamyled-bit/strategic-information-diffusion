"""
SIRF agent-based diffusion engine (Stage 2 core).

Implements the methodology's five-operation, fixed-order time-step loop on the
Higgs follower graph, vectorized over scipy.sparse CSR adjacency for speed.

States (int8): S=0, I=1, R=2, F=3.

Per-step loop (t = 1..T_max):
  1. Exposure        -- S nodes adjacent to >= 1 I node.
  2. Sharing decision-- exposed node shares (S->I) iff U_v >= theta_v, where
       U_v = alpha * c_v * attention_v
             + beta  * peer_influence_v
             - gamma * (1 - c_v) * credibility_risk_v
     peer_influence_v = influence-weighted fraction of v's neighbors that are
       transmitting (I at factor 1, F at factor eta_eff).
  3. Recovery        -- nodes that were I at the START of the step recover (->R)
       with prob rho, synchronously (new infections this step are not eligible).
  4. Intervention    -- if active and t >= tau_int, target set -> F (Stage 6).
  5. Logging         -- per-state counts, new-I count.

Termination: no I nodes remain, or t == T_max.

The `intervention` argument is the hook Stage 6 plugs into; baseline runs pass
None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp

from . import config
from .utils import make_credulity_rng, make_rng, make_seeds_rng


# --------------------------------------------------------------------------
# Experiment-level held-fixed factors (Design A randomisation protocol)
# --------------------------------------------------------------------------
def draw_fixed_credulity(n: int) -> np.ndarray:
    """Credulity c_v ~ U(0,1), drawn once and held fixed for the whole experiment."""
    return make_credulity_rng().uniform(0.0, 1.0, n)


def draw_fixed_seeds(n: int, k: int, seed_regime: int) -> np.ndarray:
    """Initial-I seed set (random-uniform regime), drawn once per seed regime.

    The structural-likelihood regime (Stage 6) supplies its seed set explicitly
    via the run() `seeds` argument.
    """
    return make_seeds_rng(seed_regime).choice(n, size=k, replace=False).astype(np.int64)


# --------------------------------------------------------------------------
# Static graph substrate (built once, shared across all runs)
# --------------------------------------------------------------------------
@dataclass
class GraphData:
    csr: sp.csr_matrix              # symmetric binary adjacency (n x n)
    influence: np.ndarray          # [0,1] normalized log-degree, per node
    degree: np.ndarray             # undirected degree, per node
    n: int = field(init=False)
    degree_f: np.ndarray = field(init=False)  # |N(v)|, the peer-influence denominator

    def __post_init__(self):
        self.n = self.csr.shape[0]
        # Peer-influence is normalized by |N(v)|, the neighbor count, per chapter
        # Eq. 2.3 (peer_influence_v = (1/|N(v)|) * sum_u [...]).
        self.degree_f = self.degree.astype(np.float64)

    @classmethod
    def load(cls) -> "GraphData":
        csr = sp.load_npz(config.GRAPH_NPZ)
        arrs = np.load(config.NODE_ARRAYS_NPZ)
        return cls(csr=csr, influence=arrs["influence"], degree=arrs["degree"])


# --------------------------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------------------------
@dataclass
class SIRFParams:
    alpha: float = config.ALPHA
    beta: float = config.BETA
    gamma: float = config.GAMMA
    rho: float = config.RHO
    t_max: int = config.T_MAX
    k_seeds: int = config.K_SEEDS
    theta_a: float = config.THETA_BETA_A
    theta_b: float = config.THETA_BETA_B
    eta_default: float = config.ETA_DEFAULT
    # attention / credibility_risk: scalar (homogeneous) or per-node array.
    attention: float | np.ndarray = config.ATTENTION_DEFAULT
    credibility_risk: float | np.ndarray = config.CREDIBILITY_RISK_DEFAULT


# --------------------------------------------------------------------------
# Per-run result
# --------------------------------------------------------------------------
@dataclass
class RunResult:
    condition_id: int
    run_index: int
    # Dependent variables
    spread_size: int            # nodes that ever entered I
    spread_speed: int           # steps until 90% of final spread size
    peak_infection: int         # max simultaneous I at any step
    peak_step: int
    duration: int               # steps executed before termination
    intervention_first_step: Optional[int]
    # Per-step time series (length = duration)
    counts_S: np.ndarray
    counts_I: np.ndarray
    counts_R: np.ndarray
    counts_F: np.ndarray
    new_I: np.ndarray
    cumulative_infected: np.ndarray
    seeds: np.ndarray
    # Per-node infection generation (step at which a node entered I; seeds = 0,
    # never-infected = -1). Populated only when run(track_structure=True); the
    # cascade-structure metrics (Stage 3 selection + Stage 4 validation) need it.
    infection_step: Optional[np.ndarray] = None

    def summary(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "run_index": self.run_index,
            "spread_size": int(self.spread_size),
            "spread_speed": int(self.spread_speed),
            "peak_infection": int(self.peak_infection),
            "peak_step": int(self.peak_step),
            "duration": int(self.duration),
            "intervention_first_step": (
                None if self.intervention_first_step is None
                else int(self.intervention_first_step)
            ),
        }


class SIRFSimulation:
    def __init__(self, graph: GraphData, params: Optional[SIRFParams] = None):
        self.g = graph
        self.p = params or SIRFParams()

    def _as_array(self, x: float | np.ndarray) -> np.ndarray:
        if np.isscalar(x):
            return np.full(self.g.n, float(x), dtype=np.float64)
        return np.asarray(x, dtype=np.float64)

    def run(
        self,
        condition_id: int = config.BASELINE_CONDITION_ID,
        run_index: int = 0,
        seed_regime: int = config.SEED_REGIME_RANDOM,
        credulity: Optional[np.ndarray] = None,
        seeds: Optional[np.ndarray] = None,
        intervention=None,
        track_structure: bool = False,
        f_duration: Optional[int] = None,
    ) -> RunResult:
        g, p = self.g, self.p
        n = g.n

        # --- Held-fixed factors (Design A randomisation protocol) ---
        # Credulity is drawn once for the whole experiment; the seed set once per
        # seed regime. Both are reused across all runs of all conditions and are
        # passed in by the experiment runner. If a caller omits them they are
        # reconstructed deterministically here (identical to the runner's draw).
        if credulity is None:
            credulity = draw_fixed_credulity(n)
        else:
            credulity = np.asarray(credulity, dtype=np.float64)
        if seeds is None:
            seeds = draw_fixed_seeds(n, p.k_seeds, seed_regime)
        seeds = np.asarray(seeds, dtype=np.int64)

        # --- Per-run RNG: threshold draws + recovery + intervention only,
        # keyed on (condition_id, run_index). This is the ONLY source of
        # run-to-run variation. ---
        rng = make_rng(condition_id, run_index)
        theta = rng.beta(p.theta_a, p.theta_b, n)
        attention = self._as_array(p.attention)
        credibility = self._as_array(p.credibility_risk)

        # --- Initial states ---
        state = np.full(n, config.S, dtype=np.int8)
        state[seeds] = config.I

        ever_infected = np.zeros(n, dtype=bool)
        ever_infected[seeds] = True

        infection_step = None
        if track_structure:
            infection_step = np.full(n, -1, dtype=np.int32)
            infection_step[seeds] = 0  # seeds are generation 0

        # Per-node effective transmission factor while in F (set by interventions).
        eta_eff = np.full(n, p.eta_default, dtype=np.float64)

        # F-state duration tracking (chapter sec 2.6 robustness): when set, F nodes
        # flip back to S after `f_duration` steps. Otherwise F is absorbing.
        f_entry_step = None
        if f_duration is not None:
            f_entry_step = np.full(n, -1, dtype=np.int32)

        deg = g.degree_f
        A = g.csr

        if intervention is not None:
            intervention.on_init(self, state, eta_eff, rng)
            # Edge rewiring (and any topology-changing intervention) may replace
            # the adjacency / degree used for the rest of the run.
            if getattr(intervention, "A_override", None) is not None:
                A = intervention.A_override
            if getattr(intervention, "deg_override", None) is not None:
                deg = intervention.deg_override
        safe_denom = deg > 0.0

        # Logs
        cS, cI, cR, cF, cNew, cCum = [], [], [], [], [], []
        peak_infection, peak_step = int((state == config.I).sum()), 0
        intervention_first_step: Optional[int] = None

        for t in range(1, p.t_max + 1):
            # ---- F-duration: flip back to S after K steps (chapter sec 2.6 sensitivity) ----
            if f_duration is not None and t > 1:
                expired = (state == config.F) & (f_entry_step >= 0) \
                          & ((t - f_entry_step) >= f_duration)
                if expired.any():
                    state[expired] = config.S
                    eta_eff[expired] = p.eta_default
                    f_entry_step[expired] = -1

            I_mask = state == config.I
            n_I = int(I_mask.sum())
            if n_I == 0:
                break  # termination: no infected remain

            # ---- 1. Exposure: S nodes adjacent to >=1 Infected OR Flagged neighbor.
            # Flagged neighbors contribute to the exposure signal at factor eta_eff
            # (chapter sec 2.4 step 1); eta_eff = 0 (node removal) means a Flagged
            # neighbor exposes nothing.
            I_float = I_mask.astype(np.float64)
            F_mask = state == config.F
            exposure_signal = A.dot(I_float)
            if F_mask.any():
                exposure_signal = exposure_signal + A.dot(eta_eff * F_mask.astype(np.float64))
            S_mask = state == config.S
            exposed = S_mask & (exposure_signal > 0.0)

            # ---- 2. Sharing decision (utility threshold) ----
            # Transmitting weight: I contributes influence*1, F contributes influence*eta_eff.
            trans = g.influence * I_float
            if F_mask.any():
                trans = trans + g.influence * eta_eff * F_mask.astype(np.float64)
            weighted_active = A.dot(trans)
            peer_influence = np.zeros(n, dtype=np.float64)
            np.divide(weighted_active, deg, out=peer_influence, where=safe_denom)

            # Optional fact-checking penalty on downstream S agents (Stage 6 hook).
            cred_eff = credibility
            if intervention is not None:
                cred_eff = intervention.credibility_adjustment(self, credibility, t)

            U = (p.alpha * credulity * attention
                 + p.beta * peer_influence
                 - p.gamma * (1.0 - credulity) * cred_eff)

            newly = exposed & (U >= theta)
            state[newly] = config.I
            ever_infected |= newly
            if track_structure:
                infection_step[newly] = t
            n_new = int(newly.sum())

            # ---- 3. Recovery (synchronous: only start-of-step I are eligible) ----
            recover = I_mask & (rng.random(n) < p.rho)
            state[recover] = config.R

            # ---- 4. Intervention ----
            F_before = (state == config.F) if f_duration is not None else None
            if intervention is not None and t >= intervention.tau_int:
                applied = intervention.apply(self, state, eta_eff, t, rng)
                if applied and intervention_first_step is None:
                    intervention_first_step = t
            # F-duration: stamp this step on nodes that just transitioned to F.
            if f_duration is not None and F_before is not None:
                newly_F = (state == config.F) & ~F_before
                if newly_F.any():
                    f_entry_step[newly_F] = t

            # ---- 5. Logging ----
            nS = int((state == config.S).sum())
            nI = int((state == config.I).sum())
            nR = int((state == config.R).sum())
            nF = int((state == config.F).sum())
            cS.append(nS); cI.append(nI); cR.append(nR); cF.append(nF)
            cNew.append(n_new)
            cCum.append(int(ever_infected.sum()))
            if nI > peak_infection:
                peak_infection, peak_step = nI, t

        duration = len(cI)
        spread_size = int(ever_infected.sum())
        cum = np.asarray(cCum, dtype=np.int64)
        spread_speed = _steps_to_fraction(cum, spread_size, 0.90)

        return RunResult(
            condition_id=condition_id,
            run_index=run_index,
            spread_size=spread_size,
            spread_speed=spread_speed,
            peak_infection=peak_infection,
            peak_step=peak_step,
            duration=duration,
            intervention_first_step=intervention_first_step,
            counts_S=np.asarray(cS), counts_I=np.asarray(cI),
            counts_R=np.asarray(cR), counts_F=np.asarray(cF),
            new_I=np.asarray(cNew), cumulative_infected=cum,
            seeds=seeds,
            infection_step=infection_step,
        )


def _steps_to_fraction(cumulative: np.ndarray, final_size: int, frac: float) -> int:
    """Steps until cumulative ever-infected reaches `frac` of final spread size."""
    if final_size <= 0 or cumulative.size == 0:
        return 0
    target = frac * final_size
    idx = np.searchsorted(cumulative, target, side="left")
    # cumulative[idx] is the first step (0-based) at or above target; report 1-based.
    return int(min(idx + 1, cumulative.size))
