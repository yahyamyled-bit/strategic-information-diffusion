"""Diagnostic: trace why early_detection spread is identical between
f_duration=None and f_duration=20 (fdur20 vs primary run).

Hypothesis to test:
  H_BENIGN: the 500 hub-targets that EarlyDetection flags at t=10 are the
            most-resistant hubs because they were not infected by t=10. When
            f_duration sends them back to S at t=30, they don't get re-
            infected (still resistant), so spread_size doesn't change.
  H_BUG:    f_duration code path silently doesn't fire for EarlyDetection.

One run with each setting; trajectories instrumented per step.
"""
from __future__ import annotations

import numpy as np

from . import config, interventions as iv
from .sirf import GraphData, SIRFParams, SIRFSimulation, draw_fixed_credulity, draw_fixed_seeds
from .stage6_experiment import COND_IDS, K_TARGET, get_targets


def instrument_run(sim, targets, intervention, credulity, seeds, f_duration):
    """Run loop variant that logs target trajectory per step.

    Returns dict: per-step state of the 500 targets + spread_size + duration.
    """
    g, p = sim.g, sim.p
    n = g.n
    from .utils import make_rng

    rng = make_rng(COND_IDS["early_detection"], 0)
    theta = rng.beta(p.theta_a, p.theta_b, n)
    attention = np.full(n, float(p.attention), dtype=np.float64) if np.isscalar(p.attention) else np.asarray(p.attention, dtype=np.float64)
    credibility = np.full(n, float(p.credibility_risk), dtype=np.float64) if np.isscalar(p.credibility_risk) else np.asarray(p.credibility_risk, dtype=np.float64)

    state = np.full(n, config.S, dtype=np.int8)
    state[seeds] = config.I
    ever_infected = np.zeros(n, dtype=bool); ever_infected[seeds] = True
    eta_eff = np.full(n, p.eta_default, dtype=np.float64)
    f_entry_step = np.full(n, -1, dtype=np.int32) if f_duration is not None else None

    deg = g.degree_f.copy()
    A = g.csr

    intervention.on_init(sim, state, eta_eff, rng)
    if getattr(intervention, "A_override", None) is not None:
        A = intervention.A_override
    if getattr(intervention, "deg_override", None) is not None:
        deg = intervention.deg_override
    safe_denom = deg > 0.0

    target_traj = {}  # step -> (n_S, n_I, n_R, n_F) restricted to targets
    expired_log = []  # list of (t, n_expired_in_targets)

    for t in range(1, p.t_max + 1):
        if f_duration is not None and t > 1:
            expired = (state == config.F) & (f_entry_step >= 0) & ((t - f_entry_step) >= f_duration)
            if expired.any():
                expired_in_tgt = int(expired[targets].sum())
                if expired_in_tgt > 0:
                    expired_log.append((t, expired_in_tgt))
                state[expired] = config.S
                eta_eff[expired] = p.eta_default
                f_entry_step[expired] = -1

        I_mask = state == config.I
        n_I = int(I_mask.sum())
        if n_I == 0:
            break

        I_float = I_mask.astype(np.float64)
        F_mask = state == config.F
        exposure_signal = A.dot(I_float)
        if F_mask.any():
            exposure_signal = exposure_signal + A.dot(eta_eff * F_mask.astype(np.float64))
        S_mask = state == config.S
        exposed = S_mask & (exposure_signal > 0.0)

        trans = g.influence * I_float
        if F_mask.any():
            trans = trans + g.influence * eta_eff * F_mask.astype(np.float64)
        weighted_active = A.dot(trans)
        peer_influence = np.zeros(n, dtype=np.float64)
        np.divide(weighted_active, deg, out=peer_influence, where=safe_denom)

        cred_eff = credibility
        if intervention is not None:
            cred_eff = intervention.credibility_adjustment(sim, credibility, t)

        U = (p.alpha * credulity * attention
             + p.beta * peer_influence
             - p.gamma * (1.0 - credulity) * cred_eff)
        newly = exposed & (U >= theta)
        state[newly] = config.I
        ever_infected |= newly

        recover = I_mask & (rng.random(n) < p.rho)
        state[recover] = config.R

        F_before = (state == config.F) if f_duration is not None else None
        if intervention is not None and t >= intervention.tau_int:
            intervention.apply(sim, state, eta_eff, t, rng)
        if f_duration is not None and F_before is not None:
            newly_F = (state == config.F) & ~F_before
            if newly_F.any():
                f_entry_step[newly_F] = t

        # log target trajectory at salient steps
        if t in (10, 11, 20, 29, 30, 31, 40, 50, 70, 100, 120, 150):
            tgt_state = state[targets]
            target_traj[t] = (
                int((tgt_state == config.S).sum()),
                int((tgt_state == config.I).sum()),
                int((tgt_state == config.R).sum()),
                int((tgt_state == config.F).sum()),
            )

    return {
        "spread_size": int(ever_infected.sum()),
        "duration": t,
        "target_traj": target_traj,
        "expired_log": expired_log[:20],
        "targets_ever_infected": int(ever_infected[targets].sum()),
    }


def main():
    g = GraphData.load()
    credulity = draw_fixed_credulity(g.n)
    seeds = draw_fixed_seeds(g.n, config.K_SEEDS, config.SEED_REGIME_RANDOM)
    targets = get_targets(g, K_TARGET, "degree")
    params = SIRFParams()
    sim = SIRFSimulation(g, params)

    print(f"n_targets={len(targets)}, k_seeds={len(seeds)}")
    print(f"Target degree range: min={g.degree[targets].min()}, max={g.degree[targets].max()}, mean={g.degree[targets].mean():.0f}")
    print(f"Target credulity: mean={credulity[targets].mean():.3f}, min={credulity[targets].min():.3f}, max={credulity[targets].max():.3f}")
    print()

    print("=" * 70)
    print("Run A: EarlyDetection, f_duration=None (primary semantics)")
    print("=" * 70)
    res_a = instrument_run(sim, targets, iv.EarlyDetection(targets, t_detect=10),
                            credulity, seeds, f_duration=None)
    print(f"spread_size = {res_a['spread_size']:,}, duration = {res_a['duration']}")
    print(f"targets ever_infected = {res_a['targets_ever_infected']} / 500")
    print("target trajectory (step: nS, nI, nR, nF):")
    for t, st in sorted(res_a["target_traj"].items()):
        print(f"  t={t:>3}: S={st[0]:>3} I={st[1]:>3} R={st[2]:>3} F={st[3]:>3}")
    print()

    print("=" * 70)
    print("Run B: EarlyDetection, f_duration=20 (fdur20 semantics)")
    print("=" * 70)
    res_b = instrument_run(sim, targets, iv.EarlyDetection(targets, t_detect=10),
                            credulity, seeds, f_duration=20)
    print(f"spread_size = {res_b['spread_size']:,}, duration = {res_b['duration']}")
    print(f"targets ever_infected = {res_b['targets_ever_infected']} / 500")
    print("target trajectory (step: nS, nI, nR, nF):")
    for t, st in sorted(res_b["target_traj"].items()):
        print(f"  t={t:>3}: S={st[0]:>3} I={st[1]:>3} R={st[2]:>3} F={st[3]:>3}")
    print()
    print("expired-in-targets log (first 20):")
    for t, n_exp in res_b["expired_log"]:
        print(f"  t={t:>3}: {n_exp} targets expired F->S")

    print()
    print("=" * 70)
    print(f"DELTA: A_spread - B_spread = {res_a['spread_size'] - res_b['spread_size']:+,}")
    print(f"  identical? {res_a['spread_size'] == res_b['spread_size']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
