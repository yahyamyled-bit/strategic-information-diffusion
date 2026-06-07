"""
Stage 3 -- Parameter calibration (alpha, beta, gamma).

Per chapter section 2.3, the utility-function weights are calibrated from the
Twitter Information Operations (IO) Archive. The IO Archive is used ONLY to set
plausible *search ranges* for the three weights; it is never loaded into the
simulation. A coarse grid search over those ranges then selects the operating
point, with each candidate triple judged against two acceptance criteria.

Acceptance criteria (order-of-magnitude reach anchor):
  (1) The baseline cascade reaches an absolute size of the same order of
      magnitude as documented large real misinformation cascades
      (~1e4 to ~1e5 users; Vosoughi et al. 2018 report false news reaching up
      to ~1e5), and is sub-saturating (does not engulf the whole graph).
      The chapter's literal criterion ("Higgs cascade size within the UPFD
      cascade-size range") is not directly applicable, because UPFD propagation
      trees are 3-497 nodes while any Higgs cascade is tens of thousands. The
      chapter itself notes this scale gap (sec 2.6). The cascade-realism
      check is structural and lives in Stage 4 (depth/breadth/virality KS tests
      vs UPFD-GossipCop + Twitter15/16); Stage 3 only needs a reach gate.
  (2) The cascade is non-degenerate in depth: it does not complete in a single
      step (steps-to-90% >= MIN_STEPS_TO_90) and terminates before T_max.

IO Archive -> range mapping:
  The three weights are dimensionless multipliers on normalized utility terms
  (attention and credibility_risk are homogeneous = 1; peer_influence in [0,1]).
  The IO Archive cannot pin their absolute scale, so we:
    * extract and REPORT real empirical distributions from the Archive
      (retweet/like counts for attention; retweet/reply amplification for peer
      influence; suspended-account follower/following/age profiles for
      credibility risk), giving citable empirical facts; and
    * center each weight's search range at a reach-plausible operating point and
      set the range WIDTH in proportion to the empirical dispersion (coefficient
      of variation) of that weight's IO-Archive driver -- so the dominant,
      highly variable peer-amplification driver yields the widest beta range,
      consistent with beta being the dominant lever on cascade size.
  The absolute operating point is then fixed by the reach-anchored grid search.

Inputs:
  data/raw/ioa_users.csv         -- IO Archive account profiles (full)
  data/raw/ioa_tweets_sample.csv -- IO Archive tweets (bounded head sample)
Outputs (outputs/stage3/):
  calibration.json   -- empirical stats, derived ranges, grid results, selection
  io_distributions.png, grid_heatmap.png
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .sirf import (GraphData, SIRFParams, SIRFSimulation,
                   draw_fixed_credulity, draw_fixed_seeds)
from .utils import write_json

# --- Acceptance-criterion constants ---
REACH_LO_FRAC = 0.02     # ~9k nodes on the 456k graph: lower order-of-magnitude bound
REACH_HI_FRAC = 0.40     # ~183k nodes: upper bound, still clearly sub-saturating
SATURATION_FRAC = 0.95   # above this the cascade is degenerate (engulfs the graph)
MIN_STEPS_TO_90 = 3      # fewer steps => single-step / degenerate take-off

STAGE3_OUT = config.OUTPUTS_DIR / "stage3"
STAGE3_OUT.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# IO Archive ingestion + empirical statistics
# ==========================================================================
def _robust_stats(x: np.ndarray) -> dict:
    """Distribution summary robust to heavy tails (IO engagement is power-law)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return {
        "n": int(x.size),
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean > 0 else 0.0,   # coefficient of variation
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }


def load_io_users(path) -> pd.DataFrame:
    """Account-level profiles of suspended (IO) accounts -> credibility-risk driver."""
    df = pd.read_csv(path, dtype=str, on_bad_lines="skip", engine="c",
                     encoding="utf-8", encoding_errors="replace")
    for c in ("follower_count", "following_count"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    dt = pd.to_datetime(df.get("account_creation_date"), errors="coerce")
    # Account age in days as of the Higgs-era reference; absolute value not load-
    # bearing, only the dispersion across suspended accounts is used.
    df["account_age_days"] = (pd.Timestamp("2022-01-01") - dt).dt.days
    return df


def load_io_tweets_sample(path) -> pd.DataFrame:
    """Bounded head-sample of IO tweets -> attention (alpha) + peer (beta) drivers.

    The file is a range-truncated CSV, so the final row may be incomplete; pandas
    with on_bad_lines='skip' drops it. A representative subset; the full file
    is ~114 GB.
    """
    cols = ["is_retweet", "retweet_count", "like_count", "reply_count",
            "quote_count", "in_reply_to_tweetid", "retweet_tweetid"]
    df = pd.read_csv(path, dtype=str, on_bad_lines="skip", engine="c",
                     usecols=lambda c: c in cols,
                     encoding="utf-8", encoding_errors="replace")
    for c in ("retweet_count", "like_count", "reply_count", "quote_count"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_retweet_bool"] = df.get("is_retweet").astype(str).str.lower().isin(["true", "1"])
    return df


def extract_empirical_drivers(users: pd.DataFrame, tweets: pd.DataFrame) -> dict:
    """Compute the real IO-Archive statistics that inform each weight's range."""
    # Attention driver (alpha): visibility reward = like + retweet counts on
    # original (non-retweet) content.
    orig = tweets[~tweets["is_retweet_bool"]] if "is_retweet_bool" in tweets else tweets
    attention_signal = (orig.get("like_count", pd.Series(dtype=float)).fillna(0)
                        + orig.get("retweet_count", pd.Series(dtype=float)).fillna(0))
    # Peer-influence driver (beta): coordinated amplification = retweet_count
    # distribution (how far a share propagates) plus the retweet propensity.
    peer_signal = tweets.get("retweet_count", pd.Series(dtype=float)).dropna()
    retweet_rate = float(tweets["is_retweet_bool"].mean()) if "is_retweet_bool" in tweets else float("nan")
    # Credibility-risk driver (gamma): suspended-account profile abnormality.
    # follower/following ratio dispersion separates manufactured from organic.
    foll = users["follower_count"]
    folg = users["following_count"]
    ratio = (foll / folg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    return {
        "attention_like_plus_retweet": _robust_stats(attention_signal.to_numpy()),
        "peer_retweet_count": _robust_stats(peer_signal.to_numpy()),
        "peer_retweet_rate": retweet_rate,
        "gamma_follower_count": _robust_stats(foll.to_numpy()),
        "gamma_following_count": _robust_stats(folg.to_numpy()),
        "gamma_follower_following_ratio": _robust_stats(ratio.to_numpy()),
        "gamma_account_age_days": _robust_stats(users["account_age_days"].to_numpy()),
    }


# ==========================================================================
# IO drivers -> search ranges (documented convention)
# ==========================================================================
# Reach-plausible center (provisional baseline operating point, Stage 2).
CENTER = {"alpha": 0.25, "beta": 1.0, "gamma": 0.30}
# Half-width scaling: range = center * (1 +/- WIDTH_K * cv_norm), clipped to a
# sane floor/ceiling. cv_norm is the driver's CV squashed to [0,1] via tanh so a
# single extreme outlier cannot blow the range open.
WIDTH_K = 0.6
WIDTH_FLOOR, WIDTH_CEIL = 0.15, 0.85   # min/max fractional half-width


def _cv_norm(cv: float) -> float:
    return float(np.tanh(cv / 5.0))     # heavy-tailed CVs are large; squash them


def derive_ranges(drivers: dict, n_levels: int = 4) -> dict:
    """Turn empirical dispersion into a centered, dispersion-scaled grid per weight."""
    cvs = {
        "alpha": drivers["attention_like_plus_retweet"].get("cv", 0.0),
        "beta": drivers["peer_retweet_count"].get("cv", 0.0),
        "gamma": drivers["gamma_follower_following_ratio"].get("cv", 0.0),
    }
    ranges = {}
    for w, c in CENTER.items():
        hw = float(np.clip(WIDTH_K * _cv_norm(cvs[w]), WIDTH_FLOOR, WIDTH_CEIL))
        lo, hi = c * (1 - hw), c * (1 + hw)
        ranges[w] = {
            "center": c, "cv": cvs[w], "half_width_frac": hw,
            "lo": round(lo, 4), "hi": round(hi, 4),
            "grid": [round(v, 4) for v in np.linspace(lo, hi, n_levels)],
        }
    return ranges


# ==========================================================================
# Reach-anchored grid search
# ==========================================================================
@dataclass
class GridPoint:
    alpha: float
    beta: float
    gamma: float
    spread_frac: float
    spread_size: int
    steps_to_90: float
    duration: float
    accepted: bool


def _evaluate(sim_graph, alpha, beta, gamma, credulity, seeds, n_runs, base_id):
    sizes, speeds, durs = [], [], []
    p = SIRFParams(alpha=alpha, beta=beta, gamma=gamma)
    sim = SIRFSimulation(sim_graph, p)
    for ri in range(n_runs):
        r = sim.run(condition_id=base_id, run_index=ri,
                    credulity=credulity, seeds=seeds)
        sizes.append(r.spread_size); speeds.append(r.spread_speed); durs.append(r.duration)
    return float(np.mean(sizes)), float(np.mean(speeds)), float(np.mean(durs))


def _accept(spread_frac, steps_to_90, n_nodes):
    return (REACH_LO_FRAC <= spread_frac <= REACH_HI_FRAC
            and spread_frac < SATURATION_FRAC
            and steps_to_90 >= MIN_STEPS_TO_90)


def grid_search(graph: GraphData, ranges: dict, n_runs_coarse: int = 1,
                n_runs_confirm: int = 5) -> dict:
    n = graph.n
    credulity = draw_fixed_credulity(n)
    seeds = draw_fixed_seeds(n, config.K_SEEDS, config.SEED_REGIME_RANDOM)

    grid = [(a, b, g)
            for a in ranges["alpha"]["grid"]
            for b in ranges["beta"]["grid"]
            for g in ranges["gamma"]["grid"]]
    print(f"[stage3] coarse grid: {len(grid)} triples x {n_runs_coarse} run(s)")

    points: list[GridPoint] = []
    t0 = time.time()
    for i, (a, b, g) in enumerate(grid):
        size, speed, dur = _evaluate(graph, a, b, g, credulity, seeds,
                                     n_runs_coarse, base_id=100 + i)
        frac = size / n
        acc = _accept(frac, speed, n)
        points.append(GridPoint(a, b, g, frac, int(size), speed, dur, acc))
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[stage3]  {i+1:>2}/{len(grid)} a={a:.2f} b={b:.2f} g={g:.2f} "
                  f"-> {frac:.1%} speed90={speed:.1f} {'ACCEPT' if acc else 'reject'} "
                  f"[{time.time()-t0:.0f}s]")

    accepted = [p for p in points if p.accepted]
    if not accepted:
        return {"accepted": [], "selected": None, "points": [vars(p) for p in points]}

    # Selected operating point: accepted triple whose reach is closest to the
    # geometric center of the reach window (most "typical" large cascade).
    target = float(np.sqrt(REACH_LO_FRAC * REACH_HI_FRAC))
    selected = min(accepted, key=lambda p: abs(np.log(p.spread_frac) - np.log(target)))

    # Confirm the selection with more runs for a stable reported value.
    size, speed, dur = _evaluate(graph, selected.alpha, selected.beta, selected.gamma,
                                 credulity, seeds, n_runs_confirm, base_id=999)
    confirm = {"alpha": selected.alpha, "beta": selected.beta, "gamma": selected.gamma,
               "n_runs": n_runs_confirm, "spread_frac": size / n,
               "spread_size": int(size), "steps_to_90": speed, "duration": dur}

    # Report accepted parameter ranges (the "ranges" the chapter asks for).
    acc_ranges = {w: {"min": min(getattr(p, w) for p in accepted),
                      "max": max(getattr(p, w) for p in accepted)}
                  for w in ("alpha", "beta", "gamma")}
    return {
        "accepted_count": len(accepted),
        "accepted_param_ranges": acc_ranges,
        "selected": confirm,
        "points": [vars(p) for p in points],
        "seeds": seeds.tolist(),
    }


def run(n_runs_coarse: int = 1, n_runs_confirm: int = 5) -> dict:
    t0 = time.time()
    users = load_io_users(config.RAW_DIR / "ioa_users.csv")
    tweets = load_io_tweets_sample(config.RAW_DIR / "ioa_tweets_sample.csv")
    print(f"[stage3] IO Archive: {len(users):,} accounts, {len(tweets):,} sampled tweets")

    drivers = extract_empirical_drivers(users, tweets)
    ranges = derive_ranges(drivers)
    print(f"[stage3] derived ranges: "
          f"alpha {ranges['alpha']['lo']}-{ranges['alpha']['hi']}, "
          f"beta {ranges['beta']['lo']}-{ranges['beta']['hi']}, "
          f"gamma {ranges['gamma']['lo']}-{ranges['gamma']['hi']}")

    graph = GraphData.load()
    search = grid_search(graph, ranges, n_runs_coarse, n_runs_confirm)

    result = {
        "io_empirical_drivers": drivers,
        "io_to_range_mapping": {
            "convention": "center at reach-plausible Stage-2 point; half-width = "
                          "clip(WIDTH_K*tanh(cv/5), floor, ceil); see module docstring",
            "center": CENTER, "width_k": WIDTH_K,
            "width_floor": WIDTH_FLOOR, "width_ceil": WIDTH_CEIL,
        },
        "search_ranges": ranges,
        "acceptance_criteria": {
            "reach_lo_frac": REACH_LO_FRAC, "reach_hi_frac": REACH_HI_FRAC,
            "saturation_frac": SATURATION_FRAC, "min_steps_to_90": MIN_STEPS_TO_90,
            "note": "order-of-magnitude reach anchor; UPFD size match is not "
                    "applicable, structural realism is Stage 4",
        },
        "grid_search": search,
        "wall_seconds": round(time.time() - t0, 1),
    }
    write_json(STAGE3_OUT / "calibration.json", result)
    _make_plots(drivers, search, graph.n)

    sel = search.get("selected")
    if sel:
        print(f"[stage3] SELECTED alpha={sel['alpha']} beta={sel['beta']} "
              f"gamma={sel['gamma']} -> {sel['spread_frac']:.1%} "
              f"({sel['spread_size']:,} nodes), {search['accepted_count']} triples accepted")
    else:
        print("[stage3] NO triple accepted -- widen ranges or revisit criteria")
    print(f"[stage3] done in {result['wall_seconds']}s -> outputs/stage3/")
    return result


def _make_plots(drivers, search, n_nodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = search.get("points", [])
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    fr = np.array([p["spread_frac"] for p in pts])
    bt = np.array([p["beta"] for p in pts])
    acc = np.array([p["accepted"] for p in pts])
    ax.scatter(bt[~acc], fr[~acc] * 100, c="lightgray", label="rejected", s=30)
    ax.scatter(bt[acc], fr[acc] * 100, c="tab:green", label="accepted", s=40)
    ax.axhspan(REACH_LO_FRAC * 100, REACH_HI_FRAC * 100, color="tab:green", alpha=0.08)
    ax.set(xlabel="beta (peer-influence weight)", ylabel="baseline spread (% of graph)",
           title="Stage 3 calibration grid: reach vs beta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(STAGE3_OUT / "grid_search.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage 3: calibrate alpha, beta, gamma.")
    p.add_argument("--coarse-runs", type=int, default=1)
    p.add_argument("--confirm-runs", type=int, default=5)
    args = p.parse_args()
    run(n_runs_coarse=args.coarse_runs, n_runs_confirm=args.confirm_runs)
