"""
Central configuration for the misinformation-diffusion simulation pipeline.

All paths, default parameters, and the master RNG seed live here so that every
stage reads from a single source of truth. Values that the methodology chapter
fixes (rho, T_max, k seeds, Beta threshold shape, eta defaults) are recorded
verbatim; values that Stage 3 calibrates (alpha, beta, gamma, attention,
credibility_risk) carry documented PROVISIONAL defaults until calibration runs.

Reference: the methodology (Chapter 2).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the project root, which is this file's grandparent)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STAGE1_OUT = OUTPUTS_DIR / "stage1"
STAGE2_OUT = OUTPUTS_DIR / "stage2"

for _d in (RAW_DIR, PROCESSED_DIR, STAGE1_OUT, STAGE2_OUT):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------
HIGGS_SOCIAL_URL = "https://snap.stanford.edu/data/higgs-social_network.edgelist.gz"
HIGGS_SOCIAL_GZ = RAW_DIR / "higgs-social_network.edgelist.gz"
HIGGS_RETWEET_URL = "https://snap.stanford.edu/data/higgs-activity_time.txt.gz"
HIGGS_RETWEET_GZ = RAW_DIR / "higgs-activity_time.txt.gz"

# Expected graph statistics (verification targets, from the spec)
EXPECTED_NODES = 456_626
EXPECTED_DIRECTED_EDGES = 14_855_842  # raw follower edges (~14.9M)

# Stage-1 processed artifacts
GRAPH_NPZ = PROCESSED_DIR / "higgs_adjacency.npz"          # CSR adjacency
NODE_ARRAYS_NPZ = PROCESSED_DIR / "higgs_node_arrays.npz"  # degree, influence, id map
GRAPH_STATS_JSON = STAGE1_OUT / "graph_stats.json"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# Single master seed; per-run seeds derive deterministically from
# (MASTER_SEED, condition_id, run_index) via numpy SeedSequence. See utils.py.
MASTER_SEED = 20260612  # thesis deadline, used as a fixed entropy root

# --------------------------------------------------------------------------
# SIRF model parameters (LOCKED by methodology chapter)
# --------------------------------------------------------------------------
RHO = 0.10              # recovery probability per step (I -> R)
T_MAX = 200             # maximum simulation steps
K_SEEDS = 10            # initial infected seeds
THETA_BETA_A = 2        # sharing-threshold Beta(a, b) shape
THETA_BETA_B = 5
ETA_DEFAULT = 0.10      # default F-state transmission factor
BFS_DEPTH_BOUND = 6     # depth-from-source bound (GNN feature, Stage 5)

# --------------------------------------------------------------------------
# Utility-function parameters (PROVISIONAL; Stage 3 calibrates from IO Archive)
# --------------------------------------------------------------------------
# U_v = ALPHA * c_v * attention_v + BETA * peer_influence_v
#       - GAMMA * (1 - c_v) * credibility_risk_v
#
# Calibrated operating point (Stage 3). Selected from the IO-Archive-bounded
# search box by the order-of-magnitude reach anchor: the baseline reaches
# ~15% (~68K nodes), the order of documented large real cascades (Vosoughi
# ~1e5), is sub-saturating, and non-degenerate in depth. All three weights
# contribute and the point is below the supercritical regime.
#
# Note on calibration choice: neither reach (38/64 triples accepted) nor
# cascade structure can discriminate the operating point -- the sim's depth/
# virality exceed real single-article retweet trees due to a model-class gap
# (multi-hop follower-graph contagion vs broadcast trees) that no parameter or
# recovery-rate setting closes. The reach-plausible point is fixed here and
# robustness is checked in Stage 6 (re-runs across points). See the Stage 3
# calibration output.
#
# NOTE: peer_influence is normalized by |N(v)| per chapter Eq. 2.3. attention_v
# and credibility_risk_v are homogeneous 1.0 (the chapter calibrates the weights,
# not per-node base values; see the methodology chapter).
ALPHA = 0.30
BETA = 0.80
GAMMA = 0.36
ATTENTION_DEFAULT = 1.0
CREDIBILITY_RISK_DEFAULT = 1.0

# --------------------------------------------------------------------------
# Baseline experiment (Stage 2)
# --------------------------------------------------------------------------
BASELINE_N_RUNS = 50            # runs per condition (spec: 50-100); 50 = feasible lower bound
BASELINE_CONDITION_ID = 0       # condition id reserved for the no-intervention baseline

# Seed regimes (Stage 6 secondary regime is structural-likelihood)
SEED_REGIME_RANDOM = 0          # primary: random uniform, k=10, without replacement
SEED_REGIME_STRUCTURAL = 1      # secondary: structural-likelihood (Stage 6)

# State encoding (int8)
S, I, R, F = 0, 1, 2, 3
STATE_NAMES = {S: "S", I: "I", R: "R", F: "F"}
