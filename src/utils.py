"""
Shared utilities: deterministic RNG derivation and small IO helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config


# Namespace tags keep the experiment-level (held-fixed) streams disjoint from
# the per-run stream within one master-seed root.
_NS_CREDULITY = 8001   # global: credulity, fixed across ALL runs of ALL conditions
_NS_SEEDS = 8002       # per seed regime: the k=10 initial-I node set


def make_credulity_rng(master_seed: int = config.MASTER_SEED) -> np.random.Generator:
    """RNG for credulity, drawn once at experiment start.

    Keyed only on the master seed (no condition or regime), so credulity is a
    single fixed draw held constant across all runs of all conditions and is not
    a source of run-to-run variation.
    """
    return np.random.default_rng(np.random.SeedSequence(entropy=[master_seed, _NS_CREDULITY]))


def make_seeds_rng(seed_regime: int,
                   master_seed: int = config.MASTER_SEED) -> np.random.Generator:
    """RNG for the initial-I seed set, drawn once per seed regime.

    The selected k=10 nodes are reused for all runs of all conditions within the
    regime.
    """
    return np.random.default_rng(
        np.random.SeedSequence(entropy=[master_seed, _NS_SEEDS, int(seed_regime)]))


def make_rng(condition_id: int, run_index: int,
             master_seed: int = config.MASTER_SEED) -> np.random.Generator:
    """Per-run RNG for run-to-run randomness, keyed on (condition_id, run_index).

    Run-to-run variation comes only from threshold draws (theta ~ Beta(2,5),
    re-sampled each run) and recovery events; intervention-internal stochastics
    also consume this stream. Credulity and the seed set are not drawn here;
    they are experiment-level held-fixed factors (see make_credulity_rng /
    make_seeds_rng). The same
    (condition_id, run_index) reproduces an identical run on any machine.
    """
    ss = np.random.SeedSequence(entropy=[master_seed, int(condition_id), int(run_index)])
    return np.random.default_rng(ss)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")
