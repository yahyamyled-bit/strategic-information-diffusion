# Strategic Information Diffusion: Simulating Fake News and Algorithmic Interventions

Agent-based simulation of misinformation spread on the Higgs Twitter follower
graph, with a matched-conditions comparison of five platform-level interventions
(node removal, fact-checking, edge rewiring, visibility reduction, early
detection). Companion code for an undergraduate thesis at Tunis Business School.

## Contents
- `src/` -- the simulation pipeline (six sequential stages) plus the GNN bridge and analysis scripts
- `outputs/` -- generated results: baseline curves, cascade-structure validation, intervention runs, the corrected-label GNN retraining (Stage 7), and figures
- `requirements.txt` -- pinned dependencies (Python 3.12)

## Running
From the project root:

    pip install -r requirements.txt
    python -m src.stage1_network
    python -m src.stage2_baseline
    python -m src.stage3_calibration
    python -m src.stage4_validation
    python -m src.stage5_gnn
    python -m src.stage6_experiment

The corrected-label GNN retraining (Stage 7) runs on top of the main pipeline:

    python -m src.stage7_inference_label       # realized cascade-contribution label (s_higgs) on Higgs
    python -m src.stage7_build_features_pivot   # node features with pivot-sampled betweenness
    python -m src.stage7_retrain_inference      # retrain GCN/GraphSAGE/GAT/MLP on the corrected label
    python -m src.stage7_fig_h2_newlabel        # regenerate the six-method H2 targeting figure

All stochastic operations descend from a single master seed (`src/config.py`),
so a given (condition, run) reproduces an identical run on any machine.

## Data
Datasets are not redistributed here; see `data/README.md` for sources.

## Model
Agents occupy Susceptible, Infected, Recovered, or Flagged states. A susceptible
agent shares when a utility threshold combining attention, peer influence, and
credibility risk is exceeded. The dynamic and parameters are described in the
thesis methodology chapter.

## Corrected-label GNN retraining (Stage 7)

The original GNN target -- descendant count on the UPFD propagation trees -- is a
near-deterministic function of node centrality on tree-structured data (Spearman
rho ~ 0.999 to degree and betweenness), so a model trained on it cannot test
whether learned targeting beats centrality. Stage 7 redefines the label as the
realized cascade contribution of each node, measured directly on the Higgs
inference graph over 100 diffusion runs (`s_higgs`), which decorrelates it from
centrality (rho ~ 0.33). All four learned models were retrained on this corrected
label; none out-ranks degree or PageRank, so hypothesis H2 remains rejected.
Stage 7 is an addition to, not a replacement of, the original Stage 5 results,
which are retained for transparency.
