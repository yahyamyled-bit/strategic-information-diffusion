# Data

The processed Higgs graph lives in `processed/` and ships with the repo, so the
pipeline runs out of the box. Raw downloads and the other corpora are not
redistributed here -- pull them from source:

- **Higgs Twitter** -- https://snap.stanford.edu/data/higgs-twitter.html.
  Rebuild `processed/` from the raw edge list with `python -m src.stage1_network`.
- **UPFD (PolitiFact / GossipCop)** -- fetched automatically by
  `torch_geometric.datasets.UPFD` on first run.
- **Twitter15/16** -- Ma et al. rumor-detection propagation trees.
- **Twitter Information Operations Archive** -- used only to set empirical
  parameter ranges, not consumed directly by the code.

`src/config.py` resolves these paths under `data/`.
