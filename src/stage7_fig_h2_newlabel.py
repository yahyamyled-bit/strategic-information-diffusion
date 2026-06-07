"""Regenerate fig_h2_targeting_comparison.png using the corrected-label (s_higgs)
GNN containment runs (matched_intensity_newlabel_gnn_{gcn,sage,gat}.json), in
place of the original-label _tgt_gnn_* inputs.

Overrides the GNN source filenames on the imported figure module, then calls the
existing plotting function; the centrality columns (Degree / PageRank /
Betweenness) are unchanged. The base figure script is not modified.

Run from project root:
    python -m src.stage7_fig_h2_newlabel
"""
from __future__ import annotations

from . import _chapter3_figures as cf

# point the GNN columns at the corrected-label runs
cf.TARGETING_METHODS = [
    ("Degree",      "matched_intensity.json"),
    ("PageRank",    "matched_intensity_tgt_pagerank.json"),
    ("Betweenness", "matched_intensity_tgt_betweenness.json"),
    ("GCN",         "matched_intensity_newlabel_gnn_gcn.json"),
    ("GraphSAGE",   "matched_intensity_newlabel_gnn_sage.json"),
    ("GAT",         "matched_intensity_newlabel_gnn_gat.json"),
]


def main() -> None:
    out = cf.figure_targeting_comparison()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
