"""Chapter 3 combined figures generator.

Reads Stage 6 JSON outputs and produces three publication figures:

  1. fig_h2_targeting_comparison.png  -- 6 targeting methods x 5 interventions
  2. fig_fdur_sensitivity.png         -- CE vs F-state duration K
  3. fig_dose_response.png            -- CE vs target budget k

Run from project root:

    python -m src._chapter3_figures

Outputs are written to ``outputs/figures/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE6_DIR = PROJECT_ROOT / "outputs" / "stage6"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
INTERVENTIONS = [
    "node_removal",
    "fact_checking",
    "edge_rewiring",
    "visibility_reduction",
    "early_detection",
]

INTERVENTION_LABELS = {
    "node_removal": "Node removal",
    "fact_checking": "Fact-checking",
    "edge_rewiring": "Edge rewiring",
    "visibility_reduction": "Visibility reduction",
    "early_detection": "Early detection",
}

# Hand-picked colourblind-friendly palette (tab10 derivatives)
INTERVENTION_COLORS = {
    "node_removal":         "#1f77b4",  # blue
    "fact_checking":        "#2ca02c",  # green
    "edge_rewiring":        "#ff7f0e",  # orange
    "visibility_reduction": "#9467bd",  # purple
    "early_detection":      "#d62728",  # red
}

TARGETING_METHODS = [
    ("Degree",      "matched_intensity.json"),
    ("PageRank",    "matched_intensity_tgt_pagerank.json"),
    ("Betweenness", "matched_intensity_tgt_betweenness.json"),
    ("GCN",         "matched_intensity_tgt_gnn_gcn.json"),
    ("GraphSAGE",   "matched_intensity_tgt_gnn_sage.json"),
    ("GAT",         "matched_intensity_tgt_gnn_gat.json"),
]

# Bar palette for targeting methods (distinct from intervention colours)
TARGETING_COLORS = [
    "#4c72b0",  # Degree   -- blue
    "#dd8452",  # PageRank -- orange
    "#55a467",  # Between  -- green
    "#c44e52",  # GCN      -- red
    "#8172b3",  # SAGE     -- purple
    "#937860",  # GAT      -- brown
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_json(name: str) -> dict:
    """Load a Stage 6 JSON file by filename."""
    path = STAGE6_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ce_means(data: dict) -> dict:
    """Pull containment_efficiency.<intervention>.mean for the 5 interventions."""
    block = data["containment_efficiency"]
    return {iv: block[iv]["mean"] for iv in INTERVENTIONS}


def setup_axes(ax) -> None:
    """Common axis styling: subtle grid, clean spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------
# Figure 1 -- targeting comparison
# ----------------------------------------------------------------------
def figure_targeting_comparison() -> Path:
    # Collect CE means: rows = targeting methods, cols = interventions
    matrix = np.zeros((len(TARGETING_METHODS), len(INTERVENTIONS)))
    for i, (_, fname) in enumerate(TARGETING_METHODS):
        means = ce_means(load_json(fname))
        for j, iv in enumerate(INTERVENTIONS):
            matrix[i, j] = means[iv]

    n_iv = len(INTERVENTIONS)
    n_t = len(TARGETING_METHODS)
    bar_w = 0.13
    x = np.arange(n_iv)

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    setup_axes(ax)

    for i, (label, _) in enumerate(TARGETING_METHODS):
        offset = (i - (n_t - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            matrix[i],
            width=bar_w,
            label=label,
            color=TARGETING_COLORS[i],
            edgecolor="white",
            linewidth=0.4,
        )
        ax.bar_label(bars, fmt="%+.1f", fontsize=6, rotation=90, padding=2)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([INTERVENTION_LABELS[iv] for iv in INTERVENTIONS], fontsize=10)
    ax.set_ylabel("Containment efficiency (%)", fontsize=11)
    ax.set_title(
        "Containment efficiency by intervention x targeting method "
        "(k=500, n=50 per cell)",
        fontsize=12,
    )
    ax.legend(
        title="Targeting method",
        loc="lower left",
        fontsize=9,
        title_fontsize=10,
        frameon=False,
        ncol=2,
    )

    # headroom for the rotated value labels (deep-negative bars + tiny near-zero ones)
    ymin = matrix.min() - 12
    ymax = max(matrix.max() + 10, 40)
    ax.set_ylim(ymin, ymax)

    out = FIG_DIR / "fig_h2_targeting_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Figure 2 -- F-state duration sensitivity
# ----------------------------------------------------------------------
def figure_fdur_sensitivity() -> Path:
    # Sources: K=20, K=50, K=100, K=infinity (=> matched_intensity.json)
    sources = [
        (20,  "matched_intensity_fdur20.json"),
        (50,  "matched_intensity_fdur50.json"),
        (100, "matched_intensity_fdur100.json"),
        # K = infinity is represented numerically at the right edge
        (200, "matched_intensity.json"),
    ]
    ks = [k for k, _ in sources]
    series = {iv: [] for iv in INTERVENTIONS}
    for _, fname in sources:
        means = ce_means(load_json(fname))
        for iv in INTERVENTIONS:
            series[iv].append(means[iv])

    markers = {
        "node_removal":         "o",
        "fact_checking":        "s",
        "edge_rewiring":        "^",
        "visibility_reduction": "D",
        "early_detection":      "v",
    }

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    setup_axes(ax)

    for iv in INTERVENTIONS:
        ax.plot(
            ks,
            series[iv],
            marker=markers[iv],
            color=INTERVENTION_COLORS[iv],
            linewidth=2.0,
            markersize=7,
            label=INTERVENTION_LABELS[iv],
        )

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)

    # Replace the right-most numeric tick (200) with the infinity glyph
    ax.set_xticks(ks)
    ax.set_xticklabels(["20", "50", "100", "infinity"])

    ax.set_xlabel("F-state duration K (rounds before recovery)", fontsize=11)
    ax.set_ylabel("Containment efficiency (%)", fontsize=11)
    ax.set_title(
        "F-state duration sensitivity -- intervention ranking reorders at K <= 50",
        fontsize=12,
    )

    # Annotate the K=20 -> K=infinity crossover between fact_checking and node_removal
    nr_inf = series["node_removal"][-1]
    fc_inf = series["fact_checking"][-1]
    fc_20 = series["fact_checking"][0]
    nr_20 = series["node_removal"][0]

    # Headroom for annotations
    ymax_data = max(max(series[iv]) for iv in INTERVENTIONS)
    ymin_data = min(min(series[iv]) for iv in INTERVENTIONS)
    ax.set_ylim(ymin_data - 8, ymax_data + 28)

    # Top: NR overtakes FC at K=infinity (annotation placed above the NR endpoint)
    ax.annotate(
        f"NR #1 at K=infinity ({nr_inf:.1f}% vs FC {fc_inf:.1f}%)",
        xy=(200, nr_inf),
        xytext=(70, nr_inf + 22),
        fontsize=9,
        color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8),
    )
    # Bottom: FC overtakes NR at K=20 (annotation placed below the K=20 point)
    ax.annotate(
        f"FC #1 at K=20 ({fc_20:.1f}% vs NR {nr_20:.1f}%)",
        xy=(20, fc_20),
        xytext=(35, -20),
        fontsize=9,
        color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8),
    )

    ax.legend(loc="center right", fontsize=9, frameon=False)

    out = FIG_DIR / "fig_fdur_sensitivity.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Figure 3 -- dose response
# ----------------------------------------------------------------------
def figure_dose_response() -> Path:
    sources = [
        (100,  "matched_intensity_k100.json"),
        (250,  "matched_intensity_k250.json"),
        (500,  "matched_intensity.json"),
        (1000, "matched_intensity_k1000.json"),
    ]
    ks = [k for k, _ in sources]
    series = {iv: [] for iv in INTERVENTIONS}
    for _, fname in sources:
        means = ce_means(load_json(fname))
        for iv in INTERVENTIONS:
            series[iv].append(means[iv])

    markers = {
        "node_removal":         "o",
        "fact_checking":        "s",
        "edge_rewiring":        "^",
        "visibility_reduction": "D",
        "early_detection":      "v",
    }

    # Elasticity ratios (k=1000 / k=100), per Chapter 3 sec 3.3.7
    elasticities = {
        "node_removal":         "1.87x",
        "fact_checking":        "1.76x",
        "edge_rewiring":        "1.70x",
        "visibility_reduction": "1.60x",
    }

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    setup_axes(ax)

    for iv in INTERVENTIONS:
        ax.plot(
            ks,
            series[iv],
            marker=markers[iv],
            color=INTERVENTION_COLORS[iv],
            linewidth=2.0,
            markersize=7,
            label=INTERVENTION_LABELS[iv]
            + (f" (elasticity {elasticities[iv]})" if iv in elasticities else ""),
        )

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)

    ax.set_xlabel("Target budget k (log scale)", fontsize=11)
    ax.set_ylabel("Containment efficiency (%)", fontsize=11)
    ax.set_title(
        "Dose-response: containment efficiency vs target budget k",
        fontsize=12,
    )

    # Annotate the early-detection backfire amplification
    ed_100 = series["early_detection"][0]
    ed_1000 = series["early_detection"][-1]
    ax.annotate(
        f"Backfire amplification\n{ed_100:.0f}% -> {ed_1000:.0f}%",
        xy=(1000, ed_1000),
        xytext=(450, ed_1000 - 10),
        fontsize=9,
        color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
    )

    # Add a bit of headroom on the bottom so the annotation isn't clipped
    ymin_data = min(min(series[iv]) for iv in INTERVENTIONS)
    ymax_data = max(max(series[iv]) for iv in INTERVENTIONS)
    ax.set_ylim(ymin_data - 12, ymax_data + 8)

    ax.legend(loc="center left", fontsize=9, frameon=False)

    out = FIG_DIR / "fig_dose_response.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------
def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    for fn in (figure_targeting_comparison, figure_fdur_sensitivity, figure_dose_response):
        path = fn()
        size_kb = path.stat().st_size / 1024
        print(f"wrote {path.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
