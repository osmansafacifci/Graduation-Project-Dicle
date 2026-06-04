#!/usr/bin/env python3
"""Paper Figure F4 — MMD / Mahalanobis reduction vs cross-dataset R² change.

The key visual for the "covariate alignment is not concept alignment" claim:
plot, for each of the 12 cross-dataset transfer directions, the geometric
shift reduction (raw → capnorm) on the x-axis against the prediction R²
change (raw → capnorm) on the y-axis. Points scattered around y=0 despite
large x-axis values show that geometric repair does not buy prediction
repair — the central paper-defining finding.

Inputs:
    data/intermediate/four_dataset_geometric_shift_raw_summary.csv
    data/intermediate/four_dataset_geometric_shift_capnorm_summary.csv
    outputs/results_v2_four_dataset_cross_34feat_raw_log/results_summary.csv
    outputs/results_v2_four_dataset_cross_34feat_capnorm_log/results_summary.csv
    data/intermediate/four_dataset_conditional_shift_summary.json

Outputs:
    outputs/paper_figures/paper_covariate_vs_concept.png
    outputs/paper_figures/paper_covariate_vs_concept.pdf

Usage:
    python 3_analysis/plot_covariate_vs_concept_scatter.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_science_style

ROOT = Path(__file__).resolve().parents[1]

# Midnight Executive palette (matches PROJECT slide deck for visual continuity)
NAVY = "#1E2761"
ICE_BLUE = "#CADCFC"
GOLD = "#F5B700"
TEXT_DARK = "#1A1A2E"
TEXT_MUTED = "#5A5C7A"
GREEN = "#1B5E20"
ORANGE = "#B26500"
RED = "#B00020"
GREY = "#8888AA"

# Regime → color map
REGIME_COLOR = {
    "strong_rank_signal": GREEN,
    "moderate_rank_signal": ORANGE,
    "weak_rank_signal": ORANGE,
    "offset_dominant_repair": ORANGE,
    "rank_signal_collapsed": RED,
    "negative_or_inverted_signal": RED,
}
REGIME_LABEL = {
    "strong_rank_signal": "Strong rank signal",
    "moderate_rank_signal": "Moderate / weak rank signal",
    "weak_rank_signal": "Moderate / weak rank signal",
    "offset_dominant_repair": "Moderate / weak rank signal",
    "rank_signal_collapsed": "Rank collapsed",
    "negative_or_inverted_signal": "Rank collapsed",
}
LEGEND_ORDER = ["Strong rank signal", "Moderate / weak rank signal", "Rank collapsed"]
LEGEND_COLOR = {"Strong rank signal": GREEN, "Moderate / weak rank signal": ORANGE, "Rank collapsed": RED}
LEGEND_MARKER = {"Strong rank signal": "o", "Moderate / weak rank signal": "s", "Rank collapsed": "^"}


def load_data():
    """Return one DataFrame with all 12 directions and their (ΔMahalanobis, ΔR², regime, pair)."""

    shift_raw = pd.read_csv(ROOT / "data/intermediate/four_dataset_geometric_shift_raw_summary.csv")
    shift_cap = pd.read_csv(ROOT / "data/intermediate/four_dataset_geometric_shift_capnorm_summary.csv")
    cross_raw = pd.read_csv(ROOT / "outputs/results_v2_four_dataset_cross_34feat_raw_log/results_summary.csv")
    cross_cap = pd.read_csv(ROOT / "outputs/results_v2_four_dataset_cross_34feat_capnorm_log/results_summary.csv")
    regimes = json.loads((ROOT / "data/intermediate/four_dataset_conditional_shift_summary.json").read_text())

    # Pair-level Mahalanobis at N=100, 34 features
    def pair_maha(df):
        sub = df[(df["n_cycles"] == 100) & (df["feature_set"] == 34)]
        return {row["pair"]: float(row["Mahalanobis"]) for _, row in sub.iterrows()}, \
               {row["pair"]: float(row["MMD"]) for _, row in sub.iterrows()}

    raw_maha, raw_mmd = pair_maha(shift_raw)
    cap_maha, cap_mmd = pair_maha(shift_cap)

    # Best R² per direction at N=100
    def best_r2_per_direction(df):
        out = {}
        for exp, grp in df[df["n_cycles"] == 100].groupby("experiment"):
            best = grp.sort_values("R2_mean", ascending=False).iloc[0]
            out[exp] = (float(best["R2_mean"]), str(best["model"]))
        return out

    raw_r2 = best_r2_per_direction(cross_raw)
    cap_r2 = best_r2_per_direction(cross_cap)

    rows = []
    for entry in regimes["direction_summary"]:
        direction = entry["experiment"]
        pair = entry["pair"]
        regime = entry["rank_signal_class"]
        raw_mah_val = raw_maha[pair]
        cap_mah_val = cap_maha[pair]
        raw_mmd_val = raw_mmd[pair]
        cap_mmd_val = cap_mmd[pair]
        raw_r2_val, _ = raw_r2[direction]
        cap_r2_val, _ = cap_r2[direction]
        rows.append({
            "direction": direction,
            "pair": pair,
            "regime": regime,
            "delta_mahalanobis": raw_mah_val - cap_mah_val,       # > 0 means capnorm reduces
            "pct_mahalanobis_reduction": 100.0 * (raw_mah_val - cap_mah_val) / max(raw_mah_val, 1e-9),
            "delta_mmd": raw_mmd_val - cap_mmd_val,
            "pct_mmd_reduction": 100.0 * (raw_mmd_val - cap_mmd_val) / max(raw_mmd_val, 1e-9),
            "raw_R2": raw_r2_val,
            "cap_R2": cap_r2_val,
            "delta_R2": cap_r2_val - raw_r2_val,
        })
    return pd.DataFrame(rows)


def short_label(direction: str) -> str:
    # e.g., "matr_to_hust" → "MATR→HUST"
    src, _, tgt = direction.partition("_to_")
    return f"{src.upper()}→{tgt.upper()}"


LABEL_OFFSETS = {
    # Hand-tuned offsets in *points* to avoid label collisions; (dx, dy, ha, va)
    "hust_to_luh":   (10,  6, "left",  "bottom"),
    "luh_to_hust":   (10, -3, "left",  "center"),
    "hust_to_matr":  (10, -10, "left", "top"),
    "matr_to_hust":  (10,  10, "left", "bottom"),
    "matr_to_luh":   (-10, 10, "right", "bottom"),
    "luh_to_matr":   (-10, -10, "right", "top"),
    "luh_to_sandia": (-10, 8, "right", "bottom"),
    "sandia_to_luh": (-10, -10, "right", "top"),
    "hust_to_sandia": (10, 8, "left", "bottom"),
    "sandia_to_hust": (10, -10, "left", "top"),
    "matr_to_sandia": (10, 8, "left", "bottom"),
    "sandia_to_matr": (10, -8, "left", "top"),
}


def plot_figure(df: pd.DataFrame, out_path: Path):
    apply_science_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.8), dpi=200)

    # Background bands: tolerance strip around ΔR²=0 highlights the
    # "geometric reduction does not translate into R² gain" finding.
    ax.axhspan(-0.5, 0.5, color="#FFF4D6", alpha=0.55, zorder=0)
    ax.axhline(0, color=TEXT_MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.axvline(0, color=TEXT_MUTED, linewidth=0.5, linestyle=":", zorder=1)

    # Plot the 12 directions
    for _, row in df.iterrows():
        regime_label = REGIME_LABEL.get(row["regime"], "Rank collapsed")
        c = LEGEND_COLOR[regime_label]
        marker = LEGEND_MARKER[regime_label]
        ax.scatter(row["pct_mahalanobis_reduction"], row["delta_R2"],
                   s=170, c=c, edgecolor="white", linewidth=1.2,
                   marker=marker, alpha=0.95, zorder=3)

    # Highlight the MATR↔HUST pair — the original key-finding directions
    matr_hust = df[df["pair"] == "matr_vs_hust"]
    for _, row in matr_hust.iterrows():
        ax.scatter(row["pct_mahalanobis_reduction"], row["delta_R2"],
                   s=310, facecolors="none", edgecolor=NAVY, linewidth=2.0,
                   zorder=4)

    # Now place labels with hand-tuned offsets
    for _, row in df.iterrows():
        dx, dy, ha, va = LABEL_OFFSETS.get(row["direction"], (8, 4, "left", "bottom"))
        ax.annotate(
            short_label(row["direction"]),
            (row["pct_mahalanobis_reduction"], row["delta_R2"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8.5, color=TEXT_DARK, ha=ha, va=va,
        )

    # Tolerance-band label, placed safely outside data area
    ax.text(
        103, 0.0, "no R² change band",
        fontsize=8.5, color="#A87B00", va="center", ha="right",
        style="italic", alpha=0.95,
    )

    ax.set_xlabel("Mahalanobis reduction (%): raw → capnorm  (per pair)",
                  fontsize=11.5, color=TEXT_DARK)
    ax.set_ylabel("Δ R²:  capnorm − raw  (per direction)",
                  fontsize=11.5, color=TEXT_DARK)
    ax.set_title("Covariate alignment is not concept alignment",
                 fontsize=14, color=NAVY, fontweight="bold", pad=14)

    # Subtitle / second-line explanation
    ax.text(
        0.5, 1.02,
        "Geometric repair does not translate into a monotone prediction gain across the 12 cross-dataset directions.",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9.5, color=TEXT_MUTED, style="italic",
    )

    # Custom legend (3 regimes + MATR↔HUST marker)
    legend_handles = []
    for lbl in LEGEND_ORDER:
        legend_handles.append(plt.Line2D(
            [0], [0], marker=LEGEND_MARKER[lbl], color="w", markerfacecolor=LEGEND_COLOR[lbl],
            markeredgecolor="white", markersize=11, label=lbl,
        ))
    legend_handles.append(plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="white",
        markeredgecolor=NAVY, markeredgewidth=2.0, markersize=13,
        label="MATR↔HUST (original key-finding pair)",
    ))
    ax.legend(
        handles=legend_handles, loc="lower left",
        frameon=True, fontsize=9,
        edgecolor=TEXT_MUTED, fancybox=False, framealpha=0.96,
    )

    # Cosmetics
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TEXT_MUTED)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=TEXT_DARK, labelsize=9.5)
    ax.grid(axis="y", linewidth=0.3, color=TEXT_MUTED, alpha=0.35)

    # Axis limits
    xmax = 110
    xmin = min(df["pct_mahalanobis_reduction"].min(), -5) - 5
    ax.set_xlim(xmin, xmax)
    ymax = max(df["delta_R2"].max(), 0.8) + 0.7
    ymin = min(df["delta_R2"].min(), -1.6) - 0.5
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path.with_suffix('.png')}")
    print(f"[save] {out_path.with_suffix('.pdf')}")


def main():
    df = load_data()
    print("\nPer-direction summary:")
    print(df[["direction", "pair", "regime",
              "pct_mahalanobis_reduction", "raw_R2", "cap_R2", "delta_R2"]].to_string(index=False))
    out_path = ROOT / "outputs/paper_figures/paper_covariate_vs_concept"
    plot_figure(df, out_path)
    # Also save the per-direction CSV alongside, so the paper can cite the
    # underlying numbers without re-running this script.
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"[save] {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
