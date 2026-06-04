"""
Pre-regime screen — leakage-aware sensitivity for the rank-signal super-regime
taxonomy of §3.4.

Question: can the super-regime of a cross-dataset transfer direction be anticipated
*before* fitting a target adapter, from committed source/target shift descriptors?

Design (per Reviewer 2 redesign, July 2026):
  * 12 directions across 6 unordered dataset pairs.
  * Predictors are SHIFT descriptors only. Pearson r, naive R², and linear-cal R²
    are excluded — they DEFINE the super-regime.
  * Two screens:
      (1) covariate-only:   MMD, Mahalanobis, discriminator_auc_mean
                            (all pair-symmetric)
      (2) target-light:     covariate-only ∪
                            {life_ratio_target_over_source,
                             slope_shifted_share}
                            (life_ratio is the only direction-asymmetric scalar.)
  * Two models per screen:
      - Nearest centroid (primary; transparent, no tuning knobs).
      - Multinomial logistic regression (sensitivity; L2, C=1.0,
        class_weight="balanced").
  * Cross-validation: leave-one-unordered-pair-out (6 folds, 2 test rows per
    fold). LOPO avoids the symmetric-feature leakage that LODO would introduce.
  * Standardisation: z-score using train-fold statistics only.
  * Accuracy CI: Clopper-Pearson exact 95 % interval (n = 12 is small).

OUTPUTS:
  data/intermediate/pre_regime_screen_lopo_predictions.csv      (48 rows)
  data/intermediate/pre_regime_screen_summary.md                (narrative)
  outputs/results_v2_pre_regime_screen/pre_regime_screen_confusion.png

Caveat: n = 12 directions across only 6 unordered pairs. This is an exploratory
leakage-aware sensitivity, NOT a powered predictive classifier.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
INTERMEDIATE = REPO / "data" / "intermediate"
OUTDIR = REPO / "outputs" / "results_v2_pre_regime_screen"

# ----- super-regime mapping (matches §3.4 frozen-threshold YAML) -----
RANK_TO_SUPER = {
    "strong_rank_signal":         "salvageable_linear_recovers",
    "moderate_rank_signal":       "salvageable_linear_recovers",
    "weak_rank_signal":           "offset_dominant_residual_only",
    "rank_signal_collapsed":      "cp_interval_only",
    "negative_or_inverted_signal": "cp_interval_only",
}

SUPER_REGIMES = [
    "salvageable_linear_recovers",
    "offset_dominant_residual_only",
    "cp_interval_only",
]

COVARIATE_ONLY = ["MMD", "Mahalanobis", "discriminator_auc_mean"]
TARGET_LIGHT_EXTRA = ["life_ratio_target_over_source", "slope_shifted_share"]


# ===== data assembly =====
def assemble_direction_table() -> pd.DataFrame:
    """Merge the geometric (pair-level) and conditional-shift (direction-level)
    CSVs into a 12-row table indexed by (source, target)."""
    geo = pd.read_csv(
        INTERMEDIATE / "four_dataset_geometric_shift_capnorm_summary.csv"
    )
    geo = geo[(geo["n_cycles"] == 100) & (geo["feature_set"] == 34)].copy()
    if len(geo) != 6:
        raise ValueError(
            f"Expected 6 pair rows after n_cycles=100 & feature_set=34 filter, "
            f"got {len(geo)}"
        )
    geo = geo[["pair", "MMD", "Mahalanobis", "discriminator_auc_mean"]]

    cond = pd.read_csv(
        INTERMEDIATE / "four_dataset_conditional_shift_direction_summary.csv"
    )
    if len(cond) != 12:
        raise ValueError(
            f"Expected 12 direction rows, got {len(cond)}"
        )
    cond = cond[
        [
            "source", "target", "pair",
            "life_ratio_target_over_source",
            "slope_shifted_share",
            "rank_signal_class",
            "pearson_r",
        ]
    ]

    df = cond.merge(geo, on="pair", how="left", validate="m:1")
    df["super_regime"] = df["rank_signal_class"].map(RANK_TO_SUPER)
    if df["super_regime"].isna().any():
        missing = df.loc[df["super_regime"].isna(), "rank_signal_class"].unique()
        raise ValueError(f"Unmapped rank_signal_class values: {missing}")
    df["direction"] = df["source"] + "->" + df["target"]
    return df.reset_index(drop=True)


# ===== CV =====
def leave_one_pair_out_folds(pairs: Sequence[str]) -> List[Tuple[List[int], List[int]]]:
    """6 folds, each holds out both directions of one unordered pair."""
    pair_arr = np.asarray(pairs)
    unique_pairs = np.unique(pair_arr)
    folds = []
    for p in unique_pairs:
        test_idx = list(np.where(pair_arr == p)[0])
        train_idx = list(np.where(pair_arr != p)[0])
        folds.append((train_idx, test_idx))
    return folds


# ===== models =====
@dataclass(frozen=True)
class ScreenConfig:
    name: str
    features: Tuple[str, ...]


def fit_and_predict(
    model_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Train / predict for a fold. Z-scores using train-fold statistics only."""
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    if model_name == "nearest_centroid":
        clf = NearestCentroid()
    elif model_name == "logistic_l2":
        clf = LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=2000,
        )
    else:
        raise ValueError(model_name)
    clf.fit(X_train_s, y_train)
    return clf.predict(X_test_s)


def run_screen(df: pd.DataFrame, screen: ScreenConfig, model_name: str) -> pd.DataFrame:
    folds = leave_one_pair_out_folds(df["pair"].tolist())
    records = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X = df[list(screen.features)].to_numpy()
        y = df["super_regime"].to_numpy()
        held_pair = df.iloc[test_idx]["pair"].iloc[0]
        y_pred = fit_and_predict(
            model_name, X[train_idx], y[train_idx], X[test_idx]
        )
        for j, idx in enumerate(test_idx):
            records.append({
                "fold": fold_idx,
                "held_out_pair": held_pair,
                "screen": screen.name,
                "model": model_name,
                "direction": df.iloc[idx]["direction"],
                "pair": df.iloc[idx]["pair"],
                "true_super_regime": df.iloc[idx]["super_regime"],
                "predicted_super_regime": y_pred[j],
                "agree": bool(y_pred[j] == df.iloc[idx]["super_regime"]),
            })
    return pd.DataFrame.from_records(records)


# ===== accuracy + Clopper-Pearson CI =====
def clopper_pearson_ci(k: int, n: int, level: float = 0.95) -> Tuple[float, float]:
    """Exact binomial (Clopper-Pearson) two-sided CI for a binomial proportion."""
    alpha = 1.0 - level
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def summarise(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (screen, model), grp in predictions.groupby(["screen", "model"], sort=False):
        k = int(grp["agree"].sum())
        n = int(len(grp))
        lo, hi = clopper_pearson_ci(k, n)
        rows.append({
            "screen": screen,
            "model": model,
            "n": n,
            "correct": k,
            "accuracy": k / n,
            "ci95_lo": lo,
            "ci95_hi": hi,
        })
    return pd.DataFrame.from_records(rows)


# ===== confusion matrix figure =====
def confusion_panel(predictions: pd.DataFrame, out_path: Path) -> None:
    combos: List[Tuple[str, str]] = (
        predictions[["screen", "model"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    combos = list(combos)
    n_panels = len(combos)
    ncols = 2
    nrows = (n_panels + 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows), squeeze=False
    )
    short_label = {
        "salvageable_linear_recovers":  "salvageable",
        "offset_dominant_residual_only": "offset-dom",
        "cp_interval_only":             "cp-only",
    }
    labels_full = SUPER_REGIMES
    labels_short = [short_label[s] for s in labels_full]

    for (screen, model), ax in zip(combos, axes.flat):
        sub = predictions[
            (predictions["screen"] == screen) & (predictions["model"] == model)
        ]
        cm = np.zeros((3, 3), dtype=int)
        for _, row in sub.iterrows():
            i = labels_full.index(row["true_super_regime"])
            j = labels_full.index(row["predicted_super_regime"])
            cm[i, j] += 1
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels_short, rotation=30, ha="right")
        ax.set_yticklabels(labels_short)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        acc = (cm.trace()) / cm.sum() if cm.sum() else float("nan")
        ax.set_title(
            f"{screen}  ·  {model}\nLOPO acc = {acc:.2f}  ({cm.trace()} / {cm.sum()})",
            fontsize=10,
        )
        for i in range(3):
            for j in range(3):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
    # blank any unused panels
    for k in range(len(combos), nrows * ncols):
        axes.flat[k].axis("off")
    fig.suptitle(
        "Pre-regime screen — LOPO confusion matrices (exploratory; n = 12 directions)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ===== markdown summary =====
def write_markdown_summary(
    df: pd.DataFrame, predictions: pd.DataFrame, summary: pd.DataFrame, out_path: Path
) -> None:
    lines = [
        "# Pre-regime sensitivity screen (exploratory)",
        "",
        "Leakage-aware sensitivity check for §3.4 super-regime taxonomy.",
        "",
        "**Design.** 12 cross-dataset directions across 6 unordered pairs. "
        "Two screens (covariate-only and target-light), two models per screen "
        "(nearest centroid as primary, multinomial logistic as sensitivity). "
        "Cross-validation is leave-one-unordered-pair-out (6 folds × 2 test "
        "rows). Predictors are shift descriptors only; Pearson *r*, naive "
        "R², and linear-cal R² are excluded because they define the regime.",
        "",
        "## Caveats",
        "",
        "- *n* = 12 directions across only **6 unordered pairs**. This is "
        "  exploratory; we do not present it as a powered classifier.",
        "- All three covariate-only descriptors are **pair-symmetric** "
        "  (MMD, Mahalanobis, discriminator AUC are unordered functions of "
        "  the two datasets), so the covariate-only screen is mathematically "
        "  incapable of producing different predictions for the two "
        "  directions of an asymmetric pair.",
        "- `life_ratio_target_over_source` is the only direction-asymmetric "
        "  scalar; `slope_shifted_share` is pair-symmetric.",
        "- Accuracy CIs are Clopper-Pearson exact binomial intervals.",
        "",
        "## LOPO accuracy",
        "",
        "| Screen | Model | n | Correct | Accuracy | 95 % CI (Clopper-Pearson) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['screen']} | {r['model']} | {r['n']} | {r['correct']} | "
            f"{r['accuracy']:.3f} | [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}] |"
        )
    lines += [
        "",
        "## Pair symmetry sanity check",
        "",
        "(true labels per direction; reveals 2 symmetric + 4 asymmetric pairs)",
        "",
    ]
    by_pair = df.sort_values(["pair", "direction"])
    for pair, grp in by_pair.groupby("pair"):
        labels = grp["super_regime"].tolist()
        symm = "symmetric" if len(set(labels)) == 1 else "asymmetric"
        directions = grp["direction"].tolist()
        lines.append(f"- **{pair}** ({symm}): " + ", ".join(
            f"{d} → {l}" for d, l in zip(directions, labels)
        ))
    lines += [
        "",
        "## Spearman correlations of individual descriptors with Pearson r",
        "",
        "Classifier-free sensitivity (n = 12 directions). Tests whether each "
        "descriptor is monotonically related to the rank signal.",
        "",
        "| Descriptor | Spearman ρ | p |",
        "|---|---:|---:|",
    ]
    from scipy.stats import spearmanr
    for col in [
        "MMD", "Mahalanobis", "discriminator_auc_mean",
        "life_ratio_target_over_source", "slope_shifted_share",
    ]:
        rho, pv = spearmanr(df[col], df["pearson_r"])
        lines.append(f"| `{col}` | {rho:+.3f} | {pv:.3f} |")
    lines += [
        "",
        "## All LOPO predictions",
        "",
        "Full per-fold predictions are in "
        "`data/intermediate/pre_regime_screen_lopo_predictions.csv`.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# ===== orchestration =====
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = assemble_direction_table()
    logger.info("Assembled %d directions across %d unordered pairs.",
                len(df), df["pair"].nunique())

    screens = [
        ScreenConfig(name="covariate_only",
                     features=tuple(COVARIATE_ONLY)),
        ScreenConfig(name="target_light",
                     features=tuple(COVARIATE_ONLY + TARGET_LIGHT_EXTRA)),
    ]
    model_names = ["nearest_centroid", "logistic_l2"]

    all_preds = []
    for screen in screens:
        for model_name in model_names:
            all_preds.append(run_screen(df, screen, model_name))
    predictions = pd.concat(all_preds, ignore_index=True)

    summary = summarise(predictions)
    logger.info("\n%s", summary.to_string(index=False))

    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    pred_csv = INTERMEDIATE / "pre_regime_screen_lopo_predictions.csv"
    predictions.to_csv(pred_csv, index=False)
    logger.info("Wrote %s (%d rows)", pred_csv, len(predictions))

    md_path = INTERMEDIATE / "pre_regime_screen_summary.md"
    write_markdown_summary(df, predictions, summary, md_path)
    logger.info("Wrote %s", md_path)

    png_path = OUTDIR / "pre_regime_screen_confusion.png"
    confusion_panel(predictions, png_path)
    logger.info("Wrote %s", png_path)


if __name__ == "__main__":
    main()
