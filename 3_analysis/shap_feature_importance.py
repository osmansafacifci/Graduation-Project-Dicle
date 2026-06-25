"""
SHAP feature attribution for the primary within-dataset battery-life models.

Scope is intentionally narrow and reproducible:

  - N = 100 by default, matching the headline protocol.
  - 34 detected feature columns, log-target fitting, train-only scaling.
  - Five official cell-level splits.
  - Primary paper models by within-dataset R2:
      MATR -> CatBoost
      HUST -> Random Forest
      Sandia -> XGBoost
      Luh -> CatBoost (best single-tree SHAP-compatible model; GP is champion)

SHAP values are computed in the fitted log-cycle prediction space. Reported
model metrics remain in original cycle units so they can be compared directly
with the main performance tables.

Outputs:
    data/intermediate/four_dataset_shap_feature_importance.csv
    data/intermediate/four_dataset_shap_feature_importance_detailed.csv
    data/intermediate/four_dataset_shap_feature_importance.json
    data/intermediate/four_dataset_shap_feature_importance_report.txt
    outputs/results_v2_four_dataset_shap/four_dataset_shap_feature_importance_top_features.png

Usage:
    python 3_analysis/shap_feature_importance.py
    python 3_analysis/shap_feature_importance.py --models catboost random_forest
    python 3_analysis/shap_feature_importance.py \
        --features-path data/intermediate/features_sop12_four_dataset_capnorm.csv \
        --splits-dir splits/sop_v2_four_dataset \
        --datasets matr hust sandia luh \
        --output-prefix four_dataset_shap_feature_importance \
        --output-dir outputs/results_v2_four_dataset_shap \
        --skip-transfer-stability \
        --conditional-shift-path data/intermediate/four_dataset_conditional_shift_feature_slopes.csv \
        --conditional-pairs sandia_vs_luh
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.preprocessing import StandardScaler

from plot_style import apply_science_style

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2_four_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_four_dataset_shap"

sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import META_COLS, SEEDS  # noqa: E402
from shared.battery_utils import (  # noqa: E402
    dataset_window,
    display_path as _display_path,
    load_split as _load_split,
)
from metrics_utils import compute_metrics, fit_with_threaded_joblib, to_cycles  # noqa: E402
import run_experiments as experiments  # noqa: E402
from run_experiments import (  # noqa: E402
    fit_catboost,
    fit_random_forest,
    fit_xgboost,
)

FEATURES_PATH = INTERMEDIATE_DIR / "features_sop12_four_dataset_capnorm.csv"
TRANSFER_PATH = INTERMEDIATE_DIR / "feature_transfer_stability.csv"

PRIMARY_MODEL_BY_DATASET = {
    "matr": "catboost",
    "hust": "random_forest",
    "sandia": "xgboost",
    "luh": "catboost",
}

DATASET_PLOT_ORDER = ["matr", "hust", "sandia", "luh"]
DATASET_LABEL = {
    "matr": "MATR",
    "hust": "HUST",
    "sandia": "Sandia",
    "luh": "Luh",
}
MODEL_LABEL = {
    "catboost": "CatBoost",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

MODEL_FITTERS = {
    "catboost": fit_catboost,
    "random_forest": fit_random_forest,
    "xgboost": fit_xgboost,
}


def display_path(path: Path) -> str:
    return _display_path(path, PROJECT_ROOT)


def load_split(dataset: str, seed: int, splits_dir: Path) -> dict:
    return _load_split(splits_dir, dataset, seed)


def model_list_for_dataset(requested: list[str], dataset: str) -> list[str]:
    """Expand ``"primary"`` to the per-dataset SHAP-compatible primary model.

    ``PRIMARY_MODEL_BY_DATASET`` maps each dataset to a tree-based primary
    (MATR→CatBoost, HUST→RandomForest, Sandia→XGBoost, Luh→CatBoost). Tree
    models are required because :func:`compute_tree_shap_values` uses
    :class:`shap.TreeExplainer`, which exactly attributes a tree-ensemble
    prediction whereas KernelSHAP / DeepSHAP would only approximate it.
    """
    if "primary" in requested:
        return [PRIMARY_MODEL_BY_DATASET[dataset]]
    return requested


def compute_tree_shap_values(model, X_test: np.ndarray) -> np.ndarray:
    """Exact TreeSHAP attributions for a fitted tree-ensemble model.

    Parameters
    ----------
    model : sklearn-like estimator
        Tree-based regressor with a SHAP-compatible interface
        (XGBoost / CatBoost / RandomForest).
    X_test : np.ndarray, shape (n_cells, n_features)
        Standardized test features.

    Returns
    -------
    np.ndarray, shape (n_cells, n_features)
        Per-cell × per-feature SHAP values in the model's log-cycle space
        (the same space the model was fit in).

    Notes
    -----
    ``check_additivity=False`` is passed because some boosters add a global
    bias term that fails the strict additivity assertion at floating-point
    precision; the resulting values are still exact attributions.
    The (n_cells, n_features, 1) return shape from some SHAP versions is
    squeezed to 2D for downstream code.
    """
    try:
        import shap
    except ImportError as exc:
        raise SystemExit("shap is not installed. Run `pip install shap` or install requirements.txt.") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_test, check_additivity=False)

    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got shape {values.shape}")
    return values


def benchmark_model_metrics(
    sub: pd.DataFrame,
    feature_cols: list[str],
    split: dict,
    *,
    n_cycles: int,
    seed: int,
    model_name: str,
) -> dict:
    """Reuse ``run_experiments.evaluate_split`` so SHAP metrics match the
    headline benchmark exactly.

    Avoids duplicating the benchmark code path: this helper routes the
    same (sub-frame, split, seed, model) through ``evaluate_split`` and
    returns the dict of point metrics. The point of going through the
    canonical helper is that the SHAP report's "Model check" R²/MAE row
    is then guaranteed to match the headline table within rounding
    (post-audit fix M2). Returns ``MAE``, ``SMAPE``, ``R2``, and the
    bootstrap 95% CI block.
    """
    experiments.SOP12_FEATURE_COLS = list(feature_cols)
    try:
        from joblib import parallel_backend
    except ImportError:
        logger.debug("joblib parallel_backend not available; falling back to sequential execution")
        parallel_backend = None

    if parallel_backend is None:
        result = experiments.evaluate_split(
            sub,
            split,
            n_cycles=n_cycles,
            models=[model_name],
            seed=seed,
            log_target=True,
            pca_variance=None,
        )
    else:
        with parallel_backend("threading"):
            result = experiments.evaluate_split(
                sub,
                split,
                n_cycles=n_cycles,
                models=[model_name],
                seed=seed,
                log_target=True,
                pca_variance=None,
            )
    model_metrics = result.get(model_name, {})
    return {
        "MAE": model_metrics.get("MAE"),
        "SMAPE": model_metrics.get("SMAPE"),
        "R2": model_metrics.get("R2"),
        "bootstrap_95_ci": model_metrics.get("bootstrap_95_ci"),
    }


def evaluate_and_explain(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    dataset: str,
    n_cycles: int,
    seed: int,
    model_name: str,
    splits_dir: Path,
) -> tuple[list[dict], dict]:
    """Fit a within-dataset model and emit per-feature SHAP attribution rows.

    For one (dataset, seed, n_cycles, model_name) combination:

    1. Slice the within-dataset split (train + test cells).
    2. Standardize features, fit on ``log(cycle_life)``.
    3. Compute exact TreeSHAP values on the test set.
    4. Aggregate to per-feature scalar attributions:
       ``mean_abs_shap``, ``mean_shap`` (signed), ``relative_importance``,
       and the within-split importance rank (1 = most important).
    5. Call :func:`benchmark_model_metrics` so the "Model check" row of
       the SHAP report matches the headline benchmark MAE/SMAPE/R².

    Returns ``(rows, info_dict)`` where ``rows`` is the per-feature SHAP
    record list and ``info_dict`` carries split-level metadata. Returns
    ``([], {skipped: True, …})`` when train < 5 or test < 2 cells.
    """
    split = load_split(dataset, seed, splits_dir)
    sub = dataset_window(df, dataset, n_cycles)
    train_df = sub[sub["cell_id"].isin(split["train"])].copy()
    test_df = sub[sub["cell_id"].isin(split["test"])].copy()

    if len(train_df) < 5 or len(test_df) < 2:
        return [], {
            "dataset": dataset,
            "model": model_name,
            "n_cycles": n_cycles,
            "seed": seed,
            "skipped": True,
            "reason": "too few cells",
            "train_cells": int(len(train_df)),
            "test_cells": int(len(test_df)),
        }

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["cycle_life"].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["cycle_life"].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    y_train_fit = np.log(y_train)

    fitter = MODEL_FITTERS[model_name]
    fitted = fit_with_threaded_joblib(fitter, X_train_s, y_train_fit, seed=seed)
    if isinstance(fitted, tuple):
        model, tuning = fitted
    else:
        model, tuning = fitted, {}

    pred_log = np.asarray(model.predict(X_test_s), dtype=float).ravel()
    pred = to_cycles(pred_log, log_target=True)
    attribution_metrics = compute_metrics(y_test, pred)
    metrics = benchmark_model_metrics(
        sub,
        feature_cols,
        split,
        n_cycles=n_cycles,
        seed=seed,
        model_name=model_name,
    )
    if metrics["MAE"] is None or metrics["SMAPE"] is None or metrics["R2"] is None:
        metrics = {
            "MAE": attribution_metrics["MAE"],
            "SMAPE": attribution_metrics["SMAPE"],
            "R2": attribution_metrics["R2"],
            "bootstrap_95_ci": None,
        }
        metric_source = "attribution_model_fallback"
    else:
        metric_source = "benchmark_evaluate_split"

    shap_values = compute_tree_shap_values(model, X_test_s)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_signed = np.mean(shap_values, axis=0)
    total_abs = float(np.sum(mean_abs))
    relative = mean_abs / total_abs if total_abs > 0 else np.zeros_like(mean_abs)
    order = np.argsort(-mean_abs)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)

    rows: list[dict] = []
    for i, feature in enumerate(feature_cols):
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "n_cycles": int(n_cycles),
                "seed": int(seed),
                "feature": feature,
                "mean_abs_shap_log_cycles": float(mean_abs[i]),
                "mean_shap_log_cycles": float(mean_signed[i]),
                "relative_importance": float(relative[i]),
                "rank": int(ranks[i]),
                "train_cells": int(len(train_df)),
                "test_cells": int(len(test_df)),
                "MAE": metrics["MAE"],
                "SMAPE": metrics["SMAPE"],
                "R2": metrics["R2"],
                "attribution_MAE": attribution_metrics["MAE"],
                "attribution_SMAPE": attribution_metrics["SMAPE"],
                "attribution_R2": attribution_metrics["R2"],
                "metric_source": metric_source,
            }
        )

    run_row = {
        "dataset": dataset,
        "model": model_name,
        "n_cycles": int(n_cycles),
        "seed": int(seed),
        "skipped": False,
        "train_cells": int(len(train_df)),
        "test_cells": int(len(test_df)),
        "MAE": metrics["MAE"],
        "SMAPE": metrics["SMAPE"],
        "R2": metrics["R2"],
        "bootstrap_95_ci": metrics.get("bootstrap_95_ci"),
        "attribution_MAE": attribution_metrics["MAE"],
        "attribution_SMAPE": attribution_metrics["SMAPE"],
        "attribution_R2": attribution_metrics["R2"],
        "metric_source": metric_source,
        "tuning": tuning,
    }
    return rows, run_row


def summarize_importance(detailed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed SHAP rows into a per-feature × per-(dataset, model) table.

    Computes:

    - mean / std of absolute SHAP values across seeds (importance with
      uncertainty)
    - mean signed SHAP (direction of feature contribution to log-cycle-life)
    - mean / std of relative importance (each cell normalized so a feature's
      share sums to 1 across all 34 features)
    - mean / std of the within-split rank (1 = most important)
    - ``top5_rate`` and ``top10_rate``: fraction of seeds where the feature
      lands in the top-5 / top-10 most important features
    - benchmark MAE / sMAPE / R² (one number per (dataset, model), repeated
      across feature rows for convenience)

    Output is sorted by (window, dataset, model, descending importance).
    Used to build the paper-facing SHAP × regime joined table.
    """
    grouped = detailed.groupby(["n_cycles", "dataset", "model", "feature"], as_index=False)
    summary = grouped.agg(
        mean_abs_shap_log_cycles_mean=("mean_abs_shap_log_cycles", "mean"),
        mean_abs_shap_log_cycles_std=("mean_abs_shap_log_cycles", "std"),
        mean_shap_log_cycles_mean=("mean_shap_log_cycles", "mean"),
        relative_importance_mean=("relative_importance", "mean"),
        relative_importance_std=("relative_importance", "std"),
        rank_mean=("rank", "mean"),
        rank_std=("rank", "std"),
        n_seed_runs=("seed", "nunique"),
        MAE_mean=("MAE", "mean"),
        SMAPE_mean=("SMAPE", "mean"),
        R2_mean=("R2", "mean"),
        attribution_MAE_mean=("attribution_MAE", "mean"),
        attribution_SMAPE_mean=("attribution_SMAPE", "mean"),
        attribution_R2_mean=("attribution_R2", "mean"),
    )

    top_rates = detailed.groupby(["n_cycles", "dataset", "model", "feature"])["rank"].agg(
        top5_rate=lambda s: float(np.mean(np.asarray(s) <= 5)),
        top10_rate=lambda s: float(np.mean(np.asarray(s) <= 10)),
    ).reset_index()
    summary = summary.merge(top_rates, on=["n_cycles", "dataset", "model", "feature"], how="left")
    summary = summary.sort_values(
        ["n_cycles", "dataset", "model", "mean_abs_shap_log_cycles_mean"],
        ascending=[True, True, True, False],
    )
    return summary


def join_transfer_stability(summary: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Left-join the SHAP summary table with the feature-transfer-stability table.

    Adds per-feature transfer diagnostics from the
    :file:`feature_transfer_stability.py` output: raw-vs-capnorm shift in
    pooled-z units, MATR/HUST Spearman ρ with cycle-life and their delta,
    sign agreement, within-dataset univariate R², residual-mean-adapted
    cross-dataset R², the composite stability score, and the categorical
    ``stability_class`` (scale_shift_fragile / relationship_unstable /
    stable_candidate / weak_or_mixed).

    Returns ``summary`` unchanged when the transfer file does not exist.
    """
    if not path.exists():
        return summary
    transfer = pd.read_csv(path)
    keep = [
        "feature",
        "n_cycles",
        "abs_mean_shift_z_raw",
        "abs_mean_shift_z_capnorm",
        "spearman_matr",
        "spearman_hust",
        "spearman_abs_delta",
        "spearman_sign_agree",
        "within_R2_mean",
        "adapted_cross_R2_mean",
        "transfer_stability_score",
        "stability_class",
    ]
    keep = [c for c in keep if c in transfer.columns]
    return summary.merge(transfer[keep], on=["feature", "n_cycles"], how="left")


def join_conditional_shift(summary: pd.DataFrame, path: Path | None, pairs: list[str]) -> pd.DataFrame:
    """Left-join the SHAP summary with per-pair conditional-shift slope labels.

    For each dataset pair in ``pairs`` (e.g., ``["matr_vs_hust",
    "sandia_vs_luh"]``), pulls the per-feature centred-log slope shift
    label (``slope_stable`` / ``slope_shifted``) and the universal
    log-life offset from the conditional-shift output, then attaches them
    to the SHAP summary with namespaced column names like
    ``matr_vs_hust_shift_class``.

    This is the join that produces the manuscript-facing claim "feature X
    is in the top-5 SHAP for MATR CatBoost AND is slope-shifted across
    MATR/HUST" — i.e., the within-domain model leans on a feature that
    does not carry consistent semantics across datasets.

    Returns ``summary`` unchanged when the path is missing or ``pairs`` is
    empty.
    """
    if path is None or not path.exists() or not pairs:
        return summary
    shifts = pd.read_csv(path)
    out = summary.copy()
    for pair in pairs:
        pair_rows = shifts[shifts["pair"].eq(pair)].copy()
        if pair_rows.empty:
            continue
        keep = [
            "feature",
            "shift_class",
            "delta_slope_b_minus_a",
            "delta_slope_fdr_bh",
            "log_life_offset_b_minus_a",
            "life_ratio_b_over_a",
        ]
        pair_rows = pair_rows[[c for c in keep if c in pair_rows.columns]].rename(
            columns={
                "shift_class": f"{pair}_shift_class",
                "delta_slope_b_minus_a": f"{pair}_delta_slope",
                "delta_slope_fdr_bh": f"{pair}_delta_slope_fdr_bh",
                "log_life_offset_b_minus_a": f"{pair}_log_life_offset",
                "life_ratio_b_over_a": f"{pair}_life_ratio",
            }
        )
        out = out.merge(pair_rows, on="feature", how="left")
    return out


def write_report(
    summary: pd.DataFrame,
    run_metrics: list[dict],
    out_path: Path,
    *,
    top_k: int,
    features_path: Path,
) -> None:
    """Render the human-readable SHAP report at ``out_path``.

    Sections written:

    1. Header (protocol, feature-table provenance, note that model-check
       metrics use the canonical benchmark helper).
    2. Per-(dataset, model) model-check MAE/SMAPE/R² aggregated across
       seeds — these *should* match the headline benchmark table.
    3. Top-``top_k`` SHAP features per (dataset, model) with their
       transfer-stability class and (where available) the conditional-
       shift slope class.
    4. "Important but transfer-fragile" subsection: features that are
       both top-``top_k`` SHAP *and* labelled fragile/unstable by the
       stability join — the manuscript's "covariate alignment is not
       concept alignment at the feature level" closing loop.

    The output is plain text rather than Markdown so it renders cleanly
    in CI logs and email summaries; the paper-facing version is produced
    by :file:`build_shap_regime_table.py` from the same CSVs.
    """
    lines: list[str] = []
    lines.append("SHAP feature attribution summary")
    lines.append("=" * 80)
    lines.append("Protocol: 34 features, N=100 default, log-target models, SHAP in log-cycle space.")
    lines.append(f"Feature table: {display_path(features_path)}")
    lines.append("Model-check metrics use the canonical benchmark evaluate_split() helper on the same feature table as the SHAP run.")
    lines.append("")

    metrics_df = pd.DataFrame([r for r in run_metrics if not r.get("skipped")])
    if not metrics_df.empty:
        lines.append("Model check across explained splits:")
        for _, row in (
            metrics_df.groupby(["n_cycles", "dataset", "model"], as_index=False)
            .agg(MAE=("MAE", "mean"), SMAPE=("SMAPE", "mean"), R2=("R2", "mean"), n_runs=("seed", "nunique"))
            .sort_values(["n_cycles", "dataset", "model"])
            .iterrows()
        ):
            lines.append(
                f"  N={int(row['n_cycles'])} {row['dataset'].upper()} {row['model']}: "
                f"MAE={row['MAE']:.1f}, sMAPE={row['SMAPE']:.1f}, R2={row['R2']:+.3f} "
                f"({int(row['n_runs'])} seeds)"
            )
        lines.append("")

    for (n_cycles, dataset, model), block in summary.groupby(["n_cycles", "dataset", "model"], sort=True):
        lines.append(f"N={int(n_cycles)} {dataset.upper()} {model} top SHAP features:")
        for _, row in block.head(top_k).iterrows():
            annotations = []
            shift = row.get("abs_mean_shift_z_raw", np.nan)
            if pd.notna(shift):
                annotations.append(f"shift_z={shift:.2f}")
            stability = row.get("stability_class", np.nan)
            if pd.notna(stability):
                annotations.append(f"class={stability}")
            for col in block.columns:
                if col.endswith("_shift_class") and pd.notna(row.get(col)):
                    annotations.append(f"{col.removesuffix('_shift_class')}={row[col]}")
            annotation_text = (" " + " ".join(annotations)) if annotations else ""
            lines.append(
                f"  {row['feature']:<24} rank={row['rank_mean']:.1f} "
                f"rel={100.0 * row['relative_importance_mean']:.1f}%"
                f"{annotation_text}"
            )
        lines.append("")

        fragile = block[
            block["stability_class"].isin(["scale_shift_fragile", "relationship_unstable"])
            & (block["rank_mean"] <= max(10, top_k))
        ] if "stability_class" in block.columns else block.iloc[0:0]
        if not fragile.empty:
            lines.append(f"N={int(n_cycles)} {dataset.upper()} {model} important but transfer-fragile:")
            for _, row in fragile.sort_values("rank_mean").head(top_k).iterrows():
                lines.append(
                    f"  {row['feature']:<24} rank={row['rank_mean']:.1f} "
                    f"shift_z={row['abs_mean_shift_z_raw']:.2f} "
                    f"rho(M,H)=({row['spearman_matr']:+.2f},{row['spearman_hust']:+.2f}) "
                    f"class={row['stability_class']}"
                )
            lines.append("")

    out_path.write_text("\n".join(lines))


def make_top_feature_plot(summary: pd.DataFrame, output_dir: Path, *, top_k: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    apply_science_style()
    grouped = dict(tuple(summary.groupby(["dataset", "model"], sort=False)))
    blocks = []
    for dataset in DATASET_PLOT_ORDER:
        model = PRIMARY_MODEL_BY_DATASET[dataset]
        key = (dataset, model)
        if key in grouped:
            blocks.append((key, grouped[key]))
    for key, block in grouped.items():
        if key not in {k for k, _ in blocks}:
            blocks.append((key, block))
    if not blocks:
        return

    ncols = 2 if len(blocks) > 2 else len(blocks)
    nrows = int(np.ceil(len(blocks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.0 * nrows), squeeze=False)
    for ax, ((dataset, model), block) in zip(axes.ravel(), blocks, strict=False):
        top = block.sort_values("mean_abs_shap_log_cycles_mean", ascending=True).tail(top_k)
        ax.barh(top["feature"], top["mean_abs_shap_log_cycles_mean"], color="#2E7D32")
        ax.set_title(f"{DATASET_LABEL.get(dataset, dataset.upper())} {MODEL_LABEL.get(model, model)}", fontsize=12)
        ax.set_xlabel(r"Mean $\left|\mathrm{SHAP}\right|$ (log-cycle units)")
        ax.grid(axis="x", alpha=0.25)
    for ax in axes.ravel()[len(blocks):]:
        ax.axis("off")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{make_top_feature_plot.filename}"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--transfer-path", type=Path, default=TRANSFER_PATH)
    parser.add_argument(
        "--skip-transfer-stability",
        action="store_true",
        help="Do not join the MATR/HUST feature_transfer_stability reference table.",
    )
    parser.add_argument(
        "--conditional-shift-path",
        type=Path,
        default=INTERMEDIATE_DIR / "four_dataset_conditional_shift_feature_slopes.csv",
    )
    parser.add_argument(
        "--conditional-pairs",
        nargs="+",
        default=[
            "hust_vs_luh",
            "hust_vs_sandia",
            "matr_vs_hust",
            "matr_vs_luh",
            "matr_vs_sandia",
            "sandia_vs_luh",
        ],
    )
    parser.add_argument("--output-prefix", default="four_dataset_shap_feature_importance")
    parser.add_argument("--windows", type=int, nargs="+", default=[100])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["matr", "hust", "sandia", "luh"],
        choices=["matr", "hust", "sandia", "luh"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["primary"],
        choices=["primary", "catboost", "random_forest", "xgboost"],
        help="'primary' runs the configured TreeSHAP-compatible model for each dataset.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = args.features_path if args.features_path.is_absolute() else PROJECT_ROOT / args.features_path
    splits_dir = args.splits_dir if args.splits_dir.is_absolute() else PROJECT_ROOT / args.splits_dir
    transfer_path = args.transfer_path if args.transfer_path.is_absolute() else PROJECT_ROOT / args.transfer_path
    conditional_shift_path = (
        args.conditional_shift_path if args.conditional_shift_path is None or args.conditional_shift_path.is_absolute()
        else PROJECT_ROOT / args.conditional_shift_path
    )
    if not features_path.exists():
        print(f"[error] missing {features_path}")
        return 1
    if not splits_dir.exists():
        print(f"[error] missing {splits_dir}")
        return 1

    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c not in META_COLS]

    make_top_feature_plot.filename = f"{args.output_prefix}_top_features.png"

    detailed_rows: list[dict] = []
    run_metrics: list[dict] = []

    for n_cycles in args.windows:
        for dataset in args.datasets:
            for model_name in model_list_for_dataset(args.models, dataset):
                for seed in args.seeds:
                    print(f"[run] SHAP {dataset} seed={seed} N={n_cycles} model={model_name}")
                    rows, run_row = evaluate_and_explain(
                        df,
                        feature_cols,
                        dataset=dataset,
                        n_cycles=n_cycles,
                        seed=seed,
                        model_name=model_name,
                        splits_dir=splits_dir,
                    )
                    detailed_rows.extend(rows)
                    run_metrics.append(run_row)

    if not detailed_rows:
        print("[error] no SHAP rows produced")
        return 1

    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = summarize_importance(detailed_df)
    if not args.skip_transfer_stability:
        summary_df = join_transfer_stability(summary_df, transfer_path)
    summary_df = join_conditional_shift(summary_df, conditional_shift_path, args.conditional_pairs)

    out_summary = INTERMEDIATE_DIR / f"{args.output_prefix}.csv"
    out_detailed = INTERMEDIATE_DIR / f"{args.output_prefix}_detailed.csv"
    out_json = INTERMEDIATE_DIR / f"{args.output_prefix}.json"
    out_report = INTERMEDIATE_DIR / f"{args.output_prefix}_report.txt"

    summary_df.to_csv(out_summary, index=False)
    detailed_df.to_csv(out_detailed, index=False)
    write_report(summary_df, run_metrics, out_report, top_k=args.top_k, features_path=features_path)
    make_top_feature_plot(summary_df, args.output_dir, top_k=args.top_k)

    payload = {
        "protocol": "shap_feature_importance_v2_benchmark_model_check",
        "features_path": display_path(features_path),
        "splits_dir": display_path(splits_dir),
        "transfer_path": None if args.skip_transfer_stability else display_path(transfer_path) if transfer_path.exists() else None,
        "skip_transfer_stability": args.skip_transfer_stability,
        "conditional_shift_path": display_path(conditional_shift_path) if conditional_shift_path is not None and conditional_shift_path.exists() else None,
        "conditional_pairs": args.conditional_pairs,
        "windows": args.windows,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "models": args.models,
        "primary_model_by_dataset": PRIMARY_MODEL_BY_DATASET,
        "feature_columns": feature_cols,
        "shap_space": "log-cycle prediction space",
        "summary": summary_df.replace({np.nan: None}).to_dict(orient="records"),
        "run_metrics": run_metrics,
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"[save] {out_summary}")
    print(f"[save] {out_detailed}")
    print(f"[save] {out_json}")
    print(f"[save] {out_report}")
    plot_path = args.output_dir / f"{args.output_prefix}_top_features.png"
    if plot_path.exists():
        print(f"[save] {plot_path}")
    print("\n" + out_report.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
