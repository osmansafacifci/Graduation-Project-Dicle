"""
SHAP feature attribution for the primary within-dataset battery-life models.

Scope is intentionally narrow and reproducible:

  - N = 100 by default, matching the headline protocol.
  - 34 detected feature columns, log-target fitting, train-only scaling.
  - Five official cell-level splits.
  - Primary paper models by within-dataset R2:
      MATR -> CatBoost
      HUST -> Random Forest

SHAP values are computed in the fitted log-cycle prediction space. Reported
model metrics remain in original cycle units so they can be compared directly
with the main performance tables.

Outputs:
    data/intermediate/shap_feature_importance.csv
    data/intermediate/shap_feature_importance_detailed.csv
    data/intermediate/shap_feature_importance.json
    data/intermediate/shap_feature_importance_report.txt
    outputs/results_v2_shap/shap_importance_top_features.png

Usage:
    python 3_analysis/shap_feature_importance.py
    python 3_analysis/shap_feature_importance.py --models catboost random_forest
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_shap"

sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
from metrics_utils import compute_metrics  # noqa: E402
from run_experiments import (  # noqa: E402
    META_COLS,
    SEEDS,
    fit_catboost,
    fit_random_forest,
)

FEATURES_PATH = INTERMEDIATE_DIR / "features_sop12_combined.csv"
TRANSFER_PATH = INTERMEDIATE_DIR / "feature_transfer_stability.csv"

PRIMARY_MODEL_BY_DATASET = {
    "matr": "catboost",
    "hust": "random_forest",
}

MODEL_FITTERS = {
    "catboost": fit_catboost,
    "random_forest": fit_random_forest,
}


def load_split(dataset: str, seed: int) -> dict:
    path = SPLITS_DIR / f"{dataset}_{seed}.json"
    with path.open() as f:
        return json.load(f)


def dataset_window(df: pd.DataFrame, dataset: str, n_cycles: int) -> pd.DataFrame:
    return df[(df["dataset"] == dataset) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()


def safe_cycles_from_log(pred_log: np.ndarray) -> np.ndarray:
    pred_log = np.clip(pred_log, np.log(1.0), np.log(1e9))
    return np.clip(np.nan_to_num(np.exp(pred_log), nan=1.0, posinf=1e9, neginf=1.0), 1.0, 1e9)


def model_list_for_dataset(requested: list[str], dataset: str) -> list[str]:
    if "primary" in requested:
        return [PRIMARY_MODEL_BY_DATASET[dataset]]
    return requested


def compute_tree_shap_values(model, X_test: np.ndarray) -> np.ndarray:
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


def evaluate_and_explain(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    dataset: str,
    n_cycles: int,
    seed: int,
    model_name: str,
) -> tuple[list[dict], dict]:
    split = load_split(dataset, seed)
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
    fitted = fitter(X_train_s, y_train_fit, seed=seed)
    if isinstance(fitted, tuple):
        model, tuning = fitted
    else:
        model, tuning = fitted, {}

    pred_log = np.asarray(model.predict(X_test_s), dtype=float).ravel()
    pred = safe_cycles_from_log(pred_log)
    metrics = compute_metrics(y_test, pred)

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
        "tuning": tuning,
    }
    return rows, run_row


def summarize_importance(detailed: pd.DataFrame) -> pd.DataFrame:
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


def join_transfer_stability(summary: pd.DataFrame) -> pd.DataFrame:
    if not TRANSFER_PATH.exists():
        return summary
    transfer = pd.read_csv(TRANSFER_PATH)
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


def write_report(summary: pd.DataFrame, run_metrics: list[dict], out_path: Path, *, top_k: int) -> None:
    lines: list[str] = []
    lines.append("SHAP feature attribution summary")
    lines.append("=" * 80)
    lines.append("Protocol: 34 features, N=100 default, log-target models, SHAP in log-cycle space.")
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
            stability = row.get("stability_class", "n/a")
            shift = row.get("abs_mean_shift_z_raw", np.nan)
            lines.append(
                f"  {row['feature']:<24} rank={row['rank_mean']:.1f} "
                f"rel={100.0 * row['relative_importance_mean']:.1f}% "
                f"shift_z={shift:.2f} class={stability}"
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

    blocks = list(summary.groupby(["dataset", "model"], sort=True))
    if not blocks:
        return

    fig, axes = plt.subplots(1, len(blocks), figsize=(6.2 * len(blocks), 5.5), squeeze=False)
    for ax, ((dataset, model), block) in zip(axes[0], blocks):
        top = block.sort_values("mean_abs_shap_log_cycles_mean", ascending=True).tail(top_k)
        colors = top["stability_class"].map(
            {
                "stable_candidate": "#2f7d32",
                "weak_or_mixed": "#5f6368",
                "scale_shift_fragile": "#b3261e",
                "relationship_unstable": "#a142f4",
            }
        ).fillna("#5f6368")
        ax.barh(top["feature"], top["mean_abs_shap_log_cycles_mean"], color=colors)
        ax.set_title(f"{dataset.upper()} {model}")
        ax.set_xlabel("Mean |SHAP| (log-cycle units)")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "shap_importance_top_features.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=int, nargs="+", default=[100])
    parser.add_argument("--datasets", nargs="+", default=["matr", "hust"], choices=["matr", "hust"])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["primary"],
        choices=["primary", "catboost", "random_forest"],
        help="'primary' runs CatBoost for MATR and Random Forest for HUST.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FEATURES_PATH.exists():
        print(f"[error] missing {FEATURES_PATH}")
        return 1
    if not SPLITS_DIR.exists():
        print(f"[error] missing {SPLITS_DIR}")
        return 1

    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in META_COLS]

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
                    )
                    detailed_rows.extend(rows)
                    run_metrics.append(run_row)

    if not detailed_rows:
        print("[error] no SHAP rows produced")
        return 1

    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = join_transfer_stability(summarize_importance(detailed_df))

    out_summary = INTERMEDIATE_DIR / "shap_feature_importance.csv"
    out_detailed = INTERMEDIATE_DIR / "shap_feature_importance_detailed.csv"
    out_json = INTERMEDIATE_DIR / "shap_feature_importance.json"
    out_report = INTERMEDIATE_DIR / "shap_feature_importance_report.txt"

    summary_df.to_csv(out_summary, index=False)
    detailed_df.to_csv(out_detailed, index=False)
    write_report(summary_df, run_metrics, out_report, top_k=args.top_k)
    make_top_feature_plot(summary_df, args.output_dir, top_k=args.top_k)

    payload = {
        "protocol": "shap_feature_importance_v1",
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
    plot_path = args.output_dir / "shap_importance_top_features.png"
    if plot_path.exists():
        print(f"[save] {plot_path}")
    print("\n" + out_report.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
