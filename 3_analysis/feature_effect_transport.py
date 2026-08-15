"""Feature-effect transport diagnostics for cross-dataset rank preservation.

The cross-dataset taxonomy says that some source models preserve target-cell
rankings while others collapse or invert them. This script asks *which learned
feature effects* carry that rank signal.

For each source dataset, seed, and tree backbone, we train a source-only
log-life model on the canonical training split and evaluate it on every other
target dataset. TreeSHAP gives additive feature contributions in log-cycle
space. For a fitted source model ``f_s`` and target cohort ``t``:

    corr(f_s(X_t), log Y_t) = sum_j Cov(phi_j(X_t), log Y_t)
                              / (sd(f_s(X_t)) sd(log Y_t))

up to floating-point SHAP additivity error. Positive terms carry transferable
rank signal; negative terms work against the target ranking. Spearman and
cycle-space R2 are reported as sensitivity checks, not as the additive
decomposition target.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_feature_effect_transport"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2_four_dataset"

sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
from metrics_utils import fit_with_threaded_joblib, symmetric_mape, to_cycles  # noqa: E402
from run_experiments import META_COLS, SEEDS, fit_catboost, fit_random_forest  # noqa: E402

FEATURES_PATH = INTERMEDIATE_DIR / "features_sop12_four_dataset_capnorm.csv"
DIRECTION_SUMMARY_PATH = INTERMEDIATE_DIR / "four_dataset_conditional_shift_direction_summary.csv"

MODEL_FITTERS = {
    "catboost": fit_catboost,
    "random_forest": fit_random_forest,
}

MODEL_LABEL = {
    "catboost": "CatBoost",
    "random_forest": "Random Forest",
}

FEATURE_GROUPS = {
    "capacity_level_retention": {
        "Qdis_N",
        "Qdis_cycle10",
        "delta_Qdis",
        "retention_ratio",
        "range_Qdis",
    },
    "trend_decay_timing": {
        "slope_linear",
        "slope_ratio",
        "slope_first_quarter",
        "slope_last_quarter",
        "exp_decay_k",
        "cycle_to_99pct",
        "cycle_to_98pct",
        "cycle_to_95pct",
        "knee_cycle",
    },
    "curvature_acceleration": {
        "poly2_a",
        "poly2_b",
        "poly2_c",
        "accel_mean",
        "accel_std",
        "accel_max_abs",
        "linearity_r2",
    },
    "variability_events": {
        "variance_Qdis",
        "max_drop",
        "std_diff",
        "skewness_Qdis",
        "mean_diff",
        "autocorr_lag1",
        "n_capacity_jumps",
        "kurtosis_Qdis",
        "mad_Qdis",
        "pos_neg_diff_ratio",
    },
    "spectral_entropy": {
        "fft_top3_energy_ratio",
        "spectral_entropy",
        "sample_entropy",
    },
}

PILOT_DIRECTIONS = {
    ("hust", "luh"),
    ("luh", "hust"),
    ("matr", "hust"),
    ("hust", "matr"),
}


def feature_to_group(feature: str) -> str:
    for group, features in FEATURE_GROUPS.items():
        if feature in features:
            return group
    return "other"


def load_split(dataset: str, seed: int, splits_dir: Path) -> dict:
    with (splits_dir / f"{dataset}_{seed}.json").open() as f:
        return json.load(f)


def finite_corr(func, x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan")
    return float(func(x, y).statistic)


def tree_shap_values(model, X: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        import shap
    except ImportError as exc:
        raise SystemExit("shap is required for feature-effect transport. Install project requirements.") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X, check_additivity=False)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got {values.shape}")
    base = explainer.expected_value
    if isinstance(base, (list, tuple, np.ndarray)):
        base_arr = np.asarray(base, dtype=float).ravel()
        base_value = float(base_arr[0])
    else:
        base_value = float(base)
    return values, base_value


def decomposition_rows(
    *,
    source: str,
    target: str,
    model_name: str,
    seed: int,
    feature_cols: list[str],
    shap_values: np.ndarray,
    pred_log: np.ndarray,
    y_log: np.ndarray,
    y_true: np.ndarray,
    pred_cycles: np.ndarray,
    base_value: float,
) -> tuple[list[dict], list[dict], dict]:
    pred_log = np.asarray(pred_log, dtype=float)
    y_log = np.asarray(y_log, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    pred_cycles = np.asarray(pred_cycles, dtype=float)

    pred_sd = float(np.std(pred_log))
    y_sd = float(np.std(y_log))
    denom = pred_sd * y_sd
    pearson_log = finite_corr(pearsonr, pred_log, y_log)
    spearman_cycle = finite_corr(spearmanr, pred_cycles, y_true)
    r2_cycle = float(r2_score(y_true, pred_cycles)) if len(y_true) >= 2 else float("nan")
    smape_cycle = symmetric_mape(y_true, pred_cycles)
    shap_recon = base_value + np.sum(shap_values, axis=1)
    additivity_rmse = float(np.sqrt(np.mean((pred_log - shap_recon) ** 2)))

    centered_y = y_log - np.mean(y_log)
    feature_rows: list[dict] = []
    for i, feature in enumerate(feature_cols):
        phi = shap_values[:, i]
        cov = float(np.mean((phi - np.mean(phi)) * centered_y))
        contribution = cov / denom if denom > 0 else float("nan")
        feature_rows.append(
            {
                "source": source,
                "target": target,
                "direction": f"{source}_to_{target}",
                "model": model_name,
                "seed": int(seed),
                "feature": feature,
                "feature_group": feature_to_group(feature),
                "cov_phi_loglife": cov,
                "pearson_log_contribution": contribution,
                "mean_abs_shap_log": float(np.mean(np.abs(phi))),
                "mean_signed_shap_log": float(np.mean(phi)),
                "pearson_log": pearson_log,
                "spearman_cycle": spearman_cycle,
                "r2_cycle": r2_cycle,
                "smape_cycle": smape_cycle,
                "additivity_rmse_log": additivity_rmse,
            }
        )

    feature_df = pd.DataFrame(feature_rows)
    group_rows: list[dict] = []
    for group, grp in feature_df.groupby("feature_group", sort=True):
        group_rows.append(
            {
                "source": source,
                "target": target,
                "direction": f"{source}_to_{target}",
                "model": model_name,
                "seed": int(seed),
                "feature_group": group,
                "pearson_log_contribution": float(grp["pearson_log_contribution"].sum()),
                "abs_contribution_sum": float(grp["pearson_log_contribution"].abs().sum()),
                "mean_abs_shap_log": float(grp["mean_abs_shap_log"].sum()),
                "pearson_log": pearson_log,
                "spearman_cycle": spearman_cycle,
                "r2_cycle": r2_cycle,
                "smape_cycle": smape_cycle,
                "additivity_rmse_log": additivity_rmse,
            }
        )

    run_row = {
        "source": source,
        "target": target,
        "direction": f"{source}_to_{target}",
        "model": model_name,
        "seed": int(seed),
        "n_target": int(len(y_true)),
        "pearson_log": pearson_log,
        "spearman_cycle": spearman_cycle,
        "r2_cycle": r2_cycle,
        "smape_cycle": smape_cycle,
        "additivity_rmse_log": additivity_rmse,
        "sum_feature_contributions": float(feature_df["pearson_log_contribution"].sum()),
        "decomposition_error": float(feature_df["pearson_log_contribution"].sum() - pearson_log),
    }
    return feature_rows, group_rows, run_row


def summarize_groups(group_df: pd.DataFrame, run_df: pd.DataFrame, direction_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_summary_rows: list[dict] = []
    for keys, grp in group_df.groupby(["source", "target", "direction", "model", "feature_group"], sort=True):
        row = dict(zip(["source", "target", "direction", "model", "feature_group"], keys))
        vals = grp["pearson_log_contribution"].to_numpy(dtype=float)
        row["contribution_mean"] = float(np.mean(vals))
        row["contribution_median"] = float(np.median(vals))
        row["contribution_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        median_sign = np.sign(row["contribution_median"])
        row["sign_consistency"] = float(np.mean(np.sign(vals) == median_sign)) if median_sign != 0 else float(np.mean(vals == 0))
        row["mean_abs_shap_log"] = float(grp["mean_abs_shap_log"].mean())
        group_summary_rows.append(row)
    group_summary = pd.DataFrame(group_summary_rows)

    direction_rows: list[dict] = []
    for keys, grp in run_df.groupby(["source", "target", "direction", "model"], sort=True):
        source, target, direction, model = keys
        gsub = group_summary[(group_summary["direction"] == direction) & (group_summary["model"] == model)].copy()
        pos = gsub.sort_values("contribution_mean", ascending=False).head(1)
        neg = gsub.sort_values("contribution_mean", ascending=True).head(1)
        abs_top = gsub.assign(abs_mean=lambda d: d["contribution_mean"].abs()).sort_values("abs_mean", ascending=False).head(1)
        row = {
            "source": source,
            "target": target,
            "direction": direction,
            "model": model,
            "pearson_log_mean": float(grp["pearson_log"].mean()),
            "pearson_log_sd": float(grp["pearson_log"].std(ddof=1)),
            "spearman_cycle_mean": float(grp["spearman_cycle"].mean()),
            "r2_cycle_mean": float(grp["r2_cycle"].mean()),
            "decomposition_error_abs_mean": float(grp["decomposition_error"].abs().mean()),
            "dominant_positive_group": pos["feature_group"].iloc[0],
            "dominant_positive_contribution": float(pos["contribution_mean"].iloc[0]),
            "dominant_negative_group": neg["feature_group"].iloc[0],
            "dominant_negative_contribution": float(neg["contribution_mean"].iloc[0]),
            "dominant_abs_group": abs_top["feature_group"].iloc[0],
            "dominant_abs_contribution": float(abs_top["contribution_mean"].iloc[0]),
        }
        direction_rows.append(row)
    direction_summary_out = pd.DataFrame(direction_rows)
    if not direction_summary.empty:
        keep = [
            "source",
            "target",
            "pearson_r",
            "raw_R2",
            "linear_R2",
            "rank_signal_class",
            "life_ratio_target_over_source",
        ]
        keep = [c for c in keep if c in direction_summary.columns]
        direction_summary_out = direction_summary_out.merge(direction_summary[keep], on=["source", "target"], how="left")
    return group_summary, direction_summary_out


def make_plots(group_summary: pd.DataFrame, direction_summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        sys.path.insert(0, str(HERE))
        from plot_style import apply_science_style

        apply_science_style()
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    for model in sorted(group_summary["model"].unique()):
        pivot = group_summary[group_summary["model"].eq(model)].pivot_table(
            index="direction",
            columns="feature_group",
            values="contribution_mean",
            aggfunc="mean",
        )
        ordering = (
            direction_summary[direction_summary["model"].eq(model)]
            .sort_values("pearson_log_mean", ascending=False)["direction"]
            .tolist()
        )
        pivot = pivot.reindex(ordering)
        fig, ax = plt.subplots(figsize=(9.6, 5.2))
        vmax = float(np.nanmax(np.abs(pivot.to_numpy()))) if pivot.size else 1.0
        vmax = max(vmax, 0.1)
        im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([d.replace("_to_", "→").upper() for d in pivot.index])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"Feature-effect transport contributions ({MODEL_LABEL.get(model, model)})")
        cbar = fig.colorbar(im, ax=ax, shrink=0.82)
        cbar.set_label("Contribution to Pearson r in log-cycle space")
        fig.tight_layout()
        fig.savefig(output_dir / f"feature_effect_transport_{model}_heatmap.png")
        fig.savefig(output_dir / f"feature_effect_transport_{model}_heatmap.pdf")
        plt.close(fig)


def write_report(direction_summary: pd.DataFrame, group_summary: pd.DataFrame, path: Path) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        render = df.copy()
        headers = [str(c) for c in render.columns]
        rows = render.astype(object).where(pd.notna(render), "").astype(str).values.tolist()
        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                *["| " + " | ".join(row) + " |" for row in rows],
            ]
        )

    pilot = direction_summary[direction_summary[["source", "target"]].apply(tuple, axis=1).isin(PILOT_DIRECTIONS)].copy()
    lines: list[str] = []
    lines.append("# Feature-Effect Transport")
    lines.append("")
    lines.append("TreeSHAP contributions are decomposed in log-cycle space; Spearman and cycle-space R2 are sensitivity checks. Positive group contributions carry source-model rank signal on the target cohort; negative groups oppose that ranking.")
    lines.append("")
    lines.append("## Pilot directions")
    lines.append(
        markdown_table(
            pilot[
                [
                    "source",
                    "target",
                    "model",
                    "pearson_log_mean",
                    "spearman_cycle_mean",
                    "r2_cycle_mean",
                    "dominant_positive_group",
                    "dominant_positive_contribution",
                    "dominant_negative_group",
                    "dominant_negative_contribution",
                    "rank_signal_class",
                ]
            ]
            .sort_values(["source", "target", "model"])
            .round(3)
        )
    )
    lines.append("")
    lines.append("## All directions, direction-level summary")
    lines.append(
        markdown_table(
            direction_summary[
                [
                    "source",
                    "target",
                    "model",
                    "pearson_log_mean",
                    "spearman_cycle_mean",
                    "dominant_positive_group",
                    "dominant_negative_group",
                    "rank_signal_class",
                ]
            ]
            .sort_values("pearson_log_mean", ascending=False)
            .round(3)
        )
    )
    lines.append("")
    lines.append("## Group contribution means")
    compact = group_summary.pivot_table(
        index=["source", "target", "model"],
        columns="feature_group",
        values="contribution_mean",
        aggfunc="mean",
    ).reset_index()
    lines.append(markdown_table(compact.round(3)))
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--direction-summary-path", type=Path, default=DIRECTION_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=["catboost", "random_forest"], choices=sorted(MODEL_FITTERS))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--datasets", nargs="+", default=["matr", "hust", "sandia", "luh"])
    parser.add_argument("--n-cycles", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = args.features_path if args.features_path.is_absolute() else PROJECT_ROOT / args.features_path
    splits_dir = args.splits_dir if args.splits_dir.is_absolute() else PROJECT_ROOT / args.splits_dir
    if not features_path.exists():
        print(f"[error] missing {features_path}")
        return 1
    if not splits_dir.exists():
        print(f"[error] missing {splits_dir}")
        return 1

    df = pd.read_csv(features_path)
    df = df[(df["n_cycles"] == args.n_cycles) & (df["is_censored"] == 0)].copy()
    feature_cols = [c for c in df.columns if c not in META_COLS]
    missing_groups = sorted(set(feature_cols) - {f for fs in FEATURE_GROUPS.values() for f in fs})
    if missing_groups:
        print(f"[warn] features not assigned to named group: {missing_groups}")

    direction_reference = pd.read_csv(args.direction_summary_path) if args.direction_summary_path.exists() else pd.DataFrame()

    feature_rows: list[dict] = []
    group_rows: list[dict] = []
    run_rows: list[dict] = []
    train_count = len(args.datasets) * len(args.models) * len(args.seeds)
    train_idx = 0
    for source in args.datasets:
        source_df = df[df["dataset"].eq(source)].copy()
        for model_name in args.models:
            fitter = MODEL_FITTERS[model_name]
            for seed in args.seeds:
                train_idx += 1
                split = load_split(source, seed, splits_dir)
                train_df = source_df[source_df["cell_id"].isin(split["train"])].copy()
                if len(train_df) < 5:
                    print(f"[skip] {source} {model_name} seed={seed}: train cells={len(train_df)}")
                    continue
                print(f"[fit {train_idx:03d}/{train_count:03d}] {source} {model_name} seed={seed} train={len(train_df)}")
                scaler = StandardScaler()
                X_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float))
                y_train = np.log(train_df["cycle_life"].to_numpy(dtype=float))
                fitted = fit_with_threaded_joblib(fitter, X_train, y_train, seed=seed)
                if isinstance(fitted, tuple):
                    model, _info = fitted
                else:
                    model = fitted

                for target in args.datasets:
                    if target == source:
                        continue
                    target_df = df[df["dataset"].eq(target)].copy()
                    X_target = scaler.transform(target_df[feature_cols].to_numpy(dtype=float))
                    y_true = target_df["cycle_life"].to_numpy(dtype=float)
                    y_log = np.log(y_true)
                    pred_log = np.asarray(model.predict(X_target), dtype=float).ravel()
                    pred_cycles = to_cycles(pred_log, log_target=True)
                    shap_values, base_value = tree_shap_values(model, X_target)
                    f_rows, g_rows, r_row = decomposition_rows(
                        source=source,
                        target=target,
                        model_name=model_name,
                        seed=seed,
                        feature_cols=feature_cols,
                        shap_values=shap_values,
                        pred_log=pred_log,
                        y_log=y_log,
                        y_true=y_true,
                        pred_cycles=pred_cycles,
                        base_value=base_value,
                    )
                    feature_rows.extend(f_rows)
                    group_rows.extend(g_rows)
                    run_rows.append(r_row)

    feature_df = pd.DataFrame(feature_rows)
    group_df = pd.DataFrame(group_rows)
    run_df = pd.DataFrame(run_rows)
    group_summary, direction_summary = summarize_groups(group_df, run_df, direction_reference)

    feature_path = INTERMEDIATE_DIR / "feature_effect_transport_feature_contributions.csv"
    group_path = INTERMEDIATE_DIR / "feature_effect_transport_group_contributions.csv"
    run_path = INTERMEDIATE_DIR / "feature_effect_transport_runs.csv"
    group_summary_path = INTERMEDIATE_DIR / "feature_effect_transport_group_summary.csv"
    direction_summary_path = INTERMEDIATE_DIR / "feature_effect_transport_direction_summary.csv"
    report_path = INTERMEDIATE_DIR / "feature_effect_transport_report.md"

    feature_df.to_csv(feature_path, index=False)
    group_df.to_csv(group_path, index=False)
    run_df.to_csv(run_path, index=False)
    group_summary.to_csv(group_summary_path, index=False)
    direction_summary.to_csv(direction_summary_path, index=False)
    write_report(direction_summary, group_summary, report_path)
    make_plots(group_summary, direction_summary, args.output_dir)

    print(f"[save] {feature_path} rows={len(feature_df)}")
    print(f"[save] {group_path} rows={len(group_df)}")
    print(f"[save] {run_path} rows={len(run_df)}")
    print(f"[save] {group_summary_path} rows={len(group_summary)}")
    print(f"[save] {direction_summary_path} rows={len(direction_summary)}")
    print(f"[save] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
