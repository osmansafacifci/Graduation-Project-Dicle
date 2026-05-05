"""
Run SOP-style within-dataset and cross-dataset baseline experiments.

This script uses the current feature table only as a transition layer while the
project moves toward the new 12-feature SOP pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from metrics_utils import bootstrap_metric_ci, compute_metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from split_protocol_utils import (
    DEFAULT_SEEDS,
    load_split_json,
)
from sop_utils import (
    load_prepared_dataset,
    parse_dataset_label_map,
    parse_feature_columns,
    resolve_feature_columns,
)
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "intermediate" / "features_top8_cycles.csv"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "splits"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "results" / "results_sop_protocol_baselines.json"
PRIMARY_WINDOWS = (100, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SOP-style baseline experiments.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--feature-set",
        type=str,
        default="top8",
        help="Named feature schema: top8, top7_no_qd_std, sop12_transition.",
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        default=None,
        help="Optional comma-separated feature list that overrides --feature-set.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="cycle_life",
        help="Default label column to score against.",
    )
    parser.add_argument(
        "--dataset-label-map",
        type=str,
        default=None,
        help="Optional per-dataset relabeling map, for example: b1:cycle_life,b2:eol_80_cycle",
    )
    parser.add_argument(
        "--censor-column",
        type=str,
        default=None,
        help="Optional censoring indicator column.",
    )
    parser.add_argument(
        "--drop-censored",
        action="store_true",
        help="Exclude rows flagged by --censor-column before training/evaluation.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=list(PRIMARY_WINDOWS),
        help="Cycle windows to evaluate.",
    )
    parser.add_argument(
        "--within-prefixes",
        nargs="+",
        default=["b1", "b2"],
        help="Dataset prefixes to evaluate with train/test splits from the same dataset.",
    )
    parser.add_argument(
        "--cross-pairs",
        nargs="*",
        default=[],
        help="Cross-dataset pairs in train:test format, for example matr:hust.",
    )
    return parser.parse_args()


def subset_by_cells(df: pd.DataFrame, cell_ids: list[str]) -> pd.DataFrame:
    return df[df["cell_id"].isin(cell_ids)].copy()


def build_model_factories() -> dict[str, object]:
    return {
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "elastic_net": lambda: make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                cv=3,
                max_iter=50000,
                random_state=42,
            ),
        ),
        "xgboost": lambda: XGBRegressor(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=400,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
        "catboost": lambda: CatBoostRegressor(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            loss_function="MAE",
            random_seed=42,
            verbose=False,
        ),
    }


def score_model(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
) -> dict:
    X_train = train_df[feature_columns].to_numpy()
    y_train = train_df[target_column].to_numpy()
    X_test = test_df[feature_columns].to_numpy()
    y_test = test_df[target_column].to_numpy()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)
    metrics["bootstrap_95_ci"] = bootstrap_metric_ci(y_test, preds)
    metrics["train_rows"] = int(len(train_df))
    metrics["test_rows"] = int(len(test_df))
    return metrics


def aggregate_seed_metrics(seed_results: list[dict]) -> dict:
    if not seed_results:
        return {}

    summary: dict[str, object] = {"num_seeds": len(seed_results), "per_seed": seed_results}
    for metric in ("MAE", "SMAPE", "R2"):
        values = [result[metric] for result in seed_results]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        lowers = [result["bootstrap_95_ci"][metric]["lower"] for result in seed_results]
        uppers = [result["bootstrap_95_ci"][metric]["upper"] for result in seed_results]
        summary[f"{metric}_bootstrap_95_ci"] = {
            "lower_mean": float(np.mean(lowers)),
            "upper_mean": float(np.mean(uppers)),
        }
    return summary


def run_experiment(
    df: pd.DataFrame,
    split_dir: Path,
    train_prefix: str,
    test_prefix: str,
    seeds: list[int],
    *,
    feature_columns: list[str],
    target_column: str,
    windows: list[int],
) -> dict:
    factories = build_model_factories()
    results: dict[str, dict] = {}
    for model_name, factory in factories.items():
        results[model_name] = {}
        for n_cycles in windows:
            per_seed: list[dict] = []
            for seed in seeds:
                train_split_path = split_dir / f"{train_prefix}_{seed}.json"
                test_split_path = split_dir / f"{test_prefix}_{seed}.json"
                if not train_split_path.exists() or not test_split_path.exists():
                    continue
                train_split = load_split_json(train_split_path)
                test_split = load_split_json(test_split_path)

                train_df = subset_by_cells(
                    df[(df["dataset_prefix"] == train_prefix) & (df["n_cycles"] == n_cycles)],
                    train_split["train"],
                )
                test_df = subset_by_cells(
                    df[(df["dataset_prefix"] == test_prefix) & (df["n_cycles"] == n_cycles)],
                    test_split["test"],
                )
                train_df.dropna(subset=feature_columns + [target_column], inplace=True)
                test_df.dropna(subset=feature_columns + [target_column], inplace=True)
                if train_df.empty or test_df.empty:
                    continue
                per_seed.append(
                    score_model(
                        factory(),
                        train_df,
                        test_df,
                        feature_columns=feature_columns,
                        target_column=target_column,
                    )
                )
            results[model_name][str(n_cycles)] = aggregate_seed_metrics(per_seed)
    return results


def main() -> None:
    args = parse_args()
    prepared = load_prepared_dataset(
        args.dataset,
        default_label_column=args.label_column,
        dataset_label_map=parse_dataset_label_map(args.dataset_label_map),
        censor_column=args.censor_column,
        drop_censored=args.drop_censored,
    )
    feature_df = prepared.frame
    feature_columns = resolve_feature_columns(
        feature_df,
        feature_set=args.feature_set,
        explicit_columns=parse_feature_columns(args.feature_columns),
    )
    within_dataset = {
        f"{prefix}_to_{prefix}": run_experiment(
            feature_df,
            args.split_dir,
            prefix,
            prefix,
            args.seeds,
            feature_columns=feature_columns,
            target_column=prepared.target_column,
            windows=args.windows,
        )
        for prefix in args.within_prefixes
    }
    cross_dataset = {}
    for pair in args.cross_pairs:
        if ":" not in pair:
            raise SystemExit(f"Cross pair must use train:test format, got: {pair}")
        train_prefix, test_prefix = pair.split(":", 1)
        cross_dataset[f"{train_prefix}_to_{test_prefix}"] = run_experiment(
            feature_df,
            args.split_dir,
            train_prefix,
            test_prefix,
            args.seeds,
            feature_columns=feature_columns,
            target_column=prepared.target_column,
            windows=args.windows,
        )

    summary = {
        "protocol_version": "sop_baseline",
        "notes": [
            "Uses SOP JSON splits, seed averaging, and bootstrap confidence intervals.",
            "Supports configurable feature sets, labels, and optional censor filtering.",
        ],
        "dataset": str(args.dataset),
        "feature_set": args.feature_set,
        "feature_columns": feature_columns,
        "target_column": prepared.target_column,
        "windows": args.windows,
        "censor_summary": prepared.censor_summary,
        "within_dataset": within_dataset,
        "cross_dataset": cross_dataset,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved SOP baseline summary to {args.output}")


if __name__ == "__main__":
    main()
