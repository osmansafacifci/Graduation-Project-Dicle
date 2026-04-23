"""
Train on one batch and test on the other batch without any adaptation.

This script implements the out-of-scope clarification from the project:
no transfer learning, no domain adaptation, and no fine-tuning. The model is
simply trained on dataset A and evaluated directly on dataset B.

Usage:
    python 2_modeling_featuring/evaluate_cross_dataset_generalization.py \
        --dataset data/intermediate/features_top8_cycles.csv \
        --train-batch b1 b2 b3 \
        --test-batch hust
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from metrics_utils import bootstrap_metric_ci, compute_metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sop_utils import (
    load_prepared_dataset,
    parse_dataset_label_map,
    parse_feature_columns,
    resolve_feature_columns,
)
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
DEFAULT_DATASET = INTERMEDIATE_DIR / "features_top8_cycles.csv"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on one battery batch and test on another batch."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Feature CSV path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--train-batch",
        type=str,
        nargs="+",
        default=["b1"],
        help="Source dataset/batch prefixes to train on (for example: b1 b2 b3).",
    )
    parser.add_argument(
        "--test-batch",
        type=str,
        nargs="+",
        default=["b2"],
        help="Target dataset/batch prefixes to test on (for example: hust).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Default is derived from train/test batch names.",
    )
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
        default=[25, 50, 100],
        help="Cycle windows to evaluate.",
    )
    return parser.parse_args()

def build_models() -> dict[str, Callable[[], object]]:
    rf_params = dict(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    xgb_params = dict(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    cat_params = dict(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        random_seed=42,
        verbose=False,
    )
    enet_params = dict(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-4, 0, 20),
        cv=5,
        max_iter=50000,
        tol=1e-3,
        random_state=42,
    )
    return {
        "random_forest": lambda: RandomForestRegressor(**rf_params),
        "xgboost": lambda: XGBRegressor(**xgb_params),
        "catboost": lambda: CatBoostRegressor(**cat_params),
        "elastic_net": lambda: make_pipeline(StandardScaler(), ElasticNetCV(**enet_params)),
    }

def evaluate_direction(
    df: pd.DataFrame,
    train_batches: list[str],
    test_batches: list[str],
    models: dict[str, Callable[[], object]],
    *,
    feature_columns: list[str],
    target_column: str,
    windows: list[int],
) -> dict[str, dict[int, dict[str, float]]]:
    summary: dict[str, dict[int, dict[str, float]]] = {}

    for model_name, build_model in models.items():
        cycle_results: dict[int, dict[str, float]] = {}
        for n_cycles in windows:
            train_df = df[
                (df["dataset_prefix"].isin(train_batches)) & (df["n_cycles"] == n_cycles)
            ].copy()
            test_df = df[
                (df["dataset_prefix"].isin(test_batches)) & (df["n_cycles"] == n_cycles)
            ].copy()
            if train_df.empty or test_df.empty:
                continue

            train_df.dropna(subset=feature_columns + [target_column], inplace=True)
            test_df.dropna(subset=feature_columns + [target_column], inplace=True)
            if train_df.empty or test_df.empty:
                continue

            model = build_model()
            model.fit(train_df[feature_columns], train_df[target_column])
            preds = model.predict(test_df[feature_columns])

            y_true = test_df[target_column].to_numpy()
            metrics = compute_metrics(y_true, preds)
            metrics["bootstrap_95_ci"] = bootstrap_metric_ci(y_true, preds)
            metrics["train_rows"] = int(len(train_df))
            metrics["test_rows"] = int(len(test_df))
            cycle_results[n_cycles] = metrics

        summary[model_name] = cycle_results
    return summary


def main() -> None:
    args = parse_args()
    train_batches = args.train_batch
    test_batches = args.test_batch
    if set(train_batches) & set(test_batches):
        raise SystemExit("train-batch ve test-batch kesisimi bos olmali.")

    output_path = args.output
    if output_path is None:
        train_name = "_".join(train_batches)
        test_name = "_".join(test_batches)
        output_path = (
            RESULTS_DIR
            / f"results_cross_dataset_{train_name}_to_{test_name}.json"
        )

    prepared = load_prepared_dataset(
        args.dataset,
        default_label_column=args.label_column,
        dataset_label_map=parse_dataset_label_map(args.dataset_label_map),
        censor_column=args.censor_column,
        drop_censored=args.drop_censored,
    )
    df = prepared.frame
    feature_columns = resolve_feature_columns(
        df,
        feature_set=args.feature_set,
        explicit_columns=parse_feature_columns(args.feature_columns),
    )
    available_batches = sorted(df["dataset_prefix"].dropna().unique().tolist())
    missing_train = [batch for batch in train_batches if batch not in available_batches]
    if missing_train:
        raise SystemExit(
            f"Train batch bulunamadi: {missing_train}. Mevcut batch'ler: {available_batches}"
        )
    missing_test = [batch for batch in test_batches if batch not in available_batches]
    if missing_test:
        raise SystemExit(
            f"Test batch bulunamadi: {missing_test}. Mevcut batch'ler: {available_batches}"
        )

    summary = {
        "protocol": "train_on_A_test_on_B_without_adaptation",
        "train_batch": train_batches,
        "test_batch": test_batches,
        "dataset": str(args.dataset),
        "feature_set": args.feature_set,
        "feature_columns": feature_columns,
        "target_column": prepared.target_column,
        "windows": args.windows,
        "censor_summary": prepared.censor_summary,
        "models": evaluate_direction(
            df,
            train_batches,
            test_batches,
            build_models(),
            feature_columns=feature_columns,
            target_column=prepared.target_column,
            windows=args.windows,
        ),
    }

    for model_name, cycles_data in summary["models"].items():
        print(f"{model_name} -> {cycles_data}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved cross-dataset results to {output_path}")


if __name__ == "__main__":
    main()
