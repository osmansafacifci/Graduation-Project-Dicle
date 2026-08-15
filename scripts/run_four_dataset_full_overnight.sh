#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

echo "[start] $(date)"
echo "[repo] $(pwd)"
echo "[commit] $(git rev-parse --short HEAD)"

echo "[step] rebuild four-dataset feature tables"
python 1_features/build_four_dataset_table.py

echo "[step] rebuild four-dataset splits"
python 2_models/generate_splits.py \
  --features-path data/intermediate/features_sop12_four_dataset.csv \
  --output-dir splits/sop_v2_four_dataset

echo "[step] full within-dataset run"
python 2_models/run_experiments.py \
  --features-path data/intermediate/features_sop12_four_dataset.csv \
  --splits-dir splits/sop_v2_four_dataset \
  --datasets matr hust sandia luh \
  --models elastic_net pls random_forest xgboost catboost gaussian_process stacking \
  --windows 50 100 \
  --output-dir outputs/results_v2_four_dataset_within_34feat_log

echo "[step] full naive cross-dataset run"
python 2_models/run_experiments.py \
  --features-path data/intermediate/features_sop12_four_dataset_capnorm.csv \
  --splits-dir splits/sop_v2_four_dataset \
  --datasets matr hust sandia luh \
  --models elastic_net pls random_forest xgboost catboost gaussian_process stacking \
  --windows 50 100 \
  --cross-dataset \
  --output-dir outputs/results_v2_four_dataset_cross_34feat_capnorm_log

echo "[done] $(date)"
