"""Unit tests for 2_models/generate_splits.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from generate_splits import make_splits_for_dataset, SEEDS


def _make_dummy_df(n_cells: int = 40, censored_frac: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_censored = int(n_cells * censored_frac)
    rows = []
    for i in range(n_cells):
        is_cens = 1 if i < n_censored else 0
        rows.append({
            "cell_id": f"cell_{i:03d}",
            "n_cycles": 100,
            "cycle_life": float("nan") if is_cens else float(rng.integers(200, 1500)),
            "is_censored": is_cens,
        })
    return pd.DataFrame(rows)


class TestMakeSplitsForDataset:
    def test_all_seeds_returned(self):
        df = _make_dummy_df()
        splits = make_splits_for_dataset(df, "test_ds")
        assert set(splits.keys()) == set(SEEDS)

    def test_no_overlap(self):
        df = _make_dummy_df()
        splits = make_splits_for_dataset(df, "test_ds")
        for seed, split in splits.items():
            train = set(split["train"])
            cal = set(split["calibration"])
            test = set(split["test"])
            assert train & cal == set()
            assert train & test == set()
            assert cal & test == set()

    def test_all_modeling_cells_assigned(self):
        df = _make_dummy_df(n_cells=40, censored_frac=0.1)
        n_modeling = int(df[df["is_censored"] == 0].shape[0])
        splits = make_splits_for_dataset(df, "test_ds")
        for seed, split in splits.items():
            total = len(split["train"]) + len(split["calibration"]) + len(split["test"])
            assert total == n_modeling

    def test_censored_cells_excluded(self):
        df = _make_dummy_df(n_cells=50, censored_frac=0.2)
        censored_ids = set(df[df["is_censored"] == 1]["cell_id"])
        splits = make_splits_for_dataset(df, "test_ds")
        for seed, split in splits.items():
            all_split_ids = set(split["train"]) | set(split["calibration"]) | set(split["test"])
            assert all_split_ids & censored_ids == set()

    def test_metadata_fields(self):
        df = _make_dummy_df()
        splits = make_splits_for_dataset(df, "myds")
        for seed, split in splits.items():
            assert split["dataset"] == "myds"
            assert split["seed"] == seed
            assert split["ratios"] == [0.70, 0.15, 0.15]
            assert split["censored_excluded"] is True

    def test_approximate_ratios(self):
        df = _make_dummy_df(n_cells=200, censored_frac=0.05)
        n_modeling = df[df["is_censored"] == 0].shape[0]
        splits = make_splits_for_dataset(df, "test_ds")
        for seed, split in splits.items():
            train_frac = len(split["train"]) / n_modeling
            assert 0.55 <= train_frac <= 0.85

    def test_deterministic(self):
        df = _make_dummy_df()
        splits1 = make_splits_for_dataset(df, "test_ds")
        splits2 = make_splits_for_dataset(df, "test_ds")
        for seed in SEEDS:
            assert splits1[seed]["train"] == splits2[seed]["train"]
            assert splits1[seed]["test"] == splits2[seed]["test"]

    def test_small_dataset(self):
        rows = [
            {"cell_id": f"c{i}", "n_cycles": 100,
             "cycle_life": float(100 + i * 50), "is_censored": 0}
            for i in range(5)
        ]
        df = pd.DataFrame(rows)
        splits = make_splits_for_dataset(df, "tiny")
        for seed, split in splits.items():
            total = len(split["train"]) + len(split["calibration"]) + len(split["test"])
            assert total == 5
            assert len(split["calibration"]) >= 1
            assert len(split["test"]) >= 1
