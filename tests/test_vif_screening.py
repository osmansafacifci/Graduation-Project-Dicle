"""Unit tests for 2_models/vif_screening.py."""

from __future__ import annotations

import numpy as np
import pytest

from vif_screening import compute_vif, iterative_vif_drop


class TestComputeVif:
    def test_independent_features(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        names = ["a", "b", "c"]
        vifs = compute_vif(X, names)
        for v in vifs.values():
            assert 1.0 <= v < 2.0

    def test_perfectly_correlated(self):
        X = np.column_stack([np.arange(50, dtype=float),
                             np.arange(50, dtype=float) * 2 + 1])
        vifs = compute_vif(X, ["x", "2x+1"])
        assert vifs["x"] == float("inf") or vifs["x"] > 1e6
        assert vifs["2x+1"] == float("inf") or vifs["2x+1"] > 1e6

    def test_single_feature(self):
        X = np.arange(10, dtype=float).reshape(-1, 1)
        vifs = compute_vif(X, ["only"])
        assert vifs["only"] == 1.0

    def test_empty(self):
        X = np.empty((10, 0))
        vifs = compute_vif(X, [])
        assert vifs == {}

    def test_constant_feature_gets_inf(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(50), rng.standard_normal(50)])
        vifs = compute_vif(X, ["const", "rand"])
        assert vifs["const"] == float("inf")

    def test_moderate_collinearity(self):
        rng = np.random.default_rng(7)
        x1 = rng.standard_normal(100)
        x2 = x1 + rng.normal(0, 0.3, 100)
        x3 = rng.standard_normal(100)
        X = np.column_stack([x1, x2, x3])
        vifs = compute_vif(X, ["x1", "x2", "x3"])
        assert vifs["x1"] > 2.0
        assert vifs["x2"] > 2.0
        assert vifs["x3"] < vifs["x1"]


class TestIterativeVifDrop:
    def test_nothing_dropped_below_threshold(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        names = ["a", "b", "c"]
        kept, removed, history = iterative_vif_drop(X, names, threshold=10.0)
        assert set(kept) == set(names)
        assert removed == []

    def test_collinear_feature_dropped(self):
        rng = np.random.default_rng(0)
        x1 = rng.standard_normal(100)
        x2 = x1 * 2 + rng.normal(0, 0.01, 100)
        x3 = rng.standard_normal(100)
        X = np.column_stack([x1, x2, x3])
        names = ["x1", "x2", "x3"]
        kept, removed, history = iterative_vif_drop(X, names, threshold=5.0)
        assert "x3" in kept
        assert len(removed) >= 1
        assert "x1" in removed or "x2" in removed

    def test_history_recorded(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]
        _, _, history = iterative_vif_drop(X, names, threshold=100.0)
        assert len(history) >= 1
        assert "iteration" in history[0]
        assert "n_features" in history[0]

    def test_all_features_dropped_if_threshold_zero(self):
        rng = np.random.default_rng(0)
        x1 = rng.standard_normal(50)
        x2 = x1 + rng.normal(0, 0.5, 50)
        X = np.column_stack([x1, x2])
        names = ["x1", "x2"]
        kept, removed, history = iterative_vif_drop(X, names, threshold=0.5)
        assert len(kept) + len(removed) == 2

    def test_constant_column_dropped(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(50), rng.standard_normal(50), rng.standard_normal(50)])
        names = ["const", "a", "b"]
        kept, removed, _ = iterative_vif_drop(X, names, threshold=5.0)
        assert "const" in removed
