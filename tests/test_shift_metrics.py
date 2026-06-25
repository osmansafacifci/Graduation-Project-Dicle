"""Unit tests for 3_analysis/shift_metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

from shift_metrics import (
    mahalanobis_centroid_distance,
    median_heuristic_bandwidth,
    mmd2_rbf,
    per_feature_shift,
)


# ── median_heuristic_bandwidth ─────────────────────────────────────────────

class TestMedianHeuristicBandwidth:
    def test_single_point_returns_one(self):
        X = np.array([[1.0, 2.0]])
        assert median_heuristic_bandwidth(X) == 1.0

    def test_two_points(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        bw = median_heuristic_bandwidth(X)
        assert bw == pytest.approx(5.0)

    def test_identical_points_returns_one(self):
        X = np.ones((10, 3))
        bw = median_heuristic_bandwidth(X)
        assert bw == 1.0

    def test_positive(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 5))
        bw = median_heuristic_bandwidth(X)
        assert bw > 0

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 5))
        bw1 = median_heuristic_bandwidth(X, seed=42)
        bw2 = median_heuristic_bandwidth(X, seed=42)
        assert bw1 == bw2


# ── mmd2_rbf ───────────────────────────────────────────────────────────────

class TestMmd2Rbf:
    def test_same_distribution_low_mmd(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        Y = rng.standard_normal((100, 3))
        result = mmd2_rbf(X, Y)
        assert abs(result["MMD2"]) < 0.3

    def test_different_distributions_high_mmd(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 2))
        Y = rng.standard_normal((50, 2)) + 5.0
        result = mmd2_rbf(X, Y)
        assert result["MMD2"] > 0.1

    def test_identical_samples_near_zero_mmd(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3))
        result = mmd2_rbf(X, X.copy())
        # Unbiased MMD² estimator can be slightly negative for identical samples
        assert abs(result["MMD2"]) < 0.1

    def test_result_keys(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 2))
        Y = rng.standard_normal((10, 2))
        result = mmd2_rbf(X, Y)
        assert "MMD2" in result
        assert "MMD" in result
        assert "sigma" in result

    def test_mmd_equals_sqrt_mmd2(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 2))
        Y = rng.standard_normal((30, 2)) + 2.0
        result = mmd2_rbf(X, Y)
        assert result["MMD"] == pytest.approx(
            np.sqrt(max(result["MMD2"], 0.0)), rel=1e-10
        )

    def test_custom_sigma(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 2))
        Y = rng.standard_normal((20, 2))
        result = mmd2_rbf(X, Y, sigma=1.0)
        assert result["sigma"] == 1.0


# ── mahalanobis_centroid_distance ──────────────────────────────────────────

class TestMahalanobisCentroidDistance:
    def test_identical_samples_zero(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 3))
        result = mahalanobis_centroid_distance(X, X.copy())
        assert result["Mahalanobis"] == pytest.approx(0.0, abs=1e-8)

    def test_shifted_mean(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))
        Y = rng.standard_normal((50, 3)) + 3.0
        result = mahalanobis_centroid_distance(X, Y)
        assert result["Mahalanobis"] > 1.0

    def test_result_keys(self):
        X = np.array([[1.0], [2.0], [3.0]])
        Y = np.array([[4.0], [5.0], [6.0]])
        result = mahalanobis_centroid_distance(X, Y)
        assert "Mahalanobis2" in result
        assert "Mahalanobis" in result
        assert "ridge" in result

    def test_d2_nonnegative(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((20, 5))
        Y = rng.standard_normal((20, 5))
        result = mahalanobis_centroid_distance(X, Y)
        assert result["Mahalanobis2"] >= -1e-10


# ── per_feature_shift ──────────────────────────────────────────────────────

class TestPerFeatureShift:
    def test_identical_distributions(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))
        rows = per_feature_shift(X, X.copy(), ["a", "b", "c"])
        for r in rows:
            assert r["abs_mean_shift_z"] == pytest.approx(0.0, abs=1e-10)

    def test_one_feature_shifted(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        Y = X.copy()
        Y[:, 1] += 5.0
        rows = per_feature_shift(X, Y, ["a", "big_shift", "c"])
        assert rows[0]["feature"] == "big_shift"
        assert rows[0]["abs_mean_shift_z"] > 1.0

    def test_sorted_descending(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        Y = rng.standard_normal((100, 4))
        Y[:, 2] += 3.0
        rows = per_feature_shift(X, Y, ["a", "b", "c", "d"])
        shifts = [r["abs_mean_shift_z"] for r in rows]
        assert shifts == sorted(shifts, reverse=True)

    def test_row_keys(self):
        X = np.array([[1.0], [2.0]])
        Y = np.array([[3.0], [4.0]])
        rows = per_feature_shift(X, Y, ["feat1"])
        assert len(rows) == 1
        assert set(rows[0].keys()) == {"feature", "abs_mean_shift_z", "mu_matr", "mu_hust", "pooled_std"}
