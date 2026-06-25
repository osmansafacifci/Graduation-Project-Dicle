"""Unit tests for 2_models/metrics_utils.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metrics_utils import (
    bootstrap_metric_ci,
    compute_metrics,
    symmetric_mape,
    to_cycles,
)


# ── to_cycles ──────────────────────────────────────────────────────────────

class TestToCycles:
    def test_log_target_basic(self):
        pred = np.log([100.0, 200.0, 500.0])
        result = to_cycles(pred, log_target=True)
        np.testing.assert_allclose(result, [100.0, 200.0, 500.0], rtol=1e-6)

    def test_linear_target_passthrough(self):
        pred = np.array([100.0, 200.0, 500.0])
        result = to_cycles(pred, log_target=False)
        np.testing.assert_array_equal(result, pred)

    def test_nan_replaced_log(self):
        pred = np.array([np.nan, np.log(50.0)])
        result = to_cycles(pred, log_target=True)
        assert result[0] == 1.0
        np.testing.assert_allclose(result[1], 50.0, rtol=1e-6)

    def test_nan_replaced_linear(self):
        pred = np.array([np.nan, 50.0])
        result = to_cycles(pred, log_target=False)
        assert result[0] == 1.0

    def test_extreme_log_clipped(self):
        pred = np.array([1000.0])
        result = to_cycles(pred, log_target=True, max_cycle=1e6)
        assert result[0] <= 1e6

    def test_negative_clipped_to_min(self):
        pred = np.array([-100.0])
        result = to_cycles(pred, log_target=True, min_cycle=1.0)
        assert result[0] == 1.0

    def test_custom_bounds(self):
        pred = np.array([np.log(5.0)])
        result = to_cycles(pred, log_target=True, min_cycle=10.0)
        assert result[0] == pytest.approx(10.0)

    def test_inf_handled_log(self):
        pred = np.array([np.inf, -np.inf])
        result = to_cycles(pred, log_target=True, max_cycle=1e9)
        assert np.isfinite(result).all()

    def test_inf_handled_linear(self):
        pred = np.array([np.inf, -np.inf])
        result = to_cycles(pred, log_target=False)
        assert np.isfinite(result).all()

    def test_empty_array(self):
        result = to_cycles(np.array([]), log_target=True)
        assert len(result) == 0


# ── symmetric_mape ─────────────────────────────────────────────────────────

class TestSymmetricMape:
    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        assert symmetric_mape(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([100.0])
        y_pred = np.array([80.0])
        denom = (100.0 + 80.0) / 2.0
        expected = abs(80 - 100) / denom * 100.0
        assert symmetric_mape(y_true, y_pred) == pytest.approx(expected, rel=1e-6)

    def test_symmetric(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        assert symmetric_mape(y_true, y_pred) == pytest.approx(
            symmetric_mape(y_pred, y_true), rel=1e-10
        )

    def test_both_zero(self):
        y_true = np.array([0.0])
        y_pred = np.array([0.0])
        assert symmetric_mape(y_true, y_pred) == 0.0

    def test_one_zero(self):
        y_true = np.array([0.0])
        y_pred = np.array([10.0])
        result = symmetric_mape(y_true, y_pred)
        assert result == pytest.approx(200.0)


# ── compute_metrics ────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect(self):
        y = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(y, y)
        assert m["MAE"] == pytest.approx(0.0)
        assert m["SMAPE"] == pytest.approx(0.0)
        assert m["R2"] == pytest.approx(1.0)

    def test_keys(self):
        m = compute_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert set(m.keys()) == {"MAE", "SMAPE", "R2"}

    def test_positive_mae(self):
        m = compute_metrics(np.array([100.0, 200.0]), np.array([110.0, 220.0]))
        assert m["MAE"] > 0

    def test_r2_negative_for_bad_pred(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        m = compute_metrics(y_true, y_pred)
        assert m["R2"] < 0


# ── bootstrap_metric_ci ───────────────────────────────────────────────────

class TestBootstrapMetricCi:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            bootstrap_metric_ci(np.array([1.0]), np.array([1.0, 2.0]))

    def test_single_sample_returns_nan(self):
        ci = bootstrap_metric_ci(np.array([1.0]), np.array([2.0]))
        for key in ("MAE", "SMAPE", "R2"):
            assert math.isnan(ci[key]["lower"])
            assert math.isnan(ci[key]["upper"])

    def test_ci_structure(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(100, 500, size=30)
        y_pred = y_true + rng.normal(0, 20, size=30)
        ci = bootstrap_metric_ci(y_true, y_pred, n_bootstrap=200, seed=42)
        for key in ("MAE", "SMAPE", "R2"):
            assert "lower" in ci[key]
            assert "upper" in ci[key]
            assert ci[key]["lower"] <= ci[key]["upper"]

    def test_perfect_pred_ci(self):
        y = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        ci = bootstrap_metric_ci(y, y, n_bootstrap=100, seed=0)
        assert ci["MAE"]["lower"] == pytest.approx(0.0)
        assert ci["MAE"]["upper"] == pytest.approx(0.0)
        assert ci["R2"]["lower"] == pytest.approx(1.0)
        assert ci["R2"]["upper"] == pytest.approx(1.0)

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(99)
        y_true = rng.uniform(100, 500, size=20)
        y_pred = y_true + rng.normal(0, 30, size=20)
        ci1 = bootstrap_metric_ci(y_true, y_pred, n_bootstrap=50, seed=7)
        ci2 = bootstrap_metric_ci(y_true, y_pred, n_bootstrap=50, seed=7)
        assert ci1 == ci2

    def test_wider_interval_with_noise(self):
        rng = np.random.default_rng(1)
        y_true = rng.uniform(100, 500, size=20)
        y_good = y_true + rng.normal(0, 5, size=20)
        y_bad = y_true + rng.normal(0, 100, size=20)
        ci_good = bootstrap_metric_ci(y_true, y_good, n_bootstrap=300, seed=42)
        ci_bad = bootstrap_metric_ci(y_true, y_bad, n_bootstrap=300, seed=42)
        width_good = ci_good["MAE"]["upper"] - ci_good["MAE"]["lower"]
        width_bad = ci_bad["MAE"]["upper"] - ci_bad["MAE"]["lower"]
        assert width_bad > width_good
