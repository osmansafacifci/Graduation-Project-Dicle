"""Unit tests for 3_analysis/target_rescaling.py."""

from __future__ import annotations

import numpy as np
import pytest

from target_rescaling import fit_point_adapter, safe_pred


# ── fit_point_adapter ──────────────────────────────────────────────────────

class TestFitPointAdapter:
    def test_residual_mean_identity_on_perfect(self):
        y_pred = np.array([100.0, 200.0, 300.0])
        y_true = np.array([100.0, 200.0, 300.0])
        slope, intercept = fit_point_adapter(y_pred, y_true, "residual_mean")
        assert slope == 1.0
        assert intercept == pytest.approx(0.0)

    def test_residual_mean_constant_offset(self):
        y_pred = np.array([100.0, 200.0, 300.0])
        y_true = np.array([110.0, 210.0, 310.0])
        slope, intercept = fit_point_adapter(y_pred, y_true, "residual_mean")
        assert slope == 1.0
        assert intercept == pytest.approx(10.0)

    def test_linear_perfect_fit(self):
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = 2.0 * y_pred + 10.0
        slope, intercept = fit_point_adapter(y_pred, y_true, "linear")
        assert slope == pytest.approx(2.0, rel=1e-6)
        assert intercept == pytest.approx(10.0, rel=1e-6)

    def test_linear_constant_pred_fallback(self):
        y_pred = np.array([5.0, 5.0, 5.0])
        y_true = np.array([10.0, 20.0, 30.0])
        slope, intercept = fit_point_adapter(y_pred, y_true, "linear")
        assert slope == 1.0
        corrected = slope * 5.0 + intercept
        assert corrected == pytest.approx(np.mean(y_true))

    def test_unknown_adapter_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            fit_point_adapter(np.array([1.0]), np.array([1.0]), "bogus")


# ── safe_pred ──────────────────────────────────────────────────────────────

class _DummyModel:
    def __init__(self, values):
        self._values = values

    def predict(self, X):
        return self._values


class TestSafePred:
    def test_normal_values(self):
        model = _DummyModel(np.array([1.0, 2.0, 3.0]))
        result = safe_pred(model, np.zeros((3, 1)))
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_nan_replaced(self):
        model = _DummyModel(np.array([np.nan, 5.0]))
        result = safe_pred(model, np.zeros((2, 1)))
        assert result[0] == 0.0
        assert result[1] == 5.0

    def test_inf_clipped(self):
        model = _DummyModel(np.array([np.inf, -np.inf]))
        result = safe_pred(model, np.zeros((2, 1)))
        assert result[0] <= 1e9
        assert result[1] >= -1e9

    def test_2d_prediction_raveled(self):
        model = _DummyModel(np.array([[1.0], [2.0], [3.0]]))
        result = safe_pred(model, np.zeros((3, 1)))
        assert result.ndim == 1
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_large_values_clipped(self):
        model = _DummyModel(np.array([1e15, -1e15]))
        result = safe_pred(model, np.zeros((2, 1)))
        assert result[0] == 1e9
        assert result[1] == -1e9
