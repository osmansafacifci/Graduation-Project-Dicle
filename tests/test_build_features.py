"""Unit tests for 1_features/build_features.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from build_features import (
    _sample_entropy,
    compute_cycle_life,
    compute_extended2,
    compute_q0,
    compute_sop12,
)


# ── compute_q0 ─────────────────────────────────────────────────────────────

class TestComputeQ0:
    def test_normal_series(self):
        qd = np.array([1.1, 1.08, 1.07, 1.06, 1.05, 1.04, 1.03])
        q0 = compute_q0(qd)
        expected = np.median([1.08, 1.07, 1.06, 1.05])
        assert q0 == pytest.approx(expected)

    def test_short_series_returns_nan(self):
        qd = np.array([1.0, 1.0])
        assert math.isnan(compute_q0(qd))

    def test_exactly_five(self):
        qd = np.array([1.0, 0.99, 0.98, 0.97, 0.96])
        q0 = compute_q0(qd)
        expected = np.median([0.99, 0.98, 0.97, 0.96])
        assert q0 == pytest.approx(expected)

    def test_nan_in_series(self):
        qd = np.array([1.0, np.nan, 0.98, 0.97, 0.96])
        q0 = compute_q0(qd)
        assert np.isfinite(q0)

    def test_all_zero_returns_nan(self):
        qd = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        q0 = compute_q0(qd)
        assert math.isnan(q0)

    def test_negative_values_excluded(self):
        qd = np.array([1.0, -1.0, -2.0, -3.0, -4.0])
        q0 = compute_q0(qd)
        assert math.isnan(q0)


# ── compute_cycle_life ─────────────────────────────────────────────────────

class TestComputeCycleLife:
    def test_eol_reached(self):
        qd = np.array([1.0, 0.95, 0.90, 0.84, 0.80])
        q0 = 1.0
        cl = compute_cycle_life(qd, q0, 0.85)
        assert cl == 4.0

    def test_no_eol_returns_nan(self):
        qd = np.array([1.0, 0.99, 0.98, 0.97, 0.96])
        q0 = 1.0
        cl = compute_cycle_life(qd, q0, 0.85)
        assert math.isnan(cl)

    def test_invalid_q0_returns_nan(self):
        qd = np.array([1.0, 0.5])
        assert math.isnan(compute_cycle_life(qd, float("nan"), 0.85))
        assert math.isnan(compute_cycle_life(qd, 0.0, 0.85))
        assert math.isnan(compute_cycle_life(qd, -1.0, 0.85))

    def test_immediate_eol(self):
        qd = np.array([1.0, 0.50])
        q0 = 1.0
        cl = compute_cycle_life(qd, q0, 0.85)
        assert cl == 2.0

    def test_fraction_080(self):
        qd = np.array([1.0, 0.95, 0.85, 0.79, 0.70])
        q0 = 1.0
        cl = compute_cycle_life(qd, q0, 0.80)
        assert cl == 4.0


# ── compute_sop12 ──────────────────────────────────────────────────────────

class TestComputeSop12:
    def test_too_few_cycles_returns_none(self):
        qd = np.arange(5, dtype=float)
        assert compute_sop12(qd, 5) is None

    def test_basic_output_keys(self):
        rng = np.random.default_rng(0)
        qd = 1.0 - 0.001 * np.arange(100) + rng.normal(0, 0.0001, 100)
        feats = compute_sop12(qd, 100)
        assert feats is not None
        expected_keys = {
            "Qdis_N", "delta_Qdis", "retention_ratio", "slope_linear",
            "variance_Qdis", "range_Qdis", "max_drop", "std_diff",
            "skewness_Qdis", "slope_ratio", "Qdis_cycle10", "mean_diff",
        }
        assert set(feats.keys()) == expected_keys

    def test_monotonic_decrease(self):
        qd = np.linspace(1.0, 0.8, 100)
        feats = compute_sop12(qd, 100)
        assert feats["delta_Qdis"] < 0
        assert feats["slope_linear"] < 0
        assert feats["retention_ratio"] < 1.0

    def test_constant_series(self):
        qd = np.ones(100)
        feats = compute_sop12(qd, 100)
        assert feats["delta_Qdis"] == pytest.approx(0.0)
        assert feats["variance_Qdis"] == pytest.approx(0.0, abs=1e-12)
        assert feats["range_Qdis"] == pytest.approx(0.0, abs=1e-12)

    def test_n_larger_than_series(self):
        qd = np.linspace(1.0, 0.9, 50)
        feats = compute_sop12(qd, 200)
        assert feats is not None
        assert feats["Qdis_N"] == pytest.approx(qd[-1], rel=1e-6)

    def test_qdis_cycle10(self):
        qd = np.arange(20, dtype=float) * 0.1
        feats = compute_sop12(qd, 20)
        assert feats["Qdis_cycle10"] == pytest.approx(qd[9])


# ── _sample_entropy ────────────────────────────────────────────────────────

class TestSampleEntropy:
    def test_constant_returns_zero(self):
        assert _sample_entropy(np.ones(50)) == 0.0

    def test_too_short_returns_zero(self):
        assert _sample_entropy(np.array([1.0, 2.0])) == 0.0

    def test_random_positive(self):
        rng = np.random.default_rng(42)
        series = rng.standard_normal(100)
        se = _sample_entropy(series)
        assert se > 0

    def test_deterministic(self):
        rng = np.random.default_rng(0)
        series = rng.standard_normal(50)
        assert _sample_entropy(series) == _sample_entropy(series)

    def test_periodic_lower_than_random(self):
        periodic = np.sin(np.linspace(0, 4 * np.pi, 100))
        rng = np.random.default_rng(5)
        random_series = rng.standard_normal(100)
        se_periodic = _sample_entropy(periodic)
        se_random = _sample_entropy(random_series)
        assert se_periodic < se_random


# ── compute_extended2 ──────────────────────────────────────────────────────

class TestComputeExtended2:
    def test_output_keys(self):
        qd = np.linspace(1.0, 0.8, 100)
        feats = compute_extended2(qd, 100, q0=1.0)
        expected_keys = {
            "accel_mean", "accel_std", "accel_max_abs",
            "linearity_r2", "kurtosis_Qdis",
            "fft_top3_energy_ratio", "spectral_entropy", "sample_entropy",
            "pos_neg_diff_ratio", "mad_Qdis",
        }
        assert set(feats.keys()) == expected_keys

    def test_linear_decay_high_r2(self):
        qd = np.linspace(1.0, 0.8, 100)
        feats = compute_extended2(qd, 100, q0=1.0)
        assert feats["linearity_r2"] > 0.99

    def test_too_short_returns_zeros(self):
        qd = np.array([1.0, 0.99])
        feats = compute_extended2(qd, 2, q0=1.0)
        for v in feats.values():
            assert v == 0.0

    def test_constant_series(self):
        qd = np.ones(50)
        feats = compute_extended2(qd, 50, q0=1.0)
        assert feats["accel_mean"] == pytest.approx(0.0)
        assert feats["accel_std"] == pytest.approx(0.0)
        assert feats["mad_Qdis"] == pytest.approx(0.0)

    def test_fft_energy_bounded(self):
        rng = np.random.default_rng(0)
        qd = 1.0 + rng.normal(0, 0.01, 100)
        feats = compute_extended2(qd, 100, q0=1.0)
        assert 0.0 <= feats["fft_top3_energy_ratio"] <= 1.0

    def test_monotonic_decay_pos_neg_ratio(self):
        qd = np.linspace(1.0, 0.8, 100)
        feats = compute_extended2(qd, 100, q0=1.0)
        assert feats["pos_neg_diff_ratio"] == pytest.approx(0.0)
