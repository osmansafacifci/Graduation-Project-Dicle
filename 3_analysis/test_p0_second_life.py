#!/usr/bin/env python3
"""Focused unit tests for the locked P0 gate implementation."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


SCRIPT = Path(__file__).with_name("run_p0_second_life.py")
SPEC = importlib.util.spec_from_file_location("p0_second_life", SCRIPT)
P0 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = P0
SPEC.loader.exec_module(P0)


class P0Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            "primary_horizon_throughput": 50,
            "history_span_throughput": 100,
            "history_grid_points": 20,
            "retirement_smoothing_points": 5,
            "retirement_band": [0.785, 0.815],
        }

    def test_trace_extraction_uses_only_trailing_history_for_features(self) -> None:
        cycles = np.arange(1, 501)
        soh = 1.0 - 0.00055 * cycles
        trace = pd.DataFrame({"cycle": cycles, "Q_discharge": 3.0 * soh})
        result = P0.extract_trace_record(trace, "synthetic", "a", {}, self.cfg)
        self.assertEqual(result["status"], "eligible_primary")
        self.assertGreater(result["target_g"], 0)
        self.assertLess(result["recent_slope"], -0.0004)
        self.assertGreater(result["recent_slope"], -0.0008)
        self.assertGreaterEqual(result["available_future_throughput"], 50)

    def test_right_censoring_is_explicit(self) -> None:
        cycles = np.arange(1, 390)
        soh = 1.0 - 0.00055 * cycles
        trace = pd.DataFrame({"cycle": cycles, "Q_discharge": 3.0 * soh})
        result = P0.extract_trace_record(trace, "synthetic", "b", {}, self.cfg)
        self.assertEqual(result["status"], "right_censored_before_horizon")
        self.assertNotIn("target_g", result)

    def test_oracle_pairing_improves_adjacent_target_difference(self) -> None:
        frame = pd.DataFrame(
            {
                "duty_stratum": ["same"] * 8,
                "target_g": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5],
                "baseline": np.arange(8),
                "oracle": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5],
            }
        )
        baseline = P0.adjacent_pair_difference(frame, "baseline", 4)
        oracle = P0.adjacent_pair_difference(frame, "oracle", 4)
        self.assertLess(oracle, baseline)

    def test_mat_discharge_capacity_parser(self) -> None:
        status = np.array(["CHA"] * 4 + ["PAU"] + ["DCH"] * 6 + ["PAU"] + ["CHA"] * 3, dtype=object)
        ah = np.array([0.0, 0.4, 0.8, 1.0, 1.0, 0.8, 0.4, 0.0, -0.4, -0.8, -1.0, -1.0, -0.5, 0.0, 0.5])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.mat"
            savemat(path, {"data": {"Status": status, "AhAccu": ah}})
            capacity = P0.discharge_capacity_from_mat(path)
        self.assertAlmostEqual(capacity, 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
