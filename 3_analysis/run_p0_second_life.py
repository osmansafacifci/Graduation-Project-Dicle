#!/usr/bin/env python3
"""Run the preregistered P0 feasibility gates for second-life battery grading.

The script deliberately separates structural feasibility, an oracle value ceiling,
prospective cell-level prediction, and genuine first-to-second-life confirmation.
It writes every decision, including negative and stopped gates, to machine-readable
files so the result cannot be reconstructed selectively after seeing the data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import requests
import scipy
from scipy.io import loadmat
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - reported clearly at runtime
    CatBoostRegressor = None


DATASETS = ("luh", "hust", "sandia", "matr")
HISTORY_FEATURES = (
    "soh_t0",
    "recent_slope",
    "early_slope",
    "slope_change",
    "curvature",
    "history_mad",
    "max_drop",
    "linearity_r2",
    "slope_ratio",
    "history_range",
)
DUTY_NUMERIC = (
    "future_temp",
    "future_charge_rate",
    "future_discharge_rate",
    "future_soc_min",
    "future_soc_max",
)
DUTY_CATEGORICAL = ("chemistry", "future_protocol")
BASELINE_FEATURES = ("soh_t0", "recent_slope", *DUTY_NUMERIC, *DUTY_CATEGORICAL)
FULL_FEATURES = (*HISTORY_FEATURES, *DUTY_NUMERIC, *DUTY_CATEGORICAL)


@dataclass
class GateResult:
    gate: int
    name: str
    decision: str
    reason: str
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "name": self.name,
            "decision": self.decision,
            "reason": self.reason,
            "metrics": json_safe(self.metrics),
        }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--gates", nargs="+", type=int, choices=(0, 1, 2, 3), default=[0, 1, 2, 3])
    parser.add_argument("--download-mowri", action="store_true")
    parser.add_argument("--respect-gates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Use one seed and 20 bootstraps.")
    return parser.parse_args()


def load_lock(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    cfg = json.loads(raw)
    return cfg, hashlib.sha256(raw).hexdigest()


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_metric(value: Any, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{100 * number:.1f}%" if percent else f"{number:.4g}"


def write_markdown_summary(path: Path, final: dict[str, Any]) -> None:
    results = {int(item["gate"]): item for item in final["results"]}
    lines = [
        "# P0 second-life grading gate report",
        "",
        f"Locked analysis decision: **{final['overall_decision']}**  ",
        f"Permitted claim scope: **{final['claim_scope']}**  ",
        f"Lock SHA-256: `{final['lock_sha256']}`",
        "",
        "| Gate | Question | Decision |",
        "|---:|---|---|",
    ]
    questions = {
        0: "Are the cohorts structurally usable?",
        1: "Is there enough oracle value to justify prediction?",
        2: "Does prospective grading clear the locked utility tests?",
        3: "Does the frozen grader confirm on genuine second-life data?",
    }
    for gate in sorted(results):
        item = results[gate]
        lines.append(f"| {gate} | {questions[gate]} | **{item['decision']}** |")
    lines.extend(["", "## Evidence", ""])
    if 0 in results:
        lines.append("Gate 0 cohort counts:")
        lines.append("")
        lines.append("| Dataset | Primary cells | Follow-up | Cell-level | Pack structure |")
        lines.append("|---|---:|---:|---|---|")
        for row in results[0]["metrics"].get("datasets", []):
            lines.append(
                f"| {row['dataset']} | {row['primary_eligible_cells']} | "
                f"{format_metric(row['followup_fraction'], percent=True)} | "
                f"{'pass' if row['cell_level_pass'] else 'fail'} | "
                f"{'pass' if row['pack_level_pass'] else 'fail'} |"
            )
    if 1 in results and results[1]["metrics"]:
        m = results[1]["metrics"]
        lines.extend(
            [
                "",
                f"Gate 1 used {m.get('n_cells', 'NA')} LUH cells. The signal-to-instability ratio was "
                f"{format_metric(m.get('signal_to_noise'))}; oracle within-duty pairing reduced the median "
                f"future-degradation mismatch by {format_metric(m.get('pair_benefit'), percent=True)}, with "
                f"{format_metric(m.get('positive_bootstrap_fraction'), percent=True)} positive resamples.",
            ]
        )
    if 2 in results and results[2]["metrics"]:
        m = results[2]["metrics"]
        pairing = m.get("development_pairing", {})
        lines.extend(
            [
                "",
                f"Gate 2 selected **{m.get('selected_model', 'NA')}**. Development MAE improved by "
                f"{format_metric(m.get('development_mae_improvement'), percent=True)} and the point estimate "
                f"for pairing benefit was {format_metric(pairing.get('pair_benefit'), percent=True)}, but only "
                f"{format_metric(pairing.get('positive_bootstrap_fraction'), percent=True)} of resamples were "
                "positive (locked requirement: 75%).",
            ]
        )
        for row in m.get("replications", []):
            if row.get("dataset") == "hust" and "mae_improvement" in row:
                lines.append(
                    f"HUST replication MAE changed by {format_metric(row['mae_improvement'], percent=True)} "
                    "(locked non-inferiority limit: -10%)."
                )
    if 3 in results:
        lines.extend(["", f"Gate 3 was **{results[3]['decision']}**: {results[3]['reason']}"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The data support the existence of decision-relevant heterogeneity near retirement, but the current "
            "history features and locked candidate models do not estimate it reliably enough for a second-life "
            "grading claim. The STOP is not evidence that second-life grading is impossible; it is evidence that "
            "the present operationalization should not yet anchor a manuscript pivot or justify consuming the "
            "genuine second-life confirmation dataset.",
            "",
            "All detailed decisions, predictions, cohort audits, model artifacts, and environment provenance are "
            "stored beside this report.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_number(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def parse_rate_pair(value: Any) -> tuple[float, float]:
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]
    if not numbers:
        return np.nan, np.nan
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return numbers[0], numbers[1]


def parse_soc_window(value: Any) -> tuple[float, float]:
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]
    return (numbers[0], numbers[1]) if len(numbers) >= 2 else (np.nan, np.nan)


def eligible_cell_ids(root: Path) -> dict[str, set[str]]:
    table = pd.read_csv(root / "data/intermediate/features_sop12_four_dataset.csv", usecols=["dataset", "cell_id"])
    result: dict[str, set[str]] = {}
    for dataset, group in table.groupby("dataset"):
        ids = set(group["cell_id"].astype(str))
        if dataset == "hust":
            ids = {x.removeprefix("hust_") for x in ids}
        result[str(dataset)] = ids
    return result


def metadata_tables(root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "luh": "luh_cell_audit.csv",
        "hust": "hust_threshold_audit.csv",
        "sandia": "sandia_cell_audit.csv",
        "matr": "matr_cell_audit_replication.csv",
    }
    return {name: pd.read_csv(root / "data/intermediate" / filename) for name, filename in paths.items()}


def standardized_metadata(dataset: str, cell_id: str, row: pd.Series | None) -> dict[str, Any]:
    row = row if row is not None else pd.Series(dtype=object)
    if dataset == "luh":
        temp = parse_number(row.get("age_temp"))
        charge = parse_number(row.get("age_chg_rate"))
        discharge = parse_number(row.get("age_dischg_rate"))
        soc_min = parse_number(row.get("age_soc_min"))
        soc_max = parse_number(row.get("age_soc_max"))
        protocol = f"T{temp:g}_C{charge:g}_D{discharge:g}_S{soc_min:g}-{soc_max:g}"
        chemistry = "NMC"
    elif dataset == "hust":
        rates = tuple(parse_number(row.get(f"dchg_rate_{i}")) for i in (1, 2, 3))
        temp, charge, discharge, soc_min, soc_max = 25.0, 1.0, float(np.nanmean(rates)), np.nan, np.nan
        protocol = "D" + "-".join("NA" if np.isnan(x) else f"{x:g}" for x in rates)
        chemistry = "LFP"
    elif dataset == "sandia":
        temp = parse_number(row.get("temperature"))
        charge, discharge = parse_rate_pair(row.get("charge_discharge_rate"))
        soc_min, soc_max = parse_soc_window(row.get("soc_window"))
        chemistry = str(row.get("cathode", "unknown"))
        protocol = f"{chemistry}_T{temp:g}_S{soc_min:g}-{soc_max:g}_C{charge:g}-D{discharge:g}"
    else:
        temp = charge = discharge = soc_min = soc_max = np.nan
        chemistry = "LFP"
        protocol = str(row.get("orig_batch", "unknown"))
    return {
        "future_temp": temp,
        "future_charge_rate": charge,
        "future_discharge_rate": discharge,
        "future_soc_min": soc_min,
        "future_soc_max": soc_max,
        "chemistry": chemistry,
        "future_protocol": protocol,
        "duty_stratum": protocol,
    }


def linear_features(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, 1)
    split = max(3, len(x) // 2)
    early_slope = float(np.polyfit(x[:split], y[:split], 1)[0])
    recent_slope = float(np.polyfit(x[-split:], y[-split:], 1)[0])
    quadratic = np.polyfit(x, y, 2)
    fitted = slope * x + intercept
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / denominator if denominator > 0 else 0.0
    scale = max(abs(early_slope), 1e-8)
    return {
        "soh_t0": float(y[-1]),
        "recent_slope": recent_slope,
        "early_slope": early_slope,
        "slope_change": recent_slope - early_slope,
        "curvature": float(2.0 * quadratic[0]),
        "history_mad": float(np.median(np.abs(y - np.median(y)))),
        "max_drop": float(max(0.0, -np.min(np.diff(y)))),
        "linearity_r2": r2,
        "slope_ratio": recent_slope / scale,
        "history_range": float(np.max(y) - np.min(y)),
    }


def extract_trace_record(
    trace: pd.DataFrame,
    dataset: str,
    cell_id: str,
    metadata: dict[str, Any],
    cfg: dict[str, Any],
    horizon: float | None = None,
) -> dict[str, Any]:
    horizon = float(horizon if horizon is not None else cfg["primary_horizon_throughput"])
    history_span = float(cfg["history_span_throughput"])
    grid_points = int(cfg["history_grid_points"])
    window = int(cfg["retirement_smoothing_points"])
    lo, hi = map(float, cfg["retirement_band"])

    frame = trace[["cycle", "Q_discharge"]].copy()
    frame["cycle"] = pd.to_numeric(frame["cycle"], errors="coerce")
    frame["Q_discharge"] = pd.to_numeric(frame["Q_discharge"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame["Q_discharge"] > 0].groupby("cycle", as_index=False)["Q_discharge"].median()
    frame = frame.sort_values("cycle")
    if len(frame) < max(20, grid_points):
        return {"dataset": dataset, "cell_id": cell_id, "status": "too_few_observations"}

    first = frame.iloc[: min(10, len(frame))]["Q_discharge"].to_numpy()
    q0 = float(np.median(first[first > 0]))
    if not np.isfinite(q0) or q0 <= 0:
        return {"dataset": dataset, "cell_id": cell_id, "status": "invalid_q0"}
    q = frame["Q_discharge"].to_numpy(float)
    soh = q / q0
    theta = np.cumsum(q / q0)
    smooth = pd.Series(soh).rolling(window, min_periods=1).median().to_numpy()
    band = np.flatnonzero((smooth >= lo) & (smooth <= hi) & (theta >= history_span))
    if not len(band):
        crossed = bool(np.any(smooth > hi) and np.any(smooth < lo))
        return {
            "dataset": dataset,
            "cell_id": cell_id,
            "status": "band_skipped" if crossed else "retirement_not_observed",
            "q0": q0,
        }
    index = int(band[0])
    theta0 = float(theta[index])
    history_grid = np.linspace(theta0 - history_span, theta0, grid_points)
    history_soh = np.interp(history_grid, theta, smooth)
    relative_grid = history_grid - theta0
    result: dict[str, Any] = {
        "dataset": dataset,
        "cell_id": cell_id,
        "status": "eligible_history",
        "q0": q0,
        "retirement_cycle": float(frame.iloc[index]["cycle"]),
        "retirement_throughput": theta0,
        "available_future_throughput": float(theta[-1] - theta0),
        **linear_features(relative_grid, history_soh),
        **metadata,
    }
    pre = (theta >= theta0 - history_span) & (theta <= theta0)
    residual = soh[pre] - smooth[pre]
    result["local_noise"] = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    if theta[-1] + 1e-9 < theta0 + horizon:
        result["status"] = "right_censored_before_horizon"
        return result
    future_soh = float(np.interp(theta0 + horizon, theta, smooth))
    valid_band = band[theta[band] + horizon <= theta[-1] + 1e-9]
    band_targets = np.asarray(
        [
            (smooth[j] - float(np.interp(theta[j] + horizon, theta, smooth))) / horizon
            for j in valid_band
        ],
        dtype=float,
    )
    target_instability = (
        float(1.4826 * np.median(np.abs(band_targets - np.median(band_targets))))
        if len(band_targets) >= 3
        else np.nan
    )
    result.update(
        {
            "status": "eligible_primary",
            "future_soh": future_soh,
            "target_g": (float(history_soh[-1]) - future_soh) / horizon,
            "target_horizon": horizon,
            "target_instability": target_instability,
        }
    )
    return result


def build_local_cohort(root: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = eligible_cell_ids(root)
    audits = metadata_tables(root)
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        cycles = pd.read_csv(root / "data/intermediate" / f"{dataset}_cycles_tidy.csv")
        cycles["cell_id"] = cycles["cell_id"].astype(str)
        cycles = cycles[cycles["cell_id"].isin(ids.get(dataset, set()))]
        audit = audits[dataset].copy()
        audit["cell_id"] = audit["cell_id"].astype(str)
        audit_rows = {row["cell_id"]: row for _, row in audit.iterrows()}
        for cell_id, trace in cycles.groupby("cell_id", sort=True):
            meta = standardized_metadata(dataset, cell_id, audit_rows.get(cell_id))
            records.append(extract_trace_record(trace, dataset, cell_id, meta, cfg))
    audit_frame = pd.DataFrame(records)
    primary = audit_frame[audit_frame["status"] == "eligible_primary"].copy()
    return audit_frame, primary


def gate0_structural(
    audit: pd.DataFrame, primary: pd.DataFrame, cfg: dict[str, Any], output: Path
) -> tuple[GateResult, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        part = audit[audit["dataset"] == dataset]
        history_n = int(part["status"].isin(["eligible_history", "eligible_primary", "right_censored_before_horizon"]).sum())
        primary_n = int((part["status"] == "eligible_primary").sum())
        followup_fraction = primary_n / history_n if history_n else 0.0
        cell_pass = primary_n >= int(cfg["minimum_primary_cells"]) and followup_fraction >= float(
            cfg["minimum_followup_fraction"]
        )
        ppart = primary[primary["dataset"] == dataset]
        group_sizes = ppart.groupby("duty_stratum").size() if len(ppart) else pd.Series(dtype=int)
        valid_groups = int((group_sizes >= int(cfg["pack_claim_min_cells_per_group"])).sum())
        pack_pass = valid_groups >= int(cfg["pack_claim_min_groups"])
        rows.append(
            {
                "dataset": dataset,
                "input_cells": len(part),
                "history_eligible_cells": history_n,
                "primary_eligible_cells": primary_n,
                "followup_fraction": followup_fraction,
                "cell_level_pass": cell_pass,
                "valid_pack_groups": valid_groups,
                "largest_duty_stratum": int(group_sizes.max()) if len(group_sizes) else 0,
                "pack_level_pass": pack_pass,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "p0_gate0_structural_summary.csv", index=False)
    audit.to_csv(output / "p0_gate0_cell_audit.csv", index=False)
    dev = str(cfg["development_dataset"])
    dev_pass = bool(summary.loc[summary["dataset"] == dev, "cell_level_pass"].iloc[0])
    rep_pass = int(summary[summary["dataset"].isin(cfg["replication_datasets"])]["cell_level_pass"].sum())
    decision = "GO" if dev_pass and rep_pass >= 1 else "STOP"
    reason = (
        "Cell-level development and replication are structurally estimable; pack claims remain conditional."
        if decision == "GO"
        else "The locked development cohort or every replication cohort fails the minimum follow-up requirement."
    )
    return GateResult(0, "structural_validity", decision, reason, {"datasets": rows}), summary


def make_preprocessor(feature_names: Iterable[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    features = list(feature_names)
    categorical = [x for x in features if x in DUTY_CATEGORICAL]
    numeric = [x for x in features if x not in categorical]
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        sparse_threshold=0.0,
    )
    return preprocessor, numeric, categorical


def model_pipeline(model_name: str, feature_names: Iterable[str], seed: int) -> Pipeline:
    preprocessor, _, _ = make_preprocessor(feature_names)
    if model_name == "baseline":
        model = Ridge(alpha=1.0)
    elif model_name == "elasticnet":
        model = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=20000, random_state=seed)
    elif model_name == "catboost":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is required for the locked P0 candidate set")
        model = CatBoostRegressor(
            iterations=350,
            depth=4,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=max(1, min(4, os.cpu_count() or 1)),
        )
    else:
        raise ValueError(f"unknown model: {model_name}")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def repeated_oof(
    frame: pd.DataFrame,
    model_name: str,
    feature_names: Iterable[str],
    seeds: Iterable[int],
    folds: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    entries: list[pd.DataFrame] = []
    x = frame[list(feature_names)]
    y = frame["target_g"].to_numpy(float)
    for seed in seeds:
        splitter = KFold(n_splits=min(folds, len(frame)), shuffle=True, random_state=int(seed))
        predictions = np.full(len(frame), np.nan)
        for train, test in splitter.split(x):
            pipe = model_pipeline(model_name, feature_names, int(seed))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(x.iloc[train], y[train])
            predictions[test] = pipe.predict(x.iloc[test])
        entries.append(
            pd.DataFrame(
                {
                    "dataset": frame["dataset"].to_numpy(),
                    "cell_id": frame["cell_id"].to_numpy(),
                    "seed": int(seed),
                    "model": model_name,
                    "observed": y,
                    "predicted": predictions,
                }
            )
        )
    long = pd.concat(entries, ignore_index=True)
    averaged = long.groupby(["dataset", "cell_id", "model"], as_index=False).agg(
        observed=("observed", "first"), predicted=("predicted", "mean")
    )
    metrics = regression_metrics(averaged["observed"], averaged["predicted"])
    return long, metrics


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) >= 3 else np.nan
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(rho),
    }


def averaged_predictions(long: pd.DataFrame, model_name: str) -> pd.Series:
    part = long[long["model"] == model_name]
    return part.groupby("cell_id")["predicted"].mean()


def adjacent_pair_difference(frame: pd.DataFrame, score: str, minimum: int) -> float:
    differences: list[float] = []
    for _, group in frame.groupby("duty_stratum"):
        if len(group) < minimum:
            continue
        ordered = group.sort_values(score)
        usable = len(ordered) - len(ordered) % 2
        values = ordered.iloc[:usable]["target_g"].to_numpy(float).reshape(-1, 2)
        differences.extend(np.abs(values[:, 0] - values[:, 1]).tolist())
    return float(np.median(differences)) if differences else np.nan


def pair_benefit_bootstrap(
    frame: pd.DataFrame,
    candidate_score: str,
    baseline_score: str,
    minimum: int,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    base = adjacent_pair_difference(frame, baseline_score, minimum)
    candidate = adjacent_pair_difference(frame, candidate_score, minimum)
    point = 1.0 - candidate / base if np.isfinite(base) and base > 0 and np.isfinite(candidate) else np.nan
    rng = np.random.default_rng(seed)
    benefits: list[float] = []
    valid_groups = [g for _, g in frame.groupby("duty_stratum") if len(g) >= minimum]
    for _ in range(iterations):
        sampled = []
        for group in valid_groups:
            take = max(minimum, int(math.ceil(0.8 * len(group))))
            sampled.append(group.iloc[rng.choice(len(group), size=take, replace=False)])
        if not sampled:
            continue
        boot = pd.concat(sampled, ignore_index=True)
        b = adjacent_pair_difference(boot, baseline_score, minimum)
        c = adjacent_pair_difference(boot, candidate_score, minimum)
        if np.isfinite(b) and b > 0 and np.isfinite(c):
            benefits.append(1.0 - c / b)
    return {
        "baseline_pair_difference": base,
        "candidate_pair_difference": candidate,
        "pair_benefit": point,
        "positive_bootstrap_fraction": float(np.mean(np.asarray(benefits) > 0)) if benefits else np.nan,
        "bootstrap_p05": float(np.quantile(benefits, 0.05)) if benefits else np.nan,
        "bootstrap_p95": float(np.quantile(benefits, 0.95)) if benefits else np.nan,
        "bootstrap_valid_iterations": len(benefits),
    }


def gate1_oracle(
    primary: pd.DataFrame,
    cfg: dict[str, Any],
    output: Path,
    seeds: list[int],
    iterations: int,
) -> tuple[GateResult, pd.DataFrame]:
    dev = primary[primary["dataset"] == cfg["development_dataset"]].copy()
    baseline_long, baseline_metrics = repeated_oof(
        dev, "baseline", BASELINE_FEATURES, seeds, int(cfg["outer_folds"])
    )
    dev["baseline_score"] = dev["cell_id"].map(averaged_predictions(baseline_long, "baseline"))
    dev["oracle_score"] = dev["target_g"]
    pair = pair_benefit_bootstrap(
        dev,
        "oracle_score",
        "baseline_score",
        int(cfg["minimum_pair_stratum_cells"]),
        iterations,
        seeds[0],
    )
    residual_std = float(np.std(dev["target_g"] - dev["baseline_score"], ddof=1))
    nonzero_instability = dev.loc[dev["target_instability"] > 0, "target_instability"]
    noise_rate = float(np.nanmedian(nonzero_instability)) if len(nonzero_instability) else np.nan
    signal_to_noise = residual_std / max(noise_rate, 1e-12)
    passed = (
        signal_to_noise >= float(cfg["gate1_min_signal_to_noise"])
        and pair["pair_benefit"] >= float(cfg["gate1_min_oracle_pair_benefit"])
        and pair["positive_bootstrap_fraction"] >= float(cfg["gate1_min_positive_bootstrap_fraction"])
    )
    dev.to_csv(output / "p0_gate1_development_scores.csv", index=False)
    baseline_long.to_csv(output / "p0_gate1_baseline_oof.csv", index=False)
    metrics = {"n_cells": len(dev), "baseline": baseline_metrics, "signal_to_noise": signal_to_noise, **pair}
    return (
        GateResult(
            1,
            "oracle_value_ceiling",
            "GO" if passed else "STOP",
            "A learnable grading signal could materially improve within-duty grouping."
            if passed
            else "Even an oracle score, or the signal-to-noise ratio, misses the locked value threshold.",
            metrics,
        ),
        dev,
    )


def gate2_prediction(
    primary: pd.DataFrame,
    gate0_summary: pd.DataFrame,
    cfg: dict[str, Any],
    output: Path,
    seeds: list[int],
    iterations: int,
) -> tuple[GateResult, str | None, dict[str, Pipeline]]:
    usable = gate0_summary.loc[gate0_summary["cell_level_pass"], "dataset"].tolist()
    metric_rows: list[dict[str, Any]] = []
    all_oof: list[pd.DataFrame] = []
    predictions: dict[tuple[str, str], pd.Series] = {}
    for dataset in usable:
        frame = primary[primary["dataset"] == dataset].copy()
        for model_name, features in (
            ("baseline", BASELINE_FEATURES),
            ("elasticnet", FULL_FEATURES),
            ("catboost", FULL_FEATURES),
        ):
            long, metrics = repeated_oof(frame, model_name, features, seeds, int(cfg["outer_folds"]))
            all_oof.append(long)
            predictions[(dataset, model_name)] = averaged_predictions(long, model_name)
            metric_rows.append({"dataset": dataset, "model": model_name, "n_cells": len(frame), **metrics})
    metrics_frame = pd.DataFrame(metric_rows)
    dev_name = str(cfg["development_dataset"])
    dev_candidates = metrics_frame[
        (metrics_frame["dataset"] == dev_name) & metrics_frame["model"].isin(["elasticnet", "catboost"])
    ]
    if dev_candidates.empty:
        result = GateResult(2, "prospective_prediction", "STOP", "No structurally valid development model.", {})
        return result, None, {}
    winner = str(dev_candidates.sort_values("mae").iloc[0]["model"])
    dev = primary[primary["dataset"] == dev_name].copy()
    dev["baseline_score"] = dev["cell_id"].map(predictions[(dev_name, "baseline")])
    dev["model_score"] = dev["cell_id"].map(predictions[(dev_name, winner)])
    pair = pair_benefit_bootstrap(
        dev,
        "model_score",
        "baseline_score",
        int(cfg["minimum_pair_stratum_cells"]),
        iterations,
        seeds[0] + 17,
    )
    dev_base = metrics_frame[(metrics_frame["dataset"] == dev_name) & (metrics_frame["model"] == "baseline")].iloc[0]
    dev_model = metrics_frame[(metrics_frame["dataset"] == dev_name) & (metrics_frame["model"] == winner)].iloc[0]
    dev_improvement = 1.0 - float(dev_model["mae"]) / float(dev_base["mae"])
    rep_details: list[dict[str, Any]] = []
    replication_passes = 0
    for dataset in cfg["replication_datasets"]:
        if dataset not in usable:
            rep_details.append({"dataset": dataset, "status": "structural_fail"})
            continue
        base_row = metrics_frame[(metrics_frame["dataset"] == dataset) & (metrics_frame["model"] == "baseline")].iloc[0]
        model_row = metrics_frame[(metrics_frame["dataset"] == dataset) & (metrics_frame["model"] == winner)].iloc[0]
        improvement = 1.0 - float(model_row["mae"]) / float(base_row["mae"])
        passed = improvement >= -float(cfg["gate2_max_replication_mae_worsening"])
        replication_passes += int(passed)
        rep_details.append(
            {
                "dataset": dataset,
                "status": "pass" if passed else "fail",
                "baseline_mae": float(base_row["mae"]),
                "model_mae": float(model_row["mae"]),
                "mae_improvement": improvement,
            }
        )
    passed = (
        dev_improvement >= float(cfg["gate2_min_mae_improvement"])
        and pair["pair_benefit"] >= float(cfg["gate2_min_pair_benefit"])
        and pair["positive_bootstrap_fraction"] >= float(cfg["gate2_min_positive_bootstrap_fraction"])
        and replication_passes >= int(cfg.get("gate2_min_replication_datasets_passing", 2))
    )
    oof_frame = pd.concat(all_oof, ignore_index=True)
    oof_frame.to_csv(output / "p0_gate2_oof_predictions.csv", index=False)
    metrics_frame.to_csv(output / "p0_gate2_model_metrics.csv", index=False)
    dev.to_csv(output / "p0_gate2_development_pair_scores.csv", index=False)

    pooled = primary[primary["dataset"].isin(usable)].copy()
    fitted: dict[str, Pipeline] = {}
    for name, features in (("baseline", BASELINE_FEATURES), (winner, FULL_FEATURES)):
        pipe = model_pipeline(name, features, seeds[0])
        pipe.fit(pooled[list(features)], pooled["target_g"])
        fitted[name] = pipe
        joblib.dump(pipe, output / f"p0_gate2_frozen_{name}.joblib")
    write_json(output / "p0_gate2_frozen_schema.json", {"winner": winner, "baseline_features": BASELINE_FEATURES, "model_features": FULL_FEATURES, "training_datasets": usable})
    details = {
        "selected_model": winner,
        "development_mae_improvement": dev_improvement,
        "development_pairing": pair,
        "replication_passes": replication_passes,
        "replications": rep_details,
    }
    return (
        GateResult(
            2,
            "prospective_prediction",
            "GO" if passed else "STOP",
            "The frozen history model improves development grading and remains non-inferior in enough replications."
            if passed
            else "The prospective model misses a locked development, pairing, or replication threshold.",
            details,
        ),
        winner,
        fitted,
    )


def mendeley_json(url: str) -> Any:
    response = requests.get(
        url,
        headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)


def flatten_folder_tree(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    stack = list(nodes)
    while stack:
        node = stack.pop()
        flattened.append(node)
        stack.extend(node.get("children", []) or [])
    return flattened


def folder_files(dataset_id: str, version: int, folder_id: str) -> list[dict[str, Any]]:
    url = f"https://data.mendeley.com/public-api/datasets/{dataset_id}/files?folder_id={folder_id}&version={version}"
    payload = mendeley_json(url)
    return payload if isinstance(payload, list) else payload.get("data", payload.get("files", []))


def file_name(record: dict[str, Any]) -> str:
    return str(record.get("filename") or record.get("name") or record.get("file_name"))


def file_download_url(record: dict[str, Any]) -> str:
    value = record.get("download_url") or record.get("downloadUrl") or record.get("url")
    if isinstance(value, dict):
        value = value.get("url") or value.get("href")
    if not value:
        raise KeyError(f"No download URL in Mendeley file record: {record.keys()}")
    return str(value)


def acquire_mowri(cfg: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    dataset_id = str(cfg["mowri_dataset_id"])
    version = int(cfg["mowri_dataset_version"])
    tree_url = f"https://data.mendeley.com/public-api/datasets/{dataset_id}/folders/{version}"
    nodes = flatten_folder_tree(mendeley_json(tree_url))
    by_name = {str(node.get("name")): node for node in nodes}
    first_root = by_name.get("1st life RPT_Final")
    second_root = by_name.get("2nd life RPT_Final")
    if not first_root or not second_root:
        raise RuntimeError("Mowri folder tree does not contain the locked RPT folders")
    first_children = first_root.get("children", []) or []
    second_children = second_root.get("children", []) or []
    downloaded = 0
    for folder in first_children:
        folder_id = str(folder.get("id") or folder.get("folder_id"))
        for record in folder_files(dataset_id, version, folder_id):
            name = file_name(record)
            if not name.lower().endswith(".mat"):
                continue
            destination = raw_dir / "first_life_rpt" / str(folder.get("name")) / name
            download_file(file_download_url(record), destination)
            downloaded += 1
    for folder in second_children:
        folder_id = str(folder.get("id") or folder.get("folder_id"))
        records = folder_files(dataset_id, version, folder_id)
        zips = [record for record in records if file_name(record).lower().endswith(".zip")]
        if not zips:
            continue
        record = zips[0]
        archive = raw_dir / "second_life_archives" / file_name(record)
        download_file(file_download_url(record), archive)
        destination = raw_dir / "second_life_rpt" / str(folder.get("name"))
        destination.mkdir(parents=True, exist_ok=True)
        marker = destination / ".extracted"
        if not marker.exists():
            with zipfile.ZipFile(archive) as handle:
                for member in handle.infolist():
                    if member.filename.lower().endswith(".mat"):
                        member.filename = Path(member.filename).name
                        handle.extract(member, destination)
            marker.touch()
        downloaded += 1
    return {"first_life_folders": len(first_children), "second_life_folders": len(second_children), "download_actions": downloaded}


def mat_field(struct: Any, name: str) -> np.ndarray:
    if hasattr(struct, name):
        value = getattr(struct, name)
    elif isinstance(struct, np.ndarray) and struct.dtype.names and name in struct.dtype.names:
        value = struct[name]
    else:
        raise KeyError(name)
    return np.asarray(value).reshape(-1)


def discharge_capacity_from_mat(path: Path) -> float:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    struct = payload.get("data")
    if struct is None:
        raise KeyError(f"No data struct in {path}")
    status = np.asarray([str(x).strip().upper() for x in mat_field(struct, "Status")])
    ah = pd.to_numeric(pd.Series(mat_field(struct, "AhAccu")), errors="coerce").to_numpy(float)
    capacities: list[float] = []
    starts = np.flatnonzero((status == "DCH") & np.r_[True, status[:-1] != "DCH"])
    for start in starts:
        later_charge = np.flatnonzero(status[start + 1 :] == "CHA")
        end = start + 1 + int(later_charge[0]) if len(later_charge) else len(status)
        segment = ah[start:end]
        if not len(segment) or not np.isfinite(segment).any():
            continue
        before = ah[max(0, start - 5) : start + 1]
        peak = float(np.nanmax(before)) if np.isfinite(before).any() else float(segment[0])
        capacity = peak - float(np.nanmin(segment))
        if 0.1 < capacity < 20:
            capacities.append(capacity)
    if not capacities:
        raise ValueError(f"No discharge episode found in {path}")
    return float(max(capacities))


def mowri_file_identity(path: Path) -> tuple[str | None, int | None]:
    cell_match = re.search(r"Cell0*(\d+)", path.name, flags=re.IGNORECASE)
    cycle_match = re.search(r"_(\d+)cyc_", path.name, flags=re.IGNORECASE)
    cell = f"mowri_{int(cell_match.group(1)):03d}" if cell_match else None
    cycle = int(cycle_match.group(1)) if cycle_match else None
    return cell, cycle


def build_mowri_cohort(raw_dir: Path, cfg: dict[str, Any], output: Path) -> pd.DataFrame:
    cache = output / "p0_gate3_mowri_capacities.csv"
    if cache.exists():
        capacities = pd.read_csv(cache)
    else:
        rows: list[dict[str, Any]] = []
        for path in sorted((raw_dir / "first_life_rpt").rglob("*.mat")):
            cell, cycle = mowri_file_identity(path)
            if cell is None or cycle is None:
                continue
            try:
                rows.append({"life": "first", "rpt": np.nan, "cell_id": cell, "cycle": cycle, "capacity": discharge_capacity_from_mat(path), "file": str(path)})
            except (KeyError, ValueError, OSError) as exc:
                rows.append({"life": "first_error", "rpt": np.nan, "cell_id": cell, "cycle": cycle, "capacity": np.nan, "file": str(path), "error": str(exc)})
        for path in sorted((raw_dir / "second_life_rpt").rglob("*.mat")):
            cell, _ = mowri_file_identity(path)
            rpt_match = re.search(r"RPT\s*0*(\d+)", str(path), flags=re.IGNORECASE)
            if cell is None or not rpt_match:
                continue
            rpt = int(rpt_match.group(1))
            try:
                rows.append({"life": "second", "rpt": rpt, "cell_id": cell, "cycle": np.nan, "capacity": discharge_capacity_from_mat(path), "file": str(path)})
            except (KeyError, ValueError, OSError) as exc:
                rows.append({"life": "second_error", "rpt": rpt, "cell_id": cell, "cycle": np.nan, "capacity": np.nan, "file": str(path), "error": str(exc)})
        capacities = pd.DataFrame(rows)
        capacities.to_csv(cache, index=False)
    records: list[dict[str, Any]] = []
    first = capacities[(capacities["life"] == "first") & capacities["capacity"].notna()]
    second = capacities[(capacities["life"] == "second") & capacities["capacity"].notna()]
    span = float(cfg["history_span_throughput"])
    points = int(cfg["history_grid_points"])
    for cell, first_cell in first.groupby("cell_id"):
        first_cell = first_cell.groupby("cycle", as_index=False)["capacity"].median().sort_values("cycle")
        second_cell = second[second["cell_id"] == cell].groupby("rpt", as_index=False)["capacity"].median().sort_values("rpt")
        if len(first_cell) < 4 or len(second_cell) < 2:
            continue
        q0 = float(first_cell.iloc[0]["capacity"])
        end_cycle = float(first_cell.iloc[-1]["cycle"])
        if end_cycle - float(first_cell.iloc[0]["cycle"]) < span:
            continue
        grid = np.linspace(end_cycle - span, end_cycle, points)
        history = np.interp(grid, first_cell["cycle"], first_cell["capacity"] / q0)
        first_rpt = second_cell.iloc[0]
        last_rpt = second_cell.iloc[-1]
        intervals = float(last_rpt["rpt"] - first_rpt["rpt"])
        if intervals <= 0:
            continue
        target = (float(first_rpt["capacity"]) - float(last_rpt["capacity"])) / q0 / intervals
        features = linear_features(grid - end_cycle, history)
        records.append(
            {
                "dataset": "mowri",
                "cell_id": cell,
                "status": "eligible_confirmation",
                "target_g": target,
                "q0": q0,
                "retirement_cycle": end_cycle,
                "second_life_first_rpt": int(first_rpt["rpt"]),
                "second_life_last_rpt": int(last_rpt["rpt"]),
                **features,
                "future_temp": 25.0,
                "future_charge_rate": np.nan,
                "future_discharge_rate": np.nan,
                "future_soc_min": np.nan,
                "future_soc_max": np.nan,
                "chemistry": "NMC",
                "future_protocol": "Mowri_FEM",
                "duty_stratum": "Mowri_FEM",
            }
        )
    return pd.DataFrame(records)


def gate3_confirmation(
    cfg: dict[str, Any],
    root: Path,
    output: Path,
    winner: str,
    fitted: dict[str, Pipeline],
    download: bool,
) -> GateResult:
    raw_dir = root / "data/raw/mowri_p0"
    acquisition: dict[str, Any] = {}
    if download:
        acquisition = acquire_mowri(cfg, raw_dir)
    if not raw_dir.exists():
        return GateResult(3, "genuine_second_life_confirmation", "NOT_RUN", "Mowri data are absent; rerun with --download-mowri.", {})
    cohort = build_mowri_cohort(raw_dir, cfg, output)
    if len(cohort) < 4:
        return GateResult(3, "genuine_second_life_confirmation", "STOP", "Fewer than four Mowri cells have linked first- and second-life RPT data.", {"n_cells": len(cohort), **acquisition})
    cohort["soh_only_score"] = -cohort["soh_t0"]
    cohort["baseline_score"] = fitted["baseline"].predict(cohort[list(BASELINE_FEATURES)])
    cohort["model_score"] = fitted[winner].predict(cohort[list(FULL_FEATURES)])
    rho_soh = float(spearmanr(cohort["target_g"], cohort["soh_only_score"]).statistic)
    rho_base = float(spearmanr(cohort["target_g"], cohort["baseline_score"]).statistic)
    rho_model = float(spearmanr(cohort["target_g"], cohort["model_score"]).statistic)
    loo: list[float] = []
    for cell in cohort["cell_id"]:
        part = cohort[cohort["cell_id"] != cell]
        loo.append(float(spearmanr(part["target_g"], part["model_score"]).statistic))
    cohort.to_csv(output / "p0_gate3_mowri_predictions.csv", index=False)
    metrics = {
        "n_cells": len(cohort),
        "spearman_soh_only": rho_soh,
        "spearman_strong_baseline": rho_base,
        "spearman_frozen_model": rho_model,
        "spearman_gain_over_best_baseline": rho_model - max(rho_soh, rho_base),
        "leave_one_out_spearman_min": float(np.nanmin(loo)),
        "leave_one_out_spearman_max": float(np.nanmax(loo)),
        **acquisition,
    }
    passed = rho_model >= float(cfg["gate3_min_spearman"]) and metrics[
        "spearman_gain_over_best_baseline"
    ] >= float(cfg["gate3_min_spearman_gain"])
    return GateResult(
        3,
        "genuine_second_life_confirmation",
        "GO" if passed else "STOP",
        "The frozen first-life grader ranks genuine second-life degradation beyond both locked baselines."
        if passed
        else "The frozen model does not clear the locked genuine second-life rank threshold.",
        metrics,
    )


def stopped_result(gate: int, name: str, previous: GateResult) -> GateResult:
    return GateResult(gate, name, "NOT_RUN", f"Stopped prospectively because Gate {previous.gate} returned {previous.decision}.", {})


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve() if args.project_root else project_root_from_script()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg, lock_hash = load_lock(config_path)
    if args.bootstrap_iterations is not None:
        cfg["bootstrap_iterations"] = int(args.bootstrap_iterations)
    seeds = [int(x) for x in cfg["seeds"]]
    if args.smoke:
        seeds = seeds[:1]
        cfg["bootstrap_iterations"] = 20
    iterations = int(cfg["bootstrap_iterations"])
    manifest = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_root": str(root),
        "config_path": str(config_path),
        "config_sha256": lock_hash,
        "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
        "requested_gates": args.gates,
        "respect_gates": args.respect_gates,
        "download_mowri": args.download_mowri,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "git_commit": git_value(root, "rev-parse", "HEAD"),
        "git_status_before": git_value(root, "status", "--short"),
    }
    write_json(output / "p0_run_manifest.json", manifest)
    (output / "p0_lock_snapshot.json").write_bytes(config_path.read_bytes())

    audit, primary = build_local_cohort(root, cfg)
    primary.to_csv(output / "p0_local_primary_cohort.csv", index=False)
    results: list[GateResult] = []
    gate0, gate0_summary = gate0_structural(audit, primary, cfg, output)
    if 0 in args.gates:
        results.append(gate0)
        print(f"Gate 0: {gate0.decision} - {gate0.reason}", flush=True)
    previous = gate0
    gate1: GateResult | None = None
    if 1 in args.gates:
        if args.respect_gates and previous.decision == "STOP":
            gate1 = stopped_result(1, "oracle_value_ceiling", previous)
        else:
            gate1, _ = gate1_oracle(primary, cfg, output, seeds, iterations)
        results.append(gate1)
        previous = gate1
        print(f"Gate 1: {gate1.decision} - {gate1.reason}", flush=True)
    winner: str | None = None
    fitted: dict[str, Pipeline] = {}
    gate2: GateResult | None = None
    if 2 in args.gates:
        if args.respect_gates and previous.decision in {"STOP", "NOT_RUN"}:
            gate2 = stopped_result(2, "prospective_prediction", previous)
        else:
            gate2, winner, fitted = gate2_prediction(primary, gate0_summary, cfg, output, seeds, iterations)
        results.append(gate2)
        previous = gate2
        print(f"Gate 2: {gate2.decision} - {gate2.reason}", flush=True)
    if 3 in args.gates:
        if args.respect_gates and previous.decision in {"STOP", "NOT_RUN"}:
            gate3 = stopped_result(3, "genuine_second_life_confirmation", previous)
        elif winner is None or not fitted:
            gate3 = GateResult(3, "genuine_second_life_confirmation", "NOT_RUN", "Gate 2 did not produce a frozen model in this invocation.", {})
        else:
            gate3 = gate3_confirmation(cfg, root, output, winner, fitted, args.download_mowri)
        results.append(gate3)
        print(f"Gate 3: {gate3.decision} - {gate3.reason}", flush=True)

    final = {
        "lock_sha256": lock_hash,
        "results": [item.as_dict() for item in results],
        "overall_decision": next((item.decision for item in reversed(results) if item.decision != "NOT_RUN"), "NOT_RUN"),
        "claim_scope": "virtual pack construction structurally eligible"
        if bool(gate0_summary["pack_level_pass"].any())
        else "cell-level grading only",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(output / "p0_gate_decisions.json", final)
    write_markdown_summary(output / "P0_RUN_REPORT.md", final)
    manifest["completed_utc"] = final["completed_utc"]
    manifest["git_status_after"] = git_value(root, "status", "--short")
    write_json(output / "p0_run_manifest.json", manifest)
    print(json.dumps(json_safe(final), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
