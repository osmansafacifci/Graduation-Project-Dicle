"""
Hankel-DMD / Koopman-style pilot for capacity trajectories.

This is a dynamical-systems diagnostic for the transfer-learning story, not a
replacement for the supervised predictors. Each cell's early Q_discharge curve
is converted to a Q/Q0 retention trajectory over cycles 2..N, embedded with
delay coordinates, and summarized with a small DMD spectrum plus mode
coefficients. The same delay-coordinate representation is also used to fit
dataset-level operators and test one-step prediction across source/target
datasets.

Inputs:
    data/intermediate/matr_cycles_tidy.csv
    data/intermediate/hust_cycles_tidy.csv
    data/intermediate/sandia_cycles_tidy.csv       (optional)
    data/intermediate/luh_cycles_tidy.csv          (optional)
    data/intermediate/features_sop12_combined.csv or
    data/intermediate/features_sop12_four_dataset.csv  (labels/censoring metadata)

Outputs:
    data/intermediate/koopman_dmd_cell_modes.csv
    data/intermediate/koopman_dmd_cell_summary.csv
    data/intermediate/koopman_dmd_operator_summary.csv
    data/intermediate/koopman_dmd_summary.json
    data/intermediate/koopman_dmd_report.txt
    outputs/results_v2_koopman_dmd/dmd_eigenvalue_complex_plane.png
    outputs/results_v2_koopman_dmd/dmd_dominant_distributions.png
    outputs/results_v2_koopman_dmd/dmd_operator_cross_prediction.png

Usage:
    python 3_analysis/koopman_dmd_pilot.py
    python 3_analysis/koopman_dmd_pilot.py --n-cycles 100 --delay-dim 12 --rank 4
    python 3_analysis/koopman_dmd_pilot.py \
        --datasets matr hust sandia luh \
        --features-path data/intermediate/features_sop12_four_dataset.csv \
        --output-prefix four_dataset_koopman_dmd \
        --output-dir outputs/results_v2_four_dataset_koopman_dmd
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plot_style import apply_science_style

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_koopman_dmd"

FEATURES_PATH = INTERMEDIATE_DIR / "features_sop12_combined.csv"
DATASET_CYCLE_PATHS = {
    "matr": INTERMEDIATE_DIR / "matr_cycles_tidy.csv",
    "hust": INTERMEDIATE_DIR / "hust_cycles_tidy.csv",
    "sandia": INTERMEDIATE_DIR / "sandia_cycles_tidy.csv",
    "luh": INTERMEDIATE_DIR / "luh_cycles_tidy.csv",
}
DATASET_COLORS = {
    "matr": "#4c78a8",
    "hust": "#f58518",
    "sandia": "#54a24b",
    "luh": "#b279a2",
}


def compute_q0(qd: np.ndarray) -> float:
    qd = np.asarray(qd, dtype=float).ravel()
    if len(qd) < 5:
        return float("nan")
    vals = qd[1:5]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(np.median(vals)) if len(vals) else float("nan")


def load_cycle_table(path: Path, dataset: str) -> dict[str, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"[error] missing {path.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(path)
    out: dict[str, np.ndarray] = {}
    for raw_cell_id, g in df.groupby("cell_id"):
        cell_id = str(raw_cell_id)
        if dataset == "hust" and not cell_id.startswith("hust_"):
            cell_id = f"hust_{cell_id}"
        g = g.sort_values("cycle")
        out[cell_id] = g["Q_discharge"].to_numpy(dtype=float)
    return out


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_label_metadata(n_cycles: int, features_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise SystemExit(f"[error] missing {display_path(features_path)}")
    df = pd.read_csv(features_path)
    cols = ["dataset", "cell_id", "n_cycles", "q0", "cycle_life", "is_censored"]
    return df.loc[df["n_cycles"] == n_cycles, cols].copy()


def trajectory_from_qd(qd: np.ndarray, n_cycles: int, observable: str) -> tuple[np.ndarray, float]:
    qd = np.asarray(qd, dtype=float).ravel()
    q0 = compute_q0(qd)
    if not np.isfinite(q0) or q0 <= 0 or len(qd) < n_cycles:
        return np.array([], dtype=float), float("nan")
    # Match the feature pipeline: drop formation cycle 1 and use cycles 2..N.
    window = qd[1:n_cycles]
    mask = np.isfinite(window) & (window > 0)
    if int(np.sum(mask)) != len(window):
        return np.array([], dtype=float), q0
    retention = window / q0
    if observable == "retention":
        y = retention
    elif observable == "fade":
        y = 1.0 - retention
    elif observable == "retention_centered":
        y = retention - float(retention[0])
    else:
        raise ValueError(f"unknown observable: {observable}")
    return np.asarray(y, dtype=float), q0


def hankel_snapshots(series: np.ndarray, delay_dim: int) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=float).ravel()
    n = len(series)
    if n <= delay_dim + 1:
        raise ValueError("series too short for requested delay dimension")
    columns = n - delay_dim
    h = np.column_stack([series[i:i + delay_dim] for i in range(columns)])
    return h[:, :-1], h[:, 1:]


def fit_truncated_dmd(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    *,
    svd_tol: float = 1e-10,
) -> dict[str, np.ndarray]:
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    effective_rank = int(np.sum(s > svd_tol * max(s[0], 1.0)))
    r = max(1, min(rank, effective_rank, x.shape[0], x.shape[1]))
    ur = u[:, :r]
    sr = s[:r]
    vhr = vh[:r, :]
    v = vhr.T
    inv_s = np.diag(1.0 / sr)
    a_tilde = ur.T @ y @ v @ inv_s
    eigvals, w = np.linalg.eig(a_tilde)
    modes = y @ v @ inv_s @ w
    a_full = y @ v @ inv_s @ ur.T
    b = np.linalg.lstsq(modes, x[:, 0], rcond=None)[0]
    return {
        "rank": np.array([r], dtype=int),
        "eigvals": eigvals,
        "modes": modes,
        "coefficients": b,
        "operator": a_full,
        "singular_values": sr,
    }


def dmd_cell_rows(
    dataset: str,
    qd_by_cell: dict[str, np.ndarray],
    labels: pd.DataFrame,
    *,
    n_cycles: int,
    delay_dim: int,
    rank: int,
    observable: str,
) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    label_lookup = labels.set_index(["dataset", "cell_id"]).to_dict("index")
    allowed_cells = set(labels.loc[labels["dataset"].eq(dataset), "cell_id"].astype(str))
    mode_rows: list[dict] = []
    summary_rows: list[dict] = []
    trajectories: dict[str, np.ndarray] = {}

    for cell_id in sorted(qd_by_cell):
        if allowed_cells and cell_id not in allowed_cells:
            continue
        series, q0 = trajectory_from_qd(qd_by_cell[cell_id], n_cycles, observable)
        if len(series) <= delay_dim + 1:
            continue
        try:
            x, y = hankel_snapshots(series, delay_dim)
            result = fit_truncated_dmd(x, y, rank)
        except np.linalg.LinAlgError:
            continue

        eigvals = result["eigvals"]
        coeffs = result["coefficients"]
        r_eff = int(result["rank"][0])
        coeff_abs = np.abs(coeffs)
        denom = float(np.sum(coeff_abs)) + 1e-12
        order = np.argsort(coeff_abs)[::-1]
        meta = label_lookup.get((dataset, cell_id), {})
        trajectories[cell_id] = series

        sorted_abs = np.abs(eigvals[order])
        spectral_radius = float(np.nanmax(np.abs(eigvals))) if len(eigvals) else float("nan")
        dominant_idx = int(order[0])
        dominant_lambda = eigvals[dominant_idx]
        dominant_abs = float(abs(dominant_lambda))
        dominant_decay = float(-math.log(max(dominant_abs, 1e-12)))

        summary_rows.append(
            {
                "dataset": dataset,
                "cell_id": cell_id,
                "n_cycles": int(n_cycles),
                "delay_dim": int(delay_dim),
                "rank": int(r_eff),
                "observable": observable,
                "q0": float(meta.get("q0", q0)),
                "cycle_life": float(meta.get("cycle_life", np.nan)),
                "is_censored": int(meta.get("is_censored", 0)),
                "dominant_eig_real": float(np.real(dominant_lambda)),
                "dominant_eig_imag": float(np.imag(dominant_lambda)),
                "dominant_eig_abs": dominant_abs,
                "dominant_eig_angle": float(np.angle(dominant_lambda)),
                "dominant_decay_rate": dominant_decay,
                "dominant_coeff_abs": float(coeff_abs[dominant_idx]),
                "dominant_coeff_fraction": float(coeff_abs[dominant_idx] / denom),
                "spectral_radius": spectral_radius,
                "eig_abs_mean": float(np.mean(np.abs(eigvals))),
                "eig_abs_std": float(np.std(np.abs(eigvals))),
                "top2_eig_abs": float(sorted_abs[1]) if len(sorted_abs) > 1 else float("nan"),
                "top3_eig_abs": float(sorted_abs[2]) if len(sorted_abs) > 2 else float("nan"),
            }
        )

        for mode_rank, idx in enumerate(order, start=1):
            lam = eigvals[idx]
            amp = coeffs[idx]
            mode_rows.append(
                {
                    "dataset": dataset,
                    "cell_id": cell_id,
                    "n_cycles": int(n_cycles),
                    "delay_dim": int(delay_dim),
                    "rank": int(r_eff),
                    "observable": observable,
                    "mode_rank_by_amplitude": int(mode_rank),
                    "eig_real": float(np.real(lam)),
                    "eig_imag": float(np.imag(lam)),
                    "eig_abs": float(abs(lam)),
                    "eig_angle": float(np.angle(lam)),
                    "decay_rate": float(-math.log(max(abs(lam), 1e-12))),
                    "coefficient_real": float(np.real(amp)),
                    "coefficient_imag": float(np.imag(amp)),
                    "coefficient_abs": float(abs(amp)),
                    "coefficient_fraction": float(abs(amp) / denom),
                    "q0": float(meta.get("q0", q0)),
                    "cycle_life": float(meta.get("cycle_life", np.nan)),
                    "is_censored": int(meta.get("is_censored", 0)),
                }
            )

    return mode_rows, summary_rows, trajectories


def pooled_snapshots(trajectories: dict[str, np.ndarray], delay_dim: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for series in trajectories.values():
        x, y = hankel_snapshots(series, delay_dim)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=1), np.concatenate(ys, axis=1)


def operator_rmse(operator: np.ndarray, trajectories: dict[str, np.ndarray], delay_dim: int) -> tuple[float, float]:
    sq = []
    abs_err = []
    for series in trajectories.values():
        x, y = hankel_snapshots(series, delay_dim)
        pred = operator @ x
        err = pred - y
        sq.append(np.ravel(err) ** 2)
        abs_err.append(np.abs(np.ravel(err)))
    if not sq:
        return float("nan"), float("nan")
    all_sq = np.concatenate(sq)
    all_abs = np.concatenate(abs_err)
    return float(np.sqrt(np.mean(all_sq))), float(np.mean(all_abs))


def operator_summary(
    trajectories_by_dataset: dict[str, dict[str, np.ndarray]],
    *,
    delay_dim: int,
    rank: int,
) -> pd.DataFrame:
    operators: dict[str, np.ndarray] = {}
    operator_eigs: dict[str, np.ndarray] = {}
    rows = []

    for dataset, trajectories in trajectories_by_dataset.items():
        x, y = pooled_snapshots(trajectories, delay_dim)
        result = fit_truncated_dmd(x, y, rank)
        operators[dataset] = result["operator"]
        operator_eigs[dataset] = result["eigvals"]
        eigvals = result["eigvals"]
        for mode_idx, lam in enumerate(eigvals, start=1):
            rows.append(
                {
                    "row_type": "operator_eigenvalue",
                    "source_operator": dataset,
                    "eval_dataset": dataset,
                    "mode_index": int(mode_idx),
                    "eig_real": float(np.real(lam)),
                    "eig_imag": float(np.imag(lam)),
                    "eig_abs": float(abs(lam)),
                    "eig_angle": float(np.angle(lam)),
                    "decay_rate": float(-math.log(max(abs(lam), 1e-12))),
                    "one_step_rmse": float("nan"),
                    "one_step_mae": float("nan"),
                    "n_cells_eval": int(len(trajectories)),
                }
            )

    for source, operator in operators.items():
        for target, trajectories in trajectories_by_dataset.items():
            rmse, mae = operator_rmse(operator, trajectories, delay_dim)
            rows.append(
                {
                    "row_type": "operator_prediction",
                    "source_operator": source,
                    "eval_dataset": target,
                    "mode_index": float("nan"),
                    "eig_real": float("nan"),
                    "eig_imag": float("nan"),
                    "eig_abs": float("nan"),
                    "eig_angle": float("nan"),
                    "decay_rate": float("nan"),
                    "one_step_rmse": rmse,
                    "one_step_mae": mae,
                    "target_self_rmse": float("nan"),
                    "rmse_ratio_vs_target_self": float("nan"),
                    "target_self_mae": float("nan"),
                    "mae_ratio_vs_target_self": float("nan"),
                    "n_cells_eval": int(len(trajectories)),
                }
            )

    out = pd.DataFrame(rows)
    pred_mask = out["row_type"] == "operator_prediction"
    self_rows = out[pred_mask & (out["source_operator"] == out["eval_dataset"])]
    self_rmse = dict(zip(self_rows["eval_dataset"], self_rows["one_step_rmse"], strict=False))
    self_mae = dict(zip(self_rows["eval_dataset"], self_rows["one_step_mae"], strict=False))
    for idx, row in out[pred_mask].iterrows():
        target = row["eval_dataset"]
        base_rmse = float(self_rmse.get(target, np.nan))
        base_mae = float(self_mae.get(target, np.nan))
        out.loc[idx, "target_self_rmse"] = base_rmse
        out.loc[idx, "target_self_mae"] = base_mae
        out.loc[idx, "rmse_ratio_vs_target_self"] = (
            float(row["one_step_rmse"]) / base_rmse if np.isfinite(base_rmse) and base_rmse > 0 else float("nan")
        )
        out.loc[idx, "mae_ratio_vs_target_self"] = (
            float(row["one_step_mae"]) / base_mae if np.isfinite(base_mae) and base_mae > 0 else float("nan")
        )
    return out


def safe_test(name: str, dataset_a: str, values_a: np.ndarray, dataset_b: str, values_b: np.ndarray) -> dict:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if len(values_a) < 2 or len(values_b) < 2:
        return {
            "metric": name,
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "dataset_a_mean": float("nan"),
            "dataset_b_mean": float("nan"),
            "mean_delta_b_minus_a": float("nan"),
            "ks_stat": float("nan"),
            "ks_p": float("nan"),
            "mannwhitney_u_p": float("nan"),
            "n_a": int(len(values_a)),
            "n_b": int(len(values_b)),
        }
    ks = ks_2samp(values_a, values_b)
    try:
        mw = mannwhitneyu(values_a, values_b, alternative="two-sided")
        mw_p = float(mw.pvalue)
    except Exception:
        mw_p = float("nan")
    return {
        "metric": name,
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "dataset_a_mean": float(np.mean(values_a)),
        "dataset_b_mean": float(np.mean(values_b)),
        "mean_delta_b_minus_a": float(np.mean(values_b) - np.mean(values_a)),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "mannwhitney_u_p": mw_p,
        "n_a": int(len(values_a)),
        "n_b": int(len(values_b)),
    }


def distribution_pairwise_tests(cell_summary: pd.DataFrame, datasets: list[str]) -> list[dict]:
    rows = []
    metrics = [
        "dominant_eig_abs",
        "dominant_decay_rate",
        "dominant_coeff_fraction",
        "spectral_radius",
        "eig_abs_mean",
    ]
    for metric in metrics:
        for i, dataset_a in enumerate(datasets):
            for dataset_b in datasets[i + 1:]:
                rows.append(
                    safe_test(
                        metric,
                        dataset_a,
                        cell_summary[cell_summary["dataset"] == dataset_a][metric].to_numpy(dtype=float),
                        dataset_b,
                        cell_summary[cell_summary["dataset"] == dataset_b][metric].to_numpy(dtype=float),
                    )
                )
    return rows


def dmd_discriminator_auc(summary: pd.DataFrame, rank: int, seed: int) -> dict:
    cols = [
        "dominant_eig_abs",
        "dominant_eig_angle",
        "dominant_decay_rate",
        "dominant_coeff_fraction",
        "spectral_radius",
        "eig_abs_mean",
        "eig_abs_std",
        "top2_eig_abs",
        "top3_eig_abs",
    ]
    available = [c for c in cols if c in summary.columns]
    sub = summary.dropna(subset=available + ["dataset"]).copy()
    n_classes = int(sub["dataset"].nunique())
    if n_classes < 2 or len(sub) < 20:
        return {"auc_mean": float("nan"), "auc_std": float("nan"), "n_cells": int(len(sub))}
    x = sub[available].to_numpy(dtype=float)
    y, class_labels = pd.factorize(sub["dataset"])
    min_class = int(pd.Series(y).value_counts().min())
    folds = max(2, min(5, min_class))
    scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed),
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, x, y, cv=cv, scoring=scoring)
    return {
        "auc_mean": float(np.mean(scores)),
        "auc_std": float(np.std(scores)),
        "n_cells": int(len(sub)),
        "features": available,
        "rank_requested": int(rank),
        "n_classes": n_classes,
        "class_labels": [str(label) for label in class_labels],
        "scoring": scoring,
    }


def pairwise_dmd_discriminator_auc(summary: pd.DataFrame, datasets: list[str], rank: int, seed: int) -> list[dict]:
    rows = []
    for i, dataset_a in enumerate(datasets):
        for dataset_b in datasets[i + 1:]:
            sub = summary[summary["dataset"].isin([dataset_a, dataset_b])].copy()
            auc = dmd_discriminator_auc(sub, rank, seed)
            rows.append(
                {
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "auc_mean": auc["auc_mean"],
                    "auc_std": auc["auc_std"],
                    "n_cells": auc["n_cells"],
                    "scoring": auc.get("scoring", "roc_auc"),
                }
            )
    return rows


def mode_life_correlations(summary: pd.DataFrame) -> list[dict]:
    rows = []
    for dataset, sub in summary[(summary["is_censored"] == 0) & summary["cycle_life"].notna()].groupby("dataset"):
        for metric in ["dominant_eig_abs", "dominant_decay_rate", "dominant_coeff_fraction", "spectral_radius"]:
            x = sub[metric].to_numpy(dtype=float)
            y = np.log(sub["cycle_life"].to_numpy(dtype=float))
            if len(sub) < 5 or np.std(x) < 1e-12:
                r, p = float("nan"), float("nan")
            else:
                res = pearsonr(x, y)
                r, p = float(res.statistic), float(res.pvalue)
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "pearson_r_with_log_life": r,
                    "pearson_p": p,
                    "n_uncensored": int(len(sub)),
                }
            )
    return rows


def write_json(path: Path, obj: dict) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def write_report(path: Path, summary: dict) -> None:
    tests = summary["distribution_tests"]
    auc = summary["dmd_discriminator_auc"]
    pairwise_auc = summary.get("pairwise_dmd_discriminator_auc", [])
    pred = summary["operator_prediction"]
    datasets = summary["config"]["datasets"]

    lines = [
        "Koopman / Hankel-DMD Pilot",
        "===========================",
        "",
        f"Observable: {summary['config']['observable']}",
        f"Cycles: 2..{summary['config']['n_cycles']}  | delay_dim={summary['config']['delay_dim']}  | rank={summary['config']['rank']}",
        "Cells analyzed: " + ", ".join(f"{dataset.upper()}={summary['n_cells'].get(dataset, 0)}" for dataset in datasets),
        "",
        "Per-cell spectrum distribution tests (dataset B minus dataset A):",
    ]
    for row in tests:
        lines.append(
            f"- {row['dataset_a']} -> {row['dataset_b']} {row['metric']}: "
            f"mean_delta={row['mean_delta_b_minus_a']:.4g}, KS={row['ks_stat']:.3f}, p={row['ks_p']:.3g}"
        )
    lines.extend(
        [
            "",
            "Dataset separability from DMD summaries:",
            f"- Logistic {auc.get('scoring', 'roc_auc')} = {auc['auc_mean']:.3f} +/- {auc['auc_std']:.3f} across CV folds",
            "",
            "Pairwise DMD-summary dataset separability:",
        ]
    )
    for row in pairwise_auc:
        lines.append(
            f"- {row['dataset_a']} vs {row['dataset_b']}: "
            f"AUC={row['auc_mean']:.3f} +/- {row['auc_std']:.3f} (n={row['n_cells']})"
        )
    lines.extend(
        [
            "",
            "Dataset-level operator one-step RMSE on retention-delay snapshots:",
        ]
    )
    for row in pred:
        lines.append(
            f"- K_{row['source_operator']} on {row['eval_dataset']}: "
            f"RMSE={row['one_step_rmse']:.5f}, MAE={row['one_step_mae']:.5f}, "
            f"RMSE ratio vs target-self={row['rmse_ratio_vs_target_self']:.2f}x"
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "- If spectra and operator errors differ by dataset, this supports a dynamics-level version of conditional shift.",
            "- If spectra overlap, keep the result as a negative pilot and avoid adding Koopman language to the paper's main claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_plots(
    cell_modes: pd.DataFrame,
    cell_summary: pd.DataFrame,
    operator_rows: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    apply_science_style()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    datasets = list(cell_summary["dataset"].dropna().drop_duplicates())
    colors = {dataset: DATASET_COLORS.get(dataset, None) for dataset in datasets}

    top_modes = cell_modes[cell_modes["mode_rank_by_amplitude"] <= 2].copy()
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="#888888", linestyle=":", linewidth=1.0, label="unit circle")
    for dataset, sub in top_modes.groupby("dataset"):
        ax.scatter(
            sub["eig_real"],
            sub["eig_imag"],
            s=18 + 95 * sub["coefficient_fraction"],
            alpha=0.55,
            color=colors.get(dataset),
            label=f"{dataset.upper()} modes",
            edgecolor="white",
            linewidth=0.25,
        )
    ax.axhline(0, color="#dddddd", linewidth=0.8)
    ax.axvline(0, color="#dddddd", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Eigenvalue real part")
    ax.set_ylabel("Eigenvalue imaginary part")
    ax.set_title("Per-cell Hankel-DMD eigenvalues (top two modes by coefficient)")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    out = output_dir / "dmd_eigenvalue_complex_plane.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    paths.append(out)

    metrics = [
        ("dominant_eig_abs", "Dominant |lambda|"),
        ("dominant_decay_rate", "Dominant decay rate -log(|lambda|)"),
        ("dominant_coeff_fraction", "Dominant coefficient fraction"),
        ("spectral_radius", "Spectral radius"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    for ax, (metric, title) in zip(axes.ravel(), metrics, strict=False):
        values = [cell_summary[cell_summary["dataset"] == ds][metric].dropna().to_numpy() for ds in datasets]
        ax.boxplot(values, tick_labels=[ds.upper() for ds in datasets], patch_artist=True, widths=0.55)
        for pos, vals, ds in zip(range(1, len(datasets) + 1), values, datasets, strict=False):
            rng = np.random.default_rng(17 + pos)
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=12, alpha=0.42, color=colors.get(ds), edgecolor="none")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Per-cell DMD summary distributions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = output_dir / "dmd_dominant_distributions.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    paths.append(out)

    pred = operator_rows[operator_rows["row_type"] == "operator_prediction"].copy()
    pred["label"] = pred["source_operator"].str.upper() + " operator\non " + pred["eval_dataset"].str.upper()
    fig, ax = plt.subplots(figsize=(max(8.2, 1.05 * len(pred)), 4.8))
    bar_colors = ["#4c78a8" if s == e else "#e45756" for s, e in zip(pred["source_operator"], pred["eval_dataset"], strict=False)]
    ax.bar(pred["label"], pred["rmse_ratio_vs_target_self"], color=bar_colors, alpha=0.88)
    for idx, row in enumerate(pred.itertuples(index=False)):
        ax.text(
            idx,
            row.rmse_ratio_vs_target_self,
            f"{row.rmse_ratio_vs_target_self:.2f}x\nRMSE {row.one_step_rmse:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1.0)
    ax.set_ylim(0, max(2.05, float(pred["rmse_ratio_vs_target_self"].max()) * 1.22))
    ax.set_ylabel("One-step RMSE ratio vs target-specific operator")
    ax.set_title("Dataset-level DMD operator transfer penalty")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = output_dir / "dmd_operator_cross_prediction.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    paths.append(out)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cycles", type=int, default=100, help="Use cycles 2..N. Default: 100")
    parser.add_argument("--delay-dim", type=int, default=12, help="Hankel delay embedding dimension. Default: 12")
    parser.add_argument("--rank", type=int, default=4, help="Truncated DMD rank. Default: 4")
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    parser.add_argument("--datasets", nargs="+", default=["matr", "hust"], choices=list(DATASET_CYCLE_PATHS))
    parser.add_argument("--output-prefix", default="koopman_dmd")
    parser.add_argument(
        "--observable",
        choices=["retention", "fade", "retention_centered"],
        default="retention",
        help="Trajectory observable before delay embedding. Default: retention = Q/Q0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_cycles <= args.delay_dim + 2:
        print("[error] n-cycles must be at least delay-dim + 3")
        return 1

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = args.features_path if args.features_path.is_absolute() else PROJECT_ROOT / args.features_path
    labels = load_label_metadata(args.n_cycles, features_path)
    qd_by_dataset = {
        dataset: load_cycle_table(DATASET_CYCLE_PATHS[dataset], dataset)
        for dataset in args.datasets
    }

    mode_rows: list[dict] = []
    summary_rows: list[dict] = []
    trajectories_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    for dataset, qd_by_cell in qd_by_dataset.items():
        modes, summaries, trajectories = dmd_cell_rows(
            dataset,
            qd_by_cell,
            labels,
            n_cycles=args.n_cycles,
            delay_dim=args.delay_dim,
            rank=args.rank,
            observable=args.observable,
        )
        mode_rows.extend(modes)
        summary_rows.extend(summaries)
        trajectories_by_dataset[dataset] = trajectories

    cell_modes = pd.DataFrame(mode_rows)
    cell_summary = pd.DataFrame(summary_rows)
    operator_rows = operator_summary(trajectories_by_dataset, delay_dim=args.delay_dim, rank=args.rank)

    mode_path = INTERMEDIATE_DIR / f"{args.output_prefix}_cell_modes.csv"
    summary_path = INTERMEDIATE_DIR / f"{args.output_prefix}_cell_summary.csv"
    operator_path = INTERMEDIATE_DIR / f"{args.output_prefix}_operator_summary.csv"
    cell_modes.to_csv(mode_path, index=False)
    cell_summary.to_csv(summary_path, index=False)
    operator_rows.to_csv(operator_path, index=False)

    tests = distribution_pairwise_tests(cell_summary, args.datasets)
    auc = dmd_discriminator_auc(cell_summary, args.rank, args.seed)
    pairwise_auc = pairwise_dmd_discriminator_auc(cell_summary, args.datasets, args.rank, args.seed)
    life_corr = mode_life_correlations(cell_summary)
    pred_rows = operator_rows[operator_rows["row_type"] == "operator_prediction"].copy()
    pred_summary = pred_rows[
        [
            "source_operator",
            "eval_dataset",
            "one_step_rmse",
            "one_step_mae",
            "target_self_rmse",
            "rmse_ratio_vs_target_self",
            "target_self_mae",
            "mae_ratio_vs_target_self",
            "n_cells_eval",
        ]
    ].to_dict("records")
    plot_paths = write_plots(cell_modes, cell_summary, operator_rows, output_dir)

    summary = {
        "protocol": "hankel_dmd_pilot_v1",
        "config": {
            "n_cycles": int(args.n_cycles),
            "delay_dim": int(args.delay_dim),
            "rank": int(args.rank),
            "observable": args.observable,
            "seed": int(args.seed),
            "datasets": args.datasets,
            "features_path": display_path(features_path),
        },
        "n_cells": {
            dataset: int(len(trajectories))
            for dataset, trajectories in trajectories_by_dataset.items()
        },
        "outputs": {
            "cell_modes_csv": str(mode_path.relative_to(PROJECT_ROOT)),
            "cell_summary_csv": str(summary_path.relative_to(PROJECT_ROOT)),
            "operator_summary_csv": str(operator_path.relative_to(PROJECT_ROOT)),
            "plots": [str(p.relative_to(PROJECT_ROOT)) for p in plot_paths],
        },
        "distribution_tests": tests,
        "dmd_discriminator_auc": auc,
        "pairwise_dmd_discriminator_auc": pairwise_auc,
        "operator_prediction": pred_summary,
        "mode_life_correlations": life_corr,
    }
    json_path = INTERMEDIATE_DIR / f"{args.output_prefix}_summary.json"
    report_path = INTERMEDIATE_DIR / f"{args.output_prefix}_report.txt"
    write_json(json_path, summary)
    write_report(report_path, summary)

    cell_text = ", ".join(f"{dataset.upper()}={summary['n_cells'].get(dataset, 0)}" for dataset in args.datasets)
    print(f"[koopman_dmd] cells: {cell_text}")
    print(f"[koopman_dmd] wrote {mode_path.relative_to(PROJECT_ROOT)}")
    print(f"[koopman_dmd] wrote {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"[koopman_dmd] wrote {operator_path.relative_to(PROJECT_ROOT)}")
    print(f"[koopman_dmd] wrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[koopman_dmd] wrote {report_path.relative_to(PROJECT_ROOT)}")
    for path in plot_paths:
        print(f"[koopman_dmd] wrote {path.relative_to(PROJECT_ROOT)}")
    print(f"[koopman_dmd] discriminator AUC={auc['auc_mean']:.3f} +/- {auc['auc_std']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
