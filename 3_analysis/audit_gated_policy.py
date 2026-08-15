"""Prospective target-audit gating for cross-dataset battery RUL transfer.

This script turns the retrospective rank-signal taxonomy into a deployed
policy simulation. For each source->target direction and seed, it repeatedly
draws three disjoint target subsets:

1. k=20 audit/adapter cells, used to estimate the target audit correlation and
   fit the selected target-side adapter.
2. a split-conformal calibration set, disjoint from the adapter cells.
3. a held-out test set, untouched by the gate, adapter, and CP calibration.

The gate uses the frozen deployment thresholds:
    r_hat >= 0.40       -> linear adapter
    0.10 <= r_hat < .40 -> residual-mean adapter
    r_hat < 0.10        -> CP interval only (point prediction abstains)

The CP-only branch still uses a residual-mean centre for interval construction,
but its point-prediction metrics are reported as unavailable. This matches the
paper's deployment claim: when rank signal collapses, the interval rather than
the point prediction is the deployable artefact.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_audit_gated_policy"

sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
from metrics_utils import symmetric_mape  # noqa: E402

PREDICTIONS_PATH = INTERMEDIATE_DIR / "four_dataset_conditional_shift_predictions.csv"
SUMMARY_PATH = INTERMEDIATE_DIR / "four_dataset_conditional_shift_direction_summary.csv"
SEEDS = [42, 123, 456, 789, 1011]

DATASET_LABEL = {
    "matr": "MATR",
    "hust": "HUST",
    "sandia": "Sandia",
    "luh": "Luh",
}

SCENARIO_LABEL = {
    "no_adaptation": "No adaptation",
    "always_residual_mean": "Always residual",
    "always_linear": "Always linear",
    "audit_gated_policy": "Audit-gated policy",
    "oracle_best_point": "Oracle point adapter",
}


@dataclass(frozen=True)
class SplitSizes:
    audit: int
    calibration: int
    test: int


def finite_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with explicit handling for constant vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def gate_from_r(r_hat: float) -> str:
    if not np.isfinite(r_hat):
        return "cp_interval_only"
    if r_hat >= 0.40:
        return "linear"
    if r_hat >= 0.10:
        return "residual_mean"
    return "cp_interval_only"


def scenario_adapter_name(scenario: str, gate: str, oracle_choice: str | None = None) -> str:
    if scenario == "no_adaptation":
        return "none"
    if scenario == "always_residual_mean":
        return "residual_mean"
    if scenario == "always_linear":
        return "linear"
    if scenario == "audit_gated_policy":
        return gate
    if scenario == "oracle_best_point":
        if oracle_choice is None:
            raise ValueError("oracle_best_point requires oracle_choice")
        return oracle_choice
    raise ValueError(f"Unknown scenario: {scenario}")


def fit_adapter(y_source_pred: np.ndarray, y_true: np.ndarray, adapter: str) -> tuple[float, float]:
    """Return ``(slope, intercept)`` for cycle-space target calibration."""
    y_source_pred = np.asarray(y_source_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if adapter == "none":
        return 1.0, 0.0
    if adapter in {"residual_mean", "cp_interval_only"}:
        return 1.0, float(np.mean(y_true - y_source_pred))
    if adapter == "linear":
        if len(y_true) < 2 or np.nanstd(y_source_pred) <= 1e-12:
            return 0.0, float(np.mean(y_true))
        slope, intercept = np.polyfit(y_source_pred, y_true, deg=1)
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return 0.0, float(np.mean(y_true))
        return float(slope), float(intercept)
    raise ValueError(f"Unknown adapter: {adapter}")


def apply_adapter(y_source_pred: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    pred = slope * np.asarray(y_source_pred, dtype=float) + intercept
    return np.clip(np.nan_to_num(pred, nan=1.0, posinf=1e9, neginf=1.0), 1.0, 1e9)


def conformal_quantile(scores: np.ndarray, confidence: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return float("nan")
    alpha = 1.0 - confidence
    level = min(1.0, math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, available: bool) -> dict[str, float]:
    if not available:
        return {"MAE": float("nan"), "SMAPE": float("nan"), "R2": float("nan")}
    if len(y_true) < 2:
        return {"MAE": float("nan"), "SMAPE": float("nan"), "R2": float("nan")}
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "SMAPE": symmetric_mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cp_metrics(y_true: np.ndarray, y_pred: np.ndarray, qhat: float) -> dict[str, float]:
    if not np.isfinite(qhat):
        return {
            "coverage": float("nan"),
            "median_width": float("nan"),
            "finite_interval_fraction": 0.0,
        }
    lo = y_pred - qhat
    hi = y_pred + qhat
    covered = (y_true >= lo) & (y_true <= hi)
    widths = hi - lo
    finite = np.isfinite(widths)
    return {
        "coverage": float(np.mean(covered)),
        "median_width": float(np.median(widths[finite])) if np.any(finite) else float("nan"),
        "finite_interval_fraction": float(np.mean(finite)),
    }


def choose_split_sizes(n: int, audit_size: int, cp_size_large: int, cp_size_small: int) -> SplitSizes:
    cp_size = cp_size_small if n <= audit_size + cp_size_large + 20 else cp_size_large
    test_size = n - audit_size - cp_size
    if test_size < 10:
        raise ValueError(
            f"Target cohort n={n} cannot support audit={audit_size}, cp={cp_size}, test>=10."
        )
    return SplitSizes(audit=audit_size, calibration=cp_size, test=test_size)


def draw_partition(cell_ids: np.ndarray, split_sizes: SplitSizes, rng: np.random.Generator) -> tuple[set, set, set]:
    shuffled = np.array(cell_ids, dtype=object).copy()
    rng.shuffle(shuffled)
    audit = set(shuffled[: split_sizes.audit])
    cal_start = split_sizes.audit
    cal_end = cal_start + split_sizes.calibration
    calibration = set(shuffled[cal_start:cal_end])
    test = set(shuffled[cal_end:])
    return audit, calibration, test


def evaluate_one_draw(
    block: pd.DataFrame,
    *,
    split_sizes: SplitSizes,
    rng: np.random.Generator,
    repeat: int,
    confidence_levels: list[float],
) -> list[dict]:
    cell_ids = block["cell_id"].drop_duplicates().to_numpy()
    audit_ids, cal_ids, test_ids = draw_partition(cell_ids, split_sizes, rng)

    audit = block[block["cell_id"].isin(audit_ids)]
    calibration = block[block["cell_id"].isin(cal_ids)]
    test = block[block["cell_id"].isin(test_ids)]

    audit_r = finite_pearson(audit["source_prediction"].to_numpy(), audit["cycle_life"].to_numpy())
    full_r = finite_pearson(block["source_prediction"].to_numpy(), block["cycle_life"].to_numpy())
    gate = gate_from_r(audit_r)
    full_gate = gate_from_r(full_r)
    gated_point_available = gate != "cp_interval_only"

    y_a = audit["cycle_life"].to_numpy(dtype=float)
    p_a = audit["source_prediction"].to_numpy(dtype=float)
    y_c = calibration["cycle_life"].to_numpy(dtype=float)
    p_c = calibration["source_prediction"].to_numpy(dtype=float)
    y_t = test["cycle_life"].to_numpy(dtype=float)
    p_t = test["source_prediction"].to_numpy(dtype=float)

    fitted: dict[str, tuple[float, float]] = {}
    pred_test: dict[str, np.ndarray] = {}
    pred_cal: dict[str, np.ndarray] = {}
    for adapter in ["none", "residual_mean", "linear", "cp_interval_only"]:
        fit_as = "residual_mean" if adapter == "cp_interval_only" else adapter
        slope, intercept = fit_adapter(p_a, y_a, fit_as)
        fitted[adapter] = (slope, intercept)
        pred_test[adapter] = apply_adapter(p_t, slope, intercept)
        pred_cal[adapter] = apply_adapter(p_c, slope, intercept)

    residual_metrics = point_metrics(y_t, pred_test["residual_mean"], available=True)
    linear_metrics = point_metrics(y_t, pred_test["linear"], available=True)
    oracle_choice = "linear" if linear_metrics["R2"] >= residual_metrics["R2"] else "residual_mean"

    rows: list[dict] = []
    for scenario in [
        "no_adaptation",
        "always_residual_mean",
        "always_linear",
        "audit_gated_policy",
        "oracle_best_point",
    ]:
        adapter = scenario_adapter_name(scenario, gate, oracle_choice)
        cp_adapter = "residual_mean" if adapter == "cp_interval_only" else adapter
        point_available = True
        if scenario == "audit_gated_policy" and adapter == "cp_interval_only":
            point_available = False
        metrics = point_metrics(y_t, pred_test[cp_adapter], available=point_available)
        scores = np.abs(y_c - pred_cal[cp_adapter])
        for confidence in confidence_levels:
            qhat = conformal_quantile(scores, confidence)
            interval_metrics = cp_metrics(y_t, pred_test[cp_adapter], qhat)
            rows.append(
                {
                    "repeat": int(repeat),
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABEL[scenario],
                    "adapter_used": adapter,
                    "cp_center_adapter": cp_adapter,
                    "confidence": float(confidence),
                    "audit_size": int(len(audit)),
                    "calibration_size": int(len(calibration)),
                    "test_size": int(len(test)),
                    "audit_r": audit_r,
                    "full_target_r": full_r,
                    "audit_gate": gate,
                    "full_target_gate": full_gate,
                    "gate_matches_full": bool(gate == full_gate),
                    "oracle_choice": oracle_choice,
                    "point_prediction_available": bool(point_available),
                    "MAE": metrics["MAE"],
                    "SMAPE": metrics["SMAPE"],
                    "R2": metrics["R2"],
                    "coverage": interval_metrics["coverage"],
                    "median_width": interval_metrics["median_width"],
                    "finite_interval_fraction": interval_metrics["finite_interval_fraction"],
                    "qhat": qhat,
                }
            )
    return rows


def summarize_draws(draws: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "MAE",
        "SMAPE",
        "R2",
        "coverage",
        "median_width",
        "finite_interval_fraction",
        "audit_r",
        "gate_matches_full",
    ]
    group_cols = ["source", "target", "model", "scenario", "scenario_label", "confidence"]
    summary_rows: list[dict] = []
    for keys, grp in draws.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_draws"] = int(len(grp))
        row["point_available_fraction"] = float(grp["point_prediction_available"].mean())
        for col in metric_cols:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{col}_median"] = float(np.median(vals)) if len(vals) else float("nan")
            row[f"{col}_p025"] = float(np.quantile(vals, 0.025)) if len(vals) else float("nan")
            row[f"{col}_p975"] = float(np.quantile(vals, 0.975)) if len(vals) else float("nan")
        summary_rows.append(row)

    gate_rows: list[dict] = []
    gate_base = draws[(draws["scenario"] == "audit_gated_policy") & (draws["confidence"] == draws["confidence"].min())]
    for keys, grp in gate_base.groupby(["source", "target", "model"], dropna=False):
        row = dict(zip(["source", "target", "model"], keys))
        row["n_draws"] = int(len(grp))
        row["full_target_r_mean"] = float(grp["full_target_r"].mean())
        row["audit_r_mean"] = float(grp["audit_r"].mean())
        row["audit_r_sd"] = float(grp["audit_r"].std(ddof=1))
        row["gate_matches_full_fraction"] = float(grp["gate_matches_full"].mean())
        for gate, frac in grp["audit_gate"].value_counts(normalize=True).items():
            row[f"gate_rate_{gate}"] = float(frac)
        gate_rows.append(row)
    gates = pd.DataFrame(gate_rows).fillna(0.0)
    return pd.DataFrame(summary_rows), gates


def make_plots(summary: pd.DataFrame, gates: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        sys.path.insert(0, str(HERE))
        from plot_style import apply_science_style

        apply_science_style()
    except Exception:
        pass

    import matplotlib.pyplot as plt

    primary = summary[summary["confidence"].eq(0.90)].copy()
    directions = (
        primary[primary["scenario"].eq("audit_gated_policy")]
        .assign(direction=lambda d: d["source"].str.upper() + "→" + d["target"].str.upper())
        .sort_values("R2_mean", ascending=False)["direction"]
        .tolist()
    )
    scenario_order = [
        "no_adaptation",
        "always_residual_mean",
        "always_linear",
        "audit_gated_policy",
        "oracle_best_point",
    ]
    colors = {
        "no_adaptation": "#757575",
        "always_residual_mean": "#1f77b4",
        "always_linear": "#2ca02c",
        "audit_gated_policy": "#d62728",
        "oracle_best_point": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    x = np.arange(len(directions))
    width = 0.15
    for i, scenario in enumerate(scenario_order):
        sub = primary[primary["scenario"].eq(scenario)].copy()
        sub["direction"] = sub["source"].str.upper() + "→" + sub["target"].str.upper()
        sub = sub.set_index("direction").reindex(directions)
        offset = (i - (len(scenario_order) - 1) / 2) * width
        vals = sub["R2_mean"].to_numpy(dtype=float)
        vals_plot = np.nan_to_num(vals, nan=-0.05)
        ax.bar(x + offset, vals_plot, width=width, label=SCENARIO_LABEL[scenario], color=colors[scenario], alpha=0.88)
        unavailable = sub["point_available_fraction"].fillna(1).to_numpy(dtype=float) < 0.5
        for xx, is_unavailable in zip(x + offset, unavailable):
            if is_unavailable:
                ax.text(xx, -0.02, "CP", ha="center", va="top", rotation=90, fontsize=6.5)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_ylabel("Held-out test $R^2$ (mean over draws)")
    ax.set_xticks(x)
    ax.set_xticklabels(directions, rotation=45, ha="right")
    ax.set_title("Prospective k=20 audit-gated policy vs fixed adapters")
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "audit_gated_policy_r2.png")
    fig.savefig(output_dir / "audit_gated_policy_r2.pdf")
    plt.close(fig)

    gate_cols = ["gate_rate_linear", "gate_rate_residual_mean", "gate_rate_cp_interval_only"]
    for col in gate_cols:
        if col not in gates.columns:
            gates[col] = 0.0
    gates_plot = gates.copy()
    gates_plot["direction"] = gates_plot["source"].str.upper() + "→" + gates_plot["target"].str.upper()
    gates_plot = gates_plot.sort_values("full_target_r_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    bottom = np.zeros(len(gates_plot))
    gate_colors = {
        "gate_rate_linear": "#2ca02c",
        "gate_rate_residual_mean": "#ff9800",
        "gate_rate_cp_interval_only": "#d62728",
    }
    gate_labels = {
        "gate_rate_linear": "linear",
        "gate_rate_residual_mean": "residual mean",
        "gate_rate_cp_interval_only": "CP interval only",
    }
    for col in gate_cols:
        vals = gates_plot[col].to_numpy(dtype=float)
        ax.bar(gates_plot["direction"], vals, bottom=bottom, color=gate_colors[col], label=gate_labels[col])
        bottom += vals
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of k=20 audit draws")
    ax.set_title("Audit-gate decisions reveal which regimes are noisy under a realistic target budget")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "audit_gated_policy_gate_rates.png")
    fig.savefig(output_dir / "audit_gated_policy_gate_rates.pdf")
    plt.close(fig)


def write_report(summary: pd.DataFrame, gates: pd.DataFrame, path: Path) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        """Render a compact Markdown table without pandas' optional tabulate dependency."""
        if df.empty:
            return "_No rows._"
        render = df.copy()
        headers = [str(c) for c in render.columns]
        rows = render.astype(object).where(pd.notna(render), "").astype(str).values.tolist()
        out = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        out.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(out)

    primary = summary[summary["confidence"].eq(0.90)].copy()
    gated = primary[primary["scenario"].eq("audit_gated_policy")].copy()
    fixed = primary[primary["scenario"].isin(["no_adaptation", "always_residual_mean", "always_linear", "oracle_best_point"])].copy()
    lines: list[str] = []
    lines.append("# Prospective k=20 Audit-Gated Policy")
    lines.append("")
    lines.append("Design: 20 target cells serve as both audit and adapter-fit cells; CP calibration and test cells are disjoint. Results are averaged over 200 random partitions per seed and five source-model seeds.")
    lines.append("")
    lines.append("## Gate Decision Rates")
    show_cols = [
        "source",
        "target",
        "model",
        "full_target_r_mean",
        "audit_r_mean",
        "audit_r_sd",
        "gate_matches_full_fraction",
        "gate_rate_linear",
        "gate_rate_residual_mean",
        "gate_rate_cp_interval_only",
    ]
    for col in show_cols:
        if col not in gates.columns:
            gates[col] = 0.0
    lines.append(markdown_table(gates[show_cols].round(3)))
    lines.append("")
    lines.append("## Audit-Gated Policy @ 90% CP")
    lines.append(
        markdown_table(
            gated[
                [
                    "source",
                    "target",
                    "model",
                    "R2_mean",
                    "MAE_mean",
                    "SMAPE_mean",
                    "coverage_mean",
                    "median_width_mean",
                    "point_available_fraction",
                    "gate_matches_full_mean",
                ]
            ]
            .sort_values("R2_mean", ascending=False)
            .round(3)
        )
    )
    lines.append("")
    lines.append("## Fixed-Policy Comparators @ 90% CP")
    lines.append(
        markdown_table(
            fixed[
                [
                    "source",
                    "target",
                    "scenario",
                    "R2_mean",
                    "MAE_mean",
                    "SMAPE_mean",
                    "coverage_mean",
                    "median_width_mean",
                    "point_available_fraction",
                ]
            ]
            .round(3)
        )
    )
    lines.append("")
    lines.append("Interpretation note: CP-only gate draws use a residual-mean centre for conformal intervals but mark point-prediction metrics as unavailable.")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--direction-summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-repeats", type=int, default=200)
    parser.add_argument("--audit-size", type=int, default=20)
    parser.add_argument("--cp-size-large", type=int, default=20)
    parser.add_argument("--cp-size-small", type=int, default=15)
    parser.add_argument("--confidence-levels", type=float, nargs="+", default=[0.90, 0.95])
    parser.add_argument("--seed-base", type=int, default=20260616)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = args.predictions_path if args.predictions_path.is_absolute() else PROJECT_ROOT / args.predictions_path
    if not predictions_path.exists():
        print(f"[error] missing predictions file: {predictions_path}")
        return 1
    pred = pd.read_csv(predictions_path)
    required = {"source", "target", "model", "seed", "cell_id", "cycle_life", "source_prediction"}
    missing = required.difference(pred.columns)
    if missing:
        print(f"[error] missing required columns in {predictions_path}: {sorted(missing)}")
        return 1

    pred = pred[list(required)].copy()
    pred["seed"] = pred["seed"].astype(int)
    pred = pred[pred["seed"].isin(SEEDS)].copy()
    pred = pred.replace([np.inf, -np.inf], np.nan).dropna(subset=["cycle_life", "source_prediction"])

    rows: list[dict] = []
    direction_blocks = list(pred.groupby(["source", "target", "model", "seed"], sort=True))
    for idx, ((source, target, model, seed), block) in enumerate(direction_blocks, start=1):
        block = block.drop_duplicates("cell_id").copy()
        split_sizes = choose_split_sizes(
            len(block),
            audit_size=args.audit_size,
            cp_size_large=args.cp_size_large,
            cp_size_small=args.cp_size_small,
        )
        print(
            f"[{idx:03d}/{len(direction_blocks):03d}] {source}->{target} {model} seed={seed} "
            f"n={len(block)} split={split_sizes.audit}/{split_sizes.calibration}/{split_sizes.test}"
        )
        for repeat in range(args.n_repeats):
            rng = np.random.default_rng(args.seed_base + seed * 100_000 + repeat)
            draw_rows = evaluate_one_draw(
                block,
                split_sizes=split_sizes,
                rng=rng,
                repeat=repeat,
                confidence_levels=args.confidence_levels,
            )
            for row in draw_rows:
                row.update({"source": source, "target": target, "model": model, "seed": int(seed)})
            rows.extend(draw_rows)

    draws = pd.DataFrame(rows)
    summary, gates = summarize_draws(draws)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draws_path = INTERMEDIATE_DIR / "audit_gated_policy_draws.csv"
    summary_path = INTERMEDIATE_DIR / "audit_gated_policy_summary.csv"
    gates_path = INTERMEDIATE_DIR / "audit_gated_policy_gate_rates.csv"
    report_path = INTERMEDIATE_DIR / "audit_gated_policy_report.md"
    draws.to_csv(draws_path, index=False)
    summary.to_csv(summary_path, index=False)
    gates.to_csv(gates_path, index=False)
    write_report(summary, gates, report_path)
    make_plots(summary, gates, args.output_dir)

    print(f"[save] {draws_path} rows={len(draws)}")
    print(f"[save] {summary_path} rows={len(summary)}")
    print(f"[save] {gates_path} rows={len(gates)}")
    print(f"[save] {report_path}")
    print(f"[save] {args.output_dir / 'audit_gated_policy_r2.png'}")
    print(f"[save] {args.output_dir / 'audit_gated_policy_gate_rates.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
