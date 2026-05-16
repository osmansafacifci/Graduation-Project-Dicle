"""
Four-dataset survival/censoring audit for the paper extension.

The regression metrics exclude censored cells because MAE/sMAPE/R2 require
observed event times. This script keeps those modeling rules unchanged and
audits the right-censoring pattern across MATR, HUST, Sandia, and Luh.

Outputs:
    data/intermediate/four_dataset_survival_censoring_summary.csv
    data/intermediate/four_dataset_survival_censoring_curves.csv
    data/intermediate/four_dataset_survival_censoring_censored_cells.csv
    data/intermediate/four_dataset_survival_censoring_pairwise_tests.csv
    data/intermediate/four_dataset_survival_censoring_rmst_bootstrap.csv
    data/intermediate/four_dataset_survival_censoring_rmst_pairwise.csv
    data/intermediate/four_dataset_survival_censoring.json
    data/intermediate/four_dataset_survival_censoring_report.md
    outputs/results_v2_four_dataset_survival/kaplan_meier_four_dataset.png

Usage:
    python 3_analysis/survival_censoring_four_dataset.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from plot_style import apply_science_style
from survival_censoring import (
    kaplan_meier,
    logrank_test,
    restricted_mean_survival_time,
    survival_at,
    median_survival,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_four_dataset_survival"

FEATURES_PATH = INTERMEDIATE_DIR / "features_sop12_four_dataset.csv"
DATASETS = ["matr", "hust", "sandia", "luh"]
N_CYCLES = 100
RMST_BOOTSTRAPS = 2000
RANDOM_SEED = 20260512

CYCLE_SOURCES = {
    "matr": (INTERMEDIATE_DIR / "matr_cycles_tidy.csv", ""),
    "hust": (INTERMEDIATE_DIR / "hust_cycles_tidy.csv", "hust_"),
    "sandia": (INTERMEDIATE_DIR / "sandia_cycles_tidy.csv", ""),
    "luh": (INTERMEDIATE_DIR / "luh_cycles_tidy.csv", ""),
}


def observed_cycles_from_tidy(path: Path, *, prefix: str = "") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["cell_id", "observed_cycles"])
    tidy = pd.read_csv(path)
    out = tidy.groupby("cell_id", as_index=False)["cycle"].max()
    out = out.rename(columns={"cycle": "observed_cycles"})
    if prefix:
        out["cell_id"] = prefix + out["cell_id"].astype(str)
    return out


def build_event_table(features: pd.DataFrame) -> pd.DataFrame:
    cells = features[(features["n_cycles"] == N_CYCLES) & features["dataset"].isin(DATASETS)].copy()
    cells = cells[["dataset", "cell_id", "q0", "cycle_life", "is_censored"]].drop_duplicates("cell_id")

    observed_parts = []
    for dataset, (path, prefix) in CYCLE_SOURCES.items():
        part = observed_cycles_from_tidy(path, prefix=prefix)
        part["dataset"] = dataset
        observed_parts.append(part)
    observed = pd.concat(observed_parts, ignore_index=True)
    cells = cells.merge(observed[["cell_id", "observed_cycles"]], on="cell_id", how="left")
    cells["event_observed"] = (cells["is_censored"] == 0).astype(int)
    cells["time"] = np.where(cells["event_observed"].eq(1), cells["cycle_life"], cells["observed_cycles"]).astype(float)
    missing = cells["time"].isna()
    if missing.any():
        bad = ", ".join(cells.loc[missing, "cell_id"].astype(str).tolist())
        raise ValueError(f"Missing survival time for: {bad}")
    return cells.sort_values(["dataset", "time", "cell_id"]).reset_index(drop=True)


def summarize_dataset(event_table: pd.DataFrame, curves: pd.DataFrame, dataset: str, tau: float) -> dict:
    sub = event_table[event_table["dataset"] == dataset].copy()
    curve = curves[curves["dataset"] == dataset]
    event_times = sub.loc[sub["event_observed"] == 1, "time"].to_numpy(dtype=float)
    lower_bound_times = sub["time"].to_numpy(dtype=float)
    return {
        "dataset": dataset,
        "n_cells": int(len(sub)),
        "n_events": int(sub["event_observed"].sum()),
        "n_censored": int((1 - sub["event_observed"]).sum()),
        "event_mean_cycles": float(np.mean(event_times)) if len(event_times) else float("nan"),
        "event_median_cycles": float(np.median(event_times)) if len(event_times) else float("nan"),
        "lower_bound_mean_cycles": float(np.mean(lower_bound_times)),
        "lower_bound_median_cycles": float(np.median(lower_bound_times)),
        "km_median_cycles": median_survival(curve),
        "rmst_tau_cycles": float(tau),
        "rmst_cycles": restricted_mean_survival_time(curve, tau),
        "survival_at_500": survival_at(curve, 500.0),
        "survival_at_1000": survival_at(curve, 1000.0),
        "survival_at_1500": survival_at(curve, 1500.0),
        "survival_at_2000": survival_at(curve, 2000.0),
        "max_time_cycles": float(sub["time"].max()),
    }


def censored_cell_details(event_table: pd.DataFrame) -> pd.DataFrame:
    censored = event_table[event_table["event_observed"] == 0].copy()
    if censored.empty:
        return pd.DataFrame()

    tidy_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    for _, row in censored.iterrows():
        dataset = row["dataset"]
        path, prefix = CYCLE_SOURCES[dataset]
        if dataset not in tidy_cache:
            tidy = pd.read_csv(path)
            if prefix:
                tidy["cell_id"] = prefix + tidy["cell_id"].astype(str)
            tidy_cache[dataset] = tidy
        tidy = tidy_cache[dataset]
        cell_id = row["cell_id"]
        g = tidy[tidy["cell_id"] == cell_id].sort_values("cycle")
        positive = g[g["Q_discharge"] > 0].copy()
        last_positive = positive.iloc[-1] if not positive.empty else None
        min_positive = positive.loc[positive["Q_discharge"].idxmin()] if not positive.empty else None
        q0 = float(row["q0"])
        rows.append(
            {
                "dataset": dataset,
                "cell_id": cell_id,
                "censor_time_cycles": float(row["time"]),
                "q0": q0,
                "last_positive_cycle": float(last_positive["cycle"]) if last_positive is not None else float("nan"),
                "last_positive_qdis": float(last_positive["Q_discharge"]) if last_positive is not None else float("nan"),
                "last_positive_retention": (
                    float(last_positive["Q_discharge"] / q0) if last_positive is not None and q0 > 0 else float("nan")
                ),
                "min_positive_cycle": float(min_positive["cycle"]) if min_positive is not None else float("nan"),
                "min_positive_qdis": float(min_positive["Q_discharge"]) if min_positive is not None else float("nan"),
                "min_positive_retention": (
                    float(min_positive["Q_discharge"] / q0) if min_positive is not None and q0 > 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "censor_time_cycles", "cell_id"])


def pairwise_tests(event_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for a, b in combinations(DATASETS, 2):
        logrank = logrank_test(event_table, a, b)
        a_events = event_table[(event_table["dataset"] == a) & (event_table["event_observed"] == 1)]["time"]
        b_events = event_table[(event_table["dataset"] == b) & (event_table["event_observed"] == 1)]["time"]
        a_lower = event_table[event_table["dataset"] == a]["time"]
        b_lower = event_table[event_table["dataset"] == b]["time"]
        ks_events = ks_2samp(a_events, b_events)
        ks_lower = ks_2samp(a_lower, b_lower)
        rows.append(
            {
                "group_a": a,
                "group_b": b,
                "logrank_chi2": logrank["chi2"],
                "logrank_p_value": logrank["p_value"],
                "ks_events_statistic": float(ks_events.statistic),
                "ks_events_p_value": float(ks_events.pvalue),
                "ks_lower_bound_statistic": float(ks_lower.statistic),
                "ks_lower_bound_p_value": float(ks_lower.pvalue),
            }
        )
    return pd.DataFrame(rows)


def rmst_bootstrap(event_table: pd.DataFrame, tau: float, *, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dataset_samples: dict[str, np.ndarray] = {}
    dataset_rows = []
    for dataset in DATASETS:
        sub = event_table[event_table["dataset"] == dataset].reset_index(drop=True)
        values = []
        for _ in range(n_boot):
            idx = rng.choice(np.arange(len(sub)), size=len(sub), replace=True)
            boot = sub.iloc[idx].copy()
            curve = kaplan_meier(boot, dataset)
            values.append(restricted_mean_survival_time(curve, tau))
        values = np.asarray(values, dtype=float)
        dataset_samples[dataset] = values
        dataset_rows.append(
            {
                "dataset": dataset,
                "rmst_tau_cycles": float(tau),
                "rmst_boot_mean": float(np.mean(values)),
                "rmst_boot_ci95_low": float(np.percentile(values, 2.5)),
                "rmst_boot_ci95_high": float(np.percentile(values, 97.5)),
                "rmst_boot_std": float(np.std(values, ddof=1)),
                "n_bootstrap": int(n_boot),
            }
        )

    pair_rows = []
    for a, b in combinations(DATASETS, 2):
        diff = dataset_samples[b] - dataset_samples[a]
        p = 2.0 * min(float(np.mean(diff <= 0.0)), float(np.mean(diff >= 0.0)))
        pair_rows.append(
            {
                "group_a": a,
                "group_b": b,
                "rmst_tau_cycles": float(tau),
                "rmst_a_mean": float(np.mean(dataset_samples[a])),
                "rmst_b_mean": float(np.mean(dataset_samples[b])),
                "rmst_diff_b_minus_a": float(np.mean(diff)),
                "rmst_diff_ci95_low": float(np.percentile(diff, 2.5)),
                "rmst_diff_ci95_high": float(np.percentile(diff, 97.5)),
                "rmst_diff_boot_p_value": float(min(1.0, max(p, 1.0 / (n_boot + 1)))),
                "n_bootstrap": int(n_boot),
            }
        )
    return pd.DataFrame(dataset_rows), pd.DataFrame(pair_rows)


def make_plot(curves: pd.DataFrame, event_table: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    apply_science_style()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {
        "matr": "#1f77b4",
        "hust": "#d62728",
        "sandia": "#2ca02c",
        "luh": "#9467bd",
    }
    labels = {"matr": "MATR", "hust": "HUST", "sandia": "Sandia 0-100", "luh": "Luh/KIT"}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for dataset, curve in curves.groupby("dataset", sort=True):
        curve = curve.sort_values("time")
        ax.step(curve["time"], curve["survival"], where="post", color=colors.get(dataset), label=labels.get(dataset, dataset))
        cens = event_table[(event_table["dataset"] == dataset) & (event_table["event_observed"] == 0)]
        if not cens.empty:
            y = [survival_at(curve, float(t)) for t in cens["time"]]
            ax.scatter(cens["time"], y, marker="+", s=70, color=colors.get(dataset), zorder=3)
    ax.set_xlabel("Cycle life / censoring time (cycles)")
    ax.set_ylabel("Kaplan-Meier survival probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kaplan_meier_four_dataset.png", dpi=200)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, float_digits: int | None = None) -> str:
    """Small dependency-free Markdown table writer."""
    formatted = df.copy()
    if float_digits is not None:
        for col in formatted.select_dtypes(include=[np.number]).columns:
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    formatted = formatted.astype(object).where(pd.notna(formatted), "")
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in formatted.to_numpy()]
    return "\n".join([header, separator, *body])


def write_report(
    summary: pd.DataFrame,
    censored: pd.DataFrame,
    pairwise: pd.DataFrame,
    rmst_bootstrap_df: pd.DataFrame,
    rmst_pairwise: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# Four-Dataset Survival/Censoring Audit",
        "",
        "Protocol: N=100 cell table; uncensored cells are EOL events; censored cells are right-censored at last observed cycle.",
        "",
        "## Dataset Summaries",
        markdown_table(summary.sort_values("dataset"), float_digits=3),
        "",
        "## Pairwise Tests",
        markdown_table(pairwise, float_digits=3),
        "",
        "## RMST Bootstrap Robustness",
        "RMST is restricted to the common follow-up horizon across all four datasets.",
        "",
        markdown_table(rmst_bootstrap_df.sort_values("dataset"), float_digits=3),
        "",
        "## Pairwise RMST Differences",
        "Positive `rmst_diff_b_minus_a` means group B has larger restricted mean survival than group A.",
        "",
        markdown_table(rmst_pairwise, float_digits=3),
        "",
        "## Censored Cells",
    ]
    if censored.empty:
        lines.append("No censored cells.")
    else:
        lines.append(markdown_table(censored, float_digits=3))
    lines.extend(
        [
            "",
            "Interpretation: Sandia and Luh introduce additional censoring checks, but the main four-dataset modeling rule remains unchanged: censored cells are excluded from MAE/sMAPE/R2 regression metrics and retained here as right-censored observations. RMST at the common follow-up horizon is the paper-facing robustness statistic because it remains interpretable when median survival and event-only means are distorted by right censoring.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    if not FEATURES_PATH.exists():
        print(f"[error] missing {FEATURES_PATH}")
        return 1
    features = pd.read_csv(FEATURES_PATH)
    event_table = build_event_table(features)
    curves = pd.concat([kaplan_meier(event_table, ds) for ds in DATASETS], ignore_index=True)
    tau = float(event_table.groupby("dataset")["time"].max().min())
    summary = pd.DataFrame([summarize_dataset(event_table, curves, ds, tau) for ds in DATASETS])
    censored = censored_cell_details(event_table)
    pairwise = pairwise_tests(event_table)
    rmst_bootstrap_df, rmst_pairwise = rmst_bootstrap(event_table, tau, n_boot=RMST_BOOTSTRAPS, seed=RANDOM_SEED)

    out_summary = INTERMEDIATE_DIR / "four_dataset_survival_censoring_summary.csv"
    out_curves = INTERMEDIATE_DIR / "four_dataset_survival_censoring_curves.csv"
    out_censored = INTERMEDIATE_DIR / "four_dataset_survival_censoring_censored_cells.csv"
    out_pairwise = INTERMEDIATE_DIR / "four_dataset_survival_censoring_pairwise_tests.csv"
    out_rmst_bootstrap = INTERMEDIATE_DIR / "four_dataset_survival_censoring_rmst_bootstrap.csv"
    out_rmst_pairwise = INTERMEDIATE_DIR / "four_dataset_survival_censoring_rmst_pairwise.csv"
    out_json = INTERMEDIATE_DIR / "four_dataset_survival_censoring.json"
    out_report = INTERMEDIATE_DIR / "four_dataset_survival_censoring_report.md"

    summary.to_csv(out_summary, index=False)
    curves.to_csv(out_curves, index=False)
    censored.to_csv(out_censored, index=False)
    pairwise.to_csv(out_pairwise, index=False)
    rmst_bootstrap_df.to_csv(out_rmst_bootstrap, index=False)
    rmst_pairwise.to_csv(out_rmst_pairwise, index=False)
    make_plot(curves, event_table)
    write_report(summary, censored, pairwise, rmst_bootstrap_df, rmst_pairwise, out_report)

    payload = {
        "protocol": "four_dataset_survival_censoring_v1",
        "n_cycles": N_CYCLES,
        "rmst_tau": tau,
        "rmst_bootstraps": RMST_BOOTSTRAPS,
        "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
        "pairwise_tests": pairwise.replace({np.nan: None}).to_dict(orient="records"),
        "rmst_bootstrap": rmst_bootstrap_df.replace({np.nan: None}).to_dict(orient="records"),
        "rmst_pairwise": rmst_pairwise.replace({np.nan: None}).to_dict(orient="records"),
        "censored_cells": censored.replace({np.nan: None}).to_dict(orient="records"),
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"[save] {out_summary.relative_to(PROJECT_ROOT)}")
    print(f"[save] {out_curves.relative_to(PROJECT_ROOT)}")
    print(f"[save] {out_censored.relative_to(PROJECT_ROOT)}")
    print(f"[save] {out_pairwise.relative_to(PROJECT_ROOT)}")
    print(f"[save] {out_rmst_bootstrap.relative_to(PROJECT_ROOT)}")
    print(f"[save] {out_rmst_pairwise.relative_to(PROJECT_ROOT)}")
    print(f"[save] {out_json.relative_to(PROJECT_ROOT)}")
    print(f"[save] {out_report.relative_to(PROJECT_ROOT)}")
    plot = OUTPUT_DIR / "kaplan_meier_four_dataset.png"
    if plot.exists():
        print(f"[save] {plot.relative_to(PROJECT_ROOT)}")
    print("\n" + out_report.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
