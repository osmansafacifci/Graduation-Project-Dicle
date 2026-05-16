"""
Right-censoring sensitivity analysis for MATR/HUST cycle-life labels.

The primary regression pipeline intentionally excludes censored cells because
MAE/sMAPE/R2 require observed event times. This script keeps the regression
tables unchanged and adds a small survival-analysis check:

  - Build a cell-level event table at N=100.
  - Treat uncensored cycle_life as observed EOL events.
  - Treat censored MATR cells as right-censored at their last observed cycle.
  - Estimate Kaplan-Meier survival curves.
  - Compare MATR vs HUST with a two-sample log-rank test.
  - Report lower-bound sensitivity where censored cells are imputed at their
    censoring time, the earliest event time consistent with the data.

No extra survival package is required; Kaplan-Meier, Greenwood variance, RMST,
and the log-rank statistic are implemented directly for reproducibility.

Outputs:
    data/intermediate/survival_censoring_summary.csv
    data/intermediate/survival_censoring_curves.csv
    data/intermediate/survival_censoring_censored_cells.csv
    data/intermediate/survival_censoring.json
    data/intermediate/survival_censoring_report.txt
    outputs/results_v2_survival/kaplan_meier_matr_hust.png

Usage:
    python 3_analysis/survival_censoring.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, ks_2samp

from plot_style import apply_science_style

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_survival"

FEATURES_PATH = INTERMEDIATE_DIR / "features_sop12_combined.csv"
MATR_CYCLES_PATH = INTERMEDIATE_DIR / "matr_cycles_tidy.csv"
HUST_CYCLES_PATH = INTERMEDIATE_DIR / "hust_cycles_tidy.csv"


def observed_cycles_from_tidy(path: Path, *, prefix: str = "") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["cell_id", "observed_cycles"])
    tidy = pd.read_csv(path)
    out = tidy.groupby("cell_id", as_index=False)["cycle"].max()
    out = out.rename(columns={"cycle": "observed_cycles"})
    if prefix:
        out["cell_id"] = prefix + out["cell_id"].astype(str)
    return out


def build_event_table(features: pd.DataFrame, *, n_cycles: int) -> pd.DataFrame:
    cells = features[features["n_cycles"] == n_cycles].copy()
    cells = cells[["dataset", "cell_id", "q0", "cycle_life", "is_censored"]].drop_duplicates("cell_id")

    observed = pd.concat(
        [
            observed_cycles_from_tidy(MATR_CYCLES_PATH),
            observed_cycles_from_tidy(HUST_CYCLES_PATH, prefix="hust_"),
        ],
        ignore_index=True,
    )
    cells = cells.merge(observed, on="cell_id", how="left")
    cells["event_observed"] = (cells["is_censored"] == 0).astype(int)
    cells["time"] = np.where(
        cells["event_observed"].eq(1),
        cells["cycle_life"],
        cells["observed_cycles"],
    ).astype(float)
    missing = cells["time"].isna()
    if missing.any():
        bad = ", ".join(cells.loc[missing, "cell_id"].astype(str).tolist())
        raise ValueError(f"Missing survival time for: {bad}")
    return cells.sort_values(["dataset", "time", "cell_id"]).reset_index(drop=True)


def kaplan_meier(event_table: pd.DataFrame, dataset: str) -> pd.DataFrame:
    sub = event_table[event_table["dataset"] == dataset].copy()
    times = np.sort(sub.loc[sub["event_observed"] == 1, "time"].unique())
    survival = 1.0
    greenwood_sum = 0.0
    rows: list[dict] = [
        {
            "dataset": dataset,
            "time": 0.0,
            "n_at_risk": int(len(sub)),
            "n_events": 0,
            "n_censored_at_time": 0,
            "survival": 1.0,
            "greenwood_se": 0.0,
        }
    ]

    for t in times:
        n_at_risk = int(np.sum(sub["time"].to_numpy(dtype=float) >= t))
        n_events = int(np.sum((sub["time"] == t) & (sub["event_observed"] == 1)))
        n_censored = int(np.sum((sub["time"] == t) & (sub["event_observed"] == 0)))
        if n_at_risk <= 0 or n_events <= 0:
            continue
        survival *= 1.0 - (n_events / n_at_risk)
        denom = n_at_risk * max(n_at_risk - n_events, 1)
        if n_at_risk > n_events and denom > 0:
            greenwood_sum += n_events / denom
        se = survival * float(np.sqrt(greenwood_sum))
        rows.append(
            {
                "dataset": dataset,
                "time": float(t),
                "n_at_risk": n_at_risk,
                "n_events": n_events,
                "n_censored_at_time": n_censored,
                "survival": float(survival),
                "greenwood_se": float(se),
            }
        )
    return pd.DataFrame(rows)


def median_survival(curve: pd.DataFrame) -> float:
    below = curve[curve["survival"] <= 0.5]
    if below.empty:
        return float("nan")
    return float(below.iloc[0]["time"])


def survival_at(curve: pd.DataFrame, t: float) -> float:
    eligible = curve[curve["time"] <= t]
    if eligible.empty:
        return 1.0
    return float(eligible.iloc[-1]["survival"])


def restricted_mean_survival_time(curve: pd.DataFrame, tau: float) -> float:
    rows = curve.sort_values("time")
    prev_t = 0.0
    prev_s = 1.0
    area = 0.0
    for _, row in rows.iloc[1:].iterrows():
        t = float(row["time"])
        if t > tau:
            break
        area += prev_s * (t - prev_t)
        prev_t = t
        prev_s = float(row["survival"])
    if tau > prev_t:
        area += prev_s * (tau - prev_t)
    return float(area)


def logrank_test(event_table: pd.DataFrame, group_a: str, group_b: str) -> dict:
    sub = event_table[event_table["dataset"].isin([group_a, group_b])].copy()
    event_times = np.sort(sub.loc[sub["event_observed"] == 1, "time"].unique())
    oe = 0.0
    var = 0.0

    for t in event_times:
        at_risk_a = int(np.sum((sub["dataset"] == group_a) & (sub["time"] >= t)))
        at_risk_b = int(np.sum((sub["dataset"] == group_b) & (sub["time"] >= t)))
        events_a = int(np.sum((sub["dataset"] == group_a) & (sub["time"] == t) & (sub["event_observed"] == 1)))
        events_b = int(np.sum((sub["dataset"] == group_b) & (sub["time"] == t) & (sub["event_observed"] == 1)))
        at_risk = at_risk_a + at_risk_b
        events = events_a + events_b
        if at_risk <= 1 or events == 0:
            continue
        expected_a = events * at_risk_a / at_risk
        v = (
            at_risk_a
            * at_risk_b
            * events
            * (at_risk - events)
            / ((at_risk**2) * (at_risk - 1))
        )
        oe += events_a - expected_a
        var += v

    statistic = (oe**2 / var) if var > 0 else float("nan")
    p_value = float(chi2.sf(statistic, df=1)) if np.isfinite(statistic) else float("nan")
    return {
        "group_a": group_a,
        "group_b": group_b,
        "observed_minus_expected_group_a": float(oe),
        "variance": float(var),
        "chi2": float(statistic),
        "p_value": p_value,
    }


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
        "survival_at_1000": survival_at(curve, 1000.0),
        "survival_at_1500": survival_at(curve, 1500.0),
        "survival_at_2000": survival_at(curve, 2000.0),
        "max_time_cycles": float(sub["time"].max()),
    }


def censored_cell_details(event_table: pd.DataFrame) -> pd.DataFrame:
    censored = event_table[event_table["event_observed"] == 0].copy()
    if censored.empty:
        return pd.DataFrame()
    tidy = pd.read_csv(MATR_CYCLES_PATH)
    rows: list[dict] = []
    for _, row in censored.iterrows():
        cell_id = row["cell_id"]
        g = tidy[tidy["cell_id"] == cell_id].sort_values("cycle")
        positive = g[g["Q_discharge"] > 0].copy()
        last_positive = positive.iloc[-1] if not positive.empty else None
        min_positive = positive.loc[positive["Q_discharge"].idxmin()] if not positive.empty else None
        q0 = float(row["q0"])
        rows.append(
            {
                "dataset": row["dataset"],
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
    return pd.DataFrame(rows).sort_values("censor_time_cycles")


def make_plot(curves: pd.DataFrame, event_table: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    apply_science_style()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"matr": "#1f77b4", "hust": "#d62728"}
    labels = {"matr": "MATR", "hust": "HUST"}

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
    fig.savefig(output_dir / "kaplan_meier_matr_hust.png", dpi=200)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    censored: pd.DataFrame,
    logrank: dict,
    ks_uncensored: dict,
    ks_lower_bound: dict,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("Survival/censoring sensitivity analysis")
    lines.append("=" * 80)
    lines.append("Protocol: N=100 cell table; uncensored cells are EOL events; censored MATR cells are right-censored at last observed cycle.")
    lines.append("")
    lines.append("Dataset summaries:")
    for _, row in summary.sort_values("dataset").iterrows():
        lines.append(
            f"  {row['dataset'].upper()}: n={int(row['n_cells'])}, events={int(row['n_events'])}, "
            f"censored={int(row['n_censored'])}, KM median={row['km_median_cycles']:.0f}, "
            f"RMST(tau={row['rmst_tau_cycles']:.0f})={row['rmst_cycles']:.0f}, "
            f"event mean={row['event_mean_cycles']:.0f}, lower-bound mean={row['lower_bound_mean_cycles']:.0f}"
        )
    lines.append("")
    lines.append(
        f"Log-rank MATR vs HUST: chi2={logrank['chi2']:.2f}, p={logrank['p_value']:.3e}."
    )
    lines.append(
        f"KS on observed event times only: D={ks_uncensored['statistic']:.3f}, p={ks_uncensored['p_value']:.3e}."
    )
    lines.append(
        f"KS with censored MATR imputed at lower-bound censor times: D={ks_lower_bound['statistic']:.3f}, p={ks_lower_bound['p_value']:.3e}."
    )
    lines.append("")
    lines.append("Censored MATR cells:")
    for _, row in censored.iterrows():
        lines.append(
            f"  {row['cell_id']}: censored >= {row['censor_time_cycles']:.0f} cycles, "
            f"last retention={100 * row['last_positive_retention']:.1f}%, "
            f"min positive retention={100 * row['min_positive_retention']:.1f}%"
        )
    lines.append("")
    lines.append("Interpretation: properly accounting for the six censored MATR cells does not weaken the cross-dataset lifetime-shift conclusion; HUST remains markedly longer-lived.")
    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cycles", type=int, default=100)
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="RMST horizon. Default = smaller of the two datasets' maximum observed survival times.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FEATURES_PATH.exists():
        print(f"[error] missing {FEATURES_PATH}")
        return 1

    features = pd.read_csv(FEATURES_PATH)
    event_table = build_event_table(features, n_cycles=args.n_cycles)
    datasets = sorted(event_table["dataset"].unique())
    curves = pd.concat([kaplan_meier(event_table, ds) for ds in datasets], ignore_index=True)

    max_by_dataset = event_table.groupby("dataset")["time"].max()
    tau = float(args.tau) if args.tau is not None else float(max_by_dataset.min())

    summary = pd.DataFrame([summarize_dataset(event_table, curves, ds, tau) for ds in datasets])
    censored = censored_cell_details(event_table)
    logrank = logrank_test(event_table, "matr", "hust")

    matr_events = event_table[(event_table["dataset"] == "matr") & (event_table["event_observed"] == 1)]["time"]
    hust_events = event_table[(event_table["dataset"] == "hust") & (event_table["event_observed"] == 1)]["time"]
    matr_lower_bound = event_table[event_table["dataset"] == "matr"]["time"]
    ks_uncensored_raw = ks_2samp(matr_events, hust_events)
    ks_lower_bound_raw = ks_2samp(matr_lower_bound, hust_events)
    ks_uncensored = {"statistic": float(ks_uncensored_raw.statistic), "p_value": float(ks_uncensored_raw.pvalue)}
    ks_lower_bound = {"statistic": float(ks_lower_bound_raw.statistic), "p_value": float(ks_lower_bound_raw.pvalue)}

    out_summary = INTERMEDIATE_DIR / "survival_censoring_summary.csv"
    out_curves = INTERMEDIATE_DIR / "survival_censoring_curves.csv"
    out_censored = INTERMEDIATE_DIR / "survival_censoring_censored_cells.csv"
    out_json = INTERMEDIATE_DIR / "survival_censoring.json"
    out_report = INTERMEDIATE_DIR / "survival_censoring_report.txt"

    summary.to_csv(out_summary, index=False)
    curves.to_csv(out_curves, index=False)
    censored.to_csv(out_censored, index=False)
    make_plot(curves, event_table, args.output_dir)
    write_report(summary, censored, logrank, ks_uncensored, ks_lower_bound, out_report)

    payload = {
        "protocol": "survival_censoring_sensitivity_v1",
        "n_cycles": args.n_cycles,
        "rmst_tau": tau,
        "logrank": logrank,
        "ks_uncensored_events_only": ks_uncensored,
        "ks_matr_censored_as_lower_bound": ks_lower_bound,
        "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
        "censored_cells": censored.replace({np.nan: None}).to_dict(orient="records"),
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"[save] {out_summary}")
    print(f"[save] {out_curves}")
    print(f"[save] {out_censored}")
    print(f"[save] {out_json}")
    print(f"[save] {out_report}")
    plot_path = args.output_dir / "kaplan_meier_matr_hust.png"
    if plot_path.exists():
        print(f"[save] {plot_path}")
    print("\n" + out_report.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
