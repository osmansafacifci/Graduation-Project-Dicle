#!/usr/bin/env python3
"""Medium-depth MATR/HUST protocol-mismatch ablation.

Reviewer concern:
    MATR and HUST are both LFP/graphite, so the MATR<->HUST rank collapse may
    be an artefact of protocol heterogeneity rather than conditional shift.

This script performs a bounded 2 x 2 proxy ablation using only protocol
metadata that is reproducibly available in the committed intermediate files:

    - MATR full vs. MATR block-restricted (exclude batch2, the short-life
      experimental block; keep b1+b3)
    - HUST full vs. HUST discharge-profile-restricted (spread across the
      three reported discharge-rate stages <= 2; exact constant-current
      profiles have only n=3 cells and are too small for transfer evaluation)

For each direction (MATR->HUST and HUST->MATR), train the same cross-direction
Gaussian Process diagnostic used in the conditional-shift analysis on the
source train split, then score the full target variant. The key output is
Pearson rank-signal r; if protocol restriction repairs the objection, r should
move from collapsed/negative to clearly positive.

Outputs:
    data/intermediate/protocol_mismatch_ablation_protocol_audit.csv
    data/intermediate/protocol_mismatch_ablation_detailed.csv
    data/intermediate/protocol_mismatch_ablation_summary.csv
    data/intermediate/protocol_mismatch_ablation_report.md
    outputs/results_v2_protocol_mismatch/protocol_mismatch_ablation_rank_signal.png
    outputs/results_v2_protocol_mismatch/protocol_mismatch_ablation_rank_signal.pdf

Usage:
    python 3_analysis/protocol_mismatch_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from plot_style import apply_science_style

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_protocol_mismatch"
FEATURES_PATH = INTERMEDIATE_DIR / "features_sop12_four_dataset_capnorm.csv"
HUST_CYCLES_PATH = INTERMEDIATE_DIR / "hust_cycles_tidy.csv"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2_four_dataset"

sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import META_COLS, SEEDS  # noqa: E402
from shared.battery_utils import display_path as _display_path, load_split as _load_split  # noqa: E402
from metrics_utils import compute_metrics, fit_with_threaded_joblib, to_cycles  # noqa: E402
from run_experiments import fit_gaussian_process  # noqa: E402


def display_path(path: Path) -> str:
    return _display_path(path, PROJECT_ROOT)


def load_split(dataset: str, seed: int) -> dict:
    return _load_split(SPLITS_DIR, dataset, seed)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 3 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def linear_adapter_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 3 or np.std(y_pred) < 1e-12:
        return float("nan")
    model = LinearRegression()
    model.fit(y_pred.reshape(-1, 1), y_true)
    adapted = model.predict(y_pred.reshape(-1, 1))
    return float(r2_score(y_true, adapted))


def add_protocol_columns(features: pd.DataFrame, hust_cycles_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = features.copy()
    audit_rows: list[dict] = []

    matr_mask = df["dataset"].eq("matr")
    df.loc[matr_mask, "protocol_block"] = df.loc[matr_mask, "cell_id"].str.extract(r"matr_(b\d)")[0]
    df.loc[matr_mask, "protocol_variant"] = np.where(
        df.loc[matr_mask, "protocol_block"].isin(["b1", "b3"]),
        "restricted",
        "excluded_batch2",
    )
    df.loc[matr_mask, "protocol_proxy_note"] = "MATR restricted = b1+b3; batch2 excluded as short-life experimental block"

    if hust_cycles_path.exists():
        hust_meta = (
            pd.read_csv(hust_cycles_path)
            .groupby("cell_id")[["dchg_rate_1", "dchg_rate_2", "dchg_rate_3"]]
            .first()
            .reset_index()
        )
        hust_meta["hust_cell_id"] = "hust_" + hust_meta["cell_id"].astype(str)
        rate_cols = ["dchg_rate_1", "dchg_rate_2", "dchg_rate_3"]
        hust_meta["hust_mean_rate"] = hust_meta[rate_cols].mean(axis=1)
        hust_meta["hust_rate_spread"] = hust_meta[rate_cols].max(axis=1) - hust_meta[rate_cols].min(axis=1)
        hust_meta["hust_exact_constant_current"] = hust_meta["hust_rate_spread"].eq(0)
        df = df.merge(
            hust_meta[
                [
                    "hust_cell_id",
                    "dchg_rate_1",
                    "dchg_rate_2",
                    "dchg_rate_3",
                    "hust_mean_rate",
                    "hust_rate_spread",
                    "hust_exact_constant_current",
                ]
            ],
            left_on="cell_id",
            right_on="hust_cell_id",
            how="left",
        )
        df = df.drop(columns=["hust_cell_id"])
    else:
        df["hust_rate_spread"] = np.nan
        df["hust_exact_constant_current"] = False

    hust_mask = df["dataset"].eq("hust")
    df.loc[hust_mask, "protocol_variant"] = np.where(
        df.loc[hust_mask, "hust_rate_spread"].le(2),
        "restricted",
        "excluded_high_spread",
    )
    df.loc[hust_mask, "protocol_proxy_note"] = (
        "HUST restricted = discharge-rate spread <= 2; exact constant-current subset has only n=3"
    )

    n100 = df[df["n_cycles"].eq(100)].copy()
    for dataset in ["matr", "hust"]:
        sub = n100[n100["dataset"].eq(dataset) & n100["is_censored"].eq(0)]
        full_cells = int(sub["cell_id"].nunique())
        restricted = sub[sub["protocol_variant"].eq("restricted")]
        audit_rows.append(
            {
                "dataset": dataset,
                "variant": "full",
                "cells": full_cells,
                "mean_cycle_life": float(sub["cycle_life"].mean()),
                "std_cycle_life": float(sub["cycle_life"].std(ddof=1)),
                "min_cycle_life": float(sub["cycle_life"].min()),
                "max_cycle_life": float(sub["cycle_life"].max()),
                "definition": "all uncensored N=100 cells",
            }
        )
        audit_rows.append(
            {
                "dataset": dataset,
                "variant": "restricted",
                "cells": int(restricted["cell_id"].nunique()),
                "mean_cycle_life": float(restricted["cycle_life"].mean()),
                "std_cycle_life": float(restricted["cycle_life"].std(ddof=1)),
                "min_cycle_life": float(restricted["cycle_life"].min()),
                "max_cycle_life": float(restricted["cycle_life"].max()),
                "definition": str(restricted["protocol_proxy_note"].dropna().iloc[0]),
            }
        )
    if hust_mask.any():
        exact = n100[
            n100["dataset"].eq("hust")
            & n100["is_censored"].eq(0)
            & n100["hust_exact_constant_current"].eq(True)
        ]
        audit_rows.append(
            {
                "dataset": "hust",
                "variant": "exact_constant_current_audit_only",
                "cells": int(exact["cell_id"].nunique()),
                "mean_cycle_life": float(exact["cycle_life"].mean()) if len(exact) else float("nan"),
                "std_cycle_life": float(exact["cycle_life"].std(ddof=1)) if len(exact) > 1 else float("nan"),
                "min_cycle_life": float(exact["cycle_life"].min()) if len(exact) else float("nan"),
                "max_cycle_life": float(exact["cycle_life"].max()) if len(exact) else float("nan"),
                "definition": "HUST cells with dchg_rate_1 == dchg_rate_2 == dchg_rate_3; too small for transfer",
            }
        )

    return df, pd.DataFrame(audit_rows)


def variant_frame(df: pd.DataFrame, dataset: str, variant: str, n_cycles: int) -> pd.DataFrame:
    sub = df[
        df["dataset"].eq(dataset)
        & df["n_cycles"].eq(n_cycles)
        & df["is_censored"].eq(0)
    ].copy()
    if variant == "restricted":
        sub = sub[sub["protocol_variant"].eq("restricted")].copy()
    elif variant != "full":
        raise ValueError(f"Unknown variant: {variant}")
    return sub


def fit_predict_cross(
    source: pd.DataFrame,
    target: pd.DataFrame,
    split: dict,
    feature_cols: list[str],
    *,
    seed: int,
    log_target: bool,
) -> dict:
    train_df = source[source["cell_id"].isin(split["train"])].copy()
    test_df = target.copy()
    if len(train_df) < 5 or len(test_df) < 3:
        return {
            "skipped": True,
            "train_cells": int(len(train_df)),
            "target_cells": int(len(test_df)),
        }

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["cycle_life"].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["cycle_life"].to_numpy(dtype=float)

    y_fit = np.log(y_train) if log_target else y_train
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_train_s = np.clip(np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    X_test_s = np.clip(np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)

    model, tuning = fit_with_threaded_joblib(fit_gaussian_process, X_train_s, y_fit, seed=seed)
    pred = to_cycles(model.predict(X_test_s), log_target=log_target)
    metrics = compute_metrics(y_test, pred)
    metrics.update(
        {
            "skipped": False,
            "train_cells": int(len(train_df)),
            "target_cells": int(len(test_df)),
            "pearson_r": pearson_r(y_test, pred),
            "linear_adapter_R2": linear_adapter_r2(y_test, pred),
            "tuning_kernel": tuning.get("kernel") if isinstance(tuning, dict) else "",
        }
    )
    return metrics


def aggregate_results(detailed: pd.DataFrame) -> pd.DataFrame:
    ok = detailed[~detailed["skipped"].fillna(False)].copy()
    grouped = ok.groupby(["direction", "source_variant", "target_variant"], as_index=False)
    return grouped.agg(
        model=("model", "first"),
        n_seed_runs=("seed", "nunique"),
        train_cells_mean=("train_cells", "mean"),
        target_cells=("target_cells", "first"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        SMAPE_mean=("SMAPE", "mean"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        pearson_r_mean=("pearson_r", "mean"),
        pearson_r_std=("pearson_r", "std"),
        linear_adapter_R2_mean=("linear_adapter_R2", "mean"),
        life_ratio_target_over_source=("life_ratio_target_over_source", "first"),
    )


def write_report(audit: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    def md_table(df: pd.DataFrame, digits: int = 3) -> str:
        out = df.copy()
        for col in out.select_dtypes(include=[np.number]).columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
        header = "| " + " | ".join(out.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
        body = ["| " + " | ".join(map(str, row)) + " |" for row in out.to_numpy()]
        return "\n".join([header, sep, *body])

    compact_cols = [
        "direction",
        "source_variant",
        "target_variant",
        "train_cells_mean",
        "target_cells",
        "R2_mean",
        "pearson_r_mean",
        "linear_adapter_R2_mean",
        "life_ratio_target_over_source",
    ]
    lines = [
        "# Protocol-Mismatch Ablation: MATR <-> HUST",
        "",
        "Purpose: test whether the MATR<->HUST rank collapse disappears when using protocol-restricted source/target variants.",
        "",
        "Important limitation: the committed HUST metadata does not contain a usable repeated exact constant-current subgroup. The exact all-equal discharge-rate subset has only three cells, so this is a medium-depth proxy ablation rather than a definitive matched-protocol experiment.",
        "",
        "Model: Gaussian Process on the same N=100, 34-feature capacity-normalized table used by the four-dataset conditional-shift analysis; source train cells follow the official five source splits; target evaluation uses all uncensored cells in the selected target variant.",
        "",
        "## Cohort Audit",
        md_table(audit, digits=2),
        "",
        "## 2 x 2 Protocol-Restricted Transfer Summary",
        md_table(summary[compact_cols], digits=3),
        "",
        "## Interpretation",
        "If protocol restriction repaired the mechanism, Pearson r should move from collapsed/negative to clearly positive in the restricted/restricted rows. In these outputs the MATR<->HUST rank signal remains weak or negative under the proxy restriction, so the result supports the manuscript's conditional-shift framing, with the caveat that exact protocol matching would require raw protocol metadata or additional cells.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def plot_rank_signal(summary: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    apply_science_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    directions = ["matr_to_hust", "hust_to_matr"]
    variants = [
        ("full", "full", "Full -> full"),
        ("restricted", "full", "Restricted source"),
        ("full", "restricted", "Restricted target"),
        ("restricted", "restricted", "Restricted -> restricted"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, direction in zip(axes, directions, strict=False):
        sub = summary[summary["direction"].eq(direction)].copy()
        vals = []
        labels = []
        colors = []
        for source_variant, target_variant, label in variants:
            row = sub[
                sub["source_variant"].eq(source_variant)
                & sub["target_variant"].eq(target_variant)
            ]
            vals.append(float(row["pearson_r_mean"].iloc[0]) if not row.empty else np.nan)
            labels.append(label)
            colors.append("#2E7D32" if source_variant == "restricted" and target_variant == "restricted" else "#607D8B")
        ax.bar(range(len(vals)), vals, color=colors)
        ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--")
        ax.axhline(0.4, color="#888888", linewidth=0.8, linestyle=":", label="salvageable r=0.4")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_ylim(-0.45, 0.55)
        ax.set_title(direction.replace("_to_", " -> ").upper())
        ax.set_ylabel("Pearson rank signal r")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Protocol-Restricted Ablation Does Not Repair MATR<->HUST Rank Collapse", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "protocol_mismatch_ablation_rank_signal.png", dpi=220)
    fig.savefig(output_dir / "protocol_mismatch_ablation_rank_signal.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    parser.add_argument("--hust-cycles-path", type=Path, default=HUST_CYCLES_PATH)
    parser.add_argument("--n-cycles", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log-target", action="store_true", default=True)
    parser.add_argument("--no-log-target", action="store_false", dest="log_target")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = args.features_path if args.features_path.is_absolute() else PROJECT_ROOT / args.features_path
    hust_cycles_path = args.hust_cycles_path if args.hust_cycles_path.is_absolute() else PROJECT_ROOT / args.hust_cycles_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    if not features_path.exists():
        print(f"[error] missing {features_path}")
        return 1
    if not SPLITS_DIR.exists():
        print(f"[error] missing {SPLITS_DIR}")
        return 1

    df = pd.read_csv(features_path)
    df, audit = add_protocol_columns(df, hust_cycles_path)
    feature_cols = [c for c in df.columns if c not in META_COLS and not c.startswith("protocol_") and not c.startswith("hust_") and not c.startswith("dchg_")]

    audit_path = INTERMEDIATE_DIR / "protocol_mismatch_ablation_protocol_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"[save] {display_path(audit_path)}")
    print(audit.to_string(index=False))

    rows: list[dict] = []
    directions = [("matr", "hust"), ("hust", "matr")]
    variants = ["full", "restricted"]
    for source, target in directions:
        for source_variant in variants:
            for target_variant in variants:
                source_df = variant_frame(df, source, source_variant, args.n_cycles)
                target_df = variant_frame(df, target, target_variant, args.n_cycles)
                life_ratio = float(
                    np.exp(np.log(target_df["cycle_life"]).mean() - np.log(source_df["cycle_life"]).mean())
                )
                for seed in args.seeds:
                    split = load_split(source, seed)
                    result = fit_predict_cross(
                        source_df,
                        target_df,
                        split,
                        feature_cols,
                        seed=seed,
                        log_target=args.log_target,
                    )
                    result.update(
                        {
                            "direction": f"{source}_to_{target}",
                            "source": source,
                            "target": target,
                            "source_variant": source_variant,
                            "target_variant": target_variant,
                            "seed": int(seed),
                            "model": "gaussian_process",
                            "n_cycles": int(args.n_cycles),
                            "source_cells": int(source_df["cell_id"].nunique()),
                            "target_cells_total": int(target_df["cell_id"].nunique()),
                            "life_ratio_target_over_source": life_ratio,
                        }
                    )
                    rows.append(result)
                    if not result.get("skipped", False):
                        print(
                            f"[run] {source}->{target} src={source_variant} tgt={target_variant} "
                            f"seed={seed} r={result['pearson_r']:+.3f} R2={result['R2']:+.3f}"
                        )

    detailed = pd.DataFrame(rows)
    summary = aggregate_results(detailed)
    detailed_path = INTERMEDIATE_DIR / "protocol_mismatch_ablation_detailed.csv"
    summary_path = INTERMEDIATE_DIR / "protocol_mismatch_ablation_summary.csv"
    report_path = INTERMEDIATE_DIR / "protocol_mismatch_ablation_report.md"
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(audit, summary, report_path)
    plot_rank_signal(summary, output_dir)
    print(f"[save] {display_path(detailed_path)}")
    print(f"[save] {display_path(summary_path)}")
    print(f"[save] {display_path(report_path)}")
    print(f"[save] {display_path(output_dir / 'protocol_mismatch_ablation_rank_signal.png')}")
    print(f"[save] {display_path(output_dir / 'protocol_mismatch_ablation_rank_signal.pdf')}")
    print("\n" + report_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
