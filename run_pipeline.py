"""
End-to-end SOPv2 pipeline runner for extraction, modeling, and analysis.

This runner is built around the supervisor's reference notebooks
(Cell_Audit_MATR_Provided_vs_Recomputed_EOL.ipynb,
 HUST_Standalone_Preprocess_and_Audit.ipynb), faithfully ported into
 0_data/build_matr_audit.py and 0_data/build_hust_audit.py.
It is independent of the student's existing build_raw_label_table.py /
build_sop12_features.py path (which has known issues with Q0, feature
naming, and batch3 source).

Quick start:
    pip install -r requirements.txt
    python run_pipeline.py --phase extract        # download + audits + features
    python run_pipeline.py --phase model          # splits + VIF + headline experiments
    python run_pipeline.py --phase analysis       # shift + feature transfer + concept + rescaling + CP
    python run_pipeline.py --stages audit_matr    # only one stage
    python run_pipeline.py --status               # which outputs exist

Stages:
    download      pulls MATR + HUST raw data from public Drive folders
                  -> data/raw/{batch1,batch2,batch3}.pkl
                  -> data/raw/HUST_data/*.pkl  (77 cells)
    audit_matr    faithful port of MATR audit notebook
                  -> data/intermediate/matr_cell_audit_strict.csv
                  -> data/intermediate/matr_cell_audit_replication.csv
                  -> data/intermediate/matr_retention_summary.csv
    audit_hust    faithful port of HUST preprocess + audit notebook
                  -> data/intermediate/hust_cycles_tidy.csv
                  -> data/intermediate/hust_threshold_audit.csv
                  -> data/intermediate/hust_threshold_summary.csv
    features      build the corrected 34-feature capacity-only table
                  -> data/intermediate/features_sop12_{matr,hust,combined}.csv
    splits        70/15/15 lifetime-stratified splits (5 seeds × 2 datasets)
                  -> splits/sop_v2/{matr,hust}_{seed}.json
    experiments   headline within-dataset protocol: 34 features + log-target
                  -> outputs/results_v2_34feat_log/results_within_{matr,hust}.json
                  -> outputs/results_v2_34feat_log/results_summary.csv
    transfer      feature-level transfer/stability analysis
                  -> data/intermediate/feature_transfer_stability.csv
    conformal     MAPIE split CP for within, naive cross, target CP, and
                  target-adapted CP
                  -> outputs/results_v2_conformal/results_summary.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
HUST_DIR = RAW_DIR / "HUST_data"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

PYTHON = sys.executable


@dataclass
class Stage:
    name: str
    description: str
    run: Callable[[], int]
    outputs: Sequence[Path]


def _run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=cwd).returncode


def stage_download() -> int:
    return _run([PYTHON, "0_data/download_data.py"])


def stage_audit_matr() -> int:
    required = [RAW_DIR / "batch1.pkl", RAW_DIR / "batch2.pkl", RAW_DIR / "batch3.pkl"]
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"[skip] audit_matr: missing inputs: {[str(m.relative_to(PROJECT_ROOT)) for m in missing]}")
        print("       run download stage first.")
        return 1
    return _run([PYTHON, "0_data/build_matr_audit.py"])


def stage_audit_hust() -> int:
    if not HUST_DIR.exists() or not any(HUST_DIR.glob("*.pkl")):
        print(f"[skip] audit_hust: no HUST .pkl files at {HUST_DIR.relative_to(PROJECT_ROOT)}")
        print("       run download stage first.")
        return 1
    return _run([PYTHON, "0_data/build_hust_audit.py"])


def stage_features() -> int:
    matr_required = [RAW_DIR / "batch1.pkl", RAW_DIR / "batch2.pkl", RAW_DIR / "batch3.pkl"]
    matr_missing = [p for p in matr_required if not p.exists()]
    hust_tidy = INTERMEDIATE_DIR / "hust_cycles_tidy.csv"
    if matr_missing:
        print(f"[skip] features: missing MATR pkls: {[str(m.relative_to(PROJECT_ROOT)) for m in matr_missing]}")
        return 1
    if not hust_tidy.exists():
        print(f"[skip] features: missing {hust_tidy.relative_to(PROJECT_ROOT)} — run audit_hust first.")
        return 1
    return _run([PYTHON, "1_features/build_features.py"])


def stage_splits() -> int:
    combined = INTERMEDIATE_DIR / "features_sop12_combined.csv"
    if not combined.exists():
        print(f"[skip] splits: missing {combined.relative_to(PROJECT_ROOT)} — run features first.")
        return 1
    return _run([PYTHON, "2_models/generate_splits.py"])


def stage_vif() -> int:
    combined = INTERMEDIATE_DIR / "features_sop12_combined.csv"
    splits_root = PROJECT_ROOT / "splits" / "sop_v2"
    if not combined.exists():
        print(f"[skip] vif: missing {combined.relative_to(PROJECT_ROOT)} — run features first.")
        return 1
    if not splits_root.exists() or not any(splits_root.glob("matr_*.json")):
        print(f"[skip] vif: missing {splits_root.relative_to(PROJECT_ROOT)} — run splits first.")
        return 1
    # Default = report-only per supervisor's request.
    return _run([PYTHON, "2_models/vif_screening.py"])


def stage_experiments() -> int:
    splits_root = PROJECT_ROOT / "splits" / "sop_v2"
    if not splits_root.exists() or not any(splits_root.glob("*.json")):
        print(f"[skip] experiments: missing splits at {splits_root.relative_to(PROJECT_ROOT)} — run splits first.")
        return 1
    return _run([
        PYTHON,
        "2_models/run_experiments.py",
        "--log-target",
        "--output-dir",
        "outputs/results_v2_34feat_log",
    ])


def stage_shift() -> int:
    combined = INTERMEDIATE_DIR / "features_sop12_combined.csv"
    if not combined.exists():
        print(f"[skip] shift: missing {combined.relative_to(PROJECT_ROOT)} — run features first.")
        return 1
    rc = _run([PYTHON, "3_analysis/shift_metrics.py"])
    if rc != 0:
        return rc
    return _run([PYTHON, "3_analysis/shift_metrics.py", "--capacity-normalize"])


def stage_concept_shift() -> int:
    combined = INTERMEDIATE_DIR / "features_sop12_combined.csv"
    splits_root = PROJECT_ROOT / "splits" / "sop_v2"
    if not combined.exists() or not splits_root.exists():
        print("[skip] concept_shift: missing features or splits.")
        return 1
    return _run([PYTHON, "3_analysis/concept_shift_diagnostics.py"])


def stage_feature_transfer() -> int:
    combined = INTERMEDIATE_DIR / "features_sop12_combined.csv"
    splits_root = PROJECT_ROOT / "splits" / "sop_v2"
    if not combined.exists() or not splits_root.exists():
        print("[skip] feature_transfer: missing features or splits.")
        return 1
    return _run([PYTHON, "3_analysis/feature_transfer_stability.py"])


def stage_target_rescale() -> int:
    combined = INTERMEDIATE_DIR / "features_sop12_combined.csv"
    splits_root = PROJECT_ROOT / "splits" / "sop_v2"
    if not combined.exists() or not splits_root.exists():
        print("[skip] target_rescale: missing features or splits.")
        return 1
    return _run([PYTHON, "3_analysis/target_rescaling.py"])


def stage_conformal() -> int:
    combined = INTERMEDIATE_DIR / "features_sop12_combined.csv"
    splits_root = PROJECT_ROOT / "splits" / "sop_v2"
    if not combined.exists() or not splits_root.exists():
        print("[skip] conformal: missing features or splits.")
        return 1
    rc = _run([PYTHON, "3_analysis/conformal_prediction.py"])
    if rc != 0:
        return rc
    return _run([PYTHON, "3_analysis/summarize_conformal_results.py"])


STAGES: dict[str, Stage] = {
    "download": Stage(
        name="download",
        description="Fetch MATR + HUST raw data from public Drive folders",
        run=stage_download,
        outputs=[
            RAW_DIR / "batch1.pkl",
            RAW_DIR / "batch2.pkl",
            RAW_DIR / "batch3.pkl",
            HUST_DIR,
        ],
    ),
    "audit_matr": Stage(
        name="audit_matr",
        description="MATR cell-level audit (notebook port: provided vs recomputed 80% Q0 EOL)",
        run=stage_audit_matr,
        outputs=[
            INTERMEDIATE_DIR / "matr_cell_audit_strict.csv",
            INTERMEDIATE_DIR / "matr_cell_audit_replication.csv",
            INTERMEDIATE_DIR / "matr_retention_summary.csv",
        ],
    ),
    "audit_hust": Stage(
        name="audit_hust",
        description="HUST preprocess + threshold audit (notebook port: 90/85/80% Q0)",
        run=stage_audit_hust,
        outputs=[
            INTERMEDIATE_DIR / "hust_cycles_tidy.csv",
            INTERMEDIATE_DIR / "hust_threshold_audit.csv",
            INTERMEDIATE_DIR / "hust_threshold_summary.csv",
        ],
    ),
    "features": Stage(
        name="features",
        description="Build corrected 34-feature capacity-only table for MATR and HUST",
        run=stage_features,
        outputs=[
            INTERMEDIATE_DIR / "features_sop12_matr.csv",
            INTERMEDIATE_DIR / "features_sop12_hust.csv",
            INTERMEDIATE_DIR / "features_sop12_combined.csv",
        ],
    ),
    "splits": Stage(
        name="splits",
        description="70/15/15 lifetime-stratified cell splits (5 seeds × 2 datasets)",
        run=stage_splits,
        outputs=[PROJECT_ROOT / "splits" / "sop_v2"],
    ),
    "vif": Stage(
        name="vif",
        description="VIF screening on MATR train (report-only by default, no features dropped)",
        run=stage_vif,
        outputs=[
            INTERMEDIATE_DIR / "vif_screening.json",
            INTERMEDIATE_DIR / "vif_report.txt",
        ],
    ),
    "experiments": Stage(
        name="experiments",
        description="Headline within-dataset experiments — 34 features + log-target",
        run=stage_experiments,
        outputs=[
            PROJECT_ROOT / "outputs" / "results_v2_34feat_log" / "results_within_matr.json",
            PROJECT_ROOT / "outputs" / "results_v2_34feat_log" / "results_within_hust.json",
            PROJECT_ROOT / "outputs" / "results_v2_34feat_log" / "results_summary.csv",
        ],
    ),
    "shift": Stage(
        name="shift",
        description="Distribution-shift quantification: MMD + Mahalanobis + per-feature attribution",
        run=stage_shift,
        outputs=[
            INTERMEDIATE_DIR / "shift_metrics.json",
            INTERMEDIATE_DIR / "shift_metrics_capnorm.json",
        ],
    ),
    "concept_shift": Stage(
        name="concept_shift",
        description="Concept-shift diagnostics: KS test + per-cell residual constant-bias decomposition",
        run=stage_concept_shift,
        outputs=[INTERMEDIATE_DIR / "concept_shift_diagnostics.json"],
    ),
    "feature_transfer": Stage(
        name="feature_transfer",
        description="Feature-level transfer/stability analysis",
        run=stage_feature_transfer,
        outputs=[
            INTERMEDIATE_DIR / "feature_transfer_stability.csv",
            INTERMEDIATE_DIR / "feature_transfer_stability_report.txt",
        ],
    ),
    "target_rescale": Stage(
        name="target_rescale",
        description="Target-mean rescaling baseline (k=5/10/20 calibration cells)",
        run=stage_target_rescale,
        outputs=[
            PROJECT_ROOT / "outputs" / "results_v2_target_rescale" / "results_summary.csv",
        ],
    ),
    "conformal": Stage(
        name="conformal",
        description="MAPIE split CP intervals (within + cross + target-adapted)",
        run=stage_conformal,
        outputs=[
            PROJECT_ROOT / "outputs" / "results_v2_conformal" / "results_summary.csv",
            PROJECT_ROOT / "outputs" / "results_v2_conformal" / "results_conformal.json",
            PROJECT_ROOT / "outputs" / "results_v2_conformal" / "paper_cp_summary.csv",
        ],
    ),
}

DEFAULT_ORDER = [
    "download", "audit_matr", "audit_hust", "features",
    "splits", "vif", "experiments",
    "shift", "feature_transfer", "concept_shift", "target_rescale", "conformal",
]

# Phase shortcuts. Two-phase workflow:
#   extract: heavy I/O — reads raw .pkl from Drive, produces feature CSVs.
#            Run this once on Colab (or anywhere with the raw data).
#   model:   CPU-only — reads the feature CSVs that extract produced and
#            runs everything downstream. Run this locally; iterate freely.
PHASES: dict[str, list[str]] = {
    "extract":  ["download", "audit_matr", "audit_hust", "features"],
    "model":    ["splits", "vif", "experiments"],
    "analysis": ["shift", "feature_transfer", "concept_shift", "target_rescale", "conformal"],
    "all":      DEFAULT_ORDER,
}


def _outputs_exist(stage: Stage) -> bool:
    for out in stage.outputs:
        if out.is_dir():
            if not out.exists() or not any(out.iterdir()):
                return False
        else:
            if not out.exists():
                return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=list(PHASES.keys()),
        default=None,
        help="Convenience shortcut. 'extract' = audit + features (heavy I/O, run on Colab). "
             "'model' = splits + vif + experiments (CPU-light, run locally). "
             "'all' = everything. Overrides --stages when set.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=DEFAULT_ORDER,
        default=DEFAULT_ORDER,
        help="Stages to run, in order. Default: all. Ignored when --phase is set.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip the download stage.")
    parser.add_argument("--resume", action="store_true", help="Skip stages whose outputs exist.")
    parser.add_argument("--status", action="store_true", help="Print which outputs exist; exit.")
    return parser.parse_args()


def print_status() -> None:
    print("Pipeline status:")
    for name in DEFAULT_ORDER:
        stage = STAGES[name]
        ok = _outputs_exist(stage)
        marker = "✓" if ok else "·"
        print(f"  [{marker}] {name:<12} {stage.description}")
        for out in stage.outputs:
            rel = out.relative_to(PROJECT_ROOT) if PROJECT_ROOT in out.parents else out
            present = "exists" if out.exists() else "missing"
            print(f"        {rel}  ({present})")


def main() -> int:
    args = parse_args()
    if args.status:
        print_status()
        return 0

    selected = list(PHASES[args.phase]) if args.phase else list(args.stages)
    if args.skip_download and "download" in selected:
        selected.remove("download")

    print(f"Pipeline plan: {' → '.join(selected)}")
    overall = time.time()
    failures: list[str] = []

    for name in selected:
        stage = STAGES[name]
        if args.resume and _outputs_exist(stage):
            print(f"\n[resume] skipping {name}: outputs already exist.")
            continue
        print(f"\n========== [{name}] {stage.description} ==========")
        t0 = time.time()
        rc = stage.run()
        elapsed = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"\n[{name}] {status} in {elapsed:.1f}s")
        if rc != 0:
            failures.append(name)

    total = time.time() - overall
    print(f"\n========== Pipeline finished in {total:.1f}s ==========")
    if failures:
        print(f"Failed stages: {', '.join(failures)}")
        return 1
    print("All stages completed successfully.")
    print_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
