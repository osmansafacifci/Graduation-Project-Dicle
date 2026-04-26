"""
End-to-end pipeline runner: raw data → labels → features → splits → experiments.

Each stage shells out to the existing student scripts. The runner handles
ordering, timing, missing-data checks, and skip/resume so you don't have
to remember the right invocation order.

Quick start (first time):
    pip install gdown
    python 0_data_prep/download_data.py        # ~15-20 GB into data/raw/
    python run_pipeline.py                     # everything

Common variations:
    python run_pipeline.py --skip-download             # data already on disk
    python run_pipeline.py --stages labels features    # only those stages
    python run_pipeline.py --resume                    # skip stages whose outputs exist

Stages (in order):
    download   data/raw/{batch1,batch2,batch3_varcharge}.pkl + data/raw/HUST_data/*.pkl
    labels     data/intermediate/raw_label_table.csv
    features   data/intermediate/features_{matr_sop12,hust_sop_common,matr_hust_sop_common}.csv
    splits     splits/sop_matr_hust/{matr,hust}_{seed}.json
    experiments outputs/results/results_*.json
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
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_matr_hust"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

PYTHON = sys.executable


@dataclass
class Stage:
    name: str
    description: str
    run: Callable[[], int]      # returns subprocess returncode (0 = success)
    outputs: Sequence[Path]     # files produced; used by --resume / status check


# ---------- stage runners ----------

def _run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=cwd).returncode


def stage_download() -> int:
    return _run([PYTHON, "0_data_prep/download_data.py"])


def stage_labels() -> int:
    required = [RAW_DIR / "batch1.pkl", RAW_DIR / "batch2.pkl"]
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"[skip] labels: missing raw inputs: {[str(m) for m in missing]}")
        print("       run with download stage first or place pkls under data/raw/")
        return 1
    return _run([PYTHON, "1_feature_engineering/build_raw_label_table.py"])


def stage_features() -> int:
    if not (INTERMEDIATE_DIR / "raw_label_table.csv").exists():
        print("[skip] features: raw_label_table.csv missing — run labels stage first.")
        return 1
    cmd = [PYTHON, "1_feature_engineering/build_sop12_features.py"]
    if HUST_DIR.exists() and any(HUST_DIR.glob("*.pkl")):
        cmd += ["--hust-dir", str(HUST_DIR)]
    else:
        print(f"[warn] features: no HUST data found at {HUST_DIR}; HUST tables will be empty.")
    return _run(cmd)


def stage_splits() -> int:
    if not (INTERMEDIATE_DIR / "features_matr_sop12.csv").exists():
        print("[skip] splits: features_matr_sop12.csv missing — run features stage first.")
        return 1
    rc1 = _run([PYTHON, "2_modeling_featuring/generate_json_splits.py"])
    rc2 = _run([PYTHON, "2_modeling_featuring/generate_observed_only_splits.py"])
    return rc1 or rc2


def stage_experiments() -> int:
    if not SPLITS_DIR.exists() or not any(SPLITS_DIR.glob("matr_*.json")):
        print(f"[skip] experiments: no splits found at {SPLITS_DIR} — run splits stage first.")
        return 1
    rc1 = _run([PYTHON, "2_modeling_featuring/run_sop_protocol_baselines.py"])
    rc2 = _run([PYTHON, "2_modeling_featuring/evaluate_cross_dataset_generalization.py"])
    return rc1 or rc2


# ---------- registry ----------

STAGES: dict[str, Stage] = {
    "download": Stage(
        name="download",
        description="Fetch MATR + HUST raw data from public Drive folders",
        run=stage_download,
        outputs=[RAW_DIR / "batch1.pkl", RAW_DIR / "batch2.pkl", HUST_DIR],
    ),
    "labels": Stage(
        name="labels",
        description="Build merged MATR raw label table (Q0, EOL, censoring)",
        run=stage_labels,
        outputs=[INTERMEDIATE_DIR / "raw_label_table.csv"],
    ),
    "features": Stage(
        name="features",
        description="Build SOP12 + cross-dataset feature tables (MATR + HUST)",
        run=stage_features,
        outputs=[
            INTERMEDIATE_DIR / "features_matr_sop12.csv",
            INTERMEDIATE_DIR / "features_hust_sop_common.csv",
            INTERMEDIATE_DIR / "features_matr_hust_sop_common.csv",
        ],
    ),
    "splits": Stage(
        name="splits",
        description="Generate 70/15/15 stratified splits for MATR and HUST (5 seeds)",
        run=stage_splits,
        outputs=[SPLITS_DIR],
    ),
    "experiments": Stage(
        name="experiments",
        description="Run within-dataset and cross-dataset experiments (Elastic Net, XGBoost, ...)",
        run=stage_experiments,
        outputs=[RESULTS_DIR],
    ),
}

DEFAULT_ORDER = ["download", "labels", "features", "splits", "experiments"]


# ---------- orchestration ----------

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
        "--stages",
        nargs="+",
        choices=DEFAULT_ORDER,
        default=DEFAULT_ORDER,
        help="Stages to run (in order). Default: all.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download stage (data already on disk).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip stages whose declared outputs already exist.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print which stage outputs exist, then exit.",
    )
    return parser.parse_args()


def print_status() -> None:
    print("Pipeline status:")
    for name in DEFAULT_ORDER:
        stage = STAGES[name]
        ok = _outputs_exist(stage)
        marker = "✓" if ok else "·"
        print(f"  [{marker}] {name:<12} {stage.description}")
        for out in stage.outputs:
            exists = "exists" if out.exists() else "missing"
            print(f"        {out.relative_to(PROJECT_ROOT) if out.is_absolute() and PROJECT_ROOT in out.parents else out} ({exists})")


def main() -> int:
    args = parse_args()
    if args.status:
        print_status()
        return 0

    selected = list(args.stages)
    if args.skip_download and "download" in selected:
        selected.remove("download")

    print(f"Pipeline plan: {' → '.join(selected)}")
    overall_start = time.time()
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

    total = time.time() - overall_start
    print(f"\n========== Pipeline finished in {total:.1f}s ==========")
    if failures:
        print(f"Failed stages: {', '.join(failures)}")
        return 1
    print("All stages completed successfully.")
    print_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
