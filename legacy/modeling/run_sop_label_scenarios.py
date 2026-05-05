"""Run a set of SOP label scenarios and save each to a separate JSON file."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_transition.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"
RUNNER = PROJECT_ROOT / "2_modeling_featuring" / "run_sop_protocol_baselines.py"


SCENARIOS = {
    "stored_cycle_life": {
        "label_column": "stored_cycle_life",
        "extra_args": [],
    },
    "eol_80pct_q0_observed_only": {
        "label_column": "eol_80pct_q0_label",
        "extra_args": [
            "--censor-column",
            "is_censored_80pct_q0",
            "--drop-censored",
        ],
    },
    "eol_88ah_observed_only": {
        "label_column": "eol_88ah_label",
        "extra_args": [
            "--censor-column",
            "is_censored_88ah",
            "--drop-censored",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SOP label scenarios.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROJECT_ROOT / "splits",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="sop12_transition",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[50, 100],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456, 789, 1011],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=list(SCENARIOS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scenario_name in args.scenarios:
        config = SCENARIOS[scenario_name]
        output_path = args.output_dir / f"results_sop_{scenario_name}.json"
        command = [
            sys.executable,
            str(RUNNER),
            "--dataset",
            str(args.dataset),
            "--feature-set",
            args.feature_set,
            "--split-dir",
            str(args.split_dir),
            "--label-column",
            config["label_column"],
            "--output",
            str(output_path),
            "--windows",
            *[str(window) for window in args.windows],
            "--seeds",
            *[str(seed) for seed in args.seeds],
            *config["extra_args"],
        ]
        print(f"Running scenario: {scenario_name}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
