"""
Convert Stanford battery .mat datasets into pickle files.

Each pickle matches the structure expected by the feature engineering
notebooks. The loader looks both in the original dataset folder and at the
project root so manually added MAT files can be used without moving them.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_ROOT = (
    PROJECT_ROOT
    / "data-driven-prediction-of-battery-cycle-life-before-capacity-degradation-master"
    / "dataset"
)
ROOT_FALLBACK = PROJECT_ROOT
DATASETS = [
    {
        "name": "batch1",
        "cell_prefix": "b1",
        "mat_filename": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    },
    {
        "name": "batch2",
        "cell_prefix": "b2",
        "mat_filename": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    },
    {
        "name": "batch3_varcharge",
        "cell_prefix": "b3",
        "mat_filename": "2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat",
    },
]


def _decode_string(dataset) -> str:
    """Decode MATLAB char arrays into UTF-8 strings."""
    return dataset.tobytes()[::2].decode().strip()


def _resolve_mat_path(filename: str) -> Path:
    candidates = [
        DATASET_ROOT / filename,
        ROOT_FALLBACK / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise SystemExit(f"MAT file not found. Looked in: {searched}")


def _load_dataset(mat_path: Path, cell_prefix: str) -> Dict[str, dict]:
    """Return the parsed dataset for ``mat_path`` with keys prefixed."""
    print(f"Loading {mat_path.name} ...")
    with h5py.File(mat_path, "r") as f:
        batch = f["batch"]
        num_cells = batch["summary"].shape[0]
        bat_dict: dict[str, dict] = {}

        for i in range(num_cells):
            if i % 10 == 0:
                print(f"  Processing cell {i + 1}/{num_cells}")

            cell_key = f"{cell_prefix}c{i}"
            cycle_life = f[batch["cycle_life"][i, 0]][()]
            policy_dataset = f[batch["policy_readable"][i, 0]][()]
            policy = _decode_string(policy_dataset)

            summary_group = f[batch["summary"][i, 0]]
            summary = {
                "IR": np.hstack(summary_group["IR"][0, :].tolist()),
                "QC": np.hstack(summary_group["QCharge"][0, :].tolist()),
                "QD": np.hstack(summary_group["QDischarge"][0, :].tolist()),
                "Tavg": np.hstack(summary_group["Tavg"][0, :].tolist()),
                "Tmin": np.hstack(summary_group["Tmin"][0, :].tolist()),
                "Tmax": np.hstack(summary_group["Tmax"][0, :].tolist()),
                "chargetime": np.hstack(summary_group["chargetime"][0, :].tolist()),
                "cycle": np.hstack(summary_group["cycle"][0, :].tolist()),
            }

            cycles_group = f[batch["cycles"][i, 0]]
            cycle_dict: dict[str, dict] = {}
            for j in range(cycles_group["I"].shape[0]):
                cycle_dict[str(j)] = {
                    "I": np.hstack((f[cycles_group["I"][j, 0]][()])),
                    "Qc": np.hstack((f[cycles_group["Qc"][j, 0]][()])),
                    "Qd": np.hstack((f[cycles_group["Qd"][j, 0]][()])),
                    "Qdlin": np.hstack((f[cycles_group["Qdlin"][j, 0]][()])),
                    "T": np.hstack((f[cycles_group["T"][j, 0]][()])),
                    "Tdlin": np.hstack((f[cycles_group["Tdlin"][j, 0]][()])),
                    "V": np.hstack((f[cycles_group["V"][j, 0]][()])),
                    "dQdV": np.hstack((f[cycles_group["discharge_dQdV"][j, 0]][()])),
                    "t": np.hstack((f[cycles_group["t"][j, 0]][()])),
                }

            bat_dict[cell_key] = {
                "cycle_life": cycle_life,
                "charge_policy": policy,
                "summary": summary,
                "cycles": cycle_dict,
            }

    return bat_dict


def main() -> None:
    args = _parse_args()
    selected = set(args.only) if args.only else None
    processed = 0

    for dataset in DATASETS:
        if selected and dataset["name"] not in selected:
            continue
        mat_path = _resolve_mat_path(dataset["mat_filename"])
        batch_dict = _load_dataset(mat_path, dataset["cell_prefix"])
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RAW_DIR / f"{dataset['name']}.pkl"
        print(f"Writing pickle to {output_path}")
        with output_path.open("wb") as fp:
            pickle.dump(batch_dict, fp)
        processed += 1

    if processed == 0:
        allowed = ", ".join(d["name"] for d in DATASETS)
        raise SystemExit(f"No datasets processed. Available names: {allowed}")

    print(f"Done. Generated {processed} pickle(s).")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Stanford battery MAT files into pickle batches."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[dataset["name"] for dataset in DATASETS],
        help="Restrict conversion to the selected dataset name(s). Defaults to all.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
