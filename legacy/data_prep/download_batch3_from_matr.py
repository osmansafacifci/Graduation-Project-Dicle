"""
Download the full Severson batch3 dataset and convert to pkl.

The repo's existing batch3_varcharge.pkl only contains 2 cells, which
underrepresents batch3 in the merged MATR table (raw_label_table.csv
ended up with 46 b1 + 47 b2 + only 2 b3 = 95 rows instead of ~140).

This script downloads the full standard Severson batch3 from data.matr.io
(file ID 5c86bd64fa2ede00015ddbb2 in project 5c48dd2bc625d700019f3204,
file name 2018-04-12_batchdata_updated_struct_errorcorrect.mat, 46 cells, ~3.2GB)
and writes it to data/raw/batch3_varcharge.pkl, overwriting the 2-cell version.

Note on naming: the student's pipeline expected the "varcharge" experiment
(2018-04-03, batch7pt5 in the chueh-ermon naming), which is a separate dataset
that isn't part of project 5c48dd2bc625d700019f3204. We substitute the standard
Severson batch3 (2018-04-12, internally batch8) — same cell count (46) and
schema, used in the original Severson 2019 Nature Energy paper. The pkl is
saved under the existing filename so the rest of the pipeline (build_raw_label_table.py,
build_sop12_features.py) picks it up without any code changes.

Usage:
    python 0_data_prep/download_batch3_from_matr.py

After it finishes, re-run:
    python 1_feature_engineering/build_raw_label_table.py
    python 1_feature_engineering/build_sop12_features.py
"""

from __future__ import annotations

import pickle
import shutil
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# data.matr.io file metadata (verified via the live API):
#   _id:  5c86bd64fa2ede00015ddbb2
#   name: 2018-04-12_batchdata_updated_struct_errorcorrect.mat
#   size: 3,236,690,412 bytes (~3.0 GiB)
FILE_ID = "5c86bd64fa2ede00015ddbb2"
DOWNLOAD_URL = f"https://data.matr.io/1/api/v1/file/{FILE_ID}/download"
MAT_FILENAME = "2018-04-12_batchdata_updated_struct_errorcorrect.mat"
MAT_PATH = RAW_DIR / MAT_FILENAME
PKL_PATH = RAW_DIR / "batch3_varcharge.pkl"
CELL_PREFIX = "b3"
EXPECTED_MAT_BYTES = 3_236_690_412


def _decode_string(dataset) -> str:
    return dataset.tobytes()[::2].decode().strip()


def download_mat() -> None:
    """Stream the .mat from matr.io to disk, resuming-aware via size check."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if MAT_PATH.exists() and MAT_PATH.stat().st_size == EXPECTED_MAT_BYTES:
        print(f"[skip] {MAT_PATH.name} already present ({EXPECTED_MAT_BYTES:,} bytes).")
        return
    if MAT_PATH.exists():
        print(f"[warn] {MAT_PATH.name} exists but size differs; re-downloading.")
        MAT_PATH.unlink()

    print(f"[download] {DOWNLOAD_URL}")
    print(f"[download] target: {MAT_PATH} (~3.0 GiB, may take several minutes)")
    with urllib.request.urlopen(DOWNLOAD_URL) as resp, MAT_PATH.open("wb") as out:
        total = 0
        chunk = 8 * 1024 * 1024
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            total += len(buf)
            if total % (256 * 1024 * 1024) < chunk:
                print(f"  {total / (1024**3):.2f} GiB downloaded")
    actual = MAT_PATH.stat().st_size
    print(f"[done] wrote {actual:,} bytes to {MAT_PATH}")
    if actual != EXPECTED_MAT_BYTES:
        print(
            f"[warn] downloaded size {actual:,} != expected {EXPECTED_MAT_BYTES:,}. "
            "Conversion may fail; consider re-running."
        )


def convert_to_pkl() -> None:
    """Replicate build_batch_from_mat.py's loader for a single .mat → pkl conversion."""
    print(f"[convert] {MAT_PATH.name} → {PKL_PATH.name} (cell_prefix={CELL_PREFIX!r})")
    with h5py.File(MAT_PATH, "r") as f:
        batch = f["batch"]
        num_cells = batch["summary"].shape[0]
        print(f"[convert] {num_cells} cells in batch")
        bat_dict: dict[str, dict] = {}
        for i in range(num_cells):
            if i % 10 == 0:
                print(f"  cell {i + 1}/{num_cells}")
            cell_key = f"{CELL_PREFIX}c{i}"
            cycle_life = f[batch["cycle_life"][i, 0]][()]
            policy = _decode_string(f[batch["policy_readable"][i, 0]][()])
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
                    "I": np.hstack(f[cycles_group["I"][j, 0]][()]),
                    "Qc": np.hstack(f[cycles_group["Qc"][j, 0]][()]),
                    "Qd": np.hstack(f[cycles_group["Qd"][j, 0]][()]),
                    "Qdlin": np.hstack(f[cycles_group["Qdlin"][j, 0]][()]),
                    "T": np.hstack(f[cycles_group["T"][j, 0]][()]),
                    "Tdlin": np.hstack(f[cycles_group["Tdlin"][j, 0]][()]),
                    "V": np.hstack(f[cycles_group["V"][j, 0]][()]),
                    "dQdV": np.hstack(f[cycles_group["discharge_dQdV"][j, 0]][()]),
                    "t": np.hstack(f[cycles_group["t"][j, 0]][()]),
                }
            bat_dict[cell_key] = {
                "cycle_life": cycle_life,
                "charge_policy": policy,
                "summary": summary,
                "cycles": cycle_dict,
            }

    with PKL_PATH.open("wb") as fp:
        pickle.dump(bat_dict, fp)
    print(f"[done] wrote {len(bat_dict)} cells to {PKL_PATH}")


def main() -> None:
    download_mat()
    convert_to_pkl()
    print()
    print("Next steps (re-run downstream pipeline):")
    print("  python 1_feature_engineering/build_raw_label_table.py")
    print("  python 1_feature_engineering/build_sop12_features.py")


if __name__ == "__main__":
    sys.exit(main() or 0)
