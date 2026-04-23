"""Utilities for SOP-compliant JSON splits."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


DEFAULT_SEEDS = (42, 123, 456, 789, 1011)


def load_feature_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Feature CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {"cell_id", "n_cycles", "cycle_life"}
    if not rows:
        raise SystemExit("Feature CSV is empty.")
    missing = required.difference(rows[0].keys())
    if missing:
        raise SystemExit(f"Feature CSV missing columns: {sorted(missing)}")
    return rows


def infer_dataset_prefix(cell_id: str) -> str:
    match = re.match(r"^(b\d+|hust|matr)", str(cell_id))
    if not match:
        raise SystemExit(f"Could not infer dataset prefix from cell_id: {cell_id}")
    return match.group(1)


def build_cell_records(rows: list[dict[str, str]], dataset_prefix: str) -> list[dict[str, float | str]]:
    by_cell: dict[str, float] = {}
    for row in rows:
        cell_id = row["cell_id"]
        row_prefix = row.get("dataset_prefix") or infer_dataset_prefix(cell_id)
        if row_prefix != dataset_prefix:
            continue
        cycle_life_raw = row["cycle_life"]
        if cycle_life_raw is None or not str(cycle_life_raw).strip():
            continue
        if cell_id not in by_cell:
            by_cell[cell_id] = float(cycle_life_raw)
    if not by_cell:
        raise SystemExit(f"No rows found for dataset prefix: {dataset_prefix}")
    return [
        {"cell_id": cell_id, "cycle_life": by_cell[cell_id]}
        for cell_id in sorted(by_cell.keys())
    ]


def rank_values(values: list[float]) -> list[int]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0] * len(values)
    for rank, (idx, _) in enumerate(ordered):
        ranks[idx] = rank
    return ranks


def quartile_labels(values: list[float], bins: int = 4) -> list[int]:
    if not values:
        return []
    if len(values) < bins:
        bins = max(1, len(values))
    ranks = rank_values(values)
    labels = []
    for rank in ranks:
        label = min(bins - 1, math.floor(rank * bins / max(1, len(values))))
        labels.append(label)
    return labels


def shuffled_indices(size: int, seed: int) -> list[int]:
    import random

    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw = [ratio * total for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    if remainder > 0:
        fractions = sorted(
            range(3),
            key=lambda idx: (raw[idx] - counts[idx], -idx),
            reverse=True,
        )
        for idx in fractions[:remainder]:
            counts[idx] += 1

    if total >= 3:
        for idx in range(3):
            if counts[idx] == 0:
                donor = max(range(3), key=lambda j: counts[j])
                if counts[donor] > 1:
                    counts[idx] += 1
                    counts[donor] -= 1
    return counts


def build_split_payload(
    cell_records: list[dict[str, float | str]],
    *,
    train_ratio: float = 0.7,
    calibration_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int,
    stratify_bins: int = 4,
) -> dict[str, list[str] | int]:
    if not math.isclose(train_ratio + calibration_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    values = [float(record["cycle_life"]) for record in cell_records]
    labels = quartile_labels(values, bins=stratify_bins)

    label_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    train_ids: list[int] = []
    calibration_ids: list[int] = []
    test_ids: list[int] = []

    for label, idxs in sorted(label_to_indices.items()):
        shuffled = [idxs[i] for i in shuffled_indices(len(idxs), seed + label)]
        train_count, calibration_count, test_count = allocate_counts(
            len(shuffled),
            (train_ratio, calibration_ratio, test_ratio),
        )
        train_ids.extend(shuffled[:train_count])
        calibration_ids.extend(shuffled[train_count : train_count + calibration_count])
        test_ids.extend(
            shuffled[train_count + calibration_count : train_count + calibration_count + test_count]
        )

    payload = {
        "seed": seed,
        "train": [str(cell_records[idx]["cell_id"]) for idx in sorted(train_ids)],
        "calibration": [str(cell_records[idx]["cell_id"]) for idx in sorted(calibration_ids)],
        "test": [str(cell_records[idx]["cell_id"]) for idx in sorted(test_ids)],
    }

    overlap = (
        set(payload["train"]) & set(payload["calibration"])
        | set(payload["train"]) & set(payload["test"])
        | set(payload["calibration"]) & set(payload["test"])
    )
    if overlap:
        raise RuntimeError(f"Split leakage detected across cell IDs: {sorted(overlap)}")
    return payload


def save_split_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_split_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Split JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
