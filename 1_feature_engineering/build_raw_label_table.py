"""Build target labels and censoring metadata from raw battery pickles."""

from __future__ import annotations

import argparse
import copy
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "raw_label_table.csv"

# Matches the merge logic used in the original Severson notebook.
BATCH1_CONTINUATION_FROM_BATCH2 = {
    "b1c0": {"source_cell": "b2c7", "add_len": 662},
    "b1c1": {"source_cell": "b2c8", "add_len": 981},
    "b1c2": {"source_cell": "b2c9", "add_len": 1060},
    "b1c3": {"source_cell": "b2c15", "add_len": 208},
    "b1c4": {"source_cell": "b2c16", "add_len": 482},
}

BATCH1_NON_FINISHED_CELLS = {"b1c8", "b1c10", "b1c12", "b1c13", "b1c22"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create raw label/censoring table.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-nonfinished",
        action="store_true",
        help="Keep the batch1 cells that the original notebook drops before modeling.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Raw pickle not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def first_positive(values: np.ndarray) -> float:
    positive = values[np.isfinite(values) & (values > 0)]
    return float(positive[0]) if len(positive) else float("nan")


def first_crossing_cycle(
    cycles: np.ndarray,
    values: np.ndarray,
    *,
    threshold: float,
    inclusive: bool,
) -> float:
    valid = np.isfinite(cycles) & np.isfinite(values) & (values > 0)
    if inclusive:
        hits = np.where(valid & (values <= threshold))[0]
    else:
        hits = np.where(valid & (values < threshold))[0]
    if len(hits) == 0:
        return float("nan")
    return float(cycles[hits[0]])


def observed_cycle_count(cycles: np.ndarray) -> int:
    valid = cycles[np.isfinite(cycles) & (cycles > 0)]
    if len(valid) == 0:
        return 0
    return int(valid[-1])


def merge_continuations(
    batch1: dict[str, Any],
    batch2: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    merged = copy.deepcopy(batch1)
    merge_status = {cell_id: "native" for cell_id in merged}
    if batch2 is None:
        return merged, merge_status

    for target_cell, spec in BATCH1_CONTINUATION_FROM_BATCH2.items():
        source_cell = spec["source_cell"]
        add_len = spec["add_len"]
        if target_cell not in merged or source_cell not in batch2:
            continue

        target = merged[target_cell]
        source = batch2[source_cell]
        target["cycle_life"] = np.asarray(target["cycle_life"]) + add_len

        for key in target["summary"]:
            if key == "cycle":
                offset = len(target["summary"][key])
                target["summary"][key] = np.hstack(
                    (target["summary"][key], source["summary"][key] + offset)
                )
            else:
                target["summary"][key] = np.hstack((target["summary"][key], source["summary"][key]))

        last_cycle = len(target["cycles"])
        for j, source_key in enumerate(sorted(source["cycles"], key=lambda item: int(item))):
            target["cycles"][str(last_cycle + j)] = copy.deepcopy(source["cycles"][source_key])

        merge_status[target_cell] = f"continued_from_{source_cell}"
    return merged, merge_status


def build_rows(
    cells: dict[str, Any],
    *,
    merge_status: dict[str, str],
    include_nonfinished: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell_id, cell_data in sorted(cells.items()):
        if not include_nonfinished and cell_id in BATCH1_NON_FINISHED_CELLS:
            continue

        qd = np.asarray(cell_data["summary"]["QD"], dtype=float).ravel()
        cycles = np.asarray(cell_data["summary"]["cycle"], dtype=float).ravel()
        q0 = first_positive(qd)
        threshold_80 = 0.8 * q0 if math.isfinite(q0) else float("nan")
        threshold_85 = 0.85 * q0 if math.isfinite(q0) else float("nan")
        eol_80 = (
            first_crossing_cycle(cycles, qd, threshold=threshold_80, inclusive=True)
            if math.isfinite(threshold_80)
            else float("nan")
        )
        eol_85 = (
            first_crossing_cycle(cycles, qd, threshold=threshold_85, inclusive=True)
            if math.isfinite(threshold_85)
            else float("nan")
        )
        eol_88 = first_crossing_cycle(cycles, qd, threshold=0.88, inclusive=False)
        observed_cycles = observed_cycle_count(cycles)
        stored_cycle_life = float(np.asarray(cell_data["cycle_life"], dtype=float).ravel()[0])
        last_valid_qd = qd[np.isfinite(qd) & (qd > 0)]
        last_valid_qd_value = float(last_valid_qd[-1]) if len(last_valid_qd) else float("nan")
        gap_to_stored = float(stored_cycle_life - observed_cycles)
        needs_relabel = int(gap_to_stored > 1)
        current_merge_status = merge_status.get(cell_id, "native")
        if needs_relabel and current_merge_status == "native":
            current_merge_status = "stored_label_exceeds_observed_cycles"

        rows.append(
            {
                "cell_id": cell_id,
                "dataset_prefix": cell_id.split("c", 1)[0],
                "stored_cycle_life": stored_cycle_life,
                "q0": q0,
                "q0_source": "first_positive_qd",
                "eol_80pct_q0_cycle": eol_80,
                "eol_80pct_q0_label": eol_80 if math.isfinite(eol_80) else float(observed_cycles + 1),
                "is_censored_80pct_q0": int(not math.isfinite(eol_80)),
                "eol_85pct_q0_cycle": eol_85,
                "eol_85pct_q0_label": eol_85 if math.isfinite(eol_85) else float(observed_cycles + 1),
                "is_censored_85pct_q0": int(not math.isfinite(eol_85)),
                "eol_88ah_cycle": eol_88,
                "eol_88ah_label": eol_88 if math.isfinite(eol_88) else float(observed_cycles + 1),
                "is_censored_88ah": int(not math.isfinite(eol_88)),
                "observed_cycle_count": observed_cycles,
                "gap_to_stored_cycle_life": gap_to_stored,
                "needs_relabel": needs_relabel,
                "last_valid_qd": last_valid_qd_value,
                "merge_status": current_merge_status,
                "kept_for_modeling": int(cell_id not in BATCH1_NON_FINISHED_CELLS),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    batch1 = load_pickle(RAW_DIR / "batch1.pkl")
    batch2 = load_pickle(RAW_DIR / "batch2.pkl") if (RAW_DIR / "batch2.pkl").exists() else None
    merged_batch1, merge_status = merge_continuations(batch1, batch2)

    rows = build_rows(
        merged_batch1,
        merge_status=merge_status,
        include_nonfinished=args.include_nonfinished,
    )
    if batch2 is not None:
        rows.extend(
            build_rows(
                batch2,
                merge_status={cell_id: "native" for cell_id in batch2},
                include_nonfinished=True,
            )
        )
    if (RAW_DIR / "batch3_varcharge.pkl").exists():
        batch3 = load_pickle(RAW_DIR / "batch3_varcharge.pkl")
        rows.extend(
            build_rows(
                batch3,
                merge_status={cell_id: "native" for cell_id in batch3},
                include_nonfinished=True,
            )
        )

    df = pd.DataFrame(rows).sort_values("cell_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} label rows to {args.output}")
    print(
        "Available targets: stored_cycle_life, eol_80pct_q0_label, eol_88ah_label; "
        "censor flags: is_censored_80pct_q0, is_censored_88ah"
    )


if __name__ == "__main__":
    main()
