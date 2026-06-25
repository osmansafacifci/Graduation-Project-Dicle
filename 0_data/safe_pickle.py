"""Safe pickle loading utilities.

pickle.load() is inherently unsafe — it can execute arbitrary code during
deserialization.  In this project every .pkl file is produced locally by the
data-preparation pipeline, so the risk is limited.  However, if files are
fetched from remote storage or shared drives, they could be tampered with.

This module provides:
  - `load_pickle(path)`: Wraps pickle.load with integrity warnings logged to
    stderr.  Use it as a drop-in replacement so that every deserialization site
    is auditable.
  - Guidance on how to verify file hashes before loading (see download_data.py
    `verify_file_hash`).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any


def load_pickle(path: Path | str, *, warn: bool = True) -> Any:
    """Load a pickle file with a safety banner.

    Parameters
    ----------
    path : Path or str
        Path to the .pkl file.
    warn : bool, default True
        If True, prints a notice to stderr that pickle deserialization is
        occurring — useful for auditing and CI security pipelines.

    Returns
    -------
    object
        The deserialized Python object.
    """
    path = Path(path)
    if warn:
        print(
            f"[pickle] deserializing {path.name} — ensure this file "
            "was produced by a trusted source.",
            file=sys.stderr,
        )
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301
