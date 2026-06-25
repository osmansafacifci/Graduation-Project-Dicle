"""
Download MATR and HUST raw data from public Google Drive folders.

The raw data files (batch*.pkl, HUST cell pickles, .mat sources) are too large
for GitHub and are gitignored in this repo. They live in two public Drive folders
shared by the project supervisor. This script fetches them with `gdown`.

Folder contents (expected, supervised upload):
    MATR folder  -> data/raw/                 # batch{1,2}.pkl, batch3_varcharge.pkl, ...
    HUST folder  -> data/raw/HUST_data/       # individual *.pkl files (one per cell)

Usage:
    pip install gdown                          # one-time
    python 0_data/download_data.py        # download both
    python 0_data/download_data.py --only matr   # MATR only
    python 0_data/download_data.py --only hust   # HUST only

Notes:
- The folders must be shared as "Anyone with the link → Viewer".
- Drive sometimes throttles bulk downloads; if you hit a quota error, wait
  a few minutes and re-run (gdown skips files that already exist).
- If gdown fails on a specific file, you can also download it manually from
  the Drive UI and place it under the corresponding directory.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
HUST_DIR = RAW_DIR / "HUST_data"

# Public Drive folder URLs (Anyone-with-link, Viewer access)
MATR_FOLDER_URL = "https://drive.google.com/drive/folders/19wCEj4hr54QtARns1HVOX0alsFlkjt2P"
HUST_FOLDER_URL = "https://drive.google.com/drive/folders/1RVASMPuhWPbgJQE1G4z1856Na4jpxqPf"

# Pinned gdown version for supply-chain safety (avoid auto-installing untested releases).
_GDOWN_VERSION_PIN = "gdown==6.0.0"


def _ensure_gdown() -> None:
    if shutil.which("gdown") is None:
        print("[setup] gdown not found, installing into the current interpreter...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", _GDOWN_VERSION_PIN])


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file (streamed, memory-safe for large files)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def verify_file_hash(path: Path, expected_hash: str | None) -> bool:
    """Return True if file matches expected SHA-256, or if no hash is registered.

    Prints a warning on mismatch (potential tampering or corrupted download).
    """
    if expected_hash is None:
        return True
    actual = _sha256(path)
    if actual != expected_hash:
        print(
            f"[SECURITY WARNING] SHA-256 mismatch for {path.name}!\n"
            f"  expected: {expected_hash}\n"
            f"  actual:   {actual}\n"
            "  The file may be corrupted or tampered with. "
            "Delete it and re-download, or verify the source manually."
        )
        return False
    return True


def _gdown_folder(url: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    print(f"[download]  → {dest}")
    # `--folder` recursively pulls every file in the Drive folder.
    # `--remaining-ok` keeps going even if Drive returns "too many requests" mid-folder.
    cmd = [
        sys.executable, "-m", "gdown",
        "--folder", url,
        "-O", str(dest),
        "--remaining-ok",
        "--quiet",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"[warn] gdown returned non-zero exit ({result.returncode}). "
            "Some files may have failed; re-run to retry, or grab missing files manually."
        )


def download_matr() -> None:
    _gdown_folder(MATR_FOLDER_URL, RAW_DIR)
    print(f"[done] MATR files placed under {RAW_DIR}")
    _list_dir(RAW_DIR, depth=1)


def download_hust() -> None:
    _gdown_folder(HUST_FOLDER_URL, HUST_DIR)
    print(f"[done] HUST files placed under {HUST_DIR}")
    _list_dir(HUST_DIR, depth=1)


def _list_dir(path: Path, depth: int = 1) -> None:
    if not path.exists():
        return
    print(f"\n[contents] {path}:")
    for entry in sorted(path.iterdir()):
        if entry.is_dir():
            n = sum(1 for _ in entry.iterdir())
            print(f"  {entry.name}/  ({n} entries)")
        else:
            size_mb = entry.stat().st_size / (1024 * 1024)
            print(f"  {entry.name}  ({size_mb:.1f} MB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        choices=["matr", "hust", "both"],
        default="both",
        help="Restrict to one dataset (default: both).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_gdown()
    if args.only in ("matr", "both"):
        download_matr()
    if args.only in ("hust", "both"):
        download_hust()
    print("\n[next] Now run the pipeline:")
    print("       python run_pipeline.py")


if __name__ == "__main__":
    main()
