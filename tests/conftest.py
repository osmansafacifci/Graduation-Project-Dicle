"""Shared fixtures and path setup for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure project packages are importable without installing
for subdir in ("2_models", "1_features", "3_analysis"):
    p = str(PROJECT_ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

# Also add project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
