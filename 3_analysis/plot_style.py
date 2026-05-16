"""Shared SciencePlots styling for paper-facing matplotlib figures."""

from __future__ import annotations


def apply_science_style() -> None:
    """Apply the repository's SciencePlots-based matplotlib style."""
    try:
        import scienceplots  # noqa: F401  # registers the style names with matplotlib
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "SciencePlots is required for plotting. Install dependencies with "
            "`pip install -r requirements.txt` or `pip install SciencePlots`."
        ) from exc

    plt.style.use(["science", "no-latex"])
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
        }
    )
