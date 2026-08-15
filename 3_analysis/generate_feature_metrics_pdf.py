#!/usr/bin/env python3
"""Generate a complete PDF of the 34-feature contract and prediction metrics.

The report covers:
  - all 34 engineered capacity-trajectory features;
  - all classical within-dataset results for MATR, HUST, Sandia, and Luh;
  - all classical naive cross-dataset directions among those four datasets;
  - held-out UMich external-validation results; and
  - CNN, Transformer, and Mamba trajectory-input sensitivity results.

Run from the repository root:
    python 3_analysis/generate_feature_metrics_pdf.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "output" / "pdf" / "battery_rul_34_features_and_metrics.pdf"

WITHIN_PATH = ROOT / "outputs/results_v2_four_dataset_within_34feat_log/results_summary.csv"
CROSS_PATH = ROOT / "outputs/results_v2_four_dataset_cross_34feat_capnorm_log/results_summary.csv"
UMICH_WITHIN_PATH = ROOT / "outputs/results_v2_external_umich_within_34feat_capnorm_log/results_summary.csv"
UMICH_CROSS_ROOT = ROOT / "outputs/results_v2_external_umich_cross_34feat_capnorm_log"
FEATURE_PATH = ROOT / "data/intermediate/features_sop12_four_dataset_capnorm.csv"
TARGET_RESCALE_PATH = ROOT / "outputs/results_v2_four_dataset_target_rescale/results_summary.csv"

DEEP_PATHS = {
    "1D-CNN": ROOT / "outputs/results_v2_four_dataset_cnn_pytorch/results_summary.csv",
    "Transformer": ROOT / "outputs/results_v2_four_dataset_transformer_pytorch/results_summary.csv",
    "Mamba": ROOT / "outputs/results_v2_four_dataset_mamba_library_pytorch/results_summary.csv",
}

MODEL_LABELS = {
    "elastic_net": "Elastic Net",
    "pls": "PLS",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "gaussian_process": "Gaussian Process",
    "stacking": "Stacking",
    "pytorch_1d_cnn": "1D-CNN",
    "pytorch_simple_transformer": "Transformer",
    "pytorch_mamba_library": "Mamba",
}

DATASET_LABELS = {
    "matr": "MATR",
    "hust": "HUST",
    "sandia": "Sandia",
    "luh": "Luh",
    "umich": "UMich",
}

FEATURE_DEFINITIONS = [
    ("Qdis_N", "Level", "Discharge capacity at the final observed cycle N."),
    ("delta_Qdis", "Change", "Qdis_N minus discharge capacity at cycle 2."),
    ("retention_ratio", "Change", "Qdis_N divided by discharge capacity at cycle 2."),
    ("slope_linear", "Trend", "OLS slope of Qdis over cycles 2 through N."),
    ("variance_Qdis", "Spread", "Variance of Qdis over cycles 2 through N."),
    ("range_Qdis", "Spread", "Maximum Qdis minus minimum Qdis in the window."),
    ("max_drop", "Change", "Largest single-cycle capacity decrease."),
    ("std_diff", "Change", "Standard deviation of first differences in Qdis."),
    ("skewness_Qdis", "Shape", "Skewness of the Qdis distribution in the window."),
    ("slope_ratio", "Trend", "Second-half OLS slope divided by first-half OLS slope."),
    ("Qdis_cycle10", "Level", "Discharge capacity measured at cycle 10."),
    ("mean_diff", "Change", "Mean first difference of Qdis."),
    ("poly2_a", "Shape", "Intercept a in the quadratic fit Q(c) = a + b*c + c2*c^2."),
    ("poly2_b", "Shape", "Linear coefficient b in the quadratic capacity fit."),
    ("poly2_c", "Shape", "Quadratic coefficient c2 in the quadratic capacity fit."),
    ("exp_decay_k", "Decay", "Fitted exponential fade-rate constant k in Q(c) = q_init*exp(-k*c)."),
    ("cycle_to_99pct", "Landmark", "First cycle below 99% of Q0; set to N if not crossed."),
    ("cycle_to_98pct", "Landmark", "First cycle below 98% of Q0; set to N if not crossed."),
    ("cycle_to_95pct", "Landmark", "First cycle below 95% of Q0; set to N if not crossed."),
    ("slope_first_quarter", "Trend", "OLS capacity slope in the first quarter of the window."),
    ("slope_last_quarter", "Trend", "OLS capacity slope in the last quarter of the window."),
    ("autocorr_lag1", "Dynamics", "Lag-1 autocorrelation of cycle-to-cycle capacity differences."),
    ("knee_cycle", "Landmark", "Cycle of maximum deviation from the line joining window endpoints."),
    ("n_capacity_jumps", "Dynamics", "Count of single-cycle drops larger than 1% of Q0."),
    ("accel_mean", "Acceleration", "Mean second difference of Qdis."),
    ("accel_std", "Acceleration", "Standard deviation of second differences of Qdis."),
    ("accel_max_abs", "Acceleration", "Maximum absolute second difference of Qdis."),
    ("linearity_r2", "Shape", "R^2 of a straight-line fit to Qdis over cycles 2 through N."),
    ("kurtosis_Qdis", "Shape", "Fisher kurtosis of the Qdis distribution."),
    ("fft_top3_energy_ratio", "Frequency", "Fraction of non-DC FFT energy in the three strongest bins."),
    ("spectral_entropy", "Frequency", "Shannon entropy of the normalized non-DC FFT power."),
    ("sample_entropy", "Complexity", "Sample entropy with m=2 and tolerance 0.2 times window SD."),
    ("pos_neg_diff_ratio", "Dynamics", "Positive first differences divided by negative first differences."),
    ("mad_Qdis", "Spread", "Median absolute deviation of Qdis."),
]


def register_fonts() -> tuple[str, str]:
    """Use DejaVu when available for reliable mathematical glyph rendering."""
    regular = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("ReportRegular", str(regular)))
        pdfmetrics.registerFont(TTFont("ReportBold", str(bold)))
        return "ReportRegular", "ReportBold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()
PAGE_SIZE = landscape(A4)
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE

NAVY = colors.HexColor("#16324F")
GREEN = colors.HexColor("#2E7D32")
TEAL = colors.HexColor("#177E89")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_GREEN = colors.HexColor("#EAF4EC")
PALE_GRAY = colors.HexColor("#F4F6F7")
MID_GRAY = colors.HexColor("#6B7280")
GRID = colors.HexColor("#C8D0D8")

BASE_STYLES = getSampleStyleSheet()
TITLE = ParagraphStyle(
    "ReportTitle",
    parent=BASE_STYLES["Title"],
    fontName=FONT_BOLD,
    fontSize=24,
    leading=29,
    textColor=NAVY,
    alignment=TA_LEFT,
    spaceAfter=10,
)
SUBTITLE = ParagraphStyle(
    "ReportSubtitle",
    parent=BASE_STYLES["Normal"],
    fontName=FONT,
    fontSize=11,
    leading=15,
    textColor=MID_GRAY,
    spaceAfter=14,
)
H1 = ParagraphStyle(
    "H1",
    parent=BASE_STYLES["Heading1"],
    fontName=FONT_BOLD,
    fontSize=16,
    leading=20,
    textColor=NAVY,
    spaceBefore=2,
    spaceAfter=8,
)
H2 = ParagraphStyle(
    "H2",
    parent=BASE_STYLES["Heading2"],
    fontName=FONT_BOLD,
    fontSize=11,
    leading=14,
    textColor=GREEN,
    spaceBefore=5,
    spaceAfter=5,
)
BODY = ParagraphStyle(
    "Body",
    parent=BASE_STYLES["BodyText"],
    fontName=FONT,
    fontSize=8.5,
    leading=12,
    textColor=colors.HexColor("#20262E"),
    spaceAfter=6,
)
SMALL = ParagraphStyle(
    "Small",
    parent=BODY,
    fontSize=7,
    leading=9,
    textColor=MID_GRAY,
)
TABLE_HEADER = ParagraphStyle(
    "TableHeader",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=7.2,
    leading=8.5,
    textColor=colors.white,
    alignment=TA_CENTER,
)
TABLE_CELL = ParagraphStyle(
    "TableCell",
    parent=BODY,
    fontSize=7.1,
    leading=8.5,
    alignment=TA_LEFT,
)
TABLE_CELL_CENTER = ParagraphStyle(
    "TableCellCenter",
    parent=TABLE_CELL,
    alignment=TA_CENTER,
)
FEATURE_NAME = ParagraphStyle(
    "FeatureName",
    parent=TABLE_CELL,
    fontName=FONT_BOLD,
    textColor=NAVY,
)


def require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing report inputs: {missing}")


def p(text: object, style: ParagraphStyle = TABLE_CELL) -> Paragraph:
    return Paragraph(str(text), style)


def dataset_label(value: str) -> str:
    return DATASET_LABELS.get(str(value).lower(), str(value))


def direction_label(experiment: str) -> str:
    source, target = str(experiment).split("_to_", maxsplit=1)
    return f"{dataset_label(source)} to {dataset_label(target)}"


def format_number(value: float, decimals: int = 1) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value:.2e}"
    if abs(value) >= 10_000:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def format_r2(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000:
        return f"{value:.2e}"
    return f"{value:.3f}"


def metric_with_sd(row: pd.Series, mean_col: str, std_col: str, *, r2: bool = False) -> str:
    formatter = format_r2 if r2 else format_number
    mean = formatter(row[mean_col])
    if std_col not in row.index or pd.isna(row[std_col]):
        return mean
    return f"{mean} +/- {formatter(row[std_col])}"


def metric_table(df: pd.DataFrame, *, include_window: bool = True) -> Table:
    headers = ["Model"]
    widths = [49 * mm]
    if include_window:
        headers.append("N")
        widths.append(13 * mm)
    headers.extend(["MAE, cycles (mean +/- SD)", "sMAPE, % (mean +/- SD)", "R^2 (mean +/- SD)"])
    widths.extend([48 * mm, 48 * mm, 48 * mm])

    rows: list[list[Paragraph]] = [[p(value, TABLE_HEADER) for value in headers]]
    sort_cols = ["n_cycles", "model"] if include_window else ["model"]
    for _, row in df.sort_values(sort_cols).iterrows():
        values: list[Paragraph] = [p(MODEL_LABELS.get(row["model"], row["model"]), TABLE_CELL)]
        if include_window:
            values.append(p(int(row["n_cycles"]), TABLE_CELL_CENTER))
        values.extend(
            [
                p(metric_with_sd(row, "MAE_mean", "MAE_std"), TABLE_CELL_CENTER),
                p(metric_with_sd(row, "SMAPE_mean", "SMAPE_std"), TABLE_CELL_CENTER),
                p(metric_with_sd(row, "R2_mean", "R2_std", r2=True), TABLE_CELL_CENTER),
            ]
        )
        rows.append(values)

    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ]
        )
    )
    for idx in range(1, len(rows)):
        if idx % 2 == 0:
            table.setStyle(TableStyle([("BACKGROUND", (0, idx), (-1, idx), PALE_GRAY)]))
    return table


def champion_table(df: pd.DataFrame, group_col: str) -> Table:
    n100 = df[df["n_cycles"] == 100].copy()
    idx = n100.groupby(group_col)["R2_mean"].idxmax()
    best = n100.loc[idx].sort_values(group_col)
    rows = [
        [
            p("Dataset / direction", TABLE_HEADER),
            p("Best model by R^2", TABLE_HEADER),
            p("MAE, cycles", TABLE_HEADER),
            p("sMAPE, %", TABLE_HEADER),
            p("R^2", TABLE_HEADER),
        ]
    ]
    for _, row in best.iterrows():
        label = (
            dataset_label(row[group_col])
            if group_col == "dataset"
            else direction_label(row[group_col])
        )
        rows.append(
            [
                p(label),
                p(MODEL_LABELS.get(row["model"], row["model"])),
                p(format_number(row["MAE_mean"]), TABLE_CELL_CENTER),
                p(format_number(row["SMAPE_mean"]), TABLE_CELL_CENTER),
                p(format_r2(row["R2_mean"]), TABLE_CELL_CENTER),
            ]
        )
    table = Table(rows, colWidths=[48 * mm, 49 * mm, 35 * mm, 35 * mm, 31 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    for idx in range(1, len(rows)):
        if idx % 2 == 0:
            table.setStyle(TableStyle([("BACKGROUND", (0, idx), (-1, idx), PALE_BLUE)]))
    return table


def feature_table() -> Table:
    rows = [
        [
            p("No.", TABLE_HEADER),
            p("Feature", TABLE_HEADER),
            p("Group", TABLE_HEADER),
            p("Definition", TABLE_HEADER),
        ]
    ]
    for number, (name, group, definition) in enumerate(FEATURE_DEFINITIONS, start=1):
        rows.append(
            [
                p(number, TABLE_CELL_CENTER),
                p(name, FEATURE_NAME),
                p(group, TABLE_CELL_CENTER),
                p(definition),
            ]
        )
    table = Table(
        rows,
        colWidths=[12 * mm, 46 * mm, 28 * mm, 170 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), GREEN),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ]
        )
    )
    for idx in range(1, len(rows)):
        if idx % 2 == 0:
            table.setStyle(TableStyle([("BACKGROUND", (0, idx), (-1, idx), PALE_GREEN)]))
    return table


def deep_comparison_table(
    frames: dict[str, pd.DataFrame],
    scenario: str,
    experiments: set[str] | None = None,
) -> Table:
    combined = []
    for backbone, frame in frames.items():
        subset = frame[frame["scenario"] == scenario].copy()
        if experiments is not None:
            subset = subset[subset["experiment"].isin(experiments)]
        subset["backbone"] = backbone
        combined.append(subset)
    df = pd.concat(combined, ignore_index=True)

    rows = [
        [
            p("Dataset / direction", TABLE_HEADER),
            p("Backbone", TABLE_HEADER),
            p("MAE, cycles (mean +/- SD)", TABLE_HEADER),
            p("sMAPE, % (mean +/- SD)", TABLE_HEADER),
            p("R^2 (mean +/- SD)", TABLE_HEADER),
        ]
    ]
    for experiment, group in df.groupby("experiment", sort=True):
        for _, row in group.sort_values("backbone").iterrows():
            label = (
                dataset_label(row["target"])
                if scenario == "within_split"
                else direction_label(experiment)
            )
            rows.append(
                [
                    p(label),
                    p(row["backbone"]),
                    p(metric_with_sd(row, "MAE_mean", "MAE_std"), TABLE_CELL_CENTER),
                    p(metric_with_sd(row, "SMAPE_mean", "SMAPE_std"), TABLE_CELL_CENTER),
                    p(metric_with_sd(row, "R2_mean", "R2_std", r2=True), TABLE_CELL_CENTER),
                ]
            )
    table = Table(
        rows,
        colWidths=[50 * mm, 37 * mm, 48 * mm, 48 * mm, 48 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ]
        )
    )
    for idx in range(1, len(rows)):
        if idx % 2 == 0:
            table.setStyle(TableStyle([("BACKGROUND", (0, idx), (-1, idx), PALE_GRAY)]))
    return table


def select_k20_conservative_protocol(target_rescale: pd.DataFrame) -> pd.DataFrame:
    """Fix the naive-best model per direction, then retain both k=20 adapters."""
    k20 = target_rescale[target_rescale["k"].eq(20)].copy()
    best_indices = k20.groupby("experiment")["baseline_R2"].idxmax()
    fixed_models = k20.loc[best_indices, ["experiment", "model"]].drop_duplicates()
    selected = k20.merge(fixed_models, on=["experiment", "model"], how="inner")
    adapter_order = pd.CategoricalDtype(["residual_mean", "linear"], ordered=True)
    selected["adapter_type"] = selected["adapter_type"].astype(adapter_order)
    return selected.sort_values(["experiment", "adapter_type"]).reset_index(drop=True)


def k20_calibration_table(df: pd.DataFrame) -> Table:
    rows = [
        [
            p("Direction", TABLE_HEADER),
            p("Fixed model", TABLE_HEADER),
            p("Adapter", TABLE_HEADER),
            p("Naive MAE", TABLE_HEADER),
            p("Naive sMAPE", TABLE_HEADER),
            p("Naive R^2", TABLE_HEADER),
            p("k=20 MAE", TABLE_HEADER),
            p("k=20 sMAPE", TABLE_HEADER),
            p("k=20 R^2", TABLE_HEADER),
        ]
    ]
    adapter_labels = {"residual_mean": "Residual mean", "linear": "Linear"}
    for _, row in df.iterrows():
        rows.append(
            [
                p(direction_label(row["experiment"])),
                p(MODEL_LABELS.get(row["model"], row["model"])),
                p(adapter_labels[str(row["adapter_type"])]),
                p(format_number(row["baseline_MAE"]), TABLE_CELL_CENTER),
                p(format_number(row["baseline_SMAPE"]), TABLE_CELL_CENTER),
                p(format_r2(row["baseline_R2"]), TABLE_CELL_CENTER),
                p(format_number(row["adapted_MAE"]), TABLE_CELL_CENTER),
                p(format_number(row["adapted_SMAPE"]), TABLE_CELL_CENTER),
                p(format_r2(row["adapted_R2"]), TABLE_CELL_CENTER),
            ]
        )
    table = Table(
        rows,
        colWidths=[
            31 * mm,
            31 * mm,
            25 * mm,
            27 * mm,
            27 * mm,
            25 * mm,
            27 * mm,
            27 * mm,
            25 * mm,
        ],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    for idx in range(1, len(rows)):
        if idx % 2 == 0:
            table.setStyle(TableStyle([("BACKGROUND", (0, idx), (-1, idx), PALE_BLUE)]))
    return table


def add_grouped_metric_pages(
    story: list,
    df: pd.DataFrame,
    *,
    group_col: str,
    section_title: str,
    intro: str,
    include_window: bool,
) -> None:
    story.extend([Paragraph(section_title, H1), Paragraph(intro, BODY), Spacer(1, 2 * mm)])
    groups = list(df.groupby(group_col, sort=True))
    for group_index, (group_value, group) in enumerate(groups):
        label = (
            dataset_label(group_value)
            if group_col == "dataset"
            else direction_label(group_value)
        )
        story.append(KeepTogether([Paragraph(label, H2), metric_table(group, include_window=include_window)]))
        if group_index < len(groups) - 1:
            story.append(PageBreak())


def validate_inputs(
    within: pd.DataFrame,
    cross: pd.DataFrame,
    umich_within: pd.DataFrame,
    umich_cross: pd.DataFrame,
    deep: dict[str, pd.DataFrame],
    target_rescale: pd.DataFrame,
) -> None:
    feature_columns = [
        column
        for column in pd.read_csv(FEATURE_PATH, nrows=1).columns
        if column
        not in {
            "dataset",
            "cell_id",
            "n_cycles",
            "q0",
            "cycle_life",
            "is_censored",
            "capacity_normalized",
        }
    ]
    expected_features = [name for name, _, _ in FEATURE_DEFINITIONS]
    if feature_columns != expected_features:
        raise ValueError("Feature-definition order does not match the committed 34-feature CSV.")
    checks = [
        ("classical within", within, 56),
        ("classical cross", cross, 168),
        ("UMich within", umich_within, 7),
        ("UMich cross", umich_cross, 56),
    ]
    for label, frame, expected_rows in checks:
        if len(frame) != expected_rows:
            raise ValueError(f"{label}: expected {expected_rows} rows, found {len(frame)}")
        if frame[["MAE_mean", "SMAPE_mean", "R2_mean"]].isna().any().any():
            raise ValueError(f"{label}: missing required metric values")
    for label, frame in deep.items():
        if len(frame) != 16:
            raise ValueError(f"{label}: expected 16 rows, found {len(frame)}")
    selected_k20 = select_k20_conservative_protocol(target_rescale)
    if len(selected_k20) != 24:
        raise ValueError(f"k=20 calibration: expected 24 rows, found {len(selected_k20)}")
    if selected_k20[["adapted_MAE", "adapted_SMAPE", "adapted_R2"]].isna().any().any():
        raise ValueError("k=20 calibration: missing required adapted metric values")


def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(15 * mm, PAGE_HEIGHT - 12 * mm, PAGE_WIDTH - 15 * mm, PAGE_HEIGHT - 12 * mm)
    canvas.setFont(FONT, 6.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(15 * mm, PAGE_HEIGHT - 9 * mm, "Battery RUL study - feature and metric report")
    canvas.drawRightString(PAGE_WIDTH - 15 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_report() -> Path:
    required = [
        WITHIN_PATH,
        CROSS_PATH,
        UMICH_WITHIN_PATH,
        FEATURE_PATH,
        TARGET_RESCALE_PATH,
        *DEEP_PATHS.values(),
    ]
    required.extend(sorted(UMICH_CROSS_ROOT.glob("pair_*/results_summary.csv")))
    require_paths(required)

    within = pd.read_csv(WITHIN_PATH)
    cross = pd.read_csv(CROSS_PATH)
    umich_within = pd.read_csv(UMICH_WITHIN_PATH)
    umich_cross = pd.concat(
        [pd.read_csv(path) for path in sorted(UMICH_CROSS_ROOT.glob("pair_*/results_summary.csv"))],
        ignore_index=True,
    )
    deep = {label: pd.read_csv(path) for label, path in DEEP_PATHS.items()}
    target_rescale = pd.read_csv(TARGET_RESCALE_PATH)
    validate_inputs(within, cross, umich_within, umich_cross, deep, target_rescale)
    selected_k20 = select_k20_conservative_protocol(target_rescale)
    k20_experiments = sorted(selected_k20["experiment"].unique())
    k20_first = selected_k20[selected_k20["experiment"].isin(k20_experiments[:6])]
    k20_second = selected_k20[selected_k20["experiment"].isin(k20_experiments[6:])]
    deep_cross_experiments = sorted(
        deep["1D-CNN"].loc[
            deep["1D-CNN"]["scenario"] == "naive_cross", "experiment"
        ].unique()
    )
    deep_cross_first = set(deep_cross_experiments[:6])
    deep_cross_second = set(deep_cross_experiments[6:])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=PAGE_SIZE,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=17 * mm,
        bottomMargin=14 * mm,
        title="Battery RUL Study: 34 Features and Complete Prediction Metrics",
        author="Graduation-Project-Dicle",
        subject="Within-dataset and cross-dataset MAE, sMAPE, and R^2 results",
    )

    story: list = [
        Spacer(1, 16 * mm),
        Paragraph("Battery RUL Study", TITLE),
        Paragraph("34 engineered features and complete within-/cross-dataset prediction metrics", TITLE),
        Paragraph(
            "Finalized classical benchmarks, held-out UMich external validation, and deep sequence-backbone sensitivity results.",
            SUBTITLE,
        ),
        Spacer(1, 6 * mm),
        Paragraph("Scope", H1),
        Paragraph(
            "This report records MAE, sMAPE, and R^2 for every committed direct within-dataset and naive "
            "cross-dataset prediction result used in the study. Classical models use the 34 engineered "
            "capacity-trajectory features. The 1D-CNN, Transformer, and Mamba sensitivity models instead "
            "consume normalized early-cycle trajectories and are reported separately.",
            BODY,
        ),
        Paragraph(
            "Classical MATR/HUST/Sandia/Luh tables include N=50 and N=100 windows. UMich external validation "
            "and the deep backbones use N=100. Metric entries are five-seed mean +/- standard deviation. "
            "MAE is in cycles; sMAPE is in percent. Conformal-prediction, k-shot adaptation, LODO, and survival "
            "analyses are outside this direct-prediction metric report.",
            BODY,
        ),
        Spacer(1, 4 * mm),
        Paragraph("Contents", H1),
        Paragraph(
            "1. Feature contract<br/>"
            "2. N=100 classical champion overview<br/>"
            "3. Full classical within-dataset metrics<br/>"
            "4. Full classical naive cross-dataset metrics<br/>"
            "5. UMich held-out external validation<br/>"
            "6. Deep trajectory-backbone sensitivity<br/>"
            "7. k=20 target-side point calibration<br/>"
            "8. Data provenance",
            BODY,
        ),
        PageBreak(),
        Paragraph("1. The 34 engineered features", H1),
        Paragraph(
            "All features are computed from discharge-capacity observations Qdis over cycles 2 through N. "
            "Q0 is the median discharge capacity over cycles 2-5. For cross-dataset capacity normalization, "
            "features with capacity units are divided by Q0 and variance_Qdis is divided by Q0^2; inherently "
            "dimensionless and cycle-index features are unchanged.",
            BODY,
        ),
        feature_table(),
        PageBreak(),
        Paragraph("2. N=100 classical champion overview", H1),
        Paragraph(
            "The tables below select the model with the highest mean R^2 within each dataset or direction. "
            "They are navigation summaries only; the following sections retain every model result, including "
            "catastrophic negative-R^2 failure cases.",
            BODY,
        ),
        Paragraph("Within-dataset champions", H2),
        champion_table(within, "dataset"),
        Spacer(1, 7 * mm),
        Paragraph("Naive cross-dataset champions", H2),
        champion_table(cross, "experiment"),
        PageBreak(),
    ]

    add_grouped_metric_pages(
        story,
        within,
        group_col="dataset",
        section_title="3. Full classical within-dataset metrics",
        intro=(
            "Each model is trained and evaluated within the same dataset under the finalized five-seed protocol. "
            "Both N=50 and N=100 early-cycle windows are shown."
        ),
        include_window=True,
    )
    story.append(PageBreak())
    add_grouped_metric_pages(
        story,
        cross,
        group_col="experiment",
        section_title="4. Full classical naive cross-dataset metrics",
        intro=(
            "Models are trained on the source dataset and evaluated directly on the full target dataset without "
            "target-label adaptation. Capacity-normalized 34-feature inputs are used. Large negative R^2 values "
            "are retained because they quantify genuine transfer failure."
        ),
        include_window=True,
    )

    story.extend(
        [
            PageBreak(),
            Paragraph("5. UMich held-out external validation", H1),
            Paragraph(
                "UMich is kept separate from the four-dataset development benchmark. It provides a fifth held-out "
                "cohort at N=100. The same 34-feature contract and classical model family are used.",
                BODY,
            ),
            Paragraph("UMich within-dataset results", H2),
            metric_table(umich_within, include_window=False),
            PageBreak(),
        ]
    )
    add_grouped_metric_pages(
        story,
        umich_cross,
        group_col="experiment",
        section_title="UMich naive cross-dataset directions",
        intro=(
            "These eight directions pair UMich with each development dataset. Values remain external-validation "
            "results rather than additions to the original 12-direction development benchmark."
        ),
        include_window=False,
    )

    story.extend(
        [
            PageBreak(),
            Paragraph("6. Deep trajectory-backbone sensitivity", H1),
            Paragraph(
                "These models do not consume the 34 engineered features. They use normalized early-cycle Q/Q0 "
                "trajectory channels under N=100 and are included to show that the observed transfer regimes are "
                "not restricted to the classical feature-based models.",
                BODY,
            ),
            Paragraph("Within-dataset deep results", H2),
            deep_comparison_table(deep, "within_split"),
            PageBreak(),
            Paragraph("Naive cross-dataset deep results", H2),
            deep_comparison_table(deep, "naive_cross", deep_cross_first),
            PageBreak(),
            Paragraph("Naive cross-dataset deep results (continued)", H2),
            deep_comparison_table(deep, "naive_cross", deep_cross_second),
            PageBreak(),
            Paragraph("7. k=20 target-side point calibration", H1),
            Paragraph(
                "These tables append the finalized four-dataset k-shot point-prediction results. For each "
                "direction, the model with the highest naive cross-dataset R^2 is fixed before target labels are "
                "used. Residual-mean and linear adapters are then fitted using k=20 labeled target cells. Values "
                "are means across the repeated calibration protocol. This is the conservative audit protocol, "
                "not the optimistic best-adapter/best-model upper envelope.",
                BODY,
            ),
            k20_calibration_table(k20_first),
            PageBreak(),
            Paragraph("k=20 target-side point calibration (continued)", H2),
            k20_calibration_table(k20_second),
            PageBreak(),
            Paragraph("8. Data provenance", H1),
            Paragraph(
                "This PDF is generated directly from committed CSV outputs. No metric was transcribed manually.",
                BODY,
            ),
        ]
    )
    source_rows = [
        [p("Content", TABLE_HEADER), p("Repository source", TABLE_HEADER)],
        [p("34-feature order"), p(FEATURE_PATH.relative_to(ROOT))],
        [p("Classical within"), p(WITHIN_PATH.relative_to(ROOT))],
        [p("Classical cross"), p(CROSS_PATH.relative_to(ROOT))],
        [p("UMich within"), p(UMICH_WITHIN_PATH.relative_to(ROOT))],
        [p("UMich cross"), p(UMICH_CROSS_ROOT.relative_to(ROOT))],
        [p("1D-CNN"), p(DEEP_PATHS["1D-CNN"].relative_to(ROOT))],
        [p("Transformer"), p(DEEP_PATHS["Transformer"].relative_to(ROOT))],
        [p("Mamba"), p(DEEP_PATHS["Mamba"].relative_to(ROOT))],
        [p("k=20 target calibration"), p(TARGET_RESCALE_PATH.relative_to(ROOT))],
    ]
    sources = Table(source_rows, colWidths=[54 * mm, 180 * mm], repeatRows=1)
    sources.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend(
        [
            sources,
            Spacer(1, 6 * mm),
            Paragraph(
                "Interpretation note: negative R^2 means the transferred model performs worse than predicting the "
                "target mean. sMAPE is symmetric mean absolute percentage error and is reported as a percentage.",
                SMALL,
            ),
        ]
    )

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_report()
    print(f"[saved] {path}")
