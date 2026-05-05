from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

FEATURES = [
    "IR_delta",
    "dQd_slope",
    "Qd_mean",
    "IR_slope",
    "Tavg_mean",
    "IR_mean",
    "Qd_std",
    "IR_std",
]
cycles = ["25 cycles", "50 cycles", "100 cycles"]
data = [["✓" for _ in cycles] for _ in FEATURES]

fig, ax = plt.subplots(figsize=(6, len(FEATURES) * 0.45 + 1.5))
ax.axis("off")
table_data = [["Feature"] + cycles] + [[feat] + row for feat, row in zip(FEATURES, data)]
table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.35)

header_color = "#111d34"
row_colors = ["#f2f4f7", "white"]
text_color = "#111d34"

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#d7d9e1")
    if row == 0:
        cell.set_facecolor(header_color)
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
        cell.set_height(cell.get_height() * 2.0)
    else:
        cell.set_facecolor(row_colors[(row - 1) % 2])
        cell.get_text().set_color(text_color)
        cell.set_height(cell.get_height() * 1.2)

ax.set_title("Eight-feature setup vs n_cycles", fontsize=14, pad=20, color=text_color)
fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
plots_dir = Path(__file__).resolve().parent.parent / "plots"
plots_dir.mkdir(exist_ok=True)
fig.savefig(plots_dir / "table_features_cycles.png", dpi=250, bbox_inches="tight")
plt.close(fig)
