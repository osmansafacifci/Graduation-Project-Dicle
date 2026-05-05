from pathlib import Path

from PIL import Image

plots_dir = Path("plots")
models = ["randomforest", "xgboost", "catboost"]
# Load images stored one level above this script (plots/ at project root)
rows = []

for n_cycles in (25, 50, 100):
    images = []
    for model in models:
        path = Path(__file__).resolve().parents[1] / plots_dir / f"feature_importance_{model}_{n_cycles}.png"
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        images.append(Image.open(path))

    widths, heights = zip(*(img.size for img in images))
    combined = Image.new("RGB", (sum(widths), max(heights)), "white")
    x_off = 0
    for img in images:
        combined.paste(img, (x_off, 0))
        x_off += img.width
    rows.append(combined)

final_width = max(img.width for img in rows)
final_height = sum(img.height for img in rows)
merged = Image.new("RGB", (final_width, final_height), "white")
y_off = 0
for img in rows:
    merged.paste(img, (0, y_off))
    y_off += img.height

output = plots_dir / "feature_importance_combined.png"
output.parent.mkdir(parents=True, exist_ok=True)
merged.save(output, dpi=(200, 200))
print(f"Saved {output}")
