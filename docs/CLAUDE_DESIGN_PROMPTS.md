# Claude Design — prompt library

Hand-tunable prompts for figures we want Claude Design (or any image-
generation / vector-design tool) to produce. Each prompt is self-contained:
copy-paste into Claude Design, attach the project palette swatches as
reference if the UI supports it.

The visual identity throughout the manuscript matches the thesis-defense
deck (`presentation/battery_rul_defense.pptx`): **Midnight Executive**
palette, Cambria headers, Calibri body, navy primary with a single gold
accent.

---

## F1 — Pipeline schematic (manuscript Figure 1, also slide 6 in the deck)

**Goal.** A clean, paper-quality block diagram showing the analysis
pipeline. Conveys three things at a glance:

1. **Single feature contract** is built once from four raw datasets.
2. **Splits + models + shift diagnostics** run on that contract.
3. **k-shot calibration + conformal prediction** is the deployment-time
   layer that closes the loop.

**Final size.** Designed for a half-page figure in a single-column journal
or 2/3-page in a double-column journal. Aspect ratio ~16:9 (works for
manuscript figure or PowerPoint slide).

### Prompt for Claude Design (copy-paste)

```
Create a clean, academic block diagram titled "Pipeline overview" for a
machine-learning research paper on cross-dataset battery lifetime
prediction. Aspect ratio 16:9. The diagram has six labeled blocks
arranged in two rows of three, connected by arrows from left to right.
A small annotation box at the right margin shows the "outputs" of the
pipeline.

Color palette (Midnight Executive):
- Primary navy: #1E2761 (block borders, text, arrows)
- Ice blue: #CADCFC (fill for the "data" stages, rows 1-3)
- Gold accent: #F5B700 (fill for the "diagnosis + deployment" stages,
  rows 4-6)
- Text dark: #1A1A2E (block titles)
- Text muted: #5A5C7A (block subtitles)
- Background: pure white #FFFFFF

Typography:
- Block titles: Cambria, bold, ~14 pt
- Block subtitles / notes: Calibri, regular, ~10 pt
- Diagram title: Cambria, bold, 18 pt, centered above the blocks

Blocks (top row, left to right; ice-blue fill, navy border):
1. "Raw cell data"
   subtitle: "MATR · HUST · Sandia · Luh-KIT (.pkl + .csv)"
2. "Audit + features"
   subtitle: "Q0 = median(QD₂…QD₅); EOL when Q_dis ≤ 0.85·Q0; 34
   capacity-only features"
3. "Splits (70/15/15)"
   subtitle: "5 seeds · cell-level · lifetime-stratified"

Blocks (bottom row, left to right; gold fill, navy border):
4. "Models"
   subtitle: "7 classical (EN · PLS · RF · XGB · CatBoost · GP ·
   Stacking) + PyTorch 1D-CNN"
5. "Shift + regime"
   subtitle: "MMD · Mahalanobis · conditional-shift slopes · rank-signal
   classifier"
6. "k-shot + Conformal Prediction"
   subtitle: "Adapter (residual / linear) → MAPIE split CP at 90% / 95%"

Arrows:
- Block 1 → Block 2 → Block 3 (top row), straight, navy, single-headed
- Block 3 → Block 4 (downward L-turn, navy)
- Block 4 → Block 5 → Block 6 (bottom row), straight, navy
- The arrow between block 5 and block 6 is slightly thicker / gold-tinted
  to suggest "this is where the deployment-time work happens"

Side note (right of the diagram, vertically centered):
"Outputs:
 • Within / cross / shift tables and figures
 • k-shot scaling curves
 • Coverage-valid CP intervals
 • SHAP × regime joined table"
This side note is a single muted-grey text block — no border — that
visually anchors the right side of the figure.

Optional bottom-right corner caption (very small):
"Companion code: github.com/osmansafacifci/Graduation-Project-Dicle"

Constraints:
- No 3D effects, no shadows beyond a very subtle drop shadow on each
  block (optional). Strictly flat academic diagram.
- All blocks the same size (~2" wide, 1" tall in print).
- Equal spacing between blocks; arrows centered between block edges.
- Text inside blocks is left-aligned, with 0.15" padding from the box
  edge.
- The figure must read well at quarter-page size in a journal (~3.5"
  wide) and at full slide size for a 16:9 deck.
```

**Notes for iteration:**
- If Claude Design renders boxes with rounded corners, ask for a tighter
  corner radius (≤ 6 pt) or no rounding at all — academic figures
  typically use square corners.
- If the gold blocks look too saturated, ask for "muted gold #DCAA45 fill
  with 30% white tint."
- After generation, ask Claude Design to export at 300 dpi PNG + a vector
  PDF/SVG so the figure stays sharp in the manuscript and in the
  presentation deck.

---

## F2 — Within-dataset predictions vs ground truth (4 panels)

**Goal.** A 4-panel scatter plot: predicted vs actual cycle life for the
best within-dataset model on each of MATR / HUST / Sandia / Luh, with
identity reference line and bootstrap CI band.

This figure can be built directly from `outputs/results_v2_four_dataset_within_34feat_log/results_predictions.csv` or from the matching CSV in the PyTorch CNN output. It does not need Claude Design — a small matplotlib script suffices. The Claude Design prompt here is included only for completeness in case a vector-style redesign is wanted later.

### Prompt for Claude Design (alternative path)

```
Create a 2x2 panel figure for an academic ML paper. Each panel is a
scatter plot showing predicted vs actual lithium-ion battery cycle life
for one of four datasets, plus an identity y=x reference line.

Panels (top-left, top-right, bottom-left, bottom-right):
1. MATR  — R² = 0.575  (best model: CatBoost)
2. HUST  — R² = 0.340  (best model: Random Forest)
3. Sandia — R² = 0.940 (best model: XGBoost)
4. Luh-KIT — R² = 0.769 (best model: Gaussian Process)

Style:
- Same Midnight Executive palette as F1 (navy + ice blue + gold)
- Markers: filled navy dots, size ~30
- Identity line: dashed muted grey
- Each panel labeled with dataset name and best-model R² in the
  top-left corner, Cambria bold
- Square axes, equal x/y scales per panel
- Grid: light grey, thin

Output: 300 dpi PNG + vector PDF, sized to fit a half-page figure in a
single-column journal.
```

**Recommended path:** build with matplotlib directly using the predictions
CSV; reproduce the deck's slide-7 style.

---

## F5 (paper-defining) — Four-dataset conditional-shift heatmap with regime labels

**Status.** The current artefact at
`outputs/results_v2_four_dataset_conditional_shift/four_dataset_conditional_shift_heatmaps.png`
is paper-quality and was generated by `conditional_shift_four_dataset.py`.
Suggested in-paper caption is in `docs/PAPER_OUTLINE.md` §6.4.

**Only refinement Claude Design might help with:** an *overlay banner* on
the existing heatmap that calls out the three regimes in plain language
("Strong rank signal pairs" / "Offset-dominant pairs" / "Rank-collapsed
pairs") with the same green/orange/red colour code used in F4 and the
slide deck. If that overlay is wanted, ask Claude Design to:

```
Take this existing heatmap and add a vertical legend strip along the
right edge with three coloured chips:
- Green (#1B5E20): "Strong rank signal — salvageable transfer"
- Orange (#B26500): "Offset-dominant — calibratable"
- Red (#B00020): "Rank-collapsed — no fix"
Strip width ~0.6" in print; chip squares ~0.15"; labels in Calibri 10pt
left-aligned next to each chip. Do not modify the heatmap itself.
```

---

## Graphical abstract (optional, journal-style)

**Goal.** A square-ish 5:4 figure for the journal's graphical-abstract
slot. Captures the paper in one image: source dataset → diagnostic →
regime label → k-shot calibration → coverage-valid intervals.

### Prompt for Claude Design

```
Create a graphical abstract for a research paper on cross-dataset
lithium-ion battery RUL prediction. Aspect ratio 5:4, target use in a
journal's "Graphical Abstract" slot. Single image, no panels.

The image is a horizontal flow with four numbered stages connected by
chevron arrows:

[1] Two source datasets (icons of two batteries with arrows pointing
into one stack of cells) labeled "Source training data"

[2] A diagnostic block labeled "Diagnose:
       MMD · Mahalanobis · centred-log slope · Pearson r"
    that emits three coloured boxes labeled
       "Strong rank" (green) / "Offset" (orange) / "Collapsed" (red)

[3] A small target dataset of ~20 cells (mini grid of 20 cell icons)
    labeled "k = 20 target cells"

[4] An output block showing a horizontal line with a coloured band
    around it (prediction + 90% CP interval), labeled "Valid CP
    intervals" — coverage 0.91 written in small text

Colours: same Midnight Executive palette as the rest of the manuscript
(navy + ice blue + gold). Text in Cambria headings and Calibri body.

At the bottom, a single bold caption (Cambria 14pt, navy):
"Cross-dataset RUL transfer: diagnose, calibrate, predict with valid
uncertainty."

Make the image crisp at 300 dpi PNG and provide a vector PDF / SVG.
Do not use any 3D effects, gradients, or photographic textures. Pure
flat technical illustration style.
```

---

## Tips for working with Claude Design

1. **Iterate on the colour palette first.** Get a single block right with
   the right navy / ice blue / gold balance; replicate from there.
2. **Request multiple resolutions on export.** 300 dpi PNG for the manuscript,
   vector PDF/SVG for the deck/poster, and a 72 dpi PNG for previews and
   web summaries (e.g., the project README).
3. **Keep figure widths consistent.** A double-column journal expects
   single-column figures at 3.5" wide or full-width figures at 7.2" wide.
   Single-column journals expect 6" - 7" wide.
4. **Avoid AI-cliché signatures.** No accent-lines under titles, no
   neon-purple gradients, no fake bokeh. Academic technical illustration
   is conservative — keep it conservative.
5. **After export, run the figure through the visual-QA loop:** open the
   PNG at 100% zoom, check label collisions and clipped descenders.
