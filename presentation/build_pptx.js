// Battery RUL — thesis-defense style PowerPoint
// Authors: Durukan Demir, Dicle Çoban, Salih Sarp, Osman Safa Çifçi

const pptxgen = require("pptxgenjs");
const path = require("path");

const PROJECT_ROOT = path.resolve(__dirname, "..");

// Color palette — Midnight Executive
const NAVY = "1E2761";
const ICE_BLUE = "CADCFC";
const WHITE = "FFFFFF";
const ACCENT_GOLD = "F5B700";
const TEXT_DARK = "1A1A2E";
const TEXT_MUTED = "5A5C7A";
const RULE = "D8DCEB";

const HEADER_FONT = "Cambria";
const BODY_FONT = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Demir, Çoban, Sarp, Çifçi";
pres.title = "Battery RUL — Cross-Dataset Transfer and Conformal Uncertainty";

// ============== HELPER: dark-themed title slide ==============
function addDarkTitleSlide(s, { title, subtitle, footer }) {
    s.background = { color: NAVY };
    // Accent bar
    s.addShape(pres.shapes.RECTANGLE, {
        x: 0.6, y: 1.7, w: 0.08, h: 2.2, fill: { color: ACCENT_GOLD }, line: { color: ACCENT_GOLD }
    });
    s.addText(title, {
        x: 0.9, y: 1.5, w: 8.5, h: 1.6, fontSize: 32, fontFace: HEADER_FONT,
        color: WHITE, bold: true, valign: "top", margin: 0
    });
    if (subtitle) {
        s.addText(subtitle, {
            x: 0.9, y: 3.0, w: 8.5, h: 1.2, fontSize: 18, fontFace: BODY_FONT,
            color: ICE_BLUE, italic: true, valign: "top", margin: 0
        });
    }
    if (footer) {
        s.addText(footer, {
            x: 0.5, y: 5.1, w: 9, h: 0.4, fontSize: 11, fontFace: BODY_FONT,
            color: ICE_BLUE, align: "center"
        });
    }
}

// ============== HELPER: content slide chrome (title + accent + page) ==============
function addContentChrome(s, slideNum, totalSlides, title) {
    s.background = { color: WHITE };
    // Left accent bar
    s.addShape(pres.shapes.RECTANGLE, {
        x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: NAVY }, line: { color: NAVY }
    });
    // Title
    s.addText(title, {
        x: 0.45, y: 0.25, w: 9.2, h: 0.6,
        fontSize: 22, fontFace: HEADER_FONT, bold: true, color: NAVY, margin: 0
    });
    // Thin rule under title
    s.addShape(pres.shapes.LINE, {
        x: 0.45, y: 0.85, w: 9.2, h: 0,
        line: { color: RULE, width: 0.8 }
    });
    // Page number
    s.addText(`${slideNum} / ${totalSlides}`, {
        x: 8.7, y: 5.35, w: 1.2, h: 0.3,
        fontSize: 9, fontFace: BODY_FONT, color: TEXT_MUTED, align: "right"
    });
}

// ============== HELPER: bullets formatter ==============
function bullets(items, options) {
    const arr = items.map((t, i) => ({
        text: t,
        options: { bullet: { code: "25A0" }, breakLine: i < items.length - 1, color: TEXT_DARK }
    }));
    return Object.assign({
        fontSize: 14, fontFace: BODY_FONT, color: TEXT_DARK, valign: "top",
        paraSpaceAfter: 6
    }, options || {});
}
function bulletsContent(items) {
    return items.map((t, i) => ({
        text: t,
        options: { bullet: { code: "25A0" }, breakLine: i < items.length - 1 }
    }));
}

// ============== HELPER: image with preserved aspect ratio ==============
function fitImage({ width, height }, maxW, maxH) {
    const aspectImg = height / width;
    const aspectBox = maxH / maxW;
    if (aspectImg > aspectBox) {
        const h = maxH;
        const w = h / aspectImg;
        return { w, h };
    } else {
        const w = maxW;
        const h = w * aspectImg;
        return { w, h };
    }
}

// =====================================================================
// All slides will be added below. We compute totalSlides after the fact
// by tracking the count, then patching the page numbers in second pass
// using slide.addText again? Simpler: count first via array.

const slideRecords = [];

function addSlide(builder) {
    const s = pres.addSlide();
    slideRecords.push({ slide: s, build: builder });
}

// Use a 2-pass approach by collecting first, then rendering with known total.
// But pptxgenjs adds slides immediately. Simplest: hardcode totalSlides.
const TOTAL_SLIDES = 26;
let slideIdx = 0;
function next() { slideIdx += 1; return slideIdx; }

// =====================================================================
// SLIDE 1 — Title
// =====================================================================
{
    const s = pres.addSlide();
    next();
    addDarkTitleSlide(s, {
        title: "When Cross-Dataset Battery RUL Transfer Fails — and How to Repair It with Valid Uncertainty",
        subtitle: "A capacity-only, four-dataset benchmark with rank-signal regime taxonomy and split conformal prediction",
        footer: "Durukan Demir · Dicle Çoban · Salih Sarp · Osman Safa Çifçi   |   2026"
    });
    s.addNotes(
        "Welcome. This work proposes a sistematic cross-dataset transfer protocol " +
        "for early-cycle battery RUL prediction with valid uncertainty intervals."
    );
}

// =====================================================================
// SLIDE 2 — Motivation
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Motivation — why cross-dataset transfer matters");
    // Left: text
    s.addText(bulletsContent([
        "Battery management systems are trained on lab data and deployed on field data with different chemistry, protocol, format.",
        "Within-dataset accuracy is now a solved problem (Severson 2019, BatLiNet 2024).",
        "But cross-dataset transfer is treated as a black-box engineering problem: train and pray.",
        "Reviewers and practitioners ask: when will my pre-trained model work on a new dataset? How can I trust the answer?",
        "We need a diagnostic framework + a deployment protocol + valid uncertainty intervals.",
    ]), {
        x: 0.5, y: 1.0, w: 5.5, h: 4.2,
        fontSize: 14, fontFace: BODY_FONT, color: TEXT_DARK, valign: "top", paraSpaceAfter: 8
    });
    // Right: large stat callout
    s.addShape(pres.shapes.RECTANGLE, {
        x: 6.2, y: 1.2, w: 3.4, h: 3.8,
        fill: { color: ICE_BLUE, transparency: 50 }, line: { color: NAVY, width: 1 }
    });
    s.addText("4", {
        x: 6.2, y: 1.4, w: 3.4, h: 1.2,
        fontSize: 64, fontFace: HEADER_FONT, bold: true, color: NAVY, align: "center", margin: 0
    });
    s.addText("public datasets", {
        x: 6.2, y: 2.55, w: 3.4, h: 0.4, fontSize: 13, fontFace: BODY_FONT, color: TEXT_MUTED, align: "center", margin: 0
    });
    s.addText("12", {
        x: 6.2, y: 3.0, w: 3.4, h: 1.0,
        fontSize: 52, fontFace: HEADER_FONT, bold: true, color: NAVY, align: "center", margin: 0
    });
    s.addText("transfer directions", {
        x: 6.2, y: 3.95, w: 3.4, h: 0.4, fontSize: 13, fontFace: BODY_FONT, color: TEXT_MUTED, align: "center", margin: 0
    });
    s.addText("3 deterministic regimes", {
        x: 6.2, y: 4.45, w: 3.4, h: 0.5,
        fontSize: 14, fontFace: BODY_FONT, italic: true, color: NAVY, align: "center", bold: true, margin: 0
    });
    s.addNotes(
        "Within-dataset accuracy is largely a solved problem. The frontier is cross-dataset reliability " +
        "and uncertainty quantification for deployment. We span 4 public datasets and 12 transfer directions."
    );
}

// =====================================================================
// SLIDE 3 — Brief literature review
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Brief literature snapshot — what others do");
    // Two columns
    s.addText("Within-dataset accuracy SOTA", {
        x: 0.5, y: 1.05, w: 4.4, h: 0.4,
        fontSize: 13, fontFace: HEADER_FONT, bold: true, color: NAVY, margin: 0
    });
    s.addText(bulletsContent([
        "Severson 2019 (Nature Energy) — voltage curves + elastic net; ~10% MAPE on MATR.",
        "BatLiNet 2024 (Nat. MI) — joint training across 401 cells; 6% MAPE on MATR-1.",
        "EES 2025 — surface temperature features; 12% MAPE on TRI.",
        "DCIR 2024 — internal impedance pulses; 150-cycle MAE across manufacturers.",
    ]), {
        x: 0.5, y: 1.45, w: 4.4, h: 2.7,
        fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, valign: "top", paraSpaceAfter: 6
    });
    s.addText("Cross-dataset / UQ approaches", {
        x: 5.1, y: 1.05, w: 4.4, h: 0.4,
        fontSize: 13, fontFace: HEADER_FONT, bold: true, color: NAVY, margin: 0
    });
    s.addText(bulletsContent([
        "Domain-Adaptive Transformer 2025 — V/Q+fine-tune; RMSE 178 cycles.",
        "HybridoNet-Adapt 2025 — MMD as training loss for UDA.",
        "Geng et al. 2025 — LSTM+attention+SSA; UQ via 50-seed MC (no coverage guarantee).",
        "Ge et al. 2025 — Deep Koopman with embedded operating conditions.",
    ]), {
        x: 5.1, y: 1.45, w: 4.4, h: 2.7,
        fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, valign: "top", paraSpaceAfter: 6
    });
    // The gap statement
    s.addShape(pres.shapes.RECTANGLE, {
        x: 0.5, y: 4.4, w: 9, h: 0.8,
        fill: { color: NAVY }, line: { color: NAVY }
    });
    s.addText([
        { text: "GAP — ", options: { bold: true, color: ACCENT_GOLD } },
        { text: "no one first diagnoses what kind of shift exists, decides whether transfer is salvageable, and provides valid prediction intervals for the deployment.",
          options: { color: WHITE } }
    ], {
        x: 0.7, y: 4.45, w: 8.6, h: 0.7,
        fontSize: 12, fontFace: BODY_FONT, valign: "middle", margin: 0
    });
    s.addNotes(
        "Two themes: within-dataset accuracy SOTA (left) and cross-dataset / UQ (right). " +
        "None of them first diagnoses the type of shift, decides whether transfer can be repaired, " +
        "and provides valid (coverage-guaranteed) prediction intervals — that is our gap."
    );
}

// =====================================================================
// SLIDE 4 — Research questions / contributions
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Research questions and contributions");
    s.addText("Three research questions", {
        x: 0.5, y: 1.0, w: 9, h: 0.4, fontSize: 14, fontFace: HEADER_FONT,
        bold: true, color: NAVY, margin: 0
    });
    s.addText(bulletsContent([
        "RQ1 — Can cross-dataset RUL transfer failure modes be predicted before training, from source/target diagnostics alone?",
        "RQ2 — Does a small target labelled set (k=20 cells) repair the failure with valid uncertainty intervals?",
        "RQ3 — Is the answer architecture-dependent, or does it survive a deep-learning backbone?",
    ]), {
        x: 0.7, y: 1.4, w: 9, h: 1.6,
        fontSize: 13, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 8, valign: "top"
    });
    s.addText("Five contributions", {
        x: 0.5, y: 3.1, w: 9, h: 0.4, fontSize: 14, fontFace: HEADER_FONT,
        bold: true, color: NAVY, margin: 0
    });
    s.addText(bulletsContent([
        "Four-dataset benchmark (MATR / HUST / Sandia / Luh-KIT) under a capacity-only feature contract.",
        "Covariate-vs-concept shift decomposition framework with falsifier (importance-weighted CP).",
        "Rank-signal regime taxonomy of 12 transfer directions (strong-rank / offset-dominant / rank-collapsed).",
        "k-shot target calibration protocol with MAPIE split conformal prediction (valid 90% / 95% coverage).",
        "Backbone-agnostic validation via PyTorch 1D-CNN baseline.",
    ]), {
        x: 0.7, y: 3.5, w: 9, h: 2.0,
        fontSize: 13, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 6, valign: "top"
    });
    s.addNotes("Three questions, five contributions. The regime taxonomy and valid CP intervals are the two paper-defining novelties.");
}

// =====================================================================
// SLIDE 5 — Datasets summary table
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Datasets — four public archives under one feature contract");

    const tableData = [
        [
            { text: "Dataset", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "left" } },
            { text: "Cells (modeled / censored)", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "Chemistry / format", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "left" } },
            { text: "Lifetime range (cycles)", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "Source", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "left" } },
        ],
        ["MATR (Severson/TRI)", "129 / 6", "LFP 18650, fast-charge", "133 – 2066", "Nature Energy 2019"],
        ["HUST", "77 / 0", "LFP, multi-stage discharge", "829 – 2024", "EES 2022"],
        ["Sandia 0-100 SOC", "50 / 11", "NCA / NMC / LFP 18650", "varies (mixed temp)", "JES 2020"],
        ["Luh / KIT RADAR", "106 / 2", "NMC, standard cycling", "Sandia/Luh narrow", "KITopen 10.35097/1947"],
    ];
    s.addTable(tableData, {
        x: 0.5, y: 1.05, w: 9.0, colW: [1.9, 1.8, 2.4, 1.6, 1.3],
        rowH: 0.45, fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK,
        border: { pt: 0.5, color: RULE }, valign: "middle",
    });
    s.addText(bulletsContent([
        "362 modeled cells total (19 censored at 0.85·Q0 EOL threshold; retained for survival audit).",
        "Same Q0/EOL definition across datasets: Q0 = median(Q_dis at cycles 2–5); EOL when Q_dis ≤ 0.85·Q0.",
        "Capacity-only feature contract — 34 features derived from Q_dis(c) trajectory only (no voltage, current, temperature).",
        "Universal contract enables apples-to-apples transfer across batteries with different chemistries and protocols.",
    ]), {
        x: 0.5, y: 3.7, w: 9.0, h: 1.7,
        fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("Four public datasets, all using the same minimal feature contract (capacity-only). " +
        "362 modeled cells, ~19 censored cells handled in the survival audit.");
}

// =====================================================================
// SLIDE 6 — Methodology overview
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Methodology — pipeline overview");

    // Pipeline boxes
    const boxes = [
        { label: "Raw cell data", note: "MATR / HUST / Sandia / Luh raw .pkl + .csv", color: ICE_BLUE },
        { label: "Audit + features", note: "Q0 = median(QD₂…QD₅); EOL = first c where Q_dis ≤ 0.85·Q0; 34 features", color: ICE_BLUE },
        { label: "Splits (70/15/15)", note: "5 seeds × cell-level lifetime-stratified", color: ICE_BLUE },
        { label: "Models", note: "7 classical (EN, PLS, RF, XGB, CB, GP, Stacking) + PyTorch 1D-CNN", color: ICE_BLUE },
        { label: "Shift + regime", note: "MMD, Mahalanobis, conditional-shift slopes, rank-signal classifier", color: ACCENT_GOLD },
        { label: "k-shot + CP", note: "Adapter (residual / linear) → MAPIE split CP at 90% / 95%", color: ACCENT_GOLD },
    ];
    const colWidth = 1.45;
    const startX = 0.5;
    const yTop = 1.1;
    boxes.forEach((b, i) => {
        const col = i % 3, row = Math.floor(i / 3);
        const x = startX + col * (colWidth + 0.25);
        const y = yTop + row * 1.8;
        s.addShape(pres.shapes.RECTANGLE, {
            x, y, w: colWidth, h: 1.6, fill: { color: b.color }, line: { color: NAVY, width: 1 }
        });
        s.addText(b.label, {
            x: x + 0.05, y: y + 0.1, w: colWidth - 0.1, h: 0.45,
            fontSize: 13, fontFace: HEADER_FONT, bold: true, color: NAVY, align: "center", margin: 0
        });
        s.addText(b.note, {
            x: x + 0.07, y: y + 0.55, w: colWidth - 0.14, h: 1.0,
            fontSize: 9.5, fontFace: BODY_FONT, color: TEXT_DARK, align: "center", valign: "top", margin: 0
        });
    });
    // Right-side third column — pipeline boxes already span 3 cols × 2 rows. Add arrows omitted for cleanliness.
    s.addText("Outputs: paper-facing tables and figures (within / cross / shift / CP / SHAP / regime).", {
        x: 5.5, y: 1.1, w: 4.0, h: 0.8, fontSize: 11, fontFace: BODY_FONT, color: TEXT_MUTED,
        italic: true, valign: "top", margin: 0
    });
    s.addText(bulletsContent([
        "All hyperparameters tuned by inner 5-fold CV (matching protocol across classical models and CNN).",
        "Per-cell cluster bootstrap CIs avoid double-counting target cells across seeds (cross-dataset).",
        "Median R² aggregation guards against catastrophic linear-adapter draws on small k_adapter.",
        "Open code + committed feature CSVs (FAIR + Zenodo deposit, MIT license).",
    ]), {
        x: 5.5, y: 1.9, w: 4.0, h: 2.6,
        fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("Pipeline: audit → features → splits → models → diagnostics → calibration + CP. " +
        "Same CV protocol on classical and CNN; cluster bootstrap; median R² for outlier-robust aggregation.");
}

// =====================================================================
// SLIDE 7 — Within-dataset results table
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Within-dataset — capacity-only feature contract performance");

    const table = [
        [
            { text: "Dataset", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
            { text: "Best model", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
            { text: "MAE", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "sMAPE", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "R²", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "R² 95% CI (pooled cluster bootstrap)", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
        ],
        ["MATR", "CatBoost", "171.7", "23.7%", "0.575", "[0.458, 0.646]"],
        ["HUST", "Random Forest", "178.0", "12.2%", "0.340", "[0.072, 0.512]"],
        ["Sandia 0-100 SOC", "XGBoost", "120.8", "23.4%", "0.940", "[0.804, 0.987]"],
        ["Luh / KIT", "Gaussian Process", "115.8", "18.4%", "0.769", "[0.675, 0.843]"],
    ];
    s.addTable(table, {
        x: 0.5, y: 1.05, w: 9.0, colW: [1.5, 1.5, 1.0, 1.0, 1.0, 3.0],
        rowH: 0.45, fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK,
        border: { pt: 0.5, color: RULE }, valign: "middle", align: "center",
    });
    s.addText(bulletsContent([
        "Sandia and Luh — fitted reasonably well by capacity-only features (R² ≥ 0.77).",
        "MATR — within the capacity-only literature ceiling (0.6 – 0.7); HUST narrow lifetime range caps R².",
        "Reference: BatLiNet (voltage curves, joint training across 401 cells) hits 6% MAPE on MATR — we trade off accuracy for data-minimality, transferability and valid CP.",
        "Bootstrap CIs from cluster-bootstrap by cell_id across pooled out-of-split predictions.",
    ]), {
        x: 0.5, y: 3.55, w: 9.0, h: 1.8,
        fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("Within-dataset accuracy is reasonable for Sandia/Luh, capped by lifetime spread for HUST, " +
        "and in the capacity-only literature ceiling for MATR. We do not chase deep-learning SOTA — we trade accuracy for valid CP.");
}

// =====================================================================
// SLIDE 8 — Cross-dataset transfer fails
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Cross-dataset transfer fails — but not uniformly");

    const table = [
        [
            { text: "Direction", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
            { text: "Best model", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
            { text: "MAE", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "R²", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
        ],
        // Strong rank signal — green-tint
        [
            { text: "Sandia → Luh", options: { fill: { color: "E8F4E8" } } },
            { text: "PLS", options: { fill: { color: "E8F4E8" } } },
            { text: "185.0", options: { fill: { color: "E8F4E8" }, align: "center" } },
            { text: "+0.477", options: { fill: { color: "E8F4E8" }, align: "center", color: "1B5E20", bold: true } },
        ],
        [
            { text: "Luh → Sandia", options: { fill: { color: "E8F4E8" } } },
            { text: "XGBoost", options: { fill: { color: "E8F4E8" } } },
            { text: "415.3", options: { fill: { color: "E8F4E8" }, align: "center" } },
            { text: "+0.492", options: { fill: { color: "E8F4E8" }, align: "center", color: "1B5E20", bold: true } },
        ],
        // Offset-dominant — yellow
        [
            { text: "HUST → Luh", options: { fill: { color: "FFF8E1" } } },
            { text: "PLS", options: { fill: { color: "FFF8E1" } } },
            { text: "352.8", options: { fill: { color: "FFF8E1" }, align: "center" } },
            { text: "−0.373", options: { fill: { color: "FFF8E1" }, align: "center", color: "B26500", bold: true } },
        ],
        [
            { text: "MATR → Sandia", options: { fill: { color: "FFF8E1" } } },
            { text: "CatBoost", options: { fill: { color: "FFF8E1" } } },
            { text: "715.0", options: { fill: { color: "FFF8E1" }, align: "center" } },
            { text: "+0.041", options: { fill: { color: "FFF8E1" }, align: "center", color: "B26500", bold: true } },
        ],
        // Collapsed — red
        [
            { text: "MATR → HUST", options: { fill: { color: "FDE8E8" } } },
            { text: "Gaussian Process", options: { fill: { color: "FDE8E8" } } },
            { text: "763.0", options: { fill: { color: "FDE8E8" }, align: "center" } },
            { text: "−7.94", options: { fill: { color: "FDE8E8" }, align: "center", color: "B00020", bold: true } },
        ],
        [
            { text: "HUST → MATR", options: { fill: { color: "FDE8E8" } } },
            { text: "Gaussian Process", options: { fill: { color: "FDE8E8" } } },
            { text: "720.4", options: { fill: { color: "FDE8E8" }, align: "center" } },
            { text: "−3.60", options: { fill: { color: "FDE8E8" }, align: "center", color: "B00020", bold: true } },
        ],
    ];
    s.addTable(table, {
        x: 0.5, y: 1.05, w: 6.0, colW: [1.7, 1.7, 1.3, 1.3],
        rowH: 0.4, fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK,
        border: { pt: 0.5, color: RULE }, valign: "middle",
    });
    // Right legend / take-away
    s.addText("Three failure regimes emerge", {
        x: 6.7, y: 1.1, w: 3.0, h: 0.4, fontSize: 13, fontFace: HEADER_FONT, bold: true, color: NAVY, margin: 0
    });
    s.addShape(pres.shapes.RECTANGLE, { x: 6.7, y: 1.6, w: 0.3, h: 0.3, fill: { color: "E8F4E8" } });
    s.addText("Strong rank signal", {
        x: 7.05, y: 1.6, w: 2.6, h: 0.3, fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, valign: "middle", margin: 0
    });
    s.addShape(pres.shapes.RECTANGLE, { x: 6.7, y: 2.0, w: 0.3, h: 0.3, fill: { color: "FFF8E1" } });
    s.addText("Offset / weak signal", {
        x: 7.05, y: 2.0, w: 2.6, h: 0.3, fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, valign: "middle", margin: 0
    });
    s.addShape(pres.shapes.RECTANGLE, { x: 6.7, y: 2.4, w: 0.3, h: 0.3, fill: { color: "FDE8E8" } });
    s.addText("Rank-collapsed", {
        x: 7.05, y: 2.4, w: 2.6, h: 0.3, fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, valign: "middle", margin: 0
    });
    s.addText(bulletsContent([
        "Sandia ↔ Luh stay positive — salvageable.",
        "MATR ↔ HUST catastrophic — no architecture helps.",
        "Worst case: Sandia → MATR with PyTorch CNN reaches R² = −47.8.",
        "Failure structure is not noise — it is predictable.",
    ]), {
        x: 6.7, y: 2.85, w: 3.0, h: 2.4,
        fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("Cross-dataset transfer fails on every direction, but the failure mode varies. " +
        "Sandia↔Luh remain positive; MATR↔HUST collapse. This is the first sign of a regime structure.");
}

// =====================================================================
// SLIDE 9 — Geometric shift quantification
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Geometric shift — capacity normalization closes 71% of the gap");

    const table = [
        [
            { text: "Setting", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
            { text: "Feature set", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "MMD", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "Mahalanobis", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
        ],
        ["Raw features (MATR vs HUST)", "12", "0.71", "13.10"],
        ["Raw features", "34", "0.57", "16.00"],
        [
            { text: "Q0-normalized", options: { fill: { color: "E8F4E8" } } },
            { text: "12", options: { fill: { color: "E8F4E8" }, align: "center" } },
            { text: "0.51", options: { fill: { color: "E8F4E8" }, align: "center" } },
            { text: "3.75", options: { fill: { color: "E8F4E8" }, align: "center", bold: true } },
        ],
        ["Q0-normalized", "34", "0.43", "6.71"],
    ];
    s.addTable(table, {
        x: 0.3, y: 1.05, w: 5.8, colW: [2.2, 1.0, 0.9, 1.7],
        rowH: 0.4, fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK,
        border: { pt: 0.5, color: RULE }, valign: "middle",
    });

    // Stat callout
    s.addShape(pres.shapes.RECTANGLE, {
        x: 6.4, y: 1.1, w: 3.2, h: 1.7,
        fill: { color: ICE_BLUE, transparency: 30 }, line: { color: NAVY, width: 1 }
    });
    s.addText("71%", {
        x: 6.4, y: 1.15, w: 3.2, h: 0.85, fontSize: 48, fontFace: HEADER_FONT, bold: true, color: NAVY,
        align: "center", margin: 0
    });
    s.addText("of geometric shift closed by Q0 normalization", {
        x: 6.4, y: 1.95, w: 3.2, h: 0.8, fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK,
        align: "center", italic: true, margin: 0
    });

    s.addText(bulletsContent([
        "Largest single contributor: Qdis_cycle10 — 10.84 σ pooled-z mean shift (MATR Q0 ≈ 1.07 Ah, HUST Q0 ≈ 1.20 Ah).",
        "Capacity normalization (SOP §2.3) divides Q-scale features by per-cell Q0, putting heterogeneous datasets on the same scale.",
        "Mahalanobis 13.10 → 3.75 with 12-feature set; MMD 0.71 → 0.51.",
        "But R² doesn't improve — sometimes degrades (next slide).",
    ]), {
        x: 0.5, y: 3.4, w: 9.0, h: 1.9,
        fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("Capacity normalization closes 71% of the geometric gap. The natural next step is to check whether prediction improves — it doesn't (next slide).");
}

// =====================================================================
// SLIDE 10 — Covariate vs concept shift (key finding)
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Key finding — covariate alignment ≠ concept alignment");

    const table = [
        [
            { text: "Direction", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
            { text: "Feature set", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "Raw R²", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "Capnorm R²", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "Δ R²", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
        ],
        ["MATR → HUST", "12", "−8.13", "−8.11", "+0.02"],
        ["HUST → MATR", "12", "−1.53", "−3.82", "−2.29"],
        ["HUST → MATR", "34", "−2.05", "−3.60", "−1.55"],
    ];
    s.addTable(table, {
        x: 0.5, y: 1.05, w: 6.0, colW: [1.8, 1.2, 1.0, 1.2, 0.8],
        rowH: 0.4, fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK,
        border: { pt: 0.5, color: RULE }, valign: "middle",
    });
    // Take-away callout
    s.addShape(pres.shapes.RECTANGLE, {
        x: 6.8, y: 1.05, w: 2.9, h: 2.4,
        fill: { color: NAVY }, line: { color: NAVY }
    });
    s.addText("MMD ↓ 28%", {
        x: 6.8, y: 1.15, w: 2.9, h: 0.5, fontSize: 18, fontFace: HEADER_FONT, bold: true, color: WHITE, align: "center", margin: 0
    });
    s.addText("Mahalanobis ↓ 71%", {
        x: 6.8, y: 1.7, w: 2.9, h: 0.5, fontSize: 18, fontFace: HEADER_FONT, bold: true, color: WHITE, align: "center", margin: 0
    });
    s.addShape(pres.shapes.LINE, { x: 6.95, y: 2.35, w: 2.6, h: 0, line: { color: ACCENT_GOLD, width: 1.5 } });
    s.addText("R² unchanged or worse", {
        x: 6.8, y: 2.5, w: 2.9, h: 0.5, fontSize: 16, fontFace: HEADER_FONT, italic: true, bold: true, color: ACCENT_GOLD, align: "center", margin: 0
    });
    s.addText("Concept shift in P(Y|X) remains", {
        x: 6.8, y: 3.0, w: 2.9, h: 0.4, fontSize: 12, fontFace: BODY_FONT, color: ICE_BLUE, align: "center", italic: true, margin: 0
    });
    s.addText(bulletsContent([
        "Classical covariate-vs-concept shift distinction shows up cleanly in battery data.",
        "Removing the dataset-identity signal (Q0 gap) breaks the implicit predictor without fixing the y-distribution mismatch.",
        "HUST → MATR even degrades after capnorm (−1.53 → −3.82 for 12-feat) — the Q0 gap was load-bearing.",
        "Motivation for the conditional-shift decomposition framework (next slide).",
    ]), {
        x: 0.5, y: 3.6, w: 9.0, h: 1.7,
        fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("The headline finding: closing the geometric shift does not fix prediction. This motivates the next layer of diagnostics.");
}

// =====================================================================
// SLIDE 11 — Conditional shift heatmap
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Conditional shift — slope tests across four-dataset pairs");

    const imgPath = path.join(PROJECT_ROOT, "outputs/results_v2_four_dataset_conditional_shift/four_dataset_conditional_shift_heatmaps.png");
    // Image aspect 2300x1838 -> 0.799
    const maxW = 5.7, maxH = 4.0;
    const dims = fitImage({ width: 2300, height: 1838 }, maxW, maxH);
    s.addImage({ path: imgPath, x: 0.4, y: 1.1, w: dims.w, h: dims.h });

    s.addText(bulletsContent([
        "Z-score each feature within dataset; centre log(cycle_life) within dataset.",
        "Compare HUST−MATR slope differences with bootstrap CIs + BH FDR correction.",
        "Universal log-life offset reported separately (e.g. HUST/MATR = 0.735, ≈ 2.09×).",
        "Slope-shifted feature counts per pair: HUST↔Luh 26/34; HUST↔Sandia 25/34; MATR↔Sandia 19/34; MATR↔HUST 14/34; Sandia↔Luh 3/34 (the cleanest pair).",
        "Decision rule: BH-q < 0.05 AND bootstrap CI excludes zero.",
    ]), {
        x: 6.3, y: 1.1, w: 3.3, h: 4.2,
        fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 5, valign: "top"
    });
    s.addNotes("Per-feature slope test removes the universal log-life offset and asks whether the within-dataset feature/y relationship changes between datasets.");
}

// =====================================================================
// SLIDE 12 — Rank-signal regime taxonomy
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Rank-signal regime taxonomy — predict success before training");

    const table = [
        [
            { text: "Direction", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
            { text: "Naive R²", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "Pearson r", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "Linear-cal R²", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "Regime", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
        ],
        [
            { text: "Luh → Sandia", options: { fill: { color: "E8F4E8" } } },
            { text: "0.492", options: { fill: { color: "E8F4E8" }, align: "center" } },
            { text: "0.912", options: { fill: { color: "E8F4E8" }, align: "center" } },
            { text: "0.840", options: { fill: { color: "E8F4E8" }, align: "center" } },
            { text: "STRONG_RANK", options: { fill: { color: "E8F4E8" }, align: "center", bold: true, color: "1B5E20" } },
        ],
        [
            { text: "Sandia → Luh", options: { fill: { color: "E8F4E8" } } },
            { text: "0.477", options: { fill: { color: "E8F4E8" }, align: "center" } },
            { text: "0.859", options: { fill: { color: "E8F4E8" }, align: "center" } },
            { text: "0.744", options: { fill: { color: "E8F4E8" }, align: "center" } },
            { text: "STRONG_RANK", options: { fill: { color: "E8F4E8" }, align: "center", bold: true, color: "1B5E20" } },
        ],
        [
            { text: "HUST → Luh", options: { fill: { color: "FFF8E1" } } },
            { text: "−0.373", options: { fill: { color: "FFF8E1" }, align: "center" } },
            { text: "0.766", options: { fill: { color: "FFF8E1" }, align: "center" } },
            { text: "0.600", options: { fill: { color: "FFF8E1" }, align: "center" } },
            { text: "OFFSET_DOM", options: { fill: { color: "FFF8E1" }, align: "center", bold: true, color: "B26500" } },
        ],
        [
            { text: "MATR → Sandia", options: { fill: { color: "FFF8E1" } } },
            { text: "+0.041", options: { fill: { color: "FFF8E1" }, align: "center" } },
            { text: "0.522", options: { fill: { color: "FFF8E1" }, align: "center" } },
            { text: "0.343", options: { fill: { color: "FFF8E1" }, align: "center" } },
            { text: "OFFSET_DOM", options: { fill: { color: "FFF8E1" }, align: "center", bold: true, color: "B26500" } },
        ],
        [
            { text: "MATR → HUST", options: { fill: { color: "FDE8E8" } } },
            { text: "−7.94", options: { fill: { color: "FDE8E8" }, align: "center" } },
            { text: "−0.12", options: { fill: { color: "FDE8E8" }, align: "center" } },
            { text: "0.025", options: { fill: { color: "FDE8E8" }, align: "center" } },
            { text: "RANK_COLLAPSED", options: { fill: { color: "FDE8E8" }, align: "center", bold: true, color: "B00020" } },
        ],
        [
            { text: "HUST → MATR", options: { fill: { color: "FDE8E8" } } },
            { text: "−3.60", options: { fill: { color: "FDE8E8" }, align: "center" } },
            { text: "−0.13", options: { fill: { color: "FDE8E8" }, align: "center" } },
            { text: "0.015", options: { fill: { color: "FDE8E8" }, align: "center" } },
            { text: "RANK_COLLAPSED", options: { fill: { color: "FDE8E8" }, align: "center", bold: true, color: "B00020" } },
        ],
    ];
    s.addTable(table, {
        x: 0.35, y: 1.0, w: 6.5, colW: [1.3, 1.0, 1.0, 1.3, 1.9],
        rowH: 0.36, fontSize: 10.5, fontFace: BODY_FONT, color: TEXT_DARK,
        border: { pt: 0.5, color: RULE }, valign: "middle",
    });
    s.addText(bulletsContent([
        "STRONG_RANK — naive R² > 0.4, high Pearson r; salvageable.",
        "OFFSET_DOMINANT — rank signal preserved (r > 0.5), but absolute scale wrong → calibratable.",
        "RANK_COLLAPSED — r near zero or negative; no architecture or k-shot fix is sufficient.",
        "Regime label can be predicted from source/target diagnostics without training the predictor (regime classifier).",
    ]), {
        x: 6.95, y: 1.05, w: 2.85, h: 4.2,
        fontSize: 10, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("The regime taxonomy is the paper-defining table. " +
        "Pearson r between source predictions and target ground truth is the deciding diagnostic.");
}

// =====================================================================
// SLIDE 13 — Directional asymmetry MATR↔HUST
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Directional asymmetry — MATR ↔ HUST exemplar");
    const imgPath = path.join(PROJECT_ROOT, "outputs/results_v2_conditional_shift/paper_directional_asymmetry_seed42.png");
    const dims = fitImage({ width: 2160, height: 1520 }, 5.6, 4.2);
    s.addImage({ path: imgPath, x: 0.4, y: 1.0, w: dims.w, h: dims.h });
    s.addText(bulletsContent([
        "Source predictions vs target ground truth — paper-facing scatter, seed=42.",
        "HUST → MATR keeps weak positive rank signal (r ≈ 0.22 – 0.27).",
        "MATR → HUST is essentially uncorrelated with target lifetime (r ≈ −0.12 to −0.14).",
        "Negative alpha in OLS is fitting noise — not a real mechanistic inversion.",
        "Theil-Sen and Huber estimators confirm the same picture.",
        "Constant share of squared error: 72 – 90% across configs → dominant component is a directional offset.",
    ]), {
        x: 6.2, y: 1.0, w: 3.4, h: 4.3,
        fontSize: 10.5, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("Two pairwise scatter panels per direction × model. The take-away is that MATR↔HUST is rank-collapsed but with a large offset that a constant adapter can shrink.");
}

// =====================================================================
// SLIDE 14 — IWCP falsifier
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Importance-weighted CP — the falsifier");
    const imgPath = path.join(PROJECT_ROOT, "outputs/results_v2_importance_weighted_cp/paper_iwcp_comparison_90.png");
    const dims = fitImage({ width: 2430, height: 2448 }, 4.6, 4.2);
    s.addImage({ path: imgPath, x: 0.5, y: 1.0, w: dims.w, h: dims.h });
    s.addText("Can a pure covariate-shift fix recover coverage?", {
        x: 5.3, y: 1.0, w: 4.3, h: 0.4,
        fontSize: 13, fontFace: HEADER_FONT, bold: true, color: NAVY, margin: 0
    });
    s.addText(bulletsContent([
        "Cross-fitted logistic discriminator AUC = 0.994 – 0.996 → near-perfect dataset separability.",
        "Raw calibration-weight ESS/n ≈ 0.55 – 0.59.",
        "Importance-weighted CP at 90% nominal: coverage reaches 99% only by returning infinite intervals 99% of the time (finite-interval fraction ≤ 0.9%).",
        "Re-weighting alone cannot deliver useful target intervals — covariate shift is not the binding constraint.",
        "Target-adapted residual-mean CP at k=20 recovers 90% coverage with finite intervals (0.905 – 0.909).",
    ]), {
        x: 5.3, y: 1.5, w: 4.3, h: 3.8,
        fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 5, valign: "top"
    });
    s.addNotes("This is the falsifier. We rule out the obvious covariate-shift fix before introducing the target-adapter solution.");
}

// =====================================================================
// SLIDE 15 — k-shot scaling
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "k-shot target calibration — scaling with target labels");
    const imgPath = path.join(PROJECT_ROOT, "outputs/results_v2_four_dataset_kshot_scaling/paper_kshot_scaling.png");
    const dims = fitImage({ width: 3274, height: 2194 }, 6.0, 4.2);
    s.addImage({ path: imgPath, x: 0.4, y: 1.0, w: dims.w, h: dims.h });
    s.addText(bulletsContent([
        "Two-parameter linear adapter fit on k random target cells, scored on the rest.",
        "MATR → HUST CatBoost: R² −10.0 → −0.13 with k=20 (≈10× MSE reduction).",
        "HUST → MATR Stacking: R² −2.66 → −0.05 with k=20.",
        "Linear adapter outperforms residual-mean on heterogeneous Sandia targets.",
        "Repaired R² near 0 = matches target marginal mean — calibration ceiling for rank-collapsed pairs.",
        "k=5 is too few; k=20 is the sweet spot for budget and recovery.",
    ]), {
        x: 6.55, y: 1.0, w: 3.1, h: 4.3,
        fontSize: 10.5, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("Calibration curve: R² climbs from catastrophic to ~0 across k = 5 → 20, with the linear adapter dominating in heterogeneous-target directions.");
}

// =====================================================================
// SLIDE 16 — CP coverage and width
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Conformal prediction — coverage and interval width");
    const imgPath = path.join(PROJECT_ROOT, "outputs/results_v2_four_dataset_conformal/paper_cp_coverage_width.png");
    const dims = fitImage({ width: 2200, height: 1400 }, 6.0, 4.2);
    s.addImage({ path: imgPath, x: 0.4, y: 1.0, w: dims.w, h: dims.h });
    s.addText(bulletsContent([
        "MAPIE split CP at 90% / 95%, absolute-residual conformity score, prefit estimator.",
        "Within-dataset CP: coverage 0.89 – 0.97 — coverage-valid by exchangeability.",
        "Naive source-calibrated cross CP: 0.00 – 0.73 — under-covers under shift.",
        "Target-domain CP (k=20): 0.89 – 0.92 — coverage restored at cost of width.",
        "Residual-mean adapted CP (k=20 + 20): 0.89 – 0.93 — best width/coverage trade-off.",
        "Median R² used in tables (linear adapter occasionally produces extreme draws — see CP outlier note).",
    ]), {
        x: 6.55, y: 1.0, w: 3.1, h: 4.3,
        fontSize: 10.5, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("Three CP modes compared. Target-adapted residual-mean is the deployment policy.");
}

// =====================================================================
// SLIDE 17 — Regime-stratified CP
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Regime-stratified CP — coverage and width organized by regime");
    const imgPath = path.join(PROJECT_ROOT, "outputs/results_v2_four_dataset_conformal/paper_cp_regime_stratified_90.png");
    const dims = fitImage({ width: 3520, height: 1936 }, 9.0, 2.8);
    // Center the image horizontally
    const imgX = (10 - dims.w) / 2;
    s.addImage({ path: imgPath, x: imgX, y: 1.0, w: dims.w, h: dims.h });
    s.addText(bulletsContent([
        "12 transfer directions sorted by rank-signal regime; four CP scenarios per row.",
        "STRONG_RANK pair (Sandia ↔ Luh) — narrow intervals after adapter.",
        "RANK_COLLAPSED pair (MATR ↔ HUST) — coverage restored but width remains large.",
        "Source-calibrated CP collapses in every regime — under-coverage is universal, not regime-specific.",
        "Median R² reporting suppresses the LUH→MATR linear-adapter outlier (was R²_mean ≈ −3.5 × 10⁵).",
    ]), {
        x: 0.5, y: dims.h + 1.05, w: 8.0, h: 5.2 - (dims.h + 1.05),
        fontSize: 10.5, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 3, valign: "top"
    });
    s.addNotes("Regime-stratified CP table — paper-facing figure that joins the CP results with the conditional-shift regime labels.");
}

// =====================================================================
// SLIDE 18 — SHAP top features × regime
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "SHAP × regime — important within-domain, fragile across");
    const imgPath = path.join(PROJECT_ROOT, "outputs/results_v2_four_dataset_shap/four_dataset_shap_feature_importance_top_features.png");
    const dims = fitImage({ width: 2480, height: 1100 }, 9.0, 2.6);
    s.addImage({ path: imgPath, x: 0.5, y: 1.0, w: dims.w, h: dims.h });
    s.addText(bulletsContent([
        "TreeSHAP attribution on the per-dataset primary model (CatBoost / RF / XGBoost / GP).",
        "MATR top: accel_mean, poly2_c, slope_last_quarter, range_Qdis — most are slope-shifted across MATR↔HUST.",
        "HUST top: Qdis_N, linearity_r2, Qdis_cycle10, poly2_a — scale-shift fragile (Qdis_cycle10 has 10.84σ shift).",
        "Sandia top: Qdis_N (55% rel. importance), mad_Qdis, slope_linear — slope-stable across Sandia↔Luh.",
        "Sandia↔Luh transferable directions have stable top SHAP features; MATR↔HUST collapsed pairs have unstable ones.",
    ]), {
        x: 0.5, y: dims.h + 1.1, w: 9.0, h: 5.2 - (dims.h + 1.1),
        fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("Closing the loop: the within-domain importance of a feature is not the same as its cross-dataset reliability.");
}

// =====================================================================
// SLIDE 19 — PyTorch 1D-CNN baseline + Pareto positioning
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Backbone-agnostic check — PyTorch 1D-CNN + Pareto positioning");

    // Top: comparison table
    const cnnTable = [
        [
            { text: "Dataset", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
            { text: "Classical best", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "PyTorch CNN", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
            { text: "Δ R²", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
        ],
        ["MATR", "CatBoost 0.575", "0.305", "−0.270"],
        ["HUST", "RF 0.340", "−0.174", "−0.514"],
        ["Sandia", "XGBoost 0.940", "0.881", "−0.059"],
        ["Luh / KIT", "GP 0.769", "0.761", "−0.008"],
    ];
    s.addTable(cnnTable, {
        x: 0.5, y: 1.05, w: 4.5, colW: [1.2, 1.3, 1.1, 0.9],
        rowH: 0.35, fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK,
        border: { pt: 0.5, color: RULE }, valign: "middle",
    });

    // Right: Pareto column
    s.addText("Pareto positioning vs literature", {
        x: 5.3, y: 1.05, w: 4.4, h: 0.4,
        fontSize: 13, fontFace: HEADER_FONT, bold: true, color: NAVY, margin: 0
    });
    s.addText([
        { text: "Severson 2019 (V(Q))", options: { bold: true, breakLine: true } },
        { text: "    ~10% MAPE  ·  no CP", options: { breakLine: true, color: TEXT_MUTED } },
        { text: "BatLiNet 2024 (V/Q + joint)", options: { bold: true, breakLine: true } },
        { text: "    6% MAPE MATR-1  ·  no CP", options: { breakLine: true, color: TEXT_MUTED } },
        { text: "EES 2025 (T features)", options: { bold: true, breakLine: true } },
        { text: "    12% MAPE TRI  ·  no CP", options: { breakLine: true, color: TEXT_MUTED } },
        { text: "This work — capacity-only + CP", options: { bold: true, color: ACCENT_GOLD, breakLine: true } },
        { text: "    24% sMAPE MATR  ·  valid CP  ·  CPU seconds", options: { color: ACCENT_GOLD } },
    ], {
        x: 5.3, y: 1.5, w: 4.4, h: 2.3,
        fontSize: 10.5, fontFace: BODY_FONT, color: TEXT_DARK, valign: "top", margin: 0, paraSpaceAfter: 2
    });

    // Bottom take-away
    s.addText(bulletsContent([
        "CNN is competitive on Sandia/Luh, falls below classical on MATR/HUST.",
        "Cross-dataset regime taxonomy preserved with CNN — same STRONG_RANK / RANK_COLLAPSED pairs.",
        "But CNN over-extrapolates on Sandia source: Sandia → MATR R² = −47.8, Sandia → HUST = −36.8 — deep architecture amplifies source-specific signal under shift.",
        "Take-away: regime taxonomy is backbone-agnostic; deep architecture does not buy cross-dataset robustness on the capacity-only contract.",
    ]), {
        x: 0.5, y: 3.9, w: 9.2, h: 1.5,
        fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 3, valign: "top"
    });
    s.addNotes("CNN serves as backbone-agnostic check. It confirms the regime taxonomy without changing the qualitative conclusion.");
}

// =====================================================================
// SLIDE 20 — LODO source-expert protocol
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "LODO source-expert — multi-source deployment");
    const imgPath = path.join(PROJECT_ROOT, "outputs/results_v2_four_dataset_lodo_source_expert/paper_lodo_main_panel.png");
    const dims = fitImage({ width: 2914, height: 1474 }, 6.0, 3.5);
    s.addImage({ path: imgPath, x: 0.4, y: 1.0, w: dims.w, h: dims.h });
    s.addText(bulletsContent([
        "Hold out each target dataset; train on the other three.",
        "Compare pooled ERM, source-expert selection, convex source-expert weighting, and pooled ERM + k-shot adapters.",
        "Best k=20 results: HUST R² = −0.115, Luh R² = +0.667, MATR R² = −0.018, Sandia R² = +0.862.",
        "Improves over naive single-source for every target, with clear k-shot scaling.",
        "Does not uniformly beat an oracle that picks the best source post-hoc — practical protocol, not a silver bullet.",
    ]), {
        x: 6.55, y: 1.0, w: 3.1, h: 4.3,
        fontSize: 10.5, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("LODO source-expert is a practical multi-source deployment protocol. " +
        "Improves over naive single-source for every held-out target.");
}

// =====================================================================
// SLIDE 21 — Survival / censoring (SI-style)
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Survival audit — censoring is not the explanation");
    const imgPath = path.join(PROJECT_ROOT, "outputs/results_v2_four_dataset_survival/kaplan_meier_four_dataset.png");
    const dims = fitImage({ width: 1700, height: 1100 }, 5.5, 4.0);
    s.addImage({ path: imgPath, x: 0.5, y: 1.05, w: dims.w, h: dims.h });
    s.addText(bulletsContent([
        "MATR 6/135 cells censored at 0.85·Q0; HUST 0/77; Sandia 11/61; Luh 2/108.",
        "Kaplan-Meier medians: MATR 773, HUST 1513, Sandia 305, Luh 508 cycles.",
        "Log-rank χ² = 61.2, p = 5.2 × 10⁻¹⁵ between MATR and HUST — separation is decisive.",
        "Imputing censored MATR cells at their censoring times moves MATR mean only 778 → 802 cycles.",
        "Censoring is not the reason HUST appears longer-lived than MATR.",
        "Supplementary diagnostic — paper-facing claim unchanged.",
    ]), {
        x: 6.2, y: 1.05, w: 3.4, h: 4.2,
        fontSize: 10.5, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("Supplementary survival audit. Confirms that the lifetime gap between datasets is not driven by right-censoring of some MATR cells.");
}

// =====================================================================
// SLIDE 22 — Koopman / DMD diagnostic (SI-style)
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Dynamics-level evidence — Hankel-DMD pilot");
    const imgPath = path.join(PROJECT_ROOT, "outputs/results_v2_four_dataset_koopman_dmd/dmd_eigenvalue_complex_plane.png");
    const dims = fitImage({ width: 1400, height: 1240 }, 4.2, 4.0);
    s.addImage({ path: imgPath, x: 0.5, y: 1.0, w: dims.w, h: dims.h });
    s.addText(bulletsContent([
        "Hankel-DMD on early Q/Q0 trajectories (cycles 2 – 100).",
        "Four-class discriminator on DMD summaries reaches AUC 0.915 ± 0.019.",
        "Sandia vs Luh dominant |λ| Pearson r with log-life: 0.80 (Sandia), 0.58 (Luh).",
        "Sandia → Luh operator transfer: 1.07× Luh self-RMSE.",
        "Sandia → MATR / Luh: operator extrapolation fails (consistent with regime collapse).",
        "Supporting mechanistic evidence — not a headline predictor.",
    ]), {
        x: 5.0, y: 1.0, w: 4.6, h: 4.2,
        fontSize: 10.5, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, valign: "top"
    });
    s.addNotes("Dynamics-level evidence supporting the regime story. Sandia/Luh dynamics are coherent; MATR/HUST are not.");
}

// =====================================================================
// SLIDE 23 — Limitations
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Limitations");
    s.addText(bulletsContent([
        "Capacity-only feature contract has a within-dataset accuracy ceiling. BatLiNet and BatteryLife clearly outperform on accuracy alone; we trade accuracy for transferability and valid CP.",
        "Rank-collapsed regimes (MATR ↔ HUST) cannot be repaired by k-shot or by deeper architecture. The framework predicts this failure but does not solve it.",
        "Linear adapter occasionally produces degenerate fits on small k_adapter — reported via median R² rather than mean to suppress outliers in paper-facing tables.",
        "Five seeds × 70/15/15 splits give relatively wide bootstrap CIs on HUST (12-cell test set); we use pooled cluster-bootstrap to mitigate but cannot eliminate this.",
        "We do not address streaming / online cell calibration — the protocol assumes a one-time k=20 collection per deployment.",
    ]), {
        x: 0.5, y: 1.05, w: 9.2, h: 4.3,
        fontSize: 13, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 6, valign: "top"
    });
    s.addNotes("Honest limitations. The accuracy ceiling and rank-collapsed regimes are the two biggest constraints.");
}

// =====================================================================
// SLIDE 24 — Conclusion
// =====================================================================
{
    const s = pres.addSlide();
    next();
    s.background = { color: NAVY };
    // Gold accent bar near title
    s.addShape(pres.shapes.RECTANGLE, {
        x: 0.6, y: 0.55, w: 0.08, h: 0.55, fill: { color: ACCENT_GOLD }, line: { color: ACCENT_GOLD }
    });
    // Title (top of slide, not centred)
    s.addText("Conclusion — four findings", {
        x: 0.9, y: 0.45, w: 8.7, h: 0.7,
        fontSize: 28, fontFace: HEADER_FONT, bold: true, color: WHITE, valign: "top", margin: 0
    });
    // Findings body
    s.addText([
        { text: "1.  Cross-dataset RUL transfer separates into three deterministic regimes", options: { bold: true, color: WHITE, breakLine: true } },
        { text: "      (strong-rank · offset-dominant · rank-collapsed). The regime is predictable from source/target diagnostics before training.", options: { color: ICE_BLUE, breakLine: true } },
        { text: " ", options: { breakLine: true } },
        { text: "2.  Covariate alignment ≠ concept alignment.", options: { bold: true, color: WHITE, breakLine: true } },
        { text: "      Capacity normalization closes 71% of the geometric gap, but cross-dataset R² is unchanged or worse.", options: { color: ICE_BLUE, breakLine: true } },
        { text: " ", options: { breakLine: true } },
        { text: "3.  Small target labelled set (k=20) + MAPIE split CP repairs the salvageable regimes with valid 90% / 95% coverage.", options: { bold: true, color: WHITE, breakLine: true } },
        { text: "      Rank-preserving directions recover to R² ≈ 0 with finite intervals; rank-collapsed pairs remain hard.", options: { color: ICE_BLUE, breakLine: true } },
        { text: " ", options: { breakLine: true } },
        { text: "4.  The regime taxonomy is backbone-agnostic.", options: { bold: true, color: WHITE, breakLine: true } },
        { text: "      PyTorch 1D-CNN preserves it — and over-extrapolates on Sandia source, reinforcing the Pareto positioning argument.", options: { color: ICE_BLUE } },
    ], {
        x: 0.7, y: 1.4, w: 8.9, h: 3.9,
        fontSize: 13, fontFace: BODY_FONT, valign: "top", margin: 0, paraSpaceAfter: 0
    });
    s.addNotes("Four key findings; each tied to a specific table or figure in the paper.");
}

// =====================================================================
// SLIDE 25 — Future work
// =====================================================================
{
    const s = pres.addSlide();
    addContentChrome(s, next(), TOTAL_SLIDES, "Future work");
    s.addText(bulletsContent([
        "Voltage / temperature / impedance feature contracts under the same regime-taxonomy framework. Does richer data shift the regime boundaries or simply add a constant to within-dataset accuracy?",
        "Streaming / online k-shot calibration. Current protocol assumes a single batch of k=20 target cells; a real deployment might add cells incrementally.",
        "Fifth-dataset held-out validation of the regime classifier. Currently the classifier is fit on 12 directions × 4 datasets; an unseen dataset would falsify or confirm the taxonomy.",
        "Physics-informed adapter families. Replace the linear adapter with a small electrochemistry-aware correction that uses Q0 + chemistry prior — might break the rank-collapsed regime.",
        "Cross-protocol transfer at fixed chemistry. Sandia includes mixed temperatures and discharge rates within one chemistry — natural follow-up benchmark.",
    ]), {
        x: 0.5, y: 1.05, w: 9.2, h: 4.3,
        fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 7, valign: "top"
    });
    s.addNotes("Future work along five axes — feature contract, online calibration, fifth-dataset validation, physics-informed adapters, cross-protocol benchmarks.");
}

// =====================================================================
// SLIDE 26 — Thank you / Q&A
// =====================================================================
{
    const s = pres.addSlide();
    s.background = { color: NAVY };
    s.addShape(pres.shapes.RECTANGLE, {
        x: 0.6, y: 1.5, w: 0.08, h: 2.6, fill: { color: ACCENT_GOLD }, line: { color: ACCENT_GOLD }
    });
    s.addText("Thank you", {
        x: 0.9, y: 1.4, w: 8.5, h: 1.2, fontSize: 56, fontFace: HEADER_FONT,
        bold: true, color: WHITE, valign: "top", margin: 0
    });
    s.addText("Questions and discussion", {
        x: 0.9, y: 2.6, w: 8.5, h: 0.6, fontSize: 22, fontFace: BODY_FONT,
        italic: true, color: ICE_BLUE, valign: "top", margin: 0
    });
    s.addText([
        { text: "Code, data, and reproduction recipe — ", options: { color: ICE_BLUE } },
        { text: "github.com/osmansafacifci/Graduation-Project-Dicle", options: { bold: true, color: WHITE } },
    ], {
        x: 0.9, y: 3.5, w: 8.5, h: 0.4, fontSize: 13, fontFace: BODY_FONT, valign: "middle", margin: 0
    });
    s.addText([
        { text: "Companion paper: ", options: { color: ICE_BLUE } },
        { text: "MIT-licensed, Zenodo DOI on release", options: { italic: true, color: WHITE } },
    ], {
        x: 0.9, y: 3.95, w: 8.5, h: 0.4, fontSize: 13, fontFace: BODY_FONT, valign: "middle", margin: 0
    });
    s.addText("Durukan Demir  ·  Dicle Çoban  ·  Salih Sarp  ·  Osman Safa Çifçi", {
        x: 0.5, y: 5.0, w: 9, h: 0.4, fontSize: 11, fontFace: BODY_FONT, color: ICE_BLUE, align: "center"
    });
    next();
    s.addNotes("Open floor. Repository and Zenodo deposit point.");
}

// =====================================================================

const outPath = path.join(__dirname, "battery_rul_defense.pptx");
pres.writeFile({ fileName: outPath }).then((fn) => {
    console.log("Wrote:", fn);
    console.log(`Slides: ${slideIdx} (expected ${TOTAL_SLIDES})`);
}).catch((err) => {
    console.error("Failed:", err);
    process.exitCode = 1;
});
