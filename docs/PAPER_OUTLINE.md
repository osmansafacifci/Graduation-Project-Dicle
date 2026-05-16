# Paper Outline

Manuscript-ready outline for the cross-dataset battery RUL paper. Designed as
a single-page reference while writing. Section-by-section structure, target
word counts, the claim each paragraph defends, and the figure/table backing
the claim. Companion to [`MANUSCRIPT_POSITIONING.md`](MANUSCRIPT_POSITIONING.md),
which carries the positioning rationale, Pareto table, and main-vs-SI split.

**Authors:** Durukan Demir, Dicle Çoban, Salih Sarp, Osman Safa Çifçi
**Last updated:** 2026-05-16
**Target length:** ~8 000 words main text, 10–12 figures, 4–5 tables, single-column or double-column depending on venue style file.

---

## 0. Title candidates

Pick one before writing the abstract; the headline phrase reappears in §6 and
the conclusion.

1. **Covariate alignment is not concept alignment: a controlled study of cross-dataset transfer for early-cycle battery lifetime prediction**
2. *When cross-dataset battery RUL transfer fails — and how to repair it with valid uncertainty*
3. *Rank-signal regime taxonomy for cross-dataset battery RUL: diagnosis, k-shot repair, and split conformal prediction*

Strongest for a methods-flavoured journal: (1). Strongest for an applied
energy journal: (3). (2) reads like a tutorial; useful as a workshop title
but soft for a research article.

## 1. Target venues (pick before submission)

| Tier | Journal | Fit | Notes |
|---|---|---|---|
| Q1 — primary | **Engineering Applications of AI** (IF ≈ 7.8) | ★★★★★ | "Methodological framework applied to a domain" format. |
| Q1 — primary | **Applied Soft Computing** (Q2 / IF ≈ 7.2) | ★★★★★ | Slightly more lenient than EAAI on prose; same fit. |
| Q1 — high stretch | **Journal of Energy Storage** (IF ≈ 9.8) | ★★★ | Needs DL baseline (done — PyTorch CNN) + sharper Pareto framing. |
| Q1 — high stretch | **Energy** (Elsevier, IF ≈ 9.0) | ★★★ | Recently accepted similar cross-domain Transformer paper. |
| Q1 — methods | **Knowledge-Based Systems** (IF ≈ 8.8) | ★★★★ | Strong fit for the regime-taxonomy + CP combination. |

See `MANUSCRIPT_POSITIONING.md` §4 for the rationale and acceptance-
probability estimates per venue.

---

## 2. Abstract — target 230–260 words

Single paragraph. Structure:

1. **Hook (~40 w)** — battery RUL deployment context; within-dataset accuracy is solved, cross-dataset is not.
2. **Gap (~40 w)** — existing domain-adaptation papers do not diagnose the *type* of shift and do not provide valid prediction intervals.
3. **What we do (~70 w)** — capacity-only four-dataset benchmark; covariate-vs-concept shift decomposition; rank-signal regime taxonomy across 12 transfer directions; k-shot target calibration + MAPIE split CP.
4. **Headline numbers (~50 w)** — Mahalanobis 71 % drop with no R² improvement (key finding); k=20 recovery on rank-preserving directions; valid 90 % / 95 % CP coverage; backbone-agnostic via PyTorch 1D-CNN.
5. **Take-away (~30 w)** — `quantify shift → predict regime → choose k → deploy with CP`; no voltage curves, no adversarial DA, valid uncertainty by construction.

Draft (placeholder, ~250 words; rewrite during final edit):

> Battery remaining-useful-life (RUL) prediction has converged to within-dataset accuracy near the capacity-only ceiling. Real deployments, however, train on lab data and apply on field data with different chemistry, protocol, or cell format, where naive cross-dataset transfer fails on every benchmark we tested. The community largely treats this as a black-box engineering problem; we treat it as a measurable failure with a predictable structure. We assemble a four-dataset benchmark (MATR / HUST / Sandia / Luh-KIT, 362 modelled cells under a single capacity-only feature contract) and run twelve cross-dataset directions. We show that capacity normalisation closes 71 % of the geometric (Mahalanobis) gap without recovering any cross-dataset prediction R²; that an importance-weighted conformal-prediction falsifier returns valid coverage only by emitting infinite intervals; and that a centered-log per-feature slope test plus a Pearson-r rank-signal diagnostic partition the twelve directions into three deterministic regimes (strong-rank / offset-dominant / rank-collapsed). On the two salvageable regimes, a two-parameter linear adapter fit on k = 20 randomly chosen target cells plus MAPIE split conformal prediction recovers near-zero R² with finite intervals at empirical coverage 0.89–0.93 against a 0.90 nominal. A PyTorch 1D-CNN baseline preserves the regime taxonomy without changing its qualitative shape and over-extrapolates when the source is heterogeneous (Sandia → MATR R² ≈ −47.8) — confirming that the result is governed by the rank-signal structure of the source/target pair rather than the choice of backbone. Code, derived feature tables, and the reproducibility recipe are public; the framework requires neither voltage curves nor adversarial domain adaptation.

---

## 3. Introduction — target 1 000 words

Three paragraphs.

### ¶1 (~300 w) — Why RUL transfer matters
- Battery deployment in EVs / grid storage / aerospace requires reliable cycle-life estimates.
- Lab data ≠ field data; chemistries and protocols vary.
- Within-dataset RUL is essentially solved by deep learning (BatLiNet, Severson, EES-temperature) on voltage / temperature curves.
- Cross-dataset transfer remains poorly characterised — papers either fine-tune (HybridoNet-Adapt, Domain-Adaptive Transformer) or report ad hoc transfer numbers without analysing *why* a particular direction works or fails.

### ¶2 (~350 w) — The gap, framed as questions
- RQ1: Can the failure mode of a transfer direction be predicted from source/target statistics *before* training?
- RQ2: When transfer fails, what minimum target-side correction recovers usable predictions with valid uncertainty?
- RQ3: Does the answer survive a deep-learning backbone, or is it an artefact of classical regression?
- The community currently lacks (a) a diagnostic that separates covariate shift from concept shift in battery RUL, (b) a deployment protocol with coverage-valid prediction intervals, (c) an architecture-agnostic empirical claim about cross-dataset failure structure.

### ¶3 (~350 w) — Contributions (five bullets)
1. **Four-dataset, capacity-only benchmark.** MATR, HUST, Sandia, Luh/KIT under a single 34-feature feature contract derived only from Q_dis(c).
2. **Shift-decomposition framework.** MMD + Mahalanobis (geometric layer); centred-log per-feature slope tests with BH-FDR (conditional layer); importance-weighted CP as a falsifier (showing the geometric fix alone is not enough).
3. **Rank-signal regime taxonomy.** Twelve transfer directions partition into three regimes determined by Pearson r between source predictions and target ground truth.
4. **k-shot target calibration + split CP.** k = 20 random target cells + residual-mean / linear adapter + MAPIE split CP at 90 % / 95 %; coverage-valid and finite-interval everywhere.
5. **Backbone-agnostic validation.** PyTorch 1D-CNN trained under the same inner-CV protocol confirms the taxonomy and exposes a deep-model over-extrapolation failure mode that classical regression avoids.

Reference figure / table from the outline: [F1] pipeline schematic + [T1] datasets summary.

---

## 4. Related work — target 800 words

Three sub-themes, ~250 words each.

### 4.1 — Within-dataset RUL accuracy SOTA (~250 w)
- Severson 2019 (Nature Energy): ΔQ(V) features + elastic net → 10 % MAPE MATR.
- BatLiNet 2024 (Nat. MI): joint training across 401 cells → 6 % MAPE MATR-1, 11 % MATR-2, 10 % HUST.
- EES 2025 (surface temperature features) → 6–17 % MAPE across SNL/UL/TRI/XJTU.
- DCIR cross-manufacturer 2024 (impedance pulses) → 150-cycle MAE in-lab.
- BatteryLife KDD 2025 — 16-dataset benchmark with 18 baselines.

Single closing sentence: capacity-only is the universal-deployment contract — voltage / temperature / impedance are often missing in archived data or in field BMS.

### 4.2 — Cross-dataset / domain adaptation (~280 w)
- Domain-Adaptive Transformer in *Energy* 2025: V/Q + fine-tuning, RMSE 178 cycles.
- HybridoNet-Adapt PLOS One 2025: MMD as a training loss for UDA.
- MMD-based DG for RUL (MDPI Batteries 2025): training-time alignment.
- Discovery Learning (Nature 2025): few-shot lifetime prediction from minimal experiments.

Closing point: these papers *do* DA but do not first diagnose the type of shift; none provides cross-dataset prediction intervals with coverage guarantees.

### 4.3 — Uncertainty quantification (~270 w)
- Geng et al. 2025 (*Energy*): LSTM + attention + SSA + DeepSHAP + 50-seed MC for UQ; UQ is heuristic.
- Conformal-RUL 2024 (RG preprint): first explicit CP-for-battery framework on NASA Ames.
- Augmented Physics-Based Li-ion + CP 2025 (arXiv 2507.00353): physics-informed CP on voltage-error dynamics.

Closing point: split CP with valid coverage on cross-dataset transfer + target-side calibration is the missing piece.

**End of §4** — explicit gap statement (≈80 w): "*None of these works first asks what kind of shift exists, decides whether transfer is salvageable, and emits coverage-valid intervals for the surviving directions. Our contribution is the missing diagnostic + repair + uncertainty triple, on a four-dataset benchmark, under the minimum viable feature contract.*"

---

## 5. Datasets and methodology — target 1 500 words

### 5.1 — Datasets (~250 w)
- [T1] table — MATR / HUST / Sandia / Luh-KIT counts, chemistry, lifetime range.
- 362 modelled cells, 19 censored at 0.85 · Q0 EOL threshold.
- Cite each dataset DOI.

### 5.2 — Feature contract (~300 w)
- Q_dis(c) only, cycles 2–N (N ∈ {50, 100}; primary results at N = 100).
- 34 features = 12 SOP + 12 shape/decay + 10 entropy/FFT/2nd-derivative.
- Define the SOP labels: Q0 = median(Q_dis at cycles 2–5); cycle_life = first c where Q_dis ≤ 0.85 · Q0.
- Capacity normalisation toggle: divide raw-Q features by per-cell Q0.

### 5.3 — Splits and modelling (~300 w)
- 70/15/15 cell-level lifetime-stratified split; 5 seeds {42, 123, 456, 789, 1011}.
- Seven classical models (Elastic Net, PLS, Random Forest, XGBoost, CatBoost, Gaussian Process, Stacking) with inner 5-fold CV.
- Log-target training, exp back to cycles for metrics.
- PyTorch 1D-CNN (channels: retention + 1st-diff; Conv1d × 2 → mean+max pool → dense + dropout; inner 5-fold CV over (filters, lr) ∈ {8, 16, 32} × {1e-3, 3e-3, 1e-2}).

### 5.4 — Shift, regime, and uncertainty (~400 w)
- Geometric shift: MMD with RBF + median-bandwidth; Mahalanobis with pooled covariance + ridge.
- Conditional shift: within-dataset z-score, within-dataset centring of log(cycle_life), univariate slope, 1000-bootstrap CI on the slope difference, BH-FDR across 34 features.
- Rank-signal classifier: Pearson r between source predictions and target ground truth; partition into STRONG_RANK / OFFSET_DOMINANT / RANK_COLLAPSED via thresholds and CI checks.
- k-shot adapters: residual-mean (constant offset) and linear (OLS slope+intercept) fit on k random target cells, scored on the rest.
- Conformal prediction: MAPIE split CP, prefit estimator, absolute conformity score, finite-sample rank correction. Four scenarios — within / source-cal / target-cal / target-adapted.

### 5.5 — Reproducibility (~250 w)
- All split JSONs and feature CSVs committed.
- Code MIT-licensed, FAIR with CITATION.cff + Zenodo deposit.
- Pinned `requirements-pinned.txt` for exact-version reproduction.
- Single-page reproduction recipe at `docs/REPRODUCIBILITY.md`.
- Reference [F1] for the pipeline diagram.

---

## 6. Results — target 3 000 words

The bulk of the paper. Split into five subsections.

### 6.1 — Within-dataset performance (~400 w)
- [T2] table: 4 datasets × {best model, MAE, sMAPE, R², CI}.
- [F2] scatter plot: predictions vs ground truth, 4 panels.
- One paragraph on each dataset's headline number.
- Compare against Severson / BatLiNet / EES — frame as Pareto positioning (data minimality vs accuracy vs CP validity).
- [T5] Pareto table (see `MANUSCRIPT_POSITIONING.md` §"Pareto Table — concrete numbers").

### 6.2 — Naive cross-dataset transfer fails — but not uniformly (~500 w)
- [T3] transfer matrix: 12 directions × {best model, MAE, R²}.
- All R² < 0.5; some catastrophic, some near zero, two positive.
- Lead the eye toward regime structure that emerges from these numbers without telling the reader the partition yet.

### 6.3 — Geometric shift is not the problem (~500 w)
- [F3] Mahalanobis heatmap raw vs capnorm for the six pairs.
- 71 % Mahalanobis reduction with capnorm.
- [F4] **KEY**: MMD-drop-vs-R²-change scatter — visual proof of geometric ≠ prediction alignment.
- Importance-weighted CP falsifier — finite-interval-fraction ≤ 0.9 % at 90 % nominal. Cited inline; full IWCP detail to SI.

### 6.4 — Conditional shift and regime taxonomy (~700 w)
- [F5] **PAPER-DEFINING**: conditional-shift heatmap (four-dataset slope-shift counts) with regime labels.
- Centred-log slope test recap; report HUST−MATR slope shifts; 14/34 features shifted; universal log-life offset = 0.735.
- Direction-level classifier: Pearson r between source predictions and target. Tabulate r and post-calibration R² for the 12 directions.
- Three regimes with explicit thresholds.
- [F10] supporting: directional-asymmetry scatter for MATR↔HUST exemplar.

### 6.5 — k-shot calibration and conformal prediction (~600 w)
- [F6] k-shot scaling: R² vs k coloured by regime.
- [T4] CP table: 12 directions × 4 scenarios at 90 % and 95 %.
- [F7] CP coverage and width chart (existing `paper_cp_coverage_width.png`).
- [F_extra] regime-stratified CP figure (existing `paper_cp_regime_stratified_90.png`).
- Median R² aggregation footnote (linear adapter on small k_adapter occasionally degenerate).

### 6.6 — Backbone-agnostic check via PyTorch CNN (~300 w)
- CNN within-dataset numbers — competitive on Sandia/Luh, falls below classical on MATR/HUST.
- CNN cross-dataset numbers — same regime taxonomy; over-extrapolation on Sandia source.
- One-paragraph claim: regime structure is backbone-agnostic.

### 6.7 — SHAP × regime closing loop (~200 w)
- [F8] SHAP top features per dataset.
- Cross-reference with regime labels: high-importance ≠ transfer-stable.
- One-sentence take-away: "*within-domain importance and cross-dataset reliability are different properties of a feature.*"

---

## 7. Discussion — target 800 words

### 7.1 — What the regime taxonomy means in practice (~250 w)
- Re-state the deployment protocol: `quantify shift → predict regime → choose k → deploy with CP`.
- Explicit guidance per regime.

### 7.2 — Why deep models do not help here (~250 w)
- CNN over-extrapolation on Sandia → MATR.
- Capacity-only contract limits gain from architecture depth.
- Reference [T5] Pareto positioning argument.

### 7.3 — Relation to the broader DA literature (~200 w)
- Importance-weighted approaches (HybridoNet-Adapt, MMD-DG) fail when the binding constraint is conditional, not covariate.
- Joint training (BatLiNet) is a different protocol that does not address the deployment-time question.

### 7.4 — Supporting mechanistic evidence (~100 w)
- Brief mention of the Hankel-DMD pilot (full detail in SI).

---

## 8. Limitations and future work — target 350 words

### Limitations (~200 w)
- Capacity-only contract has a within-dataset accuracy ceiling.
- Rank-collapsed regimes are not repairable by k-shot or by deeper models.
- Small test splits give relatively wide bootstrap CIs (mitigated by cluster bootstrap).
- One-shot batch of k=20 — no streaming/online protocol.

### Future work (~150 w)
- Voltage / temperature / impedance contracts under the same regime framework.
- Streaming k-shot calibration with sequential CP.
- 5th held-out dataset to validate the regime classifier.
- Physics-informed adapters.

---

## 9. Conclusion — target 150 words

Four numbered findings (already drafted in `docs/MANUSCRIPT_POSITIONING.md` §"Manuscript-ready discussion paragraph" — adapt verbatim):

1. Cross-dataset transfer separates into three deterministic regimes.
2. Covariate alignment ≠ concept alignment.
3. k = 20 target calibration + MAPIE split CP repairs the salvageable regimes with valid coverage.
4. The regime taxonomy is backbone-agnostic.

One closing sentence on the practical recommendation.

---

## 10. Figures — paper-facing list

These exist in `outputs/` and can be inserted directly. See
[`data/intermediate/README.md`](../data/intermediate/README.md) for the
full manifest.

| # | Title in caption | File | Role |
|---|---|---|---|
| F1 | Pipeline schematic + 4-dataset summary | (build for paper — schematic + small table) | Methods overview |
| F2 | Within-dataset predictions vs truth (4 panels) | reuse from PyTorch CNN report or build new | §6.1 |
| F3 | Geometric shift — Mahalanobis heatmap raw vs capnorm | `outputs/results_v2_four_dataset_conditional_shift/four_dataset_conditional_shift_heatmaps.png` (top half) | §6.3 |
| F4 | **Key:** MMD drop vs R² change | (build new — 1 hour of matplotlib) | §6.3 |
| F5 | **Paper-defining:** four-dataset conditional-shift heatmap with regime labels | `outputs/results_v2_four_dataset_conditional_shift/four_dataset_conditional_shift_heatmaps.png` | §6.4 |
| F6 | k-shot scaling, R² vs k coloured by regime | `outputs/results_v2_four_dataset_kshot_scaling/paper_kshot_scaling.png` (.pdf available) | §6.5 |
| F7 | CP coverage and width | `outputs/results_v2_four_dataset_conformal/paper_cp_coverage_width.png` | §6.5 |
| F8 | SHAP top features × regime | `outputs/results_v2_four_dataset_shap/four_dataset_shap_feature_importance_top_features.png` | §6.7 |
| F9 (optional) | Regime-stratified CP | `outputs/results_v2_four_dataset_conformal/paper_cp_regime_stratified_90.png` | §6.5 inset |
| F10 | Directional asymmetry MATR↔HUST | `outputs/results_v2_conditional_shift/paper_directional_asymmetry_seed42.png` | §6.4 supporting |

### Supplementary figures (SI)
- LODO source-expert main panel (`outputs/results_v2_four_dataset_lodo_source_expert/paper_lodo_main_panel.png`)
- Kaplan-Meier four-dataset (`outputs/results_v2_four_dataset_survival/kaplan_meier_four_dataset.png`)
- DMD eigenvalue plane (`outputs/results_v2_four_dataset_koopman_dmd/dmd_eigenvalue_complex_plane.png`)
- DMD operator cross-prediction (`outputs/results_v2_four_dataset_koopman_dmd/dmd_operator_cross_prediction.png`)
- IWCP comparison (`outputs/results_v2_importance_weighted_cp/paper_iwcp_comparison_90.png`)
- All seven classical models × four datasets × two windows (full grid CSV)

---

## 11. Tables — paper-facing list

| # | Title in caption | Source | Role |
|---|---|---|---|
| T1 | Datasets — cells / chemistry / lifetime range / DOI | README dataset table | §5.1 reference |
| T2 | Within-dataset performance × 4 datasets, with pooled cluster-bootstrap CIs | `outputs/results_v2_four_dataset_within_34feat_log/results_summary.csv` | §6.1 |
| T3 | **Key:** Cross-dataset transfer matrix — 12 directions × {naive R², Pearson r, linear-cal R², regime} | `data/intermediate/four_dataset_conditional_shift_direction_summary.csv` | §6.2 + §6.4 |
| T4 | CP coverage and width — 4 scenarios × 12 directions at 90 % and 95 % | `outputs/results_v2_four_dataset_conformal/paper_cp_summary.csv` | §6.5 |
| T5 | **Positioning:** This work vs Severson / BatLiNet / EES-T / DCIR / Domain-Adaptive Transformer on (data contract × accuracy × CP validity × compute) | `docs/MANUSCRIPT_POSITIONING.md` §Pareto Table | §6.1 inset / Discussion |

### Supplementary tables (SI)
- Full 7-model × 4-dataset × 2-window grid
- Raw-vs-capnorm cross-transfer detailed comparison
- All-pairs MMD / Mahalanobis tables
- Per-feature slope-shift bootstrap rows
- LODO full sweep
- Four-dataset survival audit (RMST bootstrap CIs, pairwise log-rank/KS)

---

## 12. Submission checklist

### Pre-submission (must be done)
- [ ] **Title chosen** (§0).
- [ ] **Abstract finalized** (~250 w, §2).
- [ ] **All figure files exported at journal resolution** (TIFF/PDF/PNG, 600 dpi).
- [ ] **Figure 4 built** (MMD-drop-vs-R²-change scatter — does not yet exist).
- [ ] **Figure 1 built** (pipeline schematic — schematic-style diagram).
- [ ] **Pareto table T5 typeset** with current numbers; remove placeholder cells.
- [ ] **Zenodo deposit** with v1.0.0 tag; DOI inserted into `CITATION.cff` and the manuscript bibliography (`docs/ZENODO.md`).
- [ ] **`requirements-pinned.txt`** verified against the most recent committed runs.
- [ ] **CITATION.cff and LICENSE present** at repo root (already done).
- [ ] **Cover letter draft** (highlights for editor: 4-dataset benchmark, valid CP under shift, FAIR/open code).
- [ ] **Highlight bullets** (3–5, journal-house format).
- [ ] **Graphical abstract** (optional; reuse F5 if possible).

### Manuscript-side
- [ ] Cross-references to figures/tables resolved.
- [ ] Cite all references in §4 (BatLiNet, BatteryLife, HybridoNet-Adapt, Domain-Adaptive Transformer, EES-T, DCIR, Discovery Learning, Severson 2019, Ma et al. 2022, Geng et al. 2025, Ge et al. 2025).
- [ ] All numbers in the abstract and §6 match the committed CSVs to 3 decimals.
- [ ] LUH→MATR linear-adapter median-R² footnote in the CP section.
- [ ] Data and code availability statement points to GitHub + Zenodo DOI.

### Reviewer-anticipation
- [ ] Pre-empt "why no Transformer baseline?" — answered by §6.6 (PyTorch CNN backbone-agnostic check).
- [ ] Pre-empt "why no voltage features?" — answered by §6.1 + §6.3 + T5 (data-minimality Pareto).
- [ ] Pre-empt "rank-collapsed regimes are not solved" — explicit in §8 limitations (honest limitation; future work).

---

## 13. Open issues before submission

These are concrete TODOs that block "manuscript-ready" status. None of them
requires a new experiment; all are write-up / packaging items.

1. **F4 figure** — MMD-drop-vs-R²-change scatter does not exist as a paper-facing artefact. Estimated work: 2 hours of matplotlib + caption.
2. **F1 schematic** — pipeline diagram for §5/§6 transition. Could draw in PowerPoint or matplotlib. Estimated: 2–3 hours.
3. **T5 Pareto table typesetting** — currently a markdown table in `MANUSCRIPT_POSITIONING.md`; needs to be converted to LaTeX / Word with correct formatting.
4. **Zenodo deposit** — `docs/ZENODO.md` describes the GitHub-integration workflow; one-time setup, ~10 minutes.
5. **Abstract revision** — placeholder draft above is ~250 w; trim and sharpen before final submission.
6. **Cover letter + highlights** — 1-page cover letter + 3–5 highlight bullets (most journals require both).
7. **Graphical abstract (optional)** — reuse F5 (conditional-shift heatmap with regime labels) at 300 × 150 px target.
8. **Bibliography style** — match journal house style (Elsevier / IEEE / ACM as required).
9. **Author info** — ORCIDs, corresponding-author email, affiliations.
10. **Ethics / data availability statement** — boilerplate text pointing to the public datasets (DOIs already in README) and the MIT-licensed code.

---

## 14. Companion files

| File | Role |
|---|---|
| [`README.md`](../README.md) | Public landing page, datasets DOIs, How to cite |
| [`PROJECT_SUMMARY.md`](../PROJECT_SUMMARY.md) | Methodology decision log, SOP compliance, internal status |
| [`MANUSCRIPT_POSITIONING.md`](MANUSCRIPT_POSITIONING.md) | Pareto table, related work, main-vs-SI split, manuscript-ready discussion paragraph |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Single-page reproduction recipe |
| [`ZENODO.md`](ZENODO.md) | Zenodo deposit workflow |
| [`CITATION.cff`](../CITATION.cff) | Software citation metadata |
| [`presentation/battery_rul_defense.pptx`](../presentation/battery_rul_defense.pptx) | 26-slide thesis-defense deck mirroring this outline |
