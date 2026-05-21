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

Draft (placeholder, ~260 words; rewrite during final edit):

> Battery remaining-useful-life (RUL) prediction has converged to within-dataset accuracy near the literature ceiling on rich-signal feature contracts (voltage curves, surface temperature, internal impedance). Real deployments, however, train on lab data and apply on field data with different chemistry, protocol, or cell format. Our central question is not whether richer signals beat capacity-only features on accuracy — they do — but whether a low-instrumentation capacity-only contract can *diagnose* the type of cross-dataset transfer failure and *deliver coverage-valid uncertainty intervals* for the surviving directions. We assemble a four-dataset benchmark (MATR / HUST / Sandia / Luh-KIT, 362 modelled cells under a single 34-feature capacity-only contract) and evaluate twelve cross-dataset directions. We show that capacity normalisation closes 71 % of the geometric (Mahalanobis) gap without recovering any cross-dataset prediction R²; that an importance-weighted conformal-prediction probe is unusable under the observed source-target support mismatch (discriminator AUC 0.994–0.996, finite-interval fraction ≤ 0.9 % at 90 % nominal); and that a centred-log per-feature slope test plus a Pearson-r rank-signal diagnostic — computed from source predictions on a small labelled target probe — partition the twelve directions into three regimes (strong-rank / offset-dominant / rank-collapsed). On the two salvageable regimes, a two-parameter linear adapter fit on k = 20 random target cells plus MAPIE split conformal prediction recovers near-zero R² with finite intervals at empirical coverage 0.89–0.93 against 0.90 nominal. A small PyTorch 1D-CNN trained under the same inner-CV protocol preserves the qualitative regime structure and over-extrapolates on heterogeneous-source directions, indicating that — under this feature contract and these compact-model classes — the failure is governed by the rank-signal structure of the source/target pair rather than by the choice of backbone. Code, derived feature tables, and the reproducibility recipe are public; the framework requires neither voltage curves nor adversarial domain adaptation.

---

## 3. Introduction — target 1 000 words

Three paragraphs.

### ¶1 (~300 w) — Why RUL transfer matters
- Battery deployment in EVs / grid storage / aerospace requires reliable cycle-life estimates.
- Lab data ≠ field data; chemistries and protocols vary.
- Within-dataset RUL is essentially solved by deep learning (BatLiNet, Severson, EES-temperature) on voltage / temperature curves.
- Cross-dataset transfer remains poorly characterised — papers either fine-tune (HybridoNet-Adapt, Domain-Adaptive Transformer) or report ad hoc transfer numbers without analysing *why* a particular direction works or fails.

### ¶2 (~350 w) — The gap, framed as questions
- RQ1: Can the failure mode of a transfer direction be *diagnosed* from a combination of unsupervised source-target geometric statistics and a small labelled target probe, before committing to a full retraining or adaptation effort?
- RQ2: When transfer fails, what minimum target-side correction recovers usable point predictions *with valid uncertainty intervals*?
- RQ3: Does the qualitative answer survive a compact deep-learning baseline trained under the same protocol — or is it an artefact of the classical-regression lineup we tested?
- The community currently lacks (a) a diagnostic that separates a geometric (covariate-shift) explanation from a feature-y-relationship (conditional-shift) explanation in battery RUL, (b) a deployment protocol with coverage-valid cross-dataset prediction intervals, and (c) an empirical check on whether the same failure structure appears under both classical and compact-CNN backbones. We frame this as a *diagnosis-repair-uncertainty* triple rather than an architecture-search problem.

### ¶3 (~350 w) — Contributions (five bullets)
1. **Four-dataset, capacity-only benchmark.** MATR, HUST, Sandia, Luh/KIT under a single 34-feature contract derived only from Q_dis(c) — a low-instrumentation contract that every public battery archive satisfies.
2. **Two-layer shift-decomposition.** MMD + Mahalanobis (geometric layer); centred-log per-feature slope tests with BH-FDR (conditional layer). The geometric layer is reported alongside an importance-weighted conformal probe whose collapse to infinite intervals (finite-interval fraction ≤ 0.9 %) demonstrates that the observed source-target support mismatch makes a pure-covariate-shift repair unusable.
3. **Rank-signal regime taxonomy.** Twelve transfer directions partition into three regimes — strong-rank / offset-dominant / rank-collapsed — determined by Pearson r between source predictions and the target ground truth of a small labelled probe; the regime labels predict the post-calibration R² recoverable with k = 20 target cells.
4. **k-shot target calibration + split CP.** k = 20 random target cells + residual-mean / linear adapter + MAPIE split CP at 90 % / 95 %; coverage-valid and finite-interval everywhere on the salvageable regimes.
5. **Compact-CNN backbone check.** A small PyTorch 1D-CNN trained under the same inner-CV protocol preserves the qualitative regime structure across twelve directions; we observe an additional over-extrapolation failure mode on heterogeneous-source directions (Sandia → MATR R² ≈ −47.8). This is presented as a *backbone sanity check under matched protocol*, not as a proof that no deep architecture can rescue these regimes.

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

### 5.1 — Datasets (~300 w)
- [T1] table — MATR / HUST / Sandia / Luh-KIT counts, chemistry, lifetime range.
- 362 modelled cells, 19 censored at 0.85 · Q0 EOL threshold. **Censored cells are excluded from all regression metrics (MAE / sMAPE / R²) and retained only in the supplementary Kaplan-Meier / RMST audit.** This is the most conservative policy: censored cells contain no observed event time, so including them in regression would either require imputation (creates bias) or right-censored regression (changes the metric definition). The MATR audit shows that censoring is not the explanation for the dataset-level lifetime gap.
- Cite each dataset DOI.
- **Sandia is deliberately the heterogeneous-stress dataset.** Unlike MATR (single LFP/graphite chemistry under fast-charge), HUST (single LFP chemistry across multi-stage discharge protocols), and Luh/KIT (single NMC chemistry under standard cycling), the Sandia 0-100 % SOC subset mixes three chemistries (NCA / NMC / LFP), three temperatures (15 / 25 / 35 °C), and multiple discharge rates within a single 50-cell pool. This heterogeneity is part of the design — it lets us probe whether the rank-signal regime taxonomy still organises transfer correctly when one side of the pair is itself a mixture rather than a single condition. In §6.5 we report that target-side adaptation prefers the *linear* adapter over the *residual-mean* adapter when Sandia is the target, consistent with the heterogeneity hypothesis: the residual-mean adapter assumes a single constant target offset, while the linear adapter accommodates a per-direction scale change.

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

### 5.4 — Shift, regime, and uncertainty (~500 w)
- **Geometric shift.** MMD with RBF + median-bandwidth; Mahalanobis with pooled covariance + 1e-6 ridge. Reported per unordered pair × {raw, capnorm} × {12, 34} feature set.
- **Importance-weighted CP probe.** Cross-fitted logistic discriminator on source-vs-target labels, density-ratio weights ``p_target(X) / p_source(X)``, clipping at {5, 10, 20, ∞}. We report dataset-discriminator AUC, raw and clipped ESS / n, target-mass fraction, and the finite-interval fraction at 90 % nominal. **We frame the IW-CP probe as a falsifier, not as proof of concept shift.** A finite-interval fraction near zero is consistent with two causes (poor source-target support overlap, conditional shift in P(Y \| X), or both); we use it alongside the centred-log slope tests, the rank-signal r diagnostic, and the failure of capacity normalisation to repair R² as *jointly* arguing against a purely-covariate-shift explanation. No single one of these is sufficient.
- **Conditional shift.** Within-dataset z-score, within-dataset centring of log(cycle_life), univariate OLS slope, 1000-iteration paired bootstrap CI on the slope difference, BH-FDR across the 34 features. Universal log-life offset is reported separately (one scalar per pair) so it is not confounded with the per-feature slope.
- **Rank-signal classifier.** *Diagnostic, not pre-training oracle.* Computed from source-model predictions on a small labelled target probe (in our experiments the full uncensored target set; in deployment the same k = 20 cells used to fit the target adapter). Inputs: naive cross-dataset R², Pearson r between source predictions and target labels (with its 95 % bootstrap CI), the universal log-life offset, and the constant-share-of-SS decomposition. Partition into STRONG_RANK_SIGNAL (r ≥ 0.5 and naive R² ≥ 0.3), OFFSET_DOMINANT (r ≥ 0.5 but naive R² negative or near zero), RANK_COLLAPSED (r close to zero or its CI crosses zero). Thresholds are pre-registered in the analysis script; we report a per-direction summary in §6.4.
- **k-shot adapters.** Residual-mean (alpha = 1, beta = mean residual) and linear (alpha, beta from OLS) fit on a target *adapter set* of k_adapter cells, scored on the *test set* of the remaining target cells. We sweep k_adapter ∈ {5, 10, 15, 20} with 20 random draws per k for Monte Carlo stability.
- **Conformal prediction — split structure made explicit.** For each cross-dataset direction at each seed and each k we partition the target dataset into three disjoint subsets:
   1. **Adapter set**: ``k_adapter`` random target cells; used to fit the adapter (not used by MAPIE).
   2. **Calibration set**: a further ``k_target`` random cells, disjoint from the adapter set; used by MAPIE to compute the conformity-score quantile on the *adapter-corrected* predictions.
   3. **Test set**: the remaining target cells; used to report empirical coverage, mean / median interval width, finite-interval fraction, Winkler score, and short-vs-long-life stratified coverage.
   We use MAPIE's prefit pathway with the absolute conformity score and the finite-sample rank correction ``q = ceil((n_cal + 1)·(1 − α))``. Wilson 95 % intervals are reported alongside every empirical coverage value. Twenty random adapter+calibration draws give per-direction Monte Carlo variability; we aggregate to the paper-facing table by median across draws (mean is retained as an audit column in `results_summary.csv` because it is sensitive to occasional degenerate linear-adapter fits on small samples — see CP outlier note in `MANUSCRIPT_POSITIONING.md`).
   Coverage validity assumes exchangeability *between the adapter, calibration, and test cells within the target dataset*. This is plausible because the three subsets are drawn uniformly at random from a single dataset; it does *not* require exchangeability between source and target.

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

### 6.3 — Geometric shift is necessary but not sufficient (~550 w)
- [F3] Mahalanobis heatmap raw vs capnorm for the six pairs.
- 71 % Mahalanobis reduction with capacity normalisation (MATR↔HUST 12-feature: 13.10 → 3.75); MMD also drops (0.71 → 0.51).
- [F4] **KEY**: Mahalanobis-reduction-vs-ΔR² scatter — across 12 directions, large geometric reductions do not translate into monotone prediction-R² gains.
- IW-CP probe: cross-fitted logistic dataset discriminator reaches AUC 0.994–0.996 — *near-perfect separability*. Raw ESS / n drops to 0.55–0.59; finite-interval fraction at 90 % nominal is ≤ 0.9 %.

**Interpretation paragraph (precise; avoids the IW-CP-as-proof overclaim):**

> The near-perfect dataset discriminator AUC has two non-exclusive interpretations: (a) the source and target marginal feature distributions ``p_source(X)`` and ``p_target(X)`` have very limited overlapping support, and (b) the conditional ``P(Y \| X)`` differs between datasets. The IW-CP probe cannot distinguish (a) and (b) on its own — both produce degenerate density ratios and infinite intervals — and we therefore treat it as a *falsifier of the purely-covariate-shift hypothesis*, not as proof of conditional shift. Combined with the unchanged-or-worse R² after capacity normalisation (Figure 4) and the centred-log slope-shift counts in §6.4 (which directly test feature-y relationship changes), we read the joint evidence as inconsistent with a pure-covariate-shift explanation. Whether the residual gap is conditional shift, support mismatch, or both, the practical implication is the same: a target-side calibration step is needed.

Full IW-CP sweep (clipping table, ESS curves, infinite-interval fractions across confidence levels) deferred to SI.

### 6.4 — Conditional shift and regime taxonomy (~750 w)
- [F5] **PAPER-DEFINING**: conditional-shift heatmap (four-dataset slope-shift counts) with regime labels.
- Centred-log slope test recap; HUST↔MATR shifts 14/34 features after BH-FDR; universal log-life offset = 0.735 (HUST geometric mean ≈ 2.09× MATR).
- Direction-level classifier: Pearson r between source predictions and the labelled target probe. Tabulate r and post-calibration R² for the 12 directions.
- Three regimes with explicit thresholds (see §5.4): STRONG_RANK_SIGNAL, OFFSET_DOMINANT, RANK_COLLAPSED.
- [F10] supporting: directional-asymmetry scatter for the MATR↔HUST pair, seed = 42 — explicitly using CatBoost and Random Forest as the source models so the comparison is well-defined.

**Methodological note on model-family sensitivity (write explicitly to defuse a reviewer comment):**

> The naive-best source model for each direction is selected by mean R² across five seeds; in the four-dataset extension this picks Gaussian Process for both MATR ↔ HUST directions (because both directions are catastrophic for all seven classical models, GP happens to be least bad). The Pearson r values feeding the regime classifier are therefore computed with GP. **The same MATR ↔ HUST pair evaluated with CatBoost or Random Forest retains weak positive rank signal ``r ≈ 0.22–0.27`` instead of the slightly-negative GP value.** Both observations are reported. Operationally, the regime label is robust *to the sign of the slightly-negative GP signal*: in either case the bootstrap CI for r crosses zero, no source model produces a positive post-calibration R² with k = 20, and the MATR ↔ HUST pair sits in the RANK_COLLAPSED class. We therefore frame the classifier as agreeing across reasonable model choices on the *class label* even where r values differ in sign by chance.

**Why this matters for the deployment protocol:**

> In deployment, a practitioner does not commit to a single source model before evaluating the regime. The recommended flow is: (1) train candidate source models on the source dataset; (2) run each candidate on the k=20 labelled target probe; (3) compute (r, naive R², constant-share-of-SS) per candidate; (4) take the *best* candidate per direction by Pearson r and classify the regime. If the best r is < ~0.2 with a CI crossing zero, the direction is rank-collapsed regardless of which candidate produced it — no within-source model-family change rescues that direction in our experiments.

### 6.5 — k-shot calibration and conformal prediction (~600 w)
- [F6] k-shot scaling: R² vs k coloured by regime.
- [T4] CP table: 12 directions × 4 scenarios at 90 % and 95 %.
- [F7] CP coverage and width chart (existing `paper_cp_coverage_width.png`).
- [F_extra] regime-stratified CP figure (existing `paper_cp_regime_stratified_90.png`).
- Median R² aggregation footnote (linear adapter on small k_adapter occasionally degenerate).

### 6.6 — Compact-CNN backbone check (~350 w)
- CNN within-dataset numbers — competitive on Sandia / Luh, falls below classical on MATR / HUST.
- CNN cross-dataset numbers — same qualitative regime structure (Sandia ↔ Luh remain positive, MATR ↔ HUST remain catastrophic); additionally, the CNN exhibits a deep-model over-extrapolation failure mode when the source is heterogeneous (Sandia → MATR R² = −47.8, Sandia → HUST = −36.8).
- **Carefully bounded claim (write verbatim):**

  > Across the seven classical regressors and the small PyTorch 1D-CNN tested here, the rank-collapsed regime is not rescued by changing the model family. We do *not* claim that no neural architecture can solve these regimes — testing transformer-style or graph-based models, or fine-tuning a domain-adversarial network, are out of scope for this paper. The intended reading is that the qualitative regime structure is robust under matched-protocol changes of *backbone* across the eight architectures we evaluated, which is the relevant check for the diagnostic-and-calibration framework we propose.

### 6.7 — SHAP × regime closing loop (~250 w)
- [F8] SHAP top features per dataset.
- **Attribution models, stated explicitly.** TreeSHAP requires a tree ensemble. For MATR we use the within-dataset champion CatBoost; for HUST the champion Random Forest; for Sandia the champion XGBoost. For Luh / KIT the within-dataset champion is a Gaussian Process, which is *not* tree-SHAP compatible; we therefore use **CatBoost** as the Luh attribution model (CatBoost R² = 0.741 on Luh, the strongest tree-based runner-up to the GP champion). The SHAP report's "model-check" R² uses the canonical benchmark-helper path so it matches the headline benchmark numbers to 3 decimals.
- Cross-reference with regime labels: high-importance features for MATR (`accel_mean`, `poly2_c`, `slope_last_quarter`, `range_Qdis`) are mostly slope-shifted across the MATR ↔ HUST pair; high-importance Sandia features (`Qdis_N`, `mad_Qdis`, `slope_linear`) are slope-stable across the Sandia ↔ Luh pair. The SHAP × regime joined table makes this contrast paper-facing.
- One-sentence take-away: "*within-domain feature importance and cross-dataset feature reliability are different properties of a feature; the SHAP × regime join makes this distinction operational.*"

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

## 9. Conclusion — target 180 words

Four numbered findings (carefully hedged; matches `docs/MANUSCRIPT_POSITIONING.md` §"Manuscript-ready discussion paragraph" after the reviewer-anticipation revision):

1. Cross-dataset RUL transfer on a capacity-only feature contract organises into three regimes — strong-rank, offset-dominant, rank-collapsed — defined by source-prediction rank signal on a small labelled target probe.
2. Closing the geometric (Mahalanobis / MMD) source-target gap is necessary but not sufficient for prediction-R² recovery; the IW-CP probe failure and the centred-log slope-shift counts jointly rule out a purely covariate-shift explanation.
3. A k = 20 random-cell target probe + a two-parameter linear adapter + MAPIE split CP recovers near-zero R² with finite intervals at coverage 0.89–0.93 on the salvageable regimes; rank-collapsed regimes are not rescued by k-shot or by the compact CNN backbone we tested.
4. The qualitative regime structure is preserved across seven classical regressors and one compact PyTorch 1D-CNN trained under matched inner-CV protocol; we expect rather than claim this to generalise to larger architectures, and we explicitly defer that test.

Closing sentence: *"Quantify the shift, classify the regime on a small probe, recalibrate. No voltage curves, no adversarial domain adaptation, valid uncertainty by construction."*

---

## 9.5 Reviewer-anticipation map — internal use, not for the paper

This subsection is for the writing team only; do *not* include it in the
manuscript. It maps each of the seven likely Reviewer #2 attack points to
the specific outline section + wording mitigation. Use it as a checklist
during writing and during cover-letter drafting.

| Risk | Where mitigated | Mitigation wording (verbatim) |
|---|---|---|
| 1. "Predict before training" overclaim | §2 abstract, §3 RQ1, §5.4 rank-signal classifier, §6.4 deployment-flow note | Use *"diagnose from source-target geometric statistics + a small labelled target probe"*. Never claim a fully unsupervised pre-training classifier — the regime label uses Pearson r against labelled target cells. |
| 2. HUST → MATR appears inconsistent across sources | §6.4 model-sensitivity note | Acknowledge upfront: the four-dataset best-model column picks GP; CatBoost / RF retain weak positive r. Report both; argue the *class label* (RANK_COLLAPSED) is robust to the sign of slightly-negative GP r because the CI crosses zero in every case. |
| 3. "No architecture helps" | §6.6 bounded-claim paragraph, §9 finding 3, §9 finding 4 | Use *"none of the tested classical or compact-CNN backbones rescues rank-collapsed transfer"*. Do not generalise to transformers / GNNs / domain-adversarial. |
| 4. CP validity is only valid under exchangeability | §5.4 split-structure paragraph, §6.5 caption | State the three target-side subsets (adapter / calibration / test) and that exchangeability is required *within the target dataset only*, not between source and target. |
| 5. IW-CP as falsifier, not proof | §5.4 IW-CP probe paragraph, §6.3 interpretation paragraph | Explicitly: the AUC-near-one collapse is consistent with poor support overlap or with conditional shift; we use IW-CP *jointly* with capnorm-fails-to-fix-R² + slope-shift counts to rule out the purely-covariate-shift explanation. |
| 6. TreeSHAP × GP wording | §6.7 attribution-models paragraph | Luh CatBoost is the TreeSHAP-compatible attribution model because the GP champion is not tree-based. |
| 7. Capacity-only accuracy below SOTA | §2 abstract, §3 ¶1, §6.1 Pareto table T5 | Frame the contract as a deliberate Pareto choice — low instrumentation + valid CP. We do not claim to beat BatLiNet / EES-T / DCIR on accuracy; we claim to add valid uncertainty and diagnostic structure they do not provide. |
| 8. Sandia heterogeneity | §5.1 Sandia paragraph, §6.5 linear-adapter preference | Pre-empt: Sandia is the deliberate heterogeneous-stress dataset. The linear adapter outperforms residual-mean when Sandia is the target — this is the expected outcome and we report it. |

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
| F4 | **Key:** Mahalanobis reduction vs ΔR² scatter (12 directions, regime-coloured, MATR↔HUST highlighted) | `outputs/paper_figures/paper_covariate_vs_concept.png` (.pdf, .csv) | §6.3 |
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
- [x] **Figure 4 built** — `outputs/paper_figures/paper_covariate_vs_concept.png` (built 2026-05-16 via `3_analysis/plot_covariate_vs_concept_scatter.py`).
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
