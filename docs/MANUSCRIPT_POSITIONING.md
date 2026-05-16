# Manuscript Positioning Notes

This page is the paper-facing synthesis layer. It intentionally separates the
main manuscript claim from supporting analyses so the repository does not read
like a long list of experiments.

## Recommended Claim

**Main title direction:** *Covariate alignment is not concept alignment: a
four-dataset study of early-cycle battery lifetime transfer with k-shot
calibration and conformal prediction.*

**One-sentence contribution:** Under a fixed capacity-only feature contract,
within-dataset learning works, naive cross-dataset transfer fails in
regime-specific ways, covariate alignment alone is insufficient, and a small
target calibration set can repair point estimates and conformal interval
validity when the source model retains rank signal.

The paper should not be framed as "a new battery RUL model." The safer and
stronger framing is a **diagnostic and reliability protocol**:

1. Fix the data contract and reproduce within-dataset baselines.
2. Quantify cross-dataset failure under the same contract.
3. Decompose failure into geometric shift, feature-slope shift, and retained
   rank signal.
4. Test the principled covariate-shift CP baseline and show why it becomes
   unusable under support mismatch.
5. Use k-shot target calibration plus standard MAPIE split CP to recover
   finite, near-nominal intervals.
6. Add leave-one-dataset-out source pooling/selection as the deployment
   protocol, with full details in SI.

## Related-Work Positioning

| Work | What it establishes | How this manuscript differs |
|---|---|---|
| [Severson et al., Nature Energy 2019](https://www.nature.com/articles/s41560-019-0356-8) | Early-cycle lifetime prediction can be accurate when discharge voltage-curve features are available; reported strong first-100-cycle prediction on MATR. | We deliberately use a simpler capacity-only contract and focus on cross-dataset failure, calibration, and uncertainty, not on matching voltage-curve accuracy. |
| [BatLiNet, Nature Machine Intelligence 2025](https://www.nature.com/articles/s42256-024-00972-x) | Deep inter-cell learning improves lifetime prediction across diverse aging conditions and uses pairwise reference-cell information. | We are smaller and less ambitious architecturally, but more diagnostic: the contribution is shift decomposition, rank-signal regimes, k-shot calibration, and conformal reliability under transfer. |
| [BatteryLife benchmark, arXiv 2025](https://arxiv.org/abs/2502.18807) | Large-scale benchmark across many datasets, chemistries, formats, temperatures, and protocols; tests many time-series baselines. | We are not trying to be the broadest benchmark. We contribute a controlled four-dataset transfer audit with a fixed feature contract and paper-ready uncertainty protocol. |
| [Domain-adversarial battery lifetime prediction, Renewable & Sustainable Energy Reviews 2025](https://www.sciencedirect.com/science/article/pii/S1364032124007615) | CNN/attention feature extractors plus domain adversarial training can improve unsupervised cross-domain lifetime prediction. | We explicitly test whether covariate/domain alignment is enough. Our results show alignment can reduce geometric shift without fixing prediction, motivating target-side calibration. |
| [HybridoNet-Adapt, PLOS One 2025](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0335066) | MMD-based domain adaptation can improve RUL prediction when source/target distributions differ. | We use MMD as a diagnostic and falsifier rather than a headline model: feature distribution alignment is measured, but rank signal and conditional shift decide whether transfer is useful. |
| [Domain-Adaptive Transformer, Energy 2025](https://ideas.repec.org/a/eee/energy/v341y2025ics0360544225049308.html) | Transformer-style cross-domain SOH/RUL models can exploit richer V/Q/differential trajectories with fine-tuning. | Our approach is intentionally lighter: capacity-only features, transparent shifts, and target-calibrated uncertainty. A 1D-CNN baseline can serve as the deep-learning sanity check, not the main contribution. |
| [Conformal RUL intervals, arXiv 2022](https://arxiv.org/abs/2212.14612) | CP gives distribution-free RUL intervals for generic point predictors under exchangeability. | We apply standard split CP to battery lifetime transfer and expose the practical failure mode: source-calibrated CP under dataset shift under-covers, while target-calibrated CP restores finite near-nominal coverage. |
| [Deep Koopman battery RUL, Journal of Energy Storage 2025](https://www.sciencedirect.com/science/article/pii/S2352152X25010825) | Koopman-inspired learned dynamics can be used as a predictive model under multi-condition battery scenarios. | Our DMD/Koopman analysis is not a new predictor. It should stay as supporting dynamics evidence that early capacity trajectories carry dataset-specific operators. |

## Pareto Positioning

| Axis | High-accuracy deep models | Broad benchmarks | This manuscript |
|---|---|---|---|
| Data contract | Often uses voltage, current, temperature, or V-Q maps | Varies across tasks and datasets | Fixed capacity-only features for all datasets |
| Dataset breadth | Often one to several curated tasks | Very broad | Four public datasets, all run through one contract |
| Main objective | Lowest RMSE/MAE | Benchmark coverage | Explain and repair transfer failure |
| Transfer treatment | Domain adaptation or pairwise/deep transfer | Cross-domain evaluation | Covariate shift, conditional shift, rank-signal regimes |
| Target labels | Often none or fine-tuning labels | Varies | Explicit k-shot policy: k in {5, 10, 15, 20} |
| Uncertainty | Usually absent or model-specific | Usually secondary | Standard MAPIE split CP with Wilson CI and size-stratified coverage |
| Reviewer hook | Strong model performance | Dataset scale | Reliability protocol under realistic dataset shift |

This table is the defense if the paper is challenged for not chasing the
largest deep architecture. The answer is: the contribution is orthogonal to
architecture, and the pipeline can wrap any point predictor.

## What Goes in Main Text vs SI

| Component | Main text | SI / appendix |
|---|---|---|
| Four-dataset within/cross metrics | Yes, compact table and transfer matrix | Full model table |
| Geometric shift and raw-vs-capnorm | Yes, one figure/table | Full per-pair MMD/Mahalanobis tables |
| Conditional-shift/rank regimes | Yes, central figure | Full per-feature slope tables |
| SHAP x regime | Yes, one compact table | Full SHAP feature rankings |
| k-shot target calibration | Yes, k=20 plus scaling curve | Full k/model/adapter sweep |
| Conformal prediction | Yes, regime-stratified coverage/width figure | Full CP scenario tables, 90/95, stratified coverage |
| LODO source-expert protocol | One main panel | Full protocol-family and k sweep tables |
| Koopman/DMD | One supporting paragraph | Full spectra/operator-transfer figures and tables |
| Stacking | SI sensitivity unless it is the best single cell in a required table | Full model lineup |
| 1D-CNN baseline | One short robustness sentence or compact SI table | Training details, source-range clipping, and seed variability |

## Stacking Decision

Keep stacking in the computed model lineup because it is already reproducible
and sometimes wins a sensitivity table, especially in LODO. Do not make it a
headline model. It is an engineering ensemble that can blur the clean
interpretability story; single-model champions and the source-expert protocol
are easier to defend in the main text.

## Koopman/DMD Decision

Keep Koopman/DMD as a supporting dynamics diagnostic. The useful sentence is:

> A Hankel-DMD pilot shows that early capacity trajectories carry
> dataset-specific dynamical signatures, supporting the conditional-shift
> interpretation, but the manuscript does not rely on DMD as a predictive
> model.

This avoids overclaiming novelty against recent Koopman-based RUL papers while
still giving CS/EE reviewers a systems/dynamics explanation for why naive
transfer breaks.

## 1D-CNN Baseline Decision

The 1D-CNN baseline is complete under the same four-dataset split protocol.
It uses early `Q_discharge/q0` trajectories and their first differences with a
small PyTorch Conv1D-x2 model. Hyperparameters (filters, learning rate) are
selected via inner 5-fold CV on the train split, mirroring the XGBoost /
CatBoost CV grid. Cluster bootstrap by `cell_id` is used for the pooled
confidence intervals so cross-dataset CIs are not inflated by repeated target
cells across seeds.

Outcome (5 seeds, full-grid HP search):

| Dataset / direction | Result | vs. classical headline |
|---|---|---|
| Sandia within | R² = 0.881 [0.754, 0.950] | slightly below XGBoost 0.940 |
| Luh/KIT within | R² = 0.761 [0.624, 0.857] | comparable to Gaussian Process 0.769 |
| MATR within | R² = 0.305 [-0.250, 0.587] | below CatBoost 0.575 |
| HUST within | R² = -0.174 [-0.440, -0.028] | fails; far below Random Forest 0.340 |
| Naive cross — positive | Sandia↔Luh stay positive (Sandia→Luh R² = 0.343, Luh→Sandia R² = 0.314); MATR→Sandia near zero | Same regime-positive directions as the classical lineup |
| Naive cross — collapsed | MATR↔HUST stay deeply negative (MATR→HUST R² = -6.2, HUST→MATR R² = -3.8) | Same regime-collapsed directions as the classical lineup |
| Naive cross — over-extrapolation | Sandia source extrapolates badly to MATR/HUST (Sandia→MATR R² = -47.8, Sandia→HUST R² = -36.8) | Classical was already negative here, but CNN is dramatically worse |

Decision: keep this as reviewer armor, not as a main contribution. Two
manuscript-facing takeaways:

1. **Rank-signal regime taxonomy is backbone-agnostic.** The CNN preserves the
   classical taxonomy qualitatively — strong-rank pairs (Sandia↔Luh) stay
   positive, rank-collapsed pairs (MATR↔HUST) stay catastrophic. This directly
   answers the obvious reviewer question "but maybe a deep model breaks your
   regime story?".
2. **Deep architecture does not buy cross-dataset robustness on the
   capacity-only contract.** The CNN over-extrapolates when source-target
   shift is large (Sandia source → MATR/HUST). The (data minimality ×
   within-dataset accuracy × CP validity) Pareto positioning therefore still
   favors the classical lineup with target-side calibration.
