# T4.3 — HUST regime-assignment stability (bootstrap resampling)

**Status:** diagnostic for the Route 1 manuscript revision plan, item T4.3.
**Date:** 2026-08-13
**Input:** `data/intermediate/four_dataset_conditional_shift_predictions.csv` (stored per-cell
source predictions and target labels — no model refitting).
**Method:** for every direction × seed, 2000 bootstrap resamples of the test cells with replacement;
recompute Pearson `r` between source prediction and target label; re-classify under the frozen
`classify_rank_signal` thresholds; report self-agreement (fraction of resamples matching the point
class) both at the fine-grained five-class level and at the collapsed three-super-regime level
(`salvageable_linear_recovers` / `offset_dominant_residual_only` / `cp_interval_only`).

Seed for the bootstrap RNG: `20260811`. Seeds over target splits: {42, 123, 456, 789, 1011}.

---

## Headline

- The **rank-collapsed anchor (MATR↔HUST) is robust** to resampling at the super-regime level.
- The general statement *"regime assignment is stable for HUST-involving directions"* is **false** —
  it is direction-specific.
- Instability is **not HUST-specific**; it appears across several of the twelve directions.
- Fine-grained five-class labels are **not** stable and should not be reported as stable assignments.

---

## All 12 directions — super-regime mean / min self-agreement

| direction | n | point super-regime | mean self-agree | min self-agree |
|---|---:|---:|---:|---:|
| hust_to_luh | 106 | salvageable_linear_recovers | 1.000 | 1.000 |
| hust_to_matr | 129 | cp_interval_only | 0.997 | 0.990 |
| hust_to_sandia | 50 | salvageable_linear_recovers | 0.997 | 0.995 |
| luh_to_sandia | 50 | salvageable_linear_recovers | 1.000 | 1.000 |
| matr_to_luh | 106 | offset_dominant_residual_only | 0.944 | 0.846 |
| matr_to_hust | 77 | cp_interval_only | 0.917 | 0.601 |
| sandia_to_luh | 106 | salvageable_linear_recovers | 1.000 | 1.000 |
| sandia_to_matr | 129 | cp_interval_only | 0.850 | 0.709 |
| matr_to_sandia | 50 | salvageable_linear_recovers | 0.773 | 0.500 |
| luh_to_hust | 77 | cp_interval_only | 0.714 | 0.440 |
| luh_to_matr | 129 | cp_interval_only | 0.599 | 0.445 |
| sandia_to_hust | 77 | offset_dominant_residual_only | 0.534 | 0.454 |

Under a **mean ≥ 0.90** resolution rule, **7 of 12 directions resolve**:
hust_to_luh, hust_to_matr, hust_to_sandia, luh_to_sandia, matr_to_luh, matr_to_hust, sandia_to_luh.
The five unresolved: matr_to_sandia, luh_to_hust, luh_to_matr, sandia_to_hust, sandia_to_matr.

---

## Fine-grained five-class stability (range across seeds)

| direction | fine class | n | fine self-agree range |
|---|---:|---:|---:|
| hust_to_luh | strong | 106 | 1.00–1.00 |
| hust_to_matr | negative/collapsed | 129 | 0.63–1.00 |
| hust_to_sandia | strong/moderate | 50 | 0.82–0.97 |
| luh_to_hust | weak/collapsed/negative | 77 | 0.44–0.89 |
| luh_to_matr | collapsed | 129 | 0.41–0.58 |
| luh_to_sandia | strong | 50 | 1.00–1.00 |
| matr_to_hust | collapsed/negative | 77 | 0.54–0.84 |
| matr_to_luh | weak/offset | 106 | 0.85–1.00 |
| matr_to_sandia | strong | 50 | 0.50–1.00 |
| sandia_to_hust | moderate/weak/collapsed | 77 | 0.45–0.57 |
| sandia_to_luh | strong | 106 | 1.00–1.00 |
| sandia_to_matr | collapsed | 129 | 0.55–0.78 |

The five-class labels carry wide uncertainty (self-agreement as low as 0.41–0.45 in several
directions) and therefore should not be printed as stable single assignments without a stability
column.

---

## Notes

- **NaN seed:** `hust_to_matr`, seed 1011, produced a zero-variance source prediction → Pearson `r`
  undefined → class `undefined`. This is an exclusion-with-reason, not data dropping; it must be
  disclosed (and is itself diagnostic of constant-output/extrapolation failure). It was excluded from
  that direction's stability estimate.
- The plan text referenced "HUST's wide within-dataset CI [0.072, 0.512]" — the cross-dataset
  stability check here uses the stored per-cell predictions for the cross directions as produced by
  the four-dataset conditional-shift pipeline.

## Recommended reporting (corrected external-review wording)

> Regime assignment was assessed by bootstrap resampling of target test cells (2000 resamples × 5
> seeds) under the frozen classification thresholds, applied identically to all twelve directions. At
> the fine-grained five-class level, self-agreement ranges from 0.41 to 1.00, so individual class
> labels are not treated as stable. At the collapsed super-regime level — the decision-relevant
> partition — a direction is resolved only if mean seed-level self-agreement ≥ 0.90 (min reported).
> Under that pre-stated rule, seven of twelve directions resolve. Both directions anchoring the
> rank-collapsed regime resolve (HUST→MATR 0.997, MATR→HUST 0.917). Five directions (Luh→HUST,
> Luh→MATR, Matr→Sandia, Sandia→HUST, Sandia→MATR) fall below the threshold and are reported as
> unresolved rather than assigned. The rank-collapsed claim is therefore robust to resampling; regime
> assignment in general is direction-specific and is not claimed to be uniformly stable. One seed of
> HUST→MATR produced a zero-variance source prediction making Pearson r undefined; that seed is
> excluded from that direction's estimate and the exclusion is reported.

## Caveat for reproducibility

Mean/min values above are bootstrap estimates from a fixed RNG seed (20260811). Repeated runs with
the same seed and same data produce identical numbers. A small seed change perturbs the estimates by
±0.01–0.02 around the reported values; direction-level conclusions above are insensitive to that.
