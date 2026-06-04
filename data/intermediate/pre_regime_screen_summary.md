# Pre-regime sensitivity screen (exploratory)

Leakage-aware sensitivity check for §3.4 super-regime taxonomy.

**Design.** 12 cross-dataset directions across 6 unordered pairs. Two screens (covariate-only and target-light), two models per screen (nearest centroid as primary, multinomial logistic as sensitivity). Cross-validation is leave-one-unordered-pair-out (6 folds × 2 test rows). Predictors are shift descriptors only; Pearson *r*, naive R², and linear-cal R² are excluded because they define the regime.

## Caveats

- *n* = 12 directions across only **6 unordered pairs**. This is   exploratory; we do not present it as a powered classifier.
- All three covariate-only descriptors are **pair-symmetric**   (MMD, Mahalanobis, discriminator AUC are unordered functions of   the two datasets), so the covariate-only screen is mathematically   incapable of producing different predictions for the two   directions of an asymmetric pair.
- `life_ratio_target_over_source` is the only direction-asymmetric   scalar; `slope_shifted_share` is pair-symmetric.
- Accuracy CIs are Clopper-Pearson exact binomial intervals.

## LOPO accuracy

| Screen | Model | n | Correct | Accuracy | 95 % CI (Clopper-Pearson) |
|---|---|---:|---:|---:|---|
| covariate_only | nearest_centroid | 12 | 3 | 0.250 | [0.055, 0.572] |
| covariate_only | logistic_l2 | 12 | 3 | 0.250 | [0.055, 0.572] |
| target_light | nearest_centroid | 12 | 4 | 0.333 | [0.099, 0.651] |
| target_light | logistic_l2 | 12 | 2 | 0.167 | [0.021, 0.484] |

## Pair symmetry sanity check

(true labels per direction; reveals 2 symmetric + 4 asymmetric pairs)

- **hust_vs_luh** (asymmetric): hust->luh → salvageable_linear_recovers, luh->hust → cp_interval_only
- **hust_vs_sandia** (asymmetric): hust->sandia → salvageable_linear_recovers, sandia->hust → offset_dominant_residual_only
- **matr_vs_hust** (symmetric): hust->matr → cp_interval_only, matr->hust → cp_interval_only
- **matr_vs_luh** (asymmetric): luh->matr → offset_dominant_residual_only, matr->luh → salvageable_linear_recovers
- **matr_vs_sandia** (asymmetric): matr->sandia → salvageable_linear_recovers, sandia->matr → cp_interval_only
- **sandia_vs_luh** (symmetric): luh->sandia → salvageable_linear_recovers, sandia->luh → salvageable_linear_recovers

## Spearman correlations of individual descriptors with Pearson r

Classifier-free sensitivity (n = 12 directions). Tests whether each descriptor is monotonically related to the rank signal.

| Descriptor | Spearman ρ | p |
|---|---:|---:|
| `MMD` | +0.042 | 0.896 |
| `Mahalanobis` | +0.184 | 0.568 |
| `discriminator_auc_mean` | -0.210 | 0.512 |
| `life_ratio_target_over_source` | -0.441 | 0.152 |
| `slope_shifted_share` | -0.071 | 0.827 |

## All LOPO predictions

Full per-fold predictions are in `data/intermediate/pre_regime_screen_lopo_predictions.csv`.

