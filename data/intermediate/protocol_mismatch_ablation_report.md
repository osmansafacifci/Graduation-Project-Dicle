# Protocol-Mismatch Ablation: MATR <-> HUST

Purpose: test whether the MATR<->HUST rank collapse disappears when using protocol-restricted source/target variants.

Important limitation: the committed HUST metadata does not contain a usable repeated exact constant-current subgroup. The exact all-equal discharge-rate subset has only three cells, so this is a medium-depth proxy ablation rather than a definitive matched-protocol experiment.

Model: Gaussian Process on the same N=100, 34-feature capacity-normalized table used by the four-dataset conditional-shift analysis; source train cells follow the official five source splits; target evaluation uses all uncensored cells in the selected target variant.

## Cohort Audit
| dataset | variant | cells | mean_cycle_life | std_cycle_life | min_cycle_life | max_cycle_life | definition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matr | full | 129.00 | 777.53 | 362.46 | 133.00 | 2066.00 | all uncensored N=100 cells |
| matr | restricted | 86.00 | 946.85 | 326.56 | 504.00 | 2066.00 | MATR restricted = b1+b3; batch2 excluded as short-life experimental block |
| hust | full | 77.00 | 1489.56 | 275.90 | 829.00 | 2024.00 | all uncensored N=100 cells |
| hust | restricted | 43.00 | 1501.07 | 235.80 | 1062.00 | 1870.00 | HUST restricted = discharge-rate spread <= 2; exact constant-current subset has only n=3 |
| hust | exact_constant_current_audit_only | 3.00 | 1562.33 | 216.63 | 1332.00 | 1762.00 | HUST cells with dchg_rate_1 == dchg_rate_2 == dchg_rate_3; too small for transfer |

## 2 x 2 Protocol-Restricted Transfer Summary
| direction | source_variant | target_variant | train_cells_mean | target_cells | R2_mean | pearson_r_mean | linear_adapter_R2_mean | life_ratio_target_over_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hust_to_matr | full | full | 53.000 | 129.000 | -3.600 | -0.114 | 0.014 | 0.480 |
| hust_to_matr | full | restricted | 53.000 | 86.000 | -2.523 | 0.020 | 0.003 | 0.615 |
| hust_to_matr | restricted | full | 29.400 | 129.000 | -3.734 | -0.081 | 0.007 | 0.473 |
| hust_to_matr | restricted | restricted | 29.400 | 86.000 | -2.650 | 0.077 | 0.011 | 0.608 |
| matr_to_hust | full | full | 89.000 | 77.000 | -7.937 | -0.120 | 0.025 | 2.085 |
| matr_to_hust | full | restricted | 89.000 | 43.000 | -11.545 | -0.135 | 0.027 | 2.112 |
| matr_to_hust | restricted | full | 58.800 | 77.000 | -4.809 | -0.085 | 0.018 | 1.625 |
| matr_to_hust | restricted | restricted | 58.800 | 43.000 | -7.090 | -0.093 | 0.014 | 1.646 |

## Interpretation
If protocol restriction repaired the mechanism, Pearson r should move from collapsed/negative to clearly positive in the restricted/restricted rows. In these outputs the MATR<->HUST rank signal remains weak or negative under the proxy restriction, so the result supports the manuscript's conditional-shift framing, with the caveat that exact protocol matching would require raw protocol metadata or additional cells.
