# Four-Dataset Survival/Censoring Audit

Protocol: N=100 cell table; uncensored cells are EOL events; censored cells are right-censored at last observed cycle.

## Dataset Summaries
| dataset | n_cells | n_events | n_censored | event_mean_cycles | event_median_cycles | lower_bound_mean_cycles | lower_bound_median_cycles | km_median_cycles | rmst_tau_cycles | rmst_cycles | survival_at_500 | survival_at_1000 | survival_at_1500 | survival_at_2000 | max_time_cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hust | 77.000 | 77.000 | 0.000 | 1489.558 | 1513.000 | 1489.558 | 1513.000 | 1513.000 | 1615.000 | 1429.104 | 1.000 | 0.974 | 0.506 | 0.026 | 2024.000 |
| luh | 108.000 | 106.000 | 2.000 | 558.991 | 449.500 | 569.287 | 479.500 | 508.000 | 1615.000 | 573.073 | 0.500 | 0.181 | 0.038 | 0.000 | 1615.000 |
| matr | 135.000 | 129.000 | 6.000 | 777.535 | 740.000 | 802.200 | 773.000 | 773.000 | 1615.000 | 794.238 | 0.704 | 0.236 | 0.081 | 0.024 | 2237.000 |
| sandia | 61.000 | 50.000 | 11.000 | 762.020 | 281.000 | 1272.475 | 305.000 | 305.000 | 1615.000 | 711.197 | 0.344 | 0.344 | 0.344 | 0.328 | 4551.000 |

## Pairwise Tests
| group_a | group_b | logrank_chi2 | logrank_p_value | ks_events_statistic | ks_events_p_value | ks_lower_bound_statistic | ks_lower_bound_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matr | hust | 61.189 | 0.000 | 0.827 | 0.000 | 0.818 | 0.000 |
| matr | sandia | 2.198 | 0.138 | 0.753 | 0.000 | 0.611 | 0.000 |
| matr | luh | 12.110 | 0.001 | 0.406 | 0.000 | 0.400 | 0.000 |
| hust | sandia | 0.965 | 0.326 | 0.800 | 0.000 | 0.656 | 0.000 |
| hust | luh | 124.664 | 0.000 | 0.833 | 0.000 | 0.826 | 0.000 |
| sandia | luh | 8.150 | 0.004 | 0.391 | 0.000 | 0.335 | 0.000 |

## Censored Cells
| dataset | cell_id | censor_time_cycles | q0 | last_positive_cycle | last_positive_qdis | last_positive_retention | min_positive_cycle | min_positive_qdis | min_positive_retention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| luh | luh_P037_2_S06_C09 | 666.000 | 2.940 | 666.000 | 2.551 | 0.868 | 666.000 | 2.551 | 0.868 |
| luh | luh_P049_2_S11_C09 | 1564.000 | 2.952 | 1564.000 | 2.528 | 0.856 | 1564.000 | 2.528 | 0.856 |
| matr | matr_b1c8 | 878.000 | 1.092 | 878.000 | 0.969 | 0.888 | 878.000 | 0.969 | 0.888 |
| matr | matr_b1c22 | 890.000 | 1.081 | 890.000 | 0.963 | 0.891 | 890.000 | 0.963 | 0.891 |
| matr | matr_b1c13 | 896.000 | 1.081 | 896.000 | 0.923 | 0.854 | 896.000 | 0.923 | 0.854 |
| matr | matr_b1c10 | 905.000 | 1.077 | 905.000 | 0.961 | 0.893 | 905.000 | 0.961 | 0.893 |
| matr | matr_b3c23 | 2189.000 | 1.067 | 2189.000 | 0.937 | 0.878 | 2189.000 | 0.937 | 0.878 |
| matr | matr_b3c32 | 2237.000 | 1.073 | 2237.000 | 0.974 | 0.907 | 2237.000 | 0.974 | 0.907 |
| sandia | sandia_SNL_18650_LFP_25C_0_100_0_5_1C_c | 3038.000 | 1.052 | 3038.000 | 0.973 | 0.925 | 3003.000 | 0.969 | 0.921 |
| sandia | sandia_SNL_18650_LFP_25C_0_100_0_5_1C_d | 3038.000 | 1.051 | 3038.000 | 0.973 | 0.926 | 3028.000 | 0.968 | 0.921 |
| sandia | sandia_SNL_18650_LFP_25C_0_100_0_5_0_5C_a | 3050.000 | 1.070 | 3050.000 | 0.964 | 0.901 | 2888.000 | 0.962 | 0.899 |
| sandia | sandia_SNL_18650_LFP_25C_0_100_0_5_2C_a | 3544.000 | 1.037 | 3544.000 | 0.950 | 0.916 | 3522.000 | 0.939 | 0.906 |
| sandia | sandia_SNL_18650_LFP_25C_0_100_0_5_1C_a | 3545.000 | 1.034 | 3545.000 | 0.944 | 0.913 | 3515.000 | 0.936 | 0.905 |
| sandia | sandia_SNL_18650_LFP_15C_0_100_0_5_1C_b | 3553.000 | 1.043 | 3553.000 | 0.972 | 0.932 | 3538.000 | 0.969 | 0.929 |
| sandia | sandia_SNL_18650_LFP_25C_0_100_0_5_1C_b | 3636.000 | 1.038 | 3636.000 | 0.943 | 0.908 | 3585.000 | 0.940 | 0.906 |
| sandia | sandia_SNL_18650_LFP_15C_0_100_0_5_2C_b | 3754.000 | 1.031 | 3754.000 | 0.933 | 0.905 | 3747.000 | 0.913 | 0.885 |
| sandia | sandia_SNL_18650_LFP_15C_0_100_0_5_2C_a | 3761.000 | 1.033 | 3761.000 | 0.942 | 0.912 | 3736.000 | 0.920 | 0.891 |
| sandia | sandia_SNL_18650_LFP_25C_0_100_0_5_2C_b | 4050.000 | 1.032 | 4050.000 | 0.940 | 0.911 | 2787.000 | 0.880 | 0.853 |
| sandia | sandia_SNL_18650_LFP_15C_0_100_0_5_1C_a | 4551.000 | 1.033 | 4551.000 | 0.947 | 0.917 | 4338.000 | 0.924 | 0.894 |

Interpretation: Sandia and Luh introduce additional censoring checks, but the main four-dataset modeling rule remains unchanged: censored cells are excluded from MAE/sMAPE/R2 regression metrics and retained here as right-censored observations.
