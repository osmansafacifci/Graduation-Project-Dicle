# P0 second-life grading gate report

Locked analysis decision: **STOP**  
Permitted claim scope: **virtual pack construction structurally eligible**  
Lock SHA-256: `55b30b2aacd19e87ee96127a3ce9f5db16fcda4b09248382df44f2089e6cc301`

| Gate | Question | Decision |
|---:|---|---|
| 0 | Are the cohorts structurally usable? | **GO** |
| 1 | Is there enough oracle value to justify prediction? | **GO** |
| 2 | Does prospective grading clear the locked utility tests? | **STOP** |
| 3 | Does the frozen grader confirm on genuine second-life data? | **NOT_RUN** |

## Evidence

Gate 0 cohort counts:

| Dataset | Primary cells | Follow-up | Cell-level | Pack structure |
|---|---:|---:|---|---|
| luh | 84 | 94.4% | pass | pass |
| hust | 77 | 100.0% | pass | fail |
| sandia | 38 | 86.4% | fail | fail |
| matr | 0 | 0.0% | fail | fail |

Gate 1 used 84 LUH cells. The signal-to-instability ratio was 16.33; oracle within-duty pairing reduced the median future-degradation mismatch by 61.4%, with 98.8% positive resamples.

Gate 2 selected **catboost**. Development MAE improved by 27.3% and the point estimate for pairing benefit was 26.8%, but only 35.0% of resamples were positive (locked requirement: 75%).
HUST replication MAE changed by -13.5% (locked non-inferiority limit: -10%).

Gate 3 was **NOT_RUN**: Stopped prospectively because Gate 2 returned STOP.

## Interpretation

The data support the existence of decision-relevant heterogeneity near retirement, but the current history features and locked candidate models do not estimate it reliably enough for a second-life grading claim. The STOP is not evidence that second-life grading is impossible; it is evidence that the present operationalization should not yet anchor a manuscript pivot or justify consuming the genuine second-life confirmation dataset.

All detailed decisions, predictions, cohort audits, model artifacts, and environment provenance are stored beside this report.
