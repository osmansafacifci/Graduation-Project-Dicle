# Legacy Archive

This directory preserves the student's original notebooks, earlier modeling
scripts, old plots, and external reference code used while developing the
SOPv2 pipeline.

The files here are kept for provenance only. They are not part of the active
reproducible workflow described in the root `README.md`.

## Layout

- `data_prep/` and `feature_engineering/`: earlier data extraction and feature
  scripts that predate the corrected SOPv2 definitions.
- `modeling/`: earlier modeling, reporting, conformal, and cross-dataset
  experiments kept for traceability.
- `notebooks/`: exploratory notebooks from the original project.
- `plots_modeling/` and `plots_root/`: archived plot outputs from earlier
  runs.
- `reference/`: external Severson/Nature Energy reference code.
- `splits/`: old split files superseded by `splits/sop_v2/`.
- `presentation.pptx`: archived project presentation.

Use the active root-level pipeline instead:

```bash
python run_pipeline.py --status
python run_pipeline.py --phase model
python run_pipeline.py --phase analysis
```
