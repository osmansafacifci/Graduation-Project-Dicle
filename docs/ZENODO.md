# Zenodo deposit workflow

This document captures the one-time setup and the recurring release procedure
for archiving the repository to Zenodo, which gives the software a persistent
DOI suitable for manuscript citation and long-term archival.

## One-time setup (do this once per repository)

1. Sign in to Zenodo with the GitHub account that owns the repository:
   <https://zenodo.org/account/settings/github/>
2. On the GitHub-integration page, flip the toggle for
   `osmansafacifci/Graduation-Project-Dicle` to **ON**.
3. Verify that [`CITATION.cff`](../CITATION.cff) lists all four authors with
   the correct family/given names. Zenodo reads this file for the deposit
   metadata.
4. Verify that [`LICENSE`](../LICENSE) is present at the repository root. The
   MIT identifier in `CITATION.cff` must match the licence file.
5. Optionally pre-fill the Zenodo deposit metadata in advance via the
   `.zenodo.json` file (this overrides `CITATION.cff` when present). For our
   manuscript release we rely on `CITATION.cff`; no `.zenodo.json` is needed.

## Recurring: cut a release

Every time we want a new persistent DOI (typically on each manuscript
revision and on every major analytical milestone):

```bash
# 1. Ensure main is clean and pushed
git status
git push origin main

# 2. Tag a release. Use semantic versioning:
#    - v1.0.0 = manuscript submission
#    - v1.1.0 = revisions after first review
#    - v2.0.0 = a second study using this codebase
git tag -a v1.0.0 -m "Manuscript v1.0 (Demir, Çoban, Sarp, Çifçi 2026)"
git push origin v1.0.0

# 3. Within ~2 minutes, Zenodo creates a new deposit with metadata pulled from
#    CITATION.cff. It mints a versioned DOI (e.g., 10.5281/zenodo.XXXXXXX) AND
#    a "concept DOI" that always resolves to the latest version.
```

## After the deposit

1. Copy the version-specific DOI from your Zenodo dashboard.
2. Add it to [`CITATION.cff`](../CITATION.cff) (`doi:` field + `identifiers`
   block at the bottom of the file).
3. Add a DOI badge to the top of [`README.md`](../README.md):
   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```
4. Update the manuscript bibliography to cite the **concept DOI** (not the
   version-specific one) so that future readers always reach the latest
   archived release.
5. Commit and push the updates as a `chore: add Zenodo DOI` commit. This
   commit will not itself be archived (it is post-tag), which is fine — the
   Zenodo record points to the tagged snapshot.

## What gets archived

Zenodo archives **the tag-time state of the repository**, including:

- All source code under `0_data/ 1_features/ 2_models/ 3_analysis/`
- All committed CSVs and JSONs under `data/intermediate/` and `outputs/`
- All Markdown documentation under `docs/`, `README.md`, `PROJECT_SUMMARY.md`
- [`LICENSE`](../LICENSE), [`CITATION.cff`](../CITATION.cff),
  `requirements.txt`, `requirements-pinned.txt`

Zenodo does **not** archive:

- Anything under `.gitignore` (including `data/raw/` and `.venv/`)
- Files larger than ~50 GB per file (we are far below this limit)

The total deposit size at v1.0.0 is ~10–15 MB (the predictions CSV from the
four-dataset CNN baseline is the largest single file at ~5 MB).

## Notes on data deposits

The raw cell-level data is **not** included in the Zenodo software deposit
because it is owned by the dataset providers (Severson/TRI, HUST, Sandia,
Luh/KIT). Users obtain raw data once from the dataset DOIs documented in
[`README.md`](../README.md) §Datasets and then re-run the pipeline starting
from `data/intermediate/`. This split (software-on-Zenodo vs. raw-data-from-
provider) is consistent with FAIR §A2 (metadata survives independently of
the raw data) and §R1 (clear separation of source-data licensing from
software licensing).

If a data-only deposit becomes useful later (e.g., for cite-and-reuse of
the 34-feature CSVs in a follow-up study), open a new Zenodo deposit for
just `data/intermediate/` with a CC-BY-4.0 licence and cross-reference both
deposits via `related_identifiers` in `.zenodo.json`.
