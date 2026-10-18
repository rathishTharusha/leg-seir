# Reference papers and earlier experiment checks

The main paper cites 26 works and the supplement cites 50. The team keeps their full manuscript bibliography in Overleaf. This repository keeps additional bibliography records, publisher/source URLs and checks of reference details. Third-party paper PDFs are not copied into the public repository.

## How to read these records

- `REFERENCE_COVERAGE.md` shows where references are used.
- `REPRODUCTION_SCOPE.md` says what we actually ran for each earlier method.
- `HISTORICAL_REPRODUCIBILITY_MATRIX.md` explains missing inputs and software changes in those older runs.
- `NEW_REFERENCE_GAPS.md` lists related methods that were read but not reproduced.
- JSON files keep detailed source and reference metadata. CSV files link claims to evidence.

Downloading a paper is different from reproducing its results. Implementing a similar architecture is also different from repeating the original experiment on its original data.

Seven standalone Kaggle notebooks and saved comparison scores are under `reference_reproductions/`. These are earlier experiment records, not new runs made for this release and not matched competitors in the LEG-SEIR 2026 results table. They download upstream code and data rather than redistributing those files here.

Some older audit records use paths from the original research folder. Use the current README and `artifacts.lock.json` for public code/data locations. Missing external-model reruns and component-removal tests remain open work.
