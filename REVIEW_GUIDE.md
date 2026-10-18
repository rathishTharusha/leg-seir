# Guide for lecturer review

Start with `paper/LEG_SEIR_manuscript.pdf`. The supplement gives the full equations, comparison models and experiment details. Figure files and their input numbers are in `figures/publication/`.

The main PDF is the supplied seven-page team revision from 18 October 2026, with nine authors. It is unchanged here. The public supplementary PDF is the earlier technical supplement. The team keeps manuscript source, figure/table dependencies and the Overleaf ZIP separately. The submission venue is still to be chosen.

## Main finding

The average 2026 RMSE over five neural training seeds is 79.6343 for LEG-SEIR, 81.3923 for identity MLP and 80.7569 for shared adapter. Persistence gives 87.3824. Lower RMSE is better. LEG-SEIR's advantage over shared adapter is small.

These are results on 18 overlapping forecast starting points from a 2026 sample seen earlier in the research. They are not independent outbreak replications or proof of future accuracy. Variation across seeds is not a confidence interval.

Removing the graph changes both neural messages and imported pressure. This comparison does not tell us the separate effect of each mechanism. Further component tests are still needed.

## What a reviewer can run

Follow the CPU commands in the README to check saved weights, predictions, file hashes, data-split order and recalculated scores for 265 neural fits. This does not retrain the models or refit ridge regression.

For new training, use the public-table notebook. Its model/training cells match the original, but the input downloader is new. `python scripts/train.py --smoke` only checks that the code runs with a small budget; it is not a paper experiment.

Code: https://github.com/rathishTharusha/leg-seir

Data and saved experiments: https://huggingface.co/datasets/rathishTharusha/leg-seir

The exact input version is in `artifacts.lock.json`. Original source URLs and hashes are included. Bulk original archives and third-party research PDFs are kept separately. The reference notes say which earlier experiments were rerun and which papers were used only as background.
