# Lecturer review - LEG-SEIR

Read `paper/LEG_SEIR_manuscript.pdf` first, then the supplement for equations, complete controls and reproduction scope. Vector figures and their numerical inputs are in `figures/publication/`. The paper is the team-supplied seven-page IEEE revision dated 18 October 2026, including its author list. The submission venue remains to be confirmed.

The main PDF is published byte-for-byte as supplied by the team. The supplementary PDF retains the earlier technical supplement. Manuscript LaTeX, figure dependencies and the Overleaf ZIP are maintained separately, outside the current public repository and release assets. The recorded experiments are unchanged.

The recorded retrospective study reports mean district-pooled RMSE 79.6343 for LEG-SEIR, versus persistence 87.3824, identity MLP 81.3923 and the strongest shared-adapter control 80.7569. Neural values average five seeds; seed spread is not a confidence interval. The 18 overlapping 2026 origins are not independent replications. The graph-removal control removes both graph messages and imported pressure, so isolated attribution needs further ablations.

For a quick independent code/evidence review, follow the README CPU commands. They validate all 265 saved neural fits, including checkpoint predictions, hashes, chronology and recomputed scores. This does not retrain those fits or independently refit ridge. For new training use the standalone audited-input notebook; its scientific cells match the frozen notebook but its loader and identity are new. `python scripts/train.py --smoke` checks execution only and is not a paper experiment.

Public code: https://github.com/rathishTharusha/leg-seir

Public data/evidence: https://huggingface.co/datasets/rathishTharusha/leg-seir (immutable revision in `artifacts.lock.json`). Original provider records and hashes are retained. Third-party paper PDFs and bulk raw archives are not redistributed here. Bibliographies, paper-source links and documented reproduction gaps distinguish implemented controls from papers cited only for context.
