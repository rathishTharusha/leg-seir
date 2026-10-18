# LEG-SEIR

**Lagged Exposure for Graph-Based Dengue Forecasting**

LEG-SEIR predicts dengue case counts for Sri Lanka's 25 districts. It reads the last three reports and predicts the next three. It combines a small neural network, information from other districts, a simple disease model, and the latest known case count.

[Paper](paper/LEG_SEIR_manuscript.pdf) | [Supplement](paper/LEG_SEIR_supplement.pdf) | [Dataset and saved experiments](https://huggingface.co/datasets/rathishTharusha/leg-seir) | [How to run](docs/REPRODUCIBILITY.md)

## Where to start

1. Read [the model explanation](docs/METHOD.md) to understand the main idea.
2. Use [the glossary](docs/GLOSSARY.md) for unfamiliar terms.
3. Follow [the code guide](docs/CODE_GUIDE.md) to connect each function to the model.
4. Follow [the run instructions](docs/REPRODUCIBILITY.md) to check saved results or train again.

The paper, code and saved results are available for lecturer review. [REVIEW_GUIDE.md](REVIEW_GUIDE.md) explains what was checked and what still needs more research.

## How the model works

The model makes four forecasts: repeat the latest count, use a district's own history, use other districts' histories, and use an SEIR-inspired disease calculation. It learns how much to use each forecast for each future report. The graph tells the model which districts to use together. It is learned from data; it is not a measured map of disease spread.

![LEG-SEIR architecture](figures/publication/architecture.png)

## Recorded results

RMSE measures prediction error. Lower is better. Neural model values below are the average of five training runs with different random seeds.

| Method | Mean 2026 RMSE |
|---|---:|
| Persistence: repeat the latest count | 87.382382 |
| Identity MLP | 81.392283 |
| Shared adapter | 80.756884 |
| LEG-SEIR | **79.634330** |

All methods predict the same 18 starting points, 25 districts and three future reports. The starting points overlap, so they are not 18 fully separate outbreaks. The 2026 data had already influenced earlier research choices. These results describe a retrospective comparison, meaning a study of past data. They do not prove how well the model will work on unseen future outbreaks.

Several simple models already beat persistence. LEG-SEIR improves on shared adapter by 1.39%, so the advantage over this strong comparison model is small. LEG-SEIR also does not beat every model at every forecast horizon.

## Check the saved results on CPU

Run these commands from the repository folder after setting up Python. See [the full instructions](docs/REPRODUCIBILITY.md) for environment setup.

```sh
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
python scripts/fetch_artifacts.py
python scripts/verify.py artifacts/evaluation_outputs.zip artifacts/verified
python scripts/build_figures.py
python -m unittest discover -s tests -v
```

This downloads the saved experiment, checks all 265 neural training runs and rebuilds the figures. It does not train the models again. A SHA256 hash acts like a file fingerprint: the downloader checks that the files match the exact saved versions.

## Train the models again

Use [the training notebook](notebooks/LEG_SEIR_train_from_audited_data.ipynb) in a fresh GPU environment. It downloads the checked input tables and trains from scratch. It does not use saved model weights or predictions as training inputs.

The model and training cells match the original experiment. The table downloader is new, so the complete notebook has a different code identity. Its loader was checked with a short CPU run; a new full GPU training run was not completed for this release. The original experiment remains under [experiments/frozen](experiments/frozen/).

The [original notebook](experiments/frozen/Dengue_final_study.ipynb) starts by reading government report PDFs. Their source URLs and hashes are included, but the bulk original files are kept separately. Read [the data notes](docs/DATA.md) before using them.

## Folder guide

| Folder | What it contains |
|---|---|
| `src/leg_seir/` | Model code and the function that builds a model |
| `experiments/frozen/` | The unchanged notebook used for the recorded experiment |
| `notebooks/` | A notebook for new training and a notebook for viewing saved results |
| `scripts/` | Commands to download files, check results and draw figures |
| `results/` | Scores, chosen settings and verification records |
| `figures/publication/` | Nine figures and the numbers used to make them |
| `paper/` | The paper and supplement as PDFs |
| `tests/` | Checks for model rules, gradients and agreement with the original code |
| `references/` | Reference records and notes on what was reproduced |

The study has 180 development runs, 75 main final runs and 10 extra learning-rate checks: 265 neural runs in total. It also has three separate ridge-regression fits. The verifier checks 915 file hashes, saved predictions, data splits, scaling and how the learning rate was chosen.

## What remains uncertain

The results support this model on this sample. They do not prove that learned graph links are real transmission routes or that this is the first graph model to use disease equations. We still need separate tests of the lag weights, imported pressure and pressure transition, tests of the fixed disease-model constants, and checks on independent future or external data. An alternative way of reading repeated reports is saved, but LEG-SEIR was not trained or tested on that alternative.

## License and citation

The team's original software uses the MIT license. That license does not automatically apply to the paper, dataset or other providers' material. Reference papers are linked, not included as copied PDFs. Citation details follow the author list in the supplied paper. The team keeps the LaTeX and Overleaf project separately; they are not in this repository.
