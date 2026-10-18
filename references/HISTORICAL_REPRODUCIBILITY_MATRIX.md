# Why some earlier experiments could not be fully reproduced

These notes describe the earlier source and software checks. They are not a new benchmark against LEG-SEIR. Use [REPRODUCTION_SCOPE.md](REPRODUCTION_SCOPE.md) for achieved scores and the claims those checks support.

## Sources

| Work | Source used | Available code/data |
|---|---|---|
| Weng 2024 | *Graph Representation Learning for Dengue Forecasting*, IEEE BigData | [Authors' repository](https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2), commit `45f1c08` |
| DengueGNN | Scientific Reports 16:10584, 2026 | An OpenDengue source is named, but the exact subset/code needed for the original experiment was not established |
| Basic SEIR | Gopalakrishnan's MTH 271 lecture notes, 2020 | Equations, parameters and code in the notes |
| SEIR-SEI | Phaijoo and Gurung, 2018 | Equations and parameter tables; no surveillance dataset needed for the checks |

## Weng's table: what was available?

The table has 20 rows: ten models on each of two dataset versions, shifted and unshifted. Five shifted-dataset GNN rows had enough released material to rerun: STGAT, A3TGCN, ASTGCN, DCRNN and AAGCN. Rerunning them does not imply exact numerical equality; the scope table records the differences.

The released model/evaluation files are `Models/gnn_models.py` and `Models/evaluation.py`. The runner uses `Data/Datasets/sri_lanka_2013-2022_shifted.npy`, `Models/sri_lanka_adj_list.json`, and prefixes `[0.6, 0.7, 0.8, 0.9, 1.0]`. `Models/results.txt` records the original target scores.

| Missing comparison | Why the released files were not enough |
|---|---|
| ARIMA, Random Forest, LSTM | Scripts read `MLSO2_Final.csv`, which was absent. The released vertical CSV has different columns, so it is not a direct substitute. |
| XGBoost, ARNN | No scripts/configurations for these reported rows were found in the checked source. |
| Ten unshifted rows | The runner hardcodes the shifted array; an unshifted array was not released in the checked repository. |

The vertical CSV uses `index` and `time`, while the baseline scripts expect columns such as `Week` and `minTime`. Substituting it would raise a missing-column error. Rebuilding unshifted inputs would require original source products and preprocessing decisions. That would be additional reconstruction, not simply rerunning the released experiment.

## Why the software setup had to change

The original requirements name `pandas~=2.2.0`, `torch==2.1.2`, `torch_geometric==2.5.3` and `torch_geometric_temporal==0.54.0`. In the recorded setup, they caused these problems:

1. The temporal package requires `pandas<=1.3.5`, conflicting with pandas 2.2. pip reports `ResolutionImpossible`. The old pandas requirement also cannot be satisfied with a normal compatible wheel in that Python 3.11 setup.
2. The temporal package imports `torch_geometric.utils.to_dense_adj`, which is unavailable in the pinned PyG 2.5.3 layout. The import raises `ModuleNotFoundError`. Tested PyG 2.4.0, 2.3.1 and 2.3.0 worked with the temporal package; 2.4.0 was used.
3. The Kaggle host used Python 3.12 in the earlier run. The required torch 2.1.2 and companion graph-package wheels did not support that interpreter.

The recorded notebooks therefore use a separate Python 3.11 environment, PyG 2.4.0 and `--no-deps` for the temporal package. They keep temporal package 0.54.0 because changing the package that defines the architectures could change the models. Other named versions remain torch 2.1.2, NumPy 1.26.2-compatible, pandas 2.2.0-compatible, scikit-learn 1.4.0 and statsmodels 0.14.1. These are historical environment choices, not a guarantee about every future Kaggle runtime.

## DengueGNN: why no exact original-task rerun is claimed

The paper points to OpenDengue, but the exact countries, date range, administrative level, graph nodes, chosen settings and runnable original protocol were not established in this audit. OpenDengue alone is not enough to identify one exact experiment. An earlier implementation of the architecture on Sri Lankan data is a separate exploration; its scores cannot be substituted for the paper's results.

## Basic SEIR checks

The lecture notes supply the equations and code. Six saved checks passed. They examine the example curve, parameter changes, equilibrium and outbreak/vaccination thresholds. They use NumPy/SciPy, not a dengue surveillance training task.

## SEIR-SEI checks

Seven saved checks passed, including nine sensitivity indices checked to six decimal places. The record also notes inconsistent baseline/simulation parameter columns, an R0 below one for the stated simulation values, and a printed vector-equilibrium issue. The reproduction notebook explains which formulas and parameter set it checks. Passing those checks is not confirmation of every conclusion in the source paper.
