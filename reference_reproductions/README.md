# Earlier attempts to reproduce reference results

These notebooks run earlier papers' code or check their equations. They are separate from the LEG-SEIR experiment. A notebook name containing `exact-reproduction` is a job name, not a claim that every output matches exactly.

Read [the scope table](../references/REPRODUCTION_SCOPE.md) first. It records achieved scores and limits. [The historical data/software notes](../references/HISTORICAL_REPRODUCIBILITY_MATRIX.md) explain what could not be rerun.

## Included notebooks

| Notebook folder in `kaggle/kernels/` | What it checks | Original target |
|---|---|---|
| `weng-2024-exact-reproduction-stgat` | Weng's STGAT run | MAE 25.38, RMSE 44.78 |
| `weng-2024-exact-reproduction-a3tgcn` | Weng's A3TGCN run | MAE 34.03, RMSE 58.52 |
| `weng-2024-exact-reproduction-astgcn` | Weng's ASTGCN run | MAE 33.68, RMSE 47.72 |
| `weng-2024-exact-reproduction-dcrnn` | Weng's DCRNN run | MAE 45.98, RMSE 71.61 |
| `weng-2024-exact-reproduction-aagcn` | Weng's AAGCN run | MAE 41.93, RMSE 55.83 |
| `seir-model-reproduction-gopalakrishnan` | Basic SEIR equations from lecture notes | Six checks |
| `seir-sei-sensitivity-reproduction-phaijoo-gurung` | SEIR-SEI equations and parameter sensitivity | Nine sensitivity indices and related checks |

These are target numbers, not the scores achieved by our reruns. Read the CSVs under `results/` and the scope table for achieved agreement.

## Run one on Kaggle

Each folder has a notebook and `kernel-metadata.json`. Use your own Kaggle account and API credentials. Check the notebook's setup instructions and the kernel metadata before submission. From a chosen kernel folder:

```sh
kaggle kernels push
kaggle kernels status <your-username>/<kernel-slug>
kaggle kernels output <your-username>/<kernel-slug> -p <output-folder>
```

The Weng notebooks download the authors' repository at commit `45f1c0878f002407633ed1237638734faa9ceb2b`. They run five data prefixes for 50 epochs each on CPU. Earlier runs took roughly 1-4 hours per architecture; future runtime depends on the platform. No new runs are submitted just by reading these notes.

## Why the older software setup differs

In the recorded environment, the authors' package versions did not work together without these changes:

1. Use a separate Python 3.11 environment to install `torch==2.1.2`; the Kaggle host used Python 3.12.
2. Install `torch_geometric_temporal==0.54.0` with `--no-deps` because its declared `pandas<=1.3.5` conflicts with the authors' pandas 2.2 setting.
3. Use `torch_geometric==2.4.0` instead of 2.5.3 so the older temporal package can import the required module.

The other named package versions stay as specified in the notebooks. Report these changes when citing the reruns.

## How to interpret the results

Five shifted-dataset GNN rows from Weng's table were rerun. Missing scripts or inputs prevent a complete rerun of its 20 rows. We also found that its reported cross-validation calculation includes training windows and averages per-window RMSEs. This differs from our held-out, pooled-count scoring. Those historical numbers should not be compared directly with LEG-SEIR's 2026 scores.

SEIR checks test equations and specific mathematical claims; they do not establish dengue forecast accuracy. No exact original DengueGNN experiment is included because the required original task details were not established. A separate architecture implementation would not by itself reproduce that paper's published scores.
