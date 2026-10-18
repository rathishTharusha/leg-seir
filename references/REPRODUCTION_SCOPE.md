# What earlier experiments were actually reproduced?

Running a notebook, checking an equation and reproducing a published experiment are different achievements. This page gives the limits of our earlier checks. It takes priority over stronger wording in older copied records or notebook job names.

| Reference | What was checked or run | What we can say |
|---|---|---|
| Weng 2024 dengue benchmark | Five shifted-dataset GNN runs using the authors' code; saved CSVs and an independent implementation check | Those five runs were repeated with recorded numerical differences. The full 20-row table and all data variants were not reproduced. |
| STGAT used by Weng | Weng's GAT-plus-LSTM model and runner, with a dense implementation agreement check | We checked this dengue adaptation. It is not a reproduction of an unrelated trajectory-prediction model also called STGAT. |
| Guo 2019 ASTGCN | Correct paper obtained; Weng's adapted encoder and runner checked | We reran the dengue adaptation, not the original traffic dataset scores. |
| Bai 2021 A3T-GCN | Published seven-author paper identified; Weng's adapted runner checked | We reran the dengue adaptation, not the original traffic experiment. The older preprint is a separate reference. |
| Li 2018 DCRNN | Original paper and Weng's runner; agreement check with a dense transcription | We reran the dengue adaptation, not the original traffic results. |
| Shi 2019/2020 AGCN variants | Related papers identified; Weng/PyG AAGCN with `adaptive=False`, `attention=True` checked | Name the actual library, settings and adaptation. We did not reproduce skeleton-action recognition accuracy or every AGCN variant. |
| Wu 2019 Graph WaveNet | Original paper and adaptive graph formula checked | This informed our design. Our encoders are not the complete Graph WaveNet model, and its traffic scores were not reproduced. |
| Gopalakrishnan 2020 SEIR notes | Lecture source identified; six saved equation/example checks and a standalone notebook | Basic SEIR examples were checked. This is not a peer-reviewed dengue forecasting experiment. |
| Phaijoo/Gurung 2018 SEIR-SEI | Paper, seven checks and the sensitivity-index table, with parameter inconsistencies documented | Specific mathematical results were checked using stated parameter sets. This does not validate our forecasting decoder on surveillance data. |
| Hasan 2022 vector-host SEIR-SEI | Correct paper obtained and equations reviewed | Background only; no dedicated original experiment reproduction was established. Do not label the previous two notebooks as reproductions of Hasan. |
| Liu 2025 Sri Lankan SEIR-LSTM | Original paper obtained | Related mechanistic/neural work. No exact original dataset-and-code run was established here. Our decoder is a separate implementation. |
| GulMohamed 2026 DengueGNN | Original paper and architecture exploration | We explored an architecture. The original task subset and code protocol were not established, so its published performance was not independently reproduced. |
| Raissi 2019 / Nguyen 2024 PINNs | Original papers obtained | Background only. A disease decoder that supports gradients is not automatically a physics-informed neural network (PINN). |

## Achieved agreement in Weng author-code reruns

| Encoder | Original CV RMSE | Rerun CV RMSE | Relative difference |
|---|---:|---:|---:|
| STGAT | 44.782211 | 45.431667 | +1.450% |
| A3TGCN | 58.521675 | 60.106386 | +2.708% |
| ASTGCN | 47.722691 | 48.821220 | +2.302% |
| DCRNN | 71.608346 | 71.321104 | -0.401% |
| AAGCN | 55.830171 | 55.829052 | -0.002% |

These are earlier saved reruns, not training performed while preparing this release. Five Kaggle runs were checked as COMPLETE on 10 October 2026. The CSV files record the achieved scores. A job name saying `exact reproduction` does not prove exact numerical equality.

The original software version is commit `45f1c0878f002407633ed1237638734faa9ceb2b`. The notebooks download that repository and record their software changes. An earlier offline source copy contains selected evaluation/model/plotting files, array, graph, requirements and MIT license, not every upstream file.

## Source files and execution records

Some copied notebooks have empty outputs or zero execution counts. Their saved CSVs and completed Kaggle jobs are separate evidence of what ran. Older cross-check notebooks may need files from the team's full research folder; they are not all standalone.

The seven public Kaggle notebooks include their setup and download steps. The main LEG-SEIR public-table notebook contains its model and training logic and starts from checked public tables.

For a new claim about another paper, repeat its exact task and record the data/code version, or clearly describe an adaptation. Do not treat its published score on another task as a matched LEG-SEIR competitor.
