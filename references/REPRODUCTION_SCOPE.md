# Reference-paper reproduction scope

An executable notebook, an architecture check and a reproduced published experiment are different evidence. This register supersedes any stronger wording in the copied historical READMEs.

| Work | PDF | Code / notebook evidence | Permitted claim |
|---|---|---|---|
| Weng2024 dengue benchmark | Verified original |Five pinned author-code Kaggle kernels COMPLETE; raw CSVs and summary copied. Independent R1 implementation also available. |Author-code shifted GNN experiments rerun, with explicit numerical deviations. Not all20 TableI rows or all dataset variants reproduced. |
| STGAT in Weng |Same benchmark PDF; GAT foundational PDF additionally retrieved |Weng's GAT-plus-LSTM wiring and author runner; dense implementation parity test |Benchmark adaptation reproduced/checked. Do not cite an unrelated trajectory STGAT as the exact architecture implemented. |
| Guo2019 ASTGCN |Correct AAAI PDF retrieved; wrong old file quarantined |Weng-adapted encoder and author-code runner, not a PeMS reproduction |Architecture component/adaptation verified under dengue benchmark. Original traffic scores not reproduced here. |
| Bai2021 A3T-GCN |Published version retrieved with correct seven-author list; older Zhu preprint separately retained |Weng-adapted runner; implementation parity evidence |Dengue adaptation rerun. Original traffic datasets/results not reproduced. |
| Li2018 DCRNN |Verified original |Weng runner and dense transcription parity evidence |Dengue adaptation rerun. Original traffic forecasting results not reproduced. |
| Shi2019/2020 AGCN variants |Two-stream paper plus multi-stream attention preprint |Weng/PyG AAGCN configuration, adaptive=False and attention=True; transcription parity |State the exact library/configuration and adaptation. Do not claim skeleton-action accuracy reproduction or equate every AGCN version. |
| Wu2019 Graph WaveNet |Verified original |Adaptive adjacency formula used; our historical/new adaptive encoders differ from complete Graph WaveNet |Method inspiration and explicit adaptation. Full Graph WaveNet traffic results not reproduced. |
| Gopalakrishnan2020 SEIR lecture |Correctly identified lecture notes recovered from misnamed Hasan file |R3 notebook and standalone Kaggle kernel COMPLETE; six saved qualitative/numerical checks |Basic SEIR mathematical calibration, not dengue forecast reproduction or a peer-reviewed research experiment. |
| Phaijoo/Gurung2018 SEIR–SEI |Verified article after a cover page |R4 notebook and standalone kernel COMPLETE; seven checks, sensitivity-index table and documented baseline inconsistencies |Named equilibrium/sensitivity targets reproduced; report which parameter column/version was used. Not our open-loop decoder's empirical validation. |
| Hasan2022 vector-host SEIR–SEI |Correct publisher PDF now retrieved |No dedicated Hasan reproduction notebook in the audited collection |Theory/context citation after direct equation review. Do not relabel R3/R4 as its reproduction. |
| Liu2025 Sri Lankan SEIR-LSTM |Original PLOS PDF now retrieved |No exact Liu dataset/code run established in this audit |Closest mechanistic-neural context. Our SEIR decoder is a separate implementation/ablation. |
| GulMohamed2026 DengueGNN |Verified original |R2 architecture experiments exist; no exact original subset/code protocol established |Architecture exploration only. Original performance not independently reproduced. |
| Raissi2019/Nguyen2024 PINN |Verified originals |No full original-task reproduction in this package |Related-work context only; a differentiable SEIR decoder alone is not a PINN. |

## Numerical agreement in the author-code Weng runs

| Encoder | Published CV RMSE | Rerun CV RMSE | Relative difference |
|---|---:|---:|---:|
| STGAT |44.782211|45.431667|+1.450%|
| A3TGCN |58.521675|60.106386|+2.708%|
| ASTGCN |47.722691|48.821220|+2.302%|
| DCRNN |71.608346|71.321104|−0.401%|
| AAGCN |55.830171|55.829052|−0.002%|

These are archived reruns, not newly trained during package preparation. The five original Kaggle statuses were checked COMPLETE on 10 October 2026. Dependency deviations, random-seed/stack differences and metric conventions are documented inside their notebooks. A kernel named “exact reproduction” is a task label, not proof of exact equality. CSVs are authoritative for the achieved agreement.

The source snapshot pins commit`45f1c0878f002407633ed1237638734faa9ceb2b`, includes the authors' evaluation/model/plotting files, array, graph, requirements and MIT code license, and records hashes. The online kernels clone the full pinned repository and install their stated stack. The snapshot is a small offline provenance subset; do not imply it contains every upstream asset.

## Execution versus source

Many copied notebooks have zero stored execution counts and outputs. Their generated CSVs and live completed Kaggle run status provide separate evidence. Crosscheck notebooks require their local library and original project inputs; they are archived research notebooks, not all standalone. The seven author/mathematical Kaggle kernels are standalone execution wrappers with documented internet/dependency requirements. Our main independent notebook embeds all runtime model/data logic.

If the rewritten paper needs a new external-paper result beyond these scopes, reproduce the exact original task and cite its dataset/code version. Otherwise narrow the claim rather than supply substitute numbers.
