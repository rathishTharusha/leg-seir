# Read the code step by step

Read [METHOD.md](METHOD.md) first. Use [GLOSSARY.md](GLOSSARY.md) when a term is unfamiliar. This guide connects the model to its functions; the paper and supplement contain the full equations.

## Where the code lives

The training notebook has a new table downloader followed by the same model and training code as the recorded experiment. Files under `src/leg_seir/_frozen/` are copies of parts of that original code. We leave them unchanged so saved checkpoints and code checks still match.

`runtime.py` runs those files in their original order. They share variables such as the district count and population array, just as notebook cells share variables. Do not import one `_frozen` file on its own. Use `build_candidate(config, fold, population)` to build a model.

## Array shapes and units

| Name | Shape | Meaning |
|---|---|---|
| `rawx` | `(batch, 25, 3)` | Three recent raw case counts for every district |
| `x` | `(batch, 25, 3)` | The same counts after `log1p` and training-based standardization |
| `p` | `(batch, 25, 3)` | The latest standardized log observation, repeated over three horizons |
| Model output | `(batch, 25, 3)` | Three standardized log forecasts for every district |

`batch` is the number of samples processed together. The final axis means past reports for inputs and future reports for outputs. `counts` converts predictions back to case counts using float64 and replaces negative values with zero. The disease calculation uses seven simulated days per report; this is an assumption about the reports, not proof of their exact timing.

## Data helper functions

| Function | What it does and why |
|---|---|
| `pack` | Select three past rows and three target rows, reorder the axes and build PyTorch tensors. Training values use float32; raw targets use float64 for scoring. Windows with missing rows are removed earlier. |
| `counts` | Undo standardization and `log1p`. This is why training loss and reported count errors use different units. |
| `metrics` | Calculate errors across starting points, districts and horizons. For sMAPE, a zero prediction with a zero target contributes zero. |
| `seed_all` | Set Python, NumPy and PyTorch random seeds and request deterministic algorithms to reduce avoidable run differences. |
| `predict_z` | Make standardized log predictions with gradients turned off and the model in evaluation mode. |

## Neural and graph functions

`Candidate.__init__` creates the parameters needed by the selected model. Different configurations turn branches on or off.

`base_hidden` joins three input levels with two changes between consecutive levels. The identity-MLP comparison also adds a learned district vector. Two linear layers with 64 hidden values and ReLU produce the hidden vector. `local_head` changes it into three forecasts.

`adjacency` creates the district graph. It multiplies learned receiver and sender vectors, removes self-links, keeps five senders per receiver and applies softmax. Each row sums to one. Direction matters: district A using district B does not require district B to use district A. Choosing the top five is a discrete step; gradients update the selected scores.

`lag_weights` applies softmax to three learned values per receiver. These weights choose how much to use each past report. They are shared across senders for that receiver and stay fixed during prediction.

The lagged model's `forward` follows these steps:

1. Use `base_hidden(x)` to keep a local hidden vector.
2. Use the graph to average other districts' values at each past report position.
3. Multiply each position by its lag weight and by three. Multiplying by three keeps the usual graph average when lag weights start equal.
4. Pass this history through the same `base_hidden` network.
5. Subtract the local hidden vector, apply the message layer and gate, then add the update to the local vector.
6. Use the shared `local_head` for local and graph forecasts. Use `rate_head` to produce three inputs for the disease calculation.
7. Combine the four branches using `routing_weights`.

`routing_weights` gives four weights for each horizon. In full LEG-SEIR, these are learned parameters shared across all districts and samples. The branches are persistence, local, graph and disease-model forecasts. They all use standardized log units before mixing. This is not a simple average of raw case counts. Since branches train together, their weights alone do not show their separate causal importance.

## Disease-model functions

`conservative_step` moves values from S to E, E to I and I to R. Each movement is subtracted from one group and added to the next. Exponential transition fractions keep movement within the available amount. Groups stay nonnegative and their total stays the same, apart from rounding. The function returns the new state and the amount moving from E to I. It does not include a separate mosquito model.

`mechanistic_forecast` is the local disease calculation used by comparison models. It starts from recent counts and population scaling, converts neural outputs into rates and adds seven days of E-to-I movement per horizon.

`imported` builds a historical pressure value for each receiver. It divides raw counts by population/11, clips the fractions, averages over the graph and then combines the three report positions using the lag weights. The graph and lag weights are the same ones used by the neural branch.

`exposure_forecast` runs the full graph-aware disease calculation:

1. Set infectious and exposed proxy values from the latest two counts. Start susceptible at 0.318, use the remaining amount for recovered, and normalize the total.
2. At day `k`, combine the historical pressure anchor with graph-averaged simulated infectious states. The historical weight is `exp(-k/7)`; the simulation weight is the remaining amount.
3. Use an import gate to mix that pressure with local infectious pressure. A bounded positive neural rate, beta, multiplies the mixture.
4. Run `conservative_step` and add up E-to-I movements over each seven-day horizon.
5. Scale the result back to counts. The model then converts it to standardized logs before mixing with the other branches.

The fixed starting values and rates are assumptions, not estimates of real patient states. PyTorch can calculate gradients through these repeated steps, allowing training to update the neural rates, graph scores, lag weights and import gate.

The observer model's `prior_state`, `corrected_state` and `observer_forecast` belong to a comparison model with a different way of correcting states. The adapter functions belong to the shared-adapter comparison. They are kept so those saved models can be loaded; they are not extra branches inside LEG-SEIR.

## Training and selection functions

`make_evaluation_fold` finds complete six-row windows: three inputs and three targets. Development folds validate on 2023 and 2025. The final model trains on targets before 2025, selects a checkpoint on 2025 and scores 2026. Only rows used by training samples set the normalization mean and standard deviation.

During rolling evaluation, an earlier observed 2026 report can become input to a later forecast. Test targets do not update model weights. This does not remove the wider limitation that earlier research had already inspected the 2026 sample.

`fit` sets a random seed, shuffles training samples and uses Adam to reduce standardized-log MSE. It clips the gradient norm at five to limit very large updates. After every epoch it checks validation count RMSE. It saves the best checkpoint and stops at 300 epochs or after 40 epochs without improvement.

`ridge_baseline` fits one shared ridge-regression model using three standardized levels and two differences. `baseline` repeats the latest raw count for persistence.

The study runner tries learning rates on development data before building the final test tensors. It chooses the rate using average improvement over persistence across both development folds and seeds. Main removal controls use their parent model's rate; extra runs check separately selected rates where needed. Five random seeds are used for the final neural comparison.

Each fit saves weights, training curves and predictions. The completion file records output hashes. A smoke run uses only two epochs and one seed. It checks that the code runs; its scores are not the paper's results.

## What verification proves

`scripts/verify.py` checks file hashes, rebuilds inputs and scaling, reloads neural weights on CPU and recalculates predictions and scores. It also checks the order of data splits and setting selection. It does not train new neural models or refit ridge.

The source-identity test checks that the model/training cells and extracted files agree with the original notebook. Other tests check state conservation, gradients and graph/mixing rules. These checks help detect implementation mistakes. They do not prove future accuracy, biological validity or that the architecture is the first of its kind.
