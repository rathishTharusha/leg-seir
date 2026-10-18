# Read the implementation

Start with `METHOD.md`, the architecture figure and the standalone training notebook. The notebook has a new audited-input loader followed by the exact scientific cells of the original run. The readable files under `src/leg_seir/_frozen/` are extracted source fragments, executed in notebook order by `runtime.py`; they intentionally share globals. Do not import those fragments individually. `build_candidate(config, fold, population)` is the small model-construction API.

## Shapes and units

One sample has 25 districts and three observed report slots. `rawx` has shape `(batch, 25, 3)` and contains reported counts. `x` is `log1p(rawx)` standardized by a single training-only mean and standard deviation. `p` repeats the most recent standardized observation across three forecast horizons. The model returns `(batch, 25, 3)` standardized log predictions. `counts` applies the inverse in float64 and floors at zero. The horizons are successive report slots, mapped to seven daily simulation steps each; that mapping is a modelling assumption rather than a guarantee about source-report timing.

## Data helpers

`pack` slices three historical rows and three target rows for each eligible origin. It transposes the district axis, makes training tensors float32 and retains count targets in float64 for evaluation. Any origin touching a missing source row is excluded before this function is called. `counts` reverses normalization and `log1p`; training loss and reported count RMSE therefore use different scales. `metrics` pools count errors over origins, districts and horizons; sMAPE assigns zero to zero/zero cells. `seed_all` seeds Python, NumPy and PyTorch and requests deterministic algorithms. `predict_z` disables gradients and switches the model to evaluation mode.

## Neural and graph functions

`Candidate.__init__` allocates the selected configuration's parameters. `base_hidden` concatenates three levels and two adjacent differences, optionally appending a district embedding for the identity control. Two shared width-64 linear layers with ReLU create each district's hidden vector. `local_head` maps that vector to three forecast values.

`adjacency` multiplies receiver and sender factors, excludes the diagonal, retains the top five sender scores per receiving district and applies row softmax. Thus rows sum to one, direction matters and the graph is learned rather than supplied from measured mobility. Top-k selection itself is discrete; gradients flow through the selected scores. `lag_weights` applies softmax to one three-value parameter vector per recipient. These are static learned parameters, not sample-specific attention or identified biological delays.

The lagged `forward` first preserves a local hidden vector. It graph-averages each historical coordinate, multiplies those coordinates by the recipient's lag weights and by the window length, then encodes them with the same MLP. Multiplication by three makes uniform lag weights retain the original coordinate scale. A gated linear message transforms the neighbour-minus-local hidden difference and adds it to the local hidden vector. Shared `local_head` produces both local and graph forecasts. `rate_head` produces three mechanistic rate inputs.

`routing_weights` supplies four convex mixing weights for each horizon. In the full LEG-SEIR configuration they are learned horizon parameters shared across districts and samples. The four experts are persistence, local neural, graph neural and mechanistic forecasts, all expressed in standardized log units. The mixture is formed before inverse transformation; it is not an arithmetic average of count forecasts. The expert values co-adapt during training, so their weight magnitudes alone do not establish causal importance.

## Mechanistic functions

`conservative_step` moves nonnegative mass from S to E, E to I and I to R with exponential transition fractions. Each moved quantity is subtracted from one compartment and added to the next. Nonnegative forces keep transitions bounded and conserve total mass, apart from floating-point rounding. It returns the new state and E-to-I onsets. It is a discrete conservative proxy solver, not an explicit mosquito–human epidemic model.

`mechanistic_forecast` is the local decoder used by comparator configurations. It initializes proxy states from recent counts and population scaling, transforms neural rate outputs and sums seven daily onsets per horizon.

`imported` scales each district's raw historical counts by population/11, bounds the resulting proxy fractions, graph-averages them and combines historical coordinates using the same lag weights as the neural branch. This supplies one historical imported-pressure anchor per receiving district.

`exposure_forecast` initializes infectious and exposed proxies from the latest two counts; susceptible is initialized to 0.318, recovered is the residual and the state is normalized. Those constants are assumptions, not fitted clinical estimates. At simulated day k, `exp(-k/7)` weights the historical anchor and its complement weights graph-averaged simulated infectious proxies. A learned sigmoid import gate mixes this pressure with local infectious pressure. A bounded transformed neural beta multiplies the mixture. Daily E-to-I transitions accumulate into each horizon's incidence proxy and are rescaled to counts. The differentiable unrolled solver passes gradients back to neural rates, graph scores, lag weights and the import gate.

The observer comparator's `prior_state`, `corrected_state` and `observer_forecast` provide a different bounded state-correction mechanism. They remain in the shared source for exact comparator checkpoint reloads. They are not an extra component of LEG-SEIR. Adapter helper functions similarly belong to the shared-adapter control, not the proposed model.

## Study execution

`make_evaluation_fold` uses complete six-row windows. Development target years are 2023 and 2025; final training targets precede 2025, checkpoint selection uses 2025 and scoring uses 2026. The scalar normalization statistics use rows touched by the training samples only. Some historical target values can later become inputs in rolling evaluation; no test labels are used in optimizer updates.

`fit` seeds the run, shuffles training origins, minimizes standardized-log MSE with Adam, clips gradient norm to five and evaluates validation count RMSE after every epoch. It keeps the best validation checkpoint, with a maximum of 300 epochs and patience 40. `ridge_baseline` fits a shared ridge regression to the same three standardized levels and two differences. `baseline` repeats the latest raw report for persistence.

The study runner evaluates the development learning-rate grid before constructing final test tensors. Learning-rate selection averages skill relative to persistence across both development folds and seeds. Primary removals inherit their parent learning rate; separately selected removal learning rates are recorded as sensitivity runs. Five final neural seeds are scored. Every fit exports its checkpoint, curve and prediction arrays. The completion file hashes the outputs and the runner creates a full ZIP. A smoke run changes the budget to two epochs and one seed: it tests execution, not the paper's numerical result.

## Verification versus new training

`scripts/verify.py` checks saved bytes, reloads every neural checkpoint on CPU, reconstructs inputs/scalers/targets, recomputes forecasts and scores, and checks chronology and selection. It does not perform a new neural fit or independently refit ridge. The source-identity test checks that modular fragments and the new notebook's scientific cells agree with the unchanged frozen notebook. Conservation and gradient tests check mathematical implementation contracts. Neither CPU parity nor a passing test establishes prospective accuracy, biological identification or architectural priority.

For complete equations, proxy constants and comparator definitions, use the paper and supplement. For a new empirical claim, freeze a new development protocol and evaluate on genuinely independent data rather than repeatedly adapting to the recorded 2026 outcomes.
