# How LEG-SEIR works

The goal is to predict the dengue count in each of 25 districts for the next three reports. One input contains three recent reports from every district. A forecast horizon is how far ahead we predict: horizon 1 is the next report, and horizon 3 is the third report ahead.

## 1. Put counts on a useful scale

Dengue counts can vary greatly across districts. The code first applies `log1p(count)`, which means `log(1 + count)`. This reduces the effect of very large values and still accepts zero.

It then subtracts the training mean and divides by the training standard deviation. This is called standardization. Only training data are used to calculate these two numbers. Raw counts are also kept for the disease calculation.

## 2. Learn from each district's own reports

A small multilayer perceptron, or MLP, reads three values and the two changes between them. An MLP is a neural network made of connected layers. Here it has two hidden layers, each with 64 values, and ReLU activation functions.

The same MLP is used for every district. Its output is a hidden vector: a list of learned values that summarize that district's recent history. A readout layer turns that vector into three forecasts. This local path keeps a district's own information available throughout the model.

## 3. Add useful information from other districts

The model learns a directed graph. For each district receiving information, it selects five other districts as senders. Self-connections are excluded. A softmax turns the selected scores into positive weights that sum to one. Larger weights mean the model uses more information from that sender.

The model also learns three lag weights for each receiving district. They decide how much to use each of the three past reports. These weights also sum to one. The graph and lag weights stay fixed during prediction; they are not recalculated from each new input sample.

The weighted history is passed through the same MLP. The model subtracts the local hidden vector from this neighbour vector, applies a linear layer and a learned gate, then adds the result to the local vector. The gate controls the size of the update. This is a residual connection: the local path stays available while the model adds other information.

The graph shows useful prediction relationships. It does not show measured travel or confirmed disease transmission. The lag weights are shared across senders for each receiver; they are not measured incubation delays.

## 4. Run an SEIR-inspired calculation

SEIR groups people into susceptible, exposed, infectious and recovered states. In this model these states are numerical proxies, not measured patient groups. The latest two raw counts and fixed population scaling set the starting states.

The same graph and lag weights combine past counts from other districts into a starting estimate of imported pressure. Here, pressure means an input used in the infection calculation. It is not a measured movement of infected people.

At simulated day `k`, `exp(-k/7)` weights this historical estimate. The remaining weight is given to infectious states simulated for the neighbouring districts. Early steps therefore rely more on past reports; later steps rely more on the simulation. The day counter continues across horizons rather than restarting each horizon.

A learned gate combines local and imported pressure. A neural rate then controls the susceptible-to-exposed transition. Each step moves only an amount available in its starting group. This keeps states nonnegative and preserves their total, apart from rounding errors.

The code sums exposed-to-infectious transitions over seven simulated days for each report horizon. Seven days per report, the initial susceptible fraction 0.318, population divided by 11, and the other fixed rates are modelling assumptions. The model does not explicitly simulate mosquitoes.

## 5. Combine four forecasts

The four forecasts are persistence (repeat the latest observation), local MLP, graph MLP and the SEIR-inspired output. The SEIR output is converted to the same standardized log scale as the other forecasts.

The model learns four mixing weights for each horizon. They sum to one and are shared across districts and samples. Mixing happens on the standardized log scale, not directly on case counts. The result is converted back to counts, with negative values replaced by zero.

The full model has 9,519 parameters. The four branches train together, so a large mixing weight alone does not prove that a branch has a large independent effect.

## 6. Train and evaluate

Training minimizes mean squared error on standardized log values. After each epoch, the code checks RMSE on validation case counts and saves the best model. These two error measures use different scales.

The final model trains on targets through 2024, chooses a checkpoint using 2025, and predicts eligible 2026 targets. Because the 2026 sample had already influenced earlier research choices, this is a retrospective comparison. It is not an untouched test of a future outbreak.

For exact equations and constants, read the paper and supplement. For function names and array shapes, read [CODE_GUIDE.md](CODE_GUIDE.md).
