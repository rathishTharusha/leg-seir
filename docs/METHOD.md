# The implemented architecture

For each district, three recent reports are transformed with log1p and training-only scalar statistics. A shared two-layer width-64 MLP receives three levels and two first differences. It retains a local prediction path.

Receiver/sender factors define a static learned directed graph: diagonal excluded, top five senders per receiving district, row softmax. A recipient-specific three-lag softmax weights historical graph-averaged observations. The same shared MLP encodes this weighted history; a gated residual message adds its departure to the local representation.

The graph and lag weights also aggregate population-scaled raw counts into a historical imported-pressure anchor. Inside a conservative daily SEIR-inspired proxy rollout, exp(-k/7) shifts weight from that observed anchor to simulated neighbouring infectious states. A learned scalar mixes local and imported pressure. Horizon rates are neural outputs; all proxy compartments remain nonnegative and conserve mass.

Four forecasts—persistence, local, graph and mechanistic—are mixed by a learned horizon-specific softmax in standardized log space, then transformed back to nonnegative counts. The full model has 9,519 parameters. Lag weights are receiver-specific, not edge-specific or identified clinical delays. Learned graph edges are predictive dependencies, not measured transmission.

The mathematical definitions, initialization, rate transforms, objective, controls and fixed constants are fully specified in the manuscript and supplement. The code exposes the exact evaluated parameterization rather than claiming to identify clinical SEIR states.
