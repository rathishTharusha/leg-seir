"""Frozen shortlist and attribution controls; embedded in evaluation notebook."""

common = {"identity": False, "differences": True, "router_style": "horizon"}
CONFIGS = []
for family, mode, extra in [
    ("horizon", "router", {}),
    ("lagged", "lagged", {}),
    ("observer", "observer", {"state_features": False}),
]:
    for removal in ["full", "graph_off", "seir_off"]:
        # Without SEIR, compartment-only assimilation has no remaining operation.
        # Its replacement is the ordinary neural horizon mixture, not dead observer parameters.
        actual_mode = "router" if family == "observer" and removal == "seir_off" else mode
        CONFIGS.append(
            {
                **common,
                **extra,
                "name": family + "_" + removal,
                "mode": actual_mode,
                "graph": removal != "graph_off",
                "decoder": "direct" if removal == "seir_off" else "seir",
                "parent": family + "_full",
                "role": "candidate" if removal == "full" else "removal",
            }
        )
CONFIGS += [
    {
        **common,
        "name": "local_mlp",
        "mode": "lgi",
        "graph": False,
        "decoder": "none",
        "role": "baseline",
    },
    {
        **common,
        "name": "identity_mlp",
        "mode": "lgi",
        "identity": True,
        "graph": False,
        "decoder": "none",
        "role": "baseline",
    },
    {
        **common,
        "name": "local_seir",
        "mode": "lgi",
        "graph": False,
        "decoder": "seir",
        "role": "baseline",
    },
    {
        **common,
        "name": "plain_graph",
        "mode": "lgi",
        "graph": True,
        "decoder": "none",
        "role": "baseline",
    },
    {
        **common,
        "name": "uniform_mixture",
        "mode": "router",
        "graph": True,
        "decoder": "seir",
        "router_style": "fixed",
        "role": "baseline",
    },
    {
        **common,
        "name": "shared_adapter",
        "mode": "adapters",
        "graph": True,
        "decoder": "seir",
        "district_conditioned": False,
        "adapter_rank": 8,
        "adapter_no_decay": True,
        "role": "baseline",
    },
]
assert len(CONFIGS) == 15 and len({c["name"] for c in CONFIGS}) == 15
