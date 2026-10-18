"""Lagged exposure: sparse sender graph, lag mixture and daily imported pressure."""

BaseCandidate = Candidate


class Candidate(BaseCandidate):
    def __init__(self, config, fold):
        self.is_lagged = config.get("mode") == "lagged"
        cfg = dict(config)
        if self.is_lagged:
            cfg["mode"] = "router" if config.get("routing", True) else "lgi"
        super().__init__(cfg, fold)
        self.config = dict(config)
        if self.is_lagged and self.graph:
            # Per-recipient convex mixture over the same three observed report lags.
            self.lag_logits = nn.Parameter(torch.zeros(N, WINDOW))
            if self.decoder == "seir" and config.get("exposure", True):
                self.import_gate = nn.Parameter(torch.tensor(-2.0))

    def lag_weights(self):
        return torch.softmax(self.lag_logits, -1)

    def imported(self, rawx):
        scale = torch.as_tensor(POP, device=rawx.device, dtype=rawx.dtype)[None] / 11
        fractions = (rawx / scale[:, :, None]).clamp(1e-6, 0.5)
        spatial = torch.einsum("ij,bjl->bil", self.adjacency(), fractions)
        return (spatial * self.lag_weights()[None]).sum(-1)

    def exposure_forecast(self, rawx, rates, graph_scale=1.0, return_state=False):
        scale = torch.as_tensor(POP, device=rawx.device, dtype=rawx.dtype)[None] / 11
        infectious = (rawx[:, :, -1] / scale).clamp(1e-6, 0.5)
        exposed = (rawx[:, :, -2] / scale).clamp(1e-6, 0.5)
        susceptible = torch.full_like(infectious, 0.318)
        recovered = (1 - susceptible - exposed - infectious).clamp_min(0)
        state = torch.stack([susceptible, exposed, infectious, recovered], -1)
        state = state / state.sum(-1, keepdim=True)
        mass = state.sum(-1)
        outputs, states = [], []
        gate = (
            graph_scale * torch.sigmoid(self.import_gate) if hasattr(self, "import_gate") else 0.0
        )
        # Historical pressure remains a decaying observation anchor. Daily evolving
        # neighbours progressively replace it; no future reports are consumed.
        anchor = self.imported(rawx) if self.graph else infectious
        for h in range(HORIZON):
            beta = torch.exp((rates[:, :, h] - 1.1).clamp(-9, 1.1))
            total = torch.zeros_like(infectious)
            for day in range(7):
                elapsed = h * 7 + day
                neighbours = (
                    torch.einsum("ij,bj->bi", self.adjacency(), state[:, :, 2])
                    if self.graph
                    else state[:, :, 2]
                )
                decay = math.exp(-elapsed / 7)
                pressure = decay * anchor + (1 - decay) * neighbours
                force = beta * ((1 - gate) * state[:, :, 2] + gate * pressure)
                state, onset = conservative_step(state, force)
                total = total + onset
                states.append(state)
            outputs.append(total * scale)
        result = torch.stack(outputs, -1)
        return (result, torch.stack(states, -2), mass) if return_state else result

    def forward(self, x, p, rawx, graph_scale=1.0, return_details=False, **kwargs):
        if not self.is_lagged:
            return super().forward(x, p, rawx, graph_scale, return_details, **kwargs)
        h = self.base_hidden(x)
        hg = h
        if self.graph:
            lagged = torch.einsum("ij,bjl->bil", self.adjacency(), x)
            # Learned lag selection also informs the neural branch. Each lag retains
            # its own temporal coordinate; positive weights sum to one per recipient.
            selected = lagged * self.lag_weights()[None] * WINDOW
            neighbour_h = self.base_hidden(selected)
            hg = h + graph_scale * torch.sigmoid(self.graph_gate) * self.message(neighbour_h - h)
        local = self.local_head(h)
        if self.decoder == "none":
            return self.local_head(hg)
        raw = self.rate_head(hg)
        component = (
            (torch.log1p(self.exposure_forecast(rawx, raw, graph_scale)) - self.fold["mean"])
            / self.fold["std"]
            if self.decoder == "seir"
            else raw
        )
        if self.use_router:
            weights = self.routing_weights(h)
            experts = torch.stack([p, local, self.local_head(hg), component], -1)
            result = (weights * experts).sum(-1)
            return (result, weights, experts) if return_details else result
        base = p + torch.sigmoid(self.persistence_gate) * (local - p)
        return base + torch.sigmoid(self.mix) * (component - base)
