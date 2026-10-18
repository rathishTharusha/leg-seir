"""State observer: bounded local simplex correction before conservative SEIR."""

BaseCandidate = Candidate


class Candidate(BaseCandidate):
    def __init__(self, config, fold):
        self.is_observer = config.get("mode") == "observer"
        cfg = dict(config)
        if self.is_observer:
            cfg["mode"] = "router" if config.get("routing", True) else "lgi"
        super().__init__(cfg, fold)
        self.config = dict(config)
        if self.is_observer:
            self.observer_head = nn.Linear(WIDTH, 4)
            nn.init.zeros_(self.observer_head.weight)
            nn.init.zeros_(self.observer_head.bias)
            if config.get("state_features", True):
                self.state_projection = nn.Linear(4, WIDTH, bias=False)

    def prior_state(self, rawx):
        scale = torch.as_tensor(POP, device=rawx.device, dtype=rawx.dtype)[None] / 11
        i = (rawx[:, :, -1] / scale).clamp(1e-6, 0.5)
        e = (rawx[:, :, -2] / scale).clamp(1e-6, 0.5)
        s = torch.full_like(i, 0.318)
        r = (1 - s - e - i).clamp_min(0)
        state = torch.stack([s, e, i, r], -1)
        return state / state.sum(-1, keepdim=True)

    def corrected_state(self, x, rawx, observer_scale=1.0):
        prior = self.prior_state(rawx)
        if observer_scale == 0:
            return prior, prior, torch.zeros_like(prior)
        correction = observer_scale * 0.5 * torch.tanh(self.observer_head(self.base_hidden(x)))
        state = prior * torch.exp(correction)
        state = state / state.sum(-1, keepdim=True)
        return state, prior, correction

    def observer_forecast(self, state, rates, return_states=False):
        scale = torch.as_tensor(POP, device=state.device, dtype=state.dtype)[None] / 11
        prediction, states = [], []
        for h in range(HORIZON):
            beta = torch.exp((rates[:, :, h] - 1.1).clamp(-9, 1.1))
            total = torch.zeros_like(state[:, :, 0])
            for _ in range(7):
                state, onset = conservative_step(state, beta * state[:, :, 2])
                total = total + onset
                states.append(state)
            prediction.append(total * scale)
        y = torch.stack(prediction, -1)
        return (y, torch.stack(states, -2)) if return_states else y

    def forward(
        self, x, p, rawx, graph_scale=1.0, return_details=False, observer_scale=1.0, **kwargs
    ):
        if not self.is_observer:
            return super().forward(x, p, rawx, graph_scale, return_details, **kwargs)
        state, prior, _ = self.corrected_state(x, rawx, observer_scale)
        h = self.base_hidden(x)
        if hasattr(self, "state_projection"):
            h = h + self.state_projection(state - prior)
        hg = h
        if self.graph:
            avg = torch.einsum("ij,bjw->biw", self.adjacency(), h)
            hg = h + graph_scale * torch.sigmoid(self.graph_gate) * self.message(avg - h)
        local = self.local_head(h)
        if self.decoder == "none":
            return self.local_head(hg)
        rates = self.rate_head(hg)
        component = (
            (torch.log1p(self.observer_forecast(state, rates)) - self.fold["mean"])
            / self.fold["std"]
            if self.decoder == "seir"
            else rates
        )
        if self.use_router:
            weights = self.routing_weights(h)
            experts = torch.stack([p, local, self.local_head(hg), component], -1)
            y = (weights * experts).sum(-1)
            return (y, weights, experts) if return_details else y
        base = p + torch.sigmoid(self.persistence_gate) * (local - p)
        return base + torch.sigmoid(self.mix) * (component - base)
