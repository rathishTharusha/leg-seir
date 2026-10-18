"""Local-preserving graph prototypes; embedded in each standalone notebook."""


class Candidate(nn.Module):
    def __init__(self, config, fold):
        super().__init__()
        self.config, self.fold = dict(config), fold
        self.identity = config.get("identity", True)
        self.differences = config.get("differences", True)
        self.graph = config.get("graph", True)
        self.mode = config.get("mode", "lgi")
        self.decoder = config.get("decoder", "seir")
        self.use_router = self.mode == "router" or (
            self.mode == "adapters" and config.get("routing", True)
        )
        self.embedding = nn.Parameter(torch.randn(N, 4) * 0.05) if self.identity else None
        self.first = nn.Linear(
            WINDOW + (WINDOW - 1 if self.differences else 0) + (4 if self.identity else 0), WIDTH
        )
        self.next = nn.Linear(WIDTH, WIDTH)
        self.local_head = nn.Linear(WIDTH, HORIZON)
        if self.mode == "adapters":
            self.adapter_rank = config.get("adapter_rank", 8)
            self.district_conditioned = config.get("district_conditioned", True)
            self.adapter_down = nn.Linear(WIDTH, self.adapter_rank, bias=False)
            self.adapter_up = nn.Linear(self.adapter_rank, WIDTH, bias=False)
            nn.init.zeros_(self.adapter_up.weight)
            self.adapter_gate = nn.Parameter(torch.tensor(-2.0))
            if self.district_conditioned:
                self.adapter_embedding = nn.Parameter(torch.randn(N, 4) * 0.05)
                self.adapter_condition = nn.Linear(4, self.adapter_rank, bias=False)
            else:
                self.adapter_coefficients = nn.Parameter(torch.full((self.adapter_rank,), 0.05))
        if self.mode == "growth":
            # Preserve level information locally while graph messages see only changes.
            self.level_projection = nn.Linear(1, WIDTH, bias=False)
        if self.graph:
            self.left = nn.Parameter(torch.randn(N, 8) * 0.1)
            self.right = nn.Parameter(torch.randn(N, 8) * 0.1)
            self.message = nn.Linear(WIDTH, WIDTH, bias=False)
            self.graph_gate = nn.Parameter(torch.full((1,), -2.0))
            if self.mode == "channels":
                self.channel_gate = nn.Linear(WIDTH, 3)
        if self.decoder != "none":
            self.rate_head = nn.Linear(WIDTH, HORIZON)
            if not self.use_router:
                self.mix = nn.Parameter(torch.full((HORIZON,), -2.0))
                self.persistence_gate = nn.Parameter(torch.full((HORIZON,), 2.0))
        if self.use_router and self.decoder != "none":
            self.router_style = config.get(
                "router_style", "horizon" if self.mode == "adapters" else "conditional"
            )
            if self.router_style == "conditional":
                self.router = nn.Linear(WIDTH, HORIZON * 4)
                nn.init.zeros_(self.router.weight)
                nn.init.zeros_(self.router.bias)
            elif self.router_style == "horizon":
                self.router_logits = nn.Parameter(torch.zeros(HORIZON, 4))
            elif self.router_style == "fixed":
                self.register_buffer("fixed_router_weights", torch.full((HORIZON, 4), 0.25))
            else:
                raise ValueError(self.router_style)

    def adjacency(self):
        scores = self.left @ self.right.T
        mask = torch.eye(N, device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(mask, -torch.inf)
        top = scores.topk(5, dim=-1).indices
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter(1, top, True)
        return torch.softmax(scores.masked_fill(~keep, -torch.inf), -1)

    def base_hidden(self, x):
        v = x - x[:, :, -1:] if self.mode == "growth" else x
        features = [v]
        if self.differences:
            features.append(x[:, :, 1:] - x[:, :, :-1])
        if self.identity:
            features.append(self.embedding[None].expand(x.shape[0], -1, -1))
        return F.relu(self.next(F.relu(self.first(torch.cat(features, -1)))))

    def adapter_coeffs(self):
        if self.district_conditioned:
            return torch.tanh(self.adapter_condition(self.adapter_embedding))
        return torch.tanh(self.adapter_coefficients)[None].expand(N, -1)

    def adapter_delta(self, h):
        return (
            0.1
            * torch.sigmoid(self.adapter_gate)
            * self.adapter_up(self.adapter_down(h) * self.adapter_coeffs()[None])
        )

    def local_hidden(self, x, adapter_scale=1.0):
        h = self.base_hidden(x)
        return h + adapter_scale * self.adapter_delta(h) if self.mode == "adapters" else h

    def routing_weights(self, h):
        if self.router_style == "conditional":
            return torch.softmax(self.router(h).reshape(*h.shape[:2], HORIZON, 4), -1)
        weights = (
            torch.softmax(self.router_logits, -1)
            if self.router_style == "horizon"
            else self.fixed_router_weights
        )
        return weights[None, None].expand(h.shape[0], h.shape[1], -1, -1)

    def forward(self, x, p, rawx, graph_scale=1.0, return_details=False, adapter_scale=1.0):
        h = self.local_hidden(x, adapter_scale=adapter_scale)
        level = self.level_projection(x[:, :, -1:]) if self.mode == "growth" else 0.0
        local = self.local_head(h + level) + (p if self.mode == "growth" else 0.0)
        hg = h
        if self.graph:
            avg = torch.einsum("ij,bjw->biw", self.adjacency(), h)
            delta = avg - h
            if self.mode == "channels":
                weights = torch.softmax(self.channel_gate(h), -1)
                delta = weights[:, :, 1:2] * avg + weights[:, :, 2:3] * (h - avg)
            hg = h + graph_scale * torch.sigmoid(self.graph_gate).mean() * self.message(delta)
        if self.decoder == "none":
            return self.local_head(hg + level) + (p if self.mode == "growth" else 0.0)
        raw = self.rate_head(hg + level)
        if self.decoder == "seir":
            y = mechanistic_forecast(rawx, raw, feedback=True)
            component = (torch.log1p(y) - self.fold["mean"]) / self.fold["std"]
        else:
            component = raw + (p if self.mode == "growth" else 0.0)
        if self.use_router:
            weights = self.routing_weights(h)
            graph_forecast = self.local_head(hg)
            experts = torch.stack([p, local, graph_forecast, component], -1)
            result = (weights * experts).sum(-1)
            return (result, weights, experts) if return_details else result
        base = p + torch.sigmoid(self.persistence_gate) * (local - p)
        return base + torch.sigmoid(self.mix) * (component - base)
