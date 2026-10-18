"""Full development selection precedes every 2026 retrospective forecast."""

EPOCHS, PATIENCE, BATCH = (2, 2, 128) if EVALUATION_SMOKE else (300, 40, 128)
DEV_SEEDS = [0] if EVALUATION_SMOKE else [0, 1, 2]
FINAL_SEEDS = [0] if EVALUATION_SMOKE else [0, 1, 2, 3, 4]
LRS = [0.003] if EVALUATION_SMOKE else [0.001, 0.003]
if Path("/kaggle/input").exists():
    assert DEVICE.type == "cuda"
YEARS = np.array([r["clinical_year"] for r in records])
valid = [i for i in range(3, len(CASES) - 2) if np.isfinite(CASES[i - 3 : i + 3]).all()]


def make_evaluation_fold(year, test=False):
    idx = {
        "train": np.array([i for i in valid if (YEARS[i : i + 3] < year).all()]),
        "val": np.array([i for i in valid if (YEARS[i : i + 3] == year).all()]),
    }
    if test:
        assert (OUT / "learning_rate_freeze.json").exists(), "Freeze before creating test tensors"
        idx["test"] = np.array([i for i in valid if (YEARS[i : i + 3] == 2026).all()])
    assert all(len(v) for v in idx.values())
    assert idx["train"].max() + 2 < idx["val"].min()
    if test:
        assert idx["val"].max() + 2 < idx["test"].min()
    touched = sorted({j for i in idx["train"] for j in range(i - 3, i + 3)})
    log = np.log1p(CASES[touched])
    return {"idx": idx, "mean": float(log.mean()), "std": float(log.std() + 1e-8)}


PLAN = {
    "status": "frozen_before_fit",
    "code_hash": CODE_HASH,
    "data_hash": DATA_HASH,
    "source_policy": "frozen v3 unchanged",
    "configs": CONFIGS,
    "development_years": [2023, 2025],
    "learning_rates": LRS,
    "development_seeds": DEV_SEEDS,
    "final_seeds": FINAL_SEEDS,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "batch": BATCH,
    "width": WIDTH,
    "selection": "maximum mean fold skill vs persistence across folds and seeds; exact tie lower LR",
    "primary_removals": "same selected LR as parent; separately selected removal LR sensitivity if different",
    "test": "2026 retrospective rolling forecasts, no parameter refit/tuning from 2026 targets",
    "source_cutoff": "attached version1 archive through volume53 report33, not complete calendar2026",
    "smoke": EVALUATION_SMOKE,
    "expected_development_fits": len(CONFIGS) * len(LRS) * len(DEV_SEEDS) * 2,
    "expected_primary_final_fits": len(CONFIGS) * len(FINAL_SEEDS),
    "max_extra_removal_sensitivity_fits": 6 * len(FINAL_SEEDS),
}
(OUT / "evaluation_plan.json").write_text(json.dumps(PLAN, indent=2))
(OUT / "environment.json").write_text(
    json.dumps(
        {
            "python": sys.version,
            "torch": torch.__version__,
            "device": str(DEVICE),
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        },
        indent=2,
    )
)
rows, trials = [], []


def ridge_baseline(fold, splits, token):
    training = pack(fold, "train")

    def features(data):
        x = data["x"].cpu().numpy()
        return np.concatenate([x, np.diff(x, axis=-1)], -1).reshape(-1, 5)

    fitted = Ridge(alpha=1).fit(features(training), training["y"].cpu().numpy().reshape(-1, 3))
    joblib.dump(fitted, OUT / (token + ".joblib"))
    for split in splits:
        data = pack(fold, split)
        z = fitted.predict(features(data)).reshape(len(data["idx"]), 25, 3)
        np.savez_compressed(
            OUT / (token + "_" + split + ".npz"),
            idx=data["idx"],
            truth=data["truth"],
            prediction=counts(z, fold),
        )


def baseline(fold, split, token):
    ids = fold["idx"][split]
    truth = np.stack([CASES[i : i + 3].T for i in ids])
    pred = np.repeat(CASES[ids - 1, :, None], 3, 2)
    np.savez_compressed(OUT / (token + ".npz"), idx=ids, truth=truth, prediction=pred)
    return metrics(pred, truth)


def fit(config, fold, lr, seed, token, phase, year, sensitivity=False):
    seed_all(seed)
    tr, va = pack(fold, "train"), pack(fold, "val")
    model = Candidate(config, fold).to(DEVICE)
    if config.get("adapter_no_decay"):
        groups = [
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("adapter_")],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if not n.startswith("adapter_")],
                "weight_decay": 1e-4,
            },
        ]
        optimizer = torch.optim.Adam(groups, lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best, stale, best_state, best_epoch = float("inf"), 0, None, 0
    curve = []
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(tr["idx"]), device=DEVICE)
        losses = []
        for start in range(0, len(perm), BATCH):
            ix = perm[start : start + BATCH]
            optimizer.zero_grad(set_to_none=True)
            z = model(*[tr[k][ix] for k in ["x", "p", "rawx"]])
            loss = F.mse_loss(z, tr["y"][ix])
            assert torch.isfinite(loss)
            loss.backward()
            assert all(
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            )
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append((float(loss.detach().cpu()), len(ix)))
        score = metrics(counts(predict_z(model, va), fold), va["truth"])["RMSE"]
        curve.append(
            {
                "epoch": epoch + 1,
                "train_loss": sum(v * n for v, n in losses) / sum(n for _, n in losses),
                "validation_RMSE": score,
            }
        )
        if score < best - 1e-8:
            best, best_state, stale, best_epoch = (
                score,
                copy.deepcopy(model.state_dict()),
                0,
                epoch + 1,
            )
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint = {
        "state_dict": {k: v.cpu() for k, v in best_state.items()},
        "config": config,
        "fold": fold,
        "lr": lr,
        "seed": seed,
        "code_hash": CODE_HASH,
        "data_hash": DATA_HASH,
        "phase": phase,
        "year": year,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, OUT / (token + ".pt"))
    pd.DataFrame(curve).to_csv(OUT / (token + "_curve.csv"), index=False)
    # CPU reload and parity before its validation score can enter selection.
    cpu = Candidate(config, fold)
    cpu.load_state_dict(checkpoint["state_dict"])
    cpu.eval()
    with torch.no_grad():
        cz = cpu(*[va[k].cpu() for k in ["x", "p", "rawx"]]).numpy()
    np.testing.assert_allclose(
        counts(cz, fold), counts(predict_z(model, va), fold), rtol=1e-4, atol=0.001
    )
    splits = ["val"] if phase == "development" else ["val", "test"]
    for split in splits:
        data = va if split == "val" else pack(fold, split)
        pred = counts(predict_z(model, data), fold)
        arrays = {"idx": data["idx"], "truth": data["truth"], "prediction": pred}
        with torch.no_grad():
            if model.use_router and config["decoder"] != "none":
                _, w, experts = model(data["x"], data["p"], data["rawx"], return_details=True)
                arrays.update(router_weights=w.cpu().numpy(), router_expert_z=experts.cpu().numpy())
            if config["graph"]:
                arrays["adjacency"] = model.adjacency().detach().cpu().numpy()
                arrays["graph_off_prediction"] = counts(
                    model(data["x"], data["p"], data["rawx"], graph_scale=0).cpu().numpy(), fold
                )
            if config["mode"] == "lagged" and config["graph"]:
                arrays["lag_weights"] = model.lag_weights().detach().cpu().numpy()
                if hasattr(model, "import_gate"):
                    arrays["import_gate"] = torch.sigmoid(model.import_gate).detach().cpu().numpy()
            if config["mode"] == "observer":
                state, prior, correction = model.corrected_state(data["x"], data["rawx"])
                arrays.update(
                    observer_state=state.cpu().numpy(),
                    observer_prior=prior.cpu().numpy(),
                    observer_correction=correction.cpu().numpy(),
                )
        np.savez_compressed(OUT / (token + "_" + split + ".npz"), **arrays)
        result = metrics(pred, data["truth"])
        row = {
            "token": token,
            "name": config["name"],
            "role": config["role"],
            "phase": phase,
            "year": year,
            "split": split,
            "lr": lr,
            "seed": seed,
            "sensitivity": sensitivity,
            **result,
            "best_epoch": best_epoch,
            "epochs_run": len(curve),
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        for h in range(3):
            row["RMSE_h" + str(h + 1)] = metrics(pred[:, :, h], data["truth"][:, :, h])["RMSE"]
        if phase == "development":
            persistence = metrics(np.repeat(CASES[data["idx"] - 1, :, None], 3, 2), data["truth"])[
                "RMSE"
            ]
            row["skill"] = 1 - result["RMSE"] / persistence
        rows.append(row)
    trials.append(
        {
            "token": token,
            "phase": phase,
            "config": config["name"],
            "lr": lr,
            "seed": seed,
            "year": year,
            "sensitivity": sensitivity,
        }
    )
    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
    (OUT / "trials.json").write_text(json.dumps(trials, indent=2))
    print("FIT_COMPLETE", token, "validation RMSE", best, flush=True)


for year in [2023, 2025]:
    fold = make_evaluation_fold(year)
    baseline(fold, "val", "persistence_dev_" + str(year))
    ridge_baseline(fold, ["val"], "ridge_dev_" + str(year))
    for config in CONFIGS:
        for lr in LRS:
            for seed in DEV_SEEDS:
                token = f"dev_{year}_{config['name']}_lr{lr:g}_seed{seed}"
                fit(config, fold, lr, seed, token, "development", year)
assert len(trials) == PLAN["expected_development_fits"]
development = pd.DataFrame(rows)
assert (development.year < 2026).all() and (development.split == "val").all()
selection_scores = development.groupby(["name", "lr"]).skill.mean().reset_index()
selected = {}
for name in [c["name"] for c in CONFIGS]:
    group = selection_scores[selection_scores.name == name].sort_values(
        ["skill", "lr"], ascending=[False, True]
    )
    selected[name] = float(group.iloc[0].lr)
primary = {c["name"]: selected[c.get("parent", c["name"])] for c in CONFIGS}
freeze = {
    "status": "frozen_from_pre2026_only",
    "selected": selected,
    "primary": primary,
    "development_scores": selection_scores.to_dict("records"),
    "code_hash": CODE_HASH,
    "data_hash": DATA_HASH,
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "development_fits": len(trials),
    "test_predictions_at_freeze": 0,
}
(OUT / "learning_rate_freeze.json").write_text(json.dumps(freeze, indent=2))
print("LEARNING_RATES_FROZEN_BEFORE_TEST", json.dumps(primary), flush=True)
final_fold = make_evaluation_fold(2025, test=True)
baseline(final_fold, "val", "persistence_final_val")
baseline(final_fold, "test", "persistence_final_test")
ridge_baseline(final_fold, ["val", "test"], "ridge_final")
for config in CONFIGS:
    name = config["name"]
    settings = [(primary[name], False)]
    if config["role"] == "removal" and selected[name] != primary[name]:
        settings.append((selected[name], True))
    for lr, sensitivity in settings:
        for seed in FINAL_SEEDS:
            token = f"final_{name}_lr{lr:g}_seed{seed}" + ("_sensitivity" if sensitivity else "")
            fit(config, final_fold, lr, seed, token, "final", 2026, sensitivity)
summary = (
    pd.DataFrame(rows)
    .query("phase=='final' and split=='test' and sensitivity==False")
    .groupby("name")
    .agg(mean_RMSE=("RMSE", "mean"), seed_SD=("RMSE", "std"))
)
summary.to_csv(OUT / "test_summary.csv")
primary_rows = pd.DataFrame(rows).query("phase=='final' and split=='test' and sensitivity==False")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name in ["horizon_full", "lagged_full", "observer_full"]:
    selected_rows = primary_rows[primary_rows.name == name]
    errors = selected_rows[["RMSE_h1", "RMSE_h2", "RMSE_h3"]].mean().to_numpy()
    axes[0].plot([1, 2, 3], errors, marker="o", label=name)
    arrays = [np.load(OUT / (token + "_test.npz")) for token in selected_rows.token]
    ensemble = np.mean([arr["prediction"] for arr in arrays], axis=0)
    np.savez_compressed(
        OUT / (name + "_test_ensemble.npz"),
        idx=arrays[0]["idx"],
        truth=arrays[0]["truth"],
        prediction=ensemble,
    )
    axes[1].plot(arrays[0]["idx"], ensemble[:, :, 0].sum(1), label=name)
axes[0].set(xlabel="Report-observation horizon", ylabel="Count RMSE; mean over seeds")
axes[1].plot(
    arrays[0]["idx"], arrays[0]["truth"][:, :, 0].sum(1), color="black", label="Observed total"
)
axes[1].set(xlabel="Report slot of forecast origin", ylabel="25-district total, horizon 1")
for ax in axes:
    ax.legend()
fig.suptitle("Retrospective 2026 case study; frozen pre-2026 settings")
fig.tight_layout()
for extension in ["png", "svg", "pdf"]:
    fig.savefig(OUT / ("case_study_horizons_totals." + extension), dpi=180)
plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 6))
summary.sort_values("mean_RMSE", ascending=False).mean_RMSE.plot.barh(ax=ax)
ax.axvline(
    metrics(
        np.repeat(CASES[final_fold["idx"]["test"] - 1, :, None], 3, 2),
        pack(final_fold, "test")["truth"],
    )["RMSE"],
    color="red",
    label="Persistence",
)
ax.set_xlabel("Retrospective 2026 count RMSE; five-seed mean")
ax.legend()
fig.tight_layout()
for extension in ["png", "svg", "pdf"]:
    fig.savefig(OUT / ("retrospective_test_ranking." + extension), dpi=180)
plt.close(fig)
completion = {
    "status": "complete",
    "smoke": EVALUATION_SMOKE,
    "code_hash": CODE_HASH,
    "data_hash": DATA_HASH,
    "development_fits": PLAN["expected_development_fits"],
    "final_fits": len(trials) - PLAN["expected_development_fits"],
    "fits": len(trials),
    "files": [
        {"path": p.name, "sha256": file_sha(p)}
        for p in sorted(OUT.iterdir())
        if p.is_file() and p.name != "completion.json"
    ],
}
(OUT / "completion.json").write_text(json.dumps(completion, indent=2))
with zipfile.ZipFile(BASE / "evaluation_outputs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            archive.write(p, p.name)
print("EVALUATION_COMPLETE", len(trials), "fits", flush=True)
