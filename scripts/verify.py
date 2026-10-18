"""Check saved model predictions on CPU and check the order of data splits."""

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


def verify(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and all(Path(n).name == n for n in z.namelist())
        z.extractall(dest)
    completion = json.loads((dest / "completion.json").read_text())
    assert completion["status"] == "complete"
    for item in completion["files"]:
        assert hashlib.sha256((dest / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    notebook = json.loads((dest / "submitted_notebook.ipynb").read_text())
    codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    code_hash = hashlib.sha256(json.dumps(codes, sort_keys=True).encode()).hexdigest()
    assert code_hash == completion["code_hash"]
    plan = json.loads((dest / "evaluation_plan.json").read_text())
    assert plan["code_hash"] == code_hash and plan["data_hash"] == completion["data_hash"]
    data = pd.read_csv(dest / "cases_25.csv")
    names = list(data.columns[:25])
    raw = data[names].to_numpy()
    population = json.loads((dest / "population_audit.json").read_text())["census2012"]
    env = dict(
        np=np,
        torch=torch,
        nn=nn,
        F=F,
        math=math,
        N=25,
        WIDTH=64,
        WINDOW=3,
        HORIZON=3,
        POP=np.array([population[n] for n in names], dtype=np.float32),
    )
    for code in codes:
        if (
            code.startswith("def conservative_step")
            or "class Candidate(nn.Module):" in code
            or code.startswith('"""Lagged exposure:')
            or code.startswith('"""State observer:')
        ):
            exec(code, env)
    scores = pd.read_csv(dest / "scores.csv")
    for baseline_file in dest.glob("persistence_*.npz"):
        baseline = np.load(baseline_file)
        ids = baseline["idx"]
        np.testing.assert_array_equal(
            baseline["prediction"], np.repeat(raw[ids - 1, :, None], 3, 2)
        )
        np.testing.assert_array_equal(baseline["truth"], np.stack([raw[i : i + 3].T for i in ids]))
    trials = json.loads((dest / "trials.json").read_text())
    assert len(trials) == completion["fits"] and len({t["token"] for t in trials}) == len(trials)
    torch.set_num_threads(1)
    for trial in trials:
        token = trial["token"]
        ckpt = torch.load(dest / (token + ".pt"), map_location="cpu", weights_only=False)
        assert ckpt["code_hash"] == code_hash and ckpt["data_hash"] == plan["data_hash"]
        fold = ckpt["fold"]
        valyear = trial["year"] if trial["phase"] == "development" else 2025
        train = fold["idx"]["train"]
        val = fold["idx"]["val"]
        assert train.max() + 2 < val.min()
        assert all((data.clinical_year.iloc[i : i + 3] < valyear).all() for i in train)
        assert all((data.clinical_year.iloc[i : i + 3] == valyear).all() for i in val)
        touched = sorted({j for i in train for j in range(i - 3, i + 3)})
        assert np.isclose(np.log1p(raw[touched]).mean(), fold["mean"], rtol=1e-12)
        assert np.isclose(np.log1p(raw[touched]).std() + 1e-8, fold["std"], rtol=1e-12)
        model = env["Candidate"](ckpt["config"], fold)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        curve = pd.read_csv(dest / (token + "_curve.csv"))
        assert len(curve) > 0 and np.isfinite(curve.train_loss).all()
        best_row = curve.iloc[ckpt["best_epoch"] - 1]
        assert np.isclose(
            best_row.validation_RMSE, curve.validation_RMSE.min(), rtol=1e-10, atol=1e-8
        )
        for split in ["val"] if trial["phase"] == "development" else ["val", "test"]:
            arrays = np.load(dest / (token + "_" + split + ".npz"))
            ids = arrays["idx"]
            np.testing.assert_array_equal(ids, fold["idx"][split])
            expectedyear = valyear if split == "val" else 2026
            assert all((data.clinical_year.iloc[i : i + 3] == expectedyear).all() for i in ids)
            if split == "test":
                assert val.max() + 2 < ids.min()
            truth = np.stack([raw[i : i + 3].T for i in ids])
            np.testing.assert_array_equal(truth, arrays["truth"])
            rx = np.stack([raw[i - 3 : i].T for i in ids]).astype(np.float32)
            x = (np.log1p(rx) - fold["mean"]) / fold["std"]
            p = np.repeat(x[:, :, -1:], 3, 2)
            with torch.no_grad():
                z = model(torch.tensor(x), torch.tensor(p), torch.tensor(rx)).numpy()
            pred = np.maximum(np.expm1(z.astype(np.float64) * fold["std"] + fold["mean"]), 0)
            np.testing.assert_allclose(pred, arrays["prediction"], rtol=1e-4, atol=0.001)
            row = scores[(scores.token == token) & (scores.split == split)].iloc[0]
            assert np.isclose(
                np.sqrt(((arrays["prediction"] - truth) ** 2).mean()), row.RMSE, rtol=1e-10
            )
            assert np.isclose(np.abs(arrays["prediction"] - truth).mean(), row.MAE, rtol=1e-10)
            for h in range(3):
                assert np.isclose(
                    np.sqrt(((arrays["prediction"][:, :, h] - truth[:, :, h]) ** 2).mean()),
                    row["RMSE_h" + str(h + 1)],
                    rtol=1e-10,
                )
    dev = scores[scores.phase == "development"]
    assert len(dev) == plan["expected_development_fits"]
    for row in dev.itertuples():
        base = np.load(dest / ("persistence_dev_" + str(row.year) + ".npz"))
        persistence = np.sqrt(((base["prediction"] - base["truth"]) ** 2).mean())
        assert np.isclose(row.skill, 1 - row.RMSE / persistence, rtol=1e-10)
    freeze = json.loads((dest / "learning_rate_freeze.json").read_text())
    assert freeze["test_predictions_at_freeze"] == 0 and freeze["development_fits"] == len(dev)
    aggregate = dev.groupby(["name", "lr"]).skill.mean().reset_index()
    for name in freeze["selected"]:
        winner = (
            aggregate[aggregate.name == name]
            .sort_values(["skill", "lr"], ascending=[False, True])
            .iloc[0]
        )
        assert freeze["selected"][name] == winner.lr
    for cfg in plan["configs"]:
        expected = freeze["selected"][cfg.get("parent", cfg["name"])]
        assert freeze["primary"][cfg["name"]] == expected
        primary = scores[
            (scores.phase == "final") & (scores.name == cfg["name"]) & (~scores.sensitivity)
        ]
        assert set(primary.seed) == set(plan["final_seeds"]) and set(primary.lr) == {expected}
    result = {
        "verified": True,
        "smoke": completion["smoke"],
        "fits": len(trials),
        "code_hash": code_hash,
        "hashes_checked": len(completion["files"]),
        "cpu_parity": True,
        "chronology_and_selection": True,
    }
    (dest / "independent_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    verify(args.archive, args.destination)
