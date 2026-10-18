# Reproduce, verify, or retrain

Start in the repository root. Use Python 3.11+ in a new virtual environment:

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
python scripts/fetch_artifacts.py
python scripts/verify.py artifacts/evaluation_outputs.zip artifacts/verified
python scripts/build_figures.py
python -m unittest discover -s tests -v
```

`artifacts.lock.json` pins the Hugging Face commit and every downloaded byte hash. The CPU verification covers the original recorded run. It is distinct from training again. See `results/environment.json` and the original `pip_freeze.txt` inside `evaluation_outputs.zip` for the Kaggle GPU runtime; the CPU requirements describe the tested release-verification environment. Runtime differences may change training weights. The complete platform package inventory is preserved unchanged in that archive; it includes preinstalled software unrelated to this experiment. Repository requirements list the dependencies needed for training or verification.

## Train from the public audited input tables

On Kaggle, enable GPU and Internet and preserve its CUDA-enabled PyTorch. **Do not install `requirements-cpu.txt` there:** its CPU-only PyTorch is intended for local verification. `requirements-training.txt` records the non-PyTorch dependencies from the original GPU run; install them in a setup cell if needed, restart the kernel after environment changes, then execute the standalone notebook. Check `torch.cuda.is_available()` before starting. The new loader was smoke-tested on CPU; a fresh full GPU retraining was not performed for the release. The recorded GPU environment is preserved separately so runtime differences remain visible.

Run `notebooks/LEG_SEIR_train_from_audited_data.ipynb` in a fresh GPU kernel, with internet enabled or `LEG_SEIR_DATA_DIR` pointing to a previously downloaded snapshot. It uses only audited case/population/provenance inputs, not predictions or checkpoints, to fit the models. The scientific model and training cells are unchanged from the evaluated notebook; its input loader is different and has a new code identity. The full study performs 180 development and 75 primary final fits plus selected removal-LR sensitivities (85 final fits in the original run); exact retrained selections/sensitivity counts can differ by runtime. Outputs go to a new timestamped directory. Never replace archived evidence with new outputs.

The original immutable full-study notebook remains under `experiments/frozen/`. It parses the original report PDFs and requires the original source snapshot. `scripts/retrieve_original_sources.py` retrieves the publicly referenced source URLs with strict hashes; some providers may change or remove old files. That failure is explicit. The raw third-party snapshot is retained privately by the research team and is not covered by MIT.

For that original workflow, first fetch the public artifact metadata, run `python scripts/retrieve_original_sources.py` to populate `sources/` and its `manifest.json`, then run `python scripts/train_original.py` from a fresh checkout. The wrapper supplies the original notebook path and repository working directory without changing its code. It refuses to overwrite an existing `evaluation_outputs/` run. Reacquiring 714 original files can be slow; this source-retrieval workflow was not rerun end to end for the lecturer release. The audited-input notebook avoids those PDF/geography dependencies and is the recommended public training path.

## Model source

Readable scientific fragments are extracted from the frozen notebook under `src/leg_seir/_frozen/`. They intentionally use a shared namespace, matching the evaluated implementation. `leg_seir.build_candidate(config, fold, population)` supplies that namespace and preserves checkpoint parameter names. The original notebook, not an adapted external architecture, defines the recorded experiment.

## Paper and figures

`scripts/build_figures.py` regenerates all nine publication figures from verified saved outputs. `notebooks/LEG_SEIR_results_gallery.ipynb` is an inspection notebook. It does not train models. Compiled manuscript and supplementary PDFs are in `paper/`. The team maintains the LaTeX/Overleaf project separately; sources and the Overleaf ZIP are excluded from the current public repository and release assets. Bibliographic research records remain under `references/`. The team-supplied manuscript PDF is preserved as received; its wording about dataset release predates the public Hugging Face release linked in the README.

## Important distinctions

- Seed-mean RMSE is not ensemble RMSE; seed SD is not a confidence interval.
- The 2026 case study was previously inspected and is retrospective.
- Alternative printed-repeat data are archived but not a completed LEG-SEIR sensitivity experiment.
- CPU parity verifies stored weights, not a fresh fit or biological validity.
- Related-work papers were not all reproduced as matched numerical competitors.
