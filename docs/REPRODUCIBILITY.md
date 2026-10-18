# Check saved results or train again

These are different tasks. Checking saved results reloads existing weights and recalculates errors. Training again starts with new weights and can give slightly different results on a different computer.

## Option 1: check the recorded experiment on CPU

Use Python 3.11 or later. Open a terminal in the repository folder and create a virtual environment, which keeps this project's packages separate:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. Then run:

```sh
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
python scripts/fetch_artifacts.py
python scripts/verify.py artifacts/evaluation_outputs.zip artifacts/verified
python scripts/build_figures.py
python -m unittest discover -s tests -v
```

The downloader uses `artifacts.lock.json` to choose an exact Hugging Face version and check file hashes. The verifier checks the original 265 saved neural fits. It rebuilds their predictions from saved weights, rather than training them again. Figures are written to `figures/publication/`.

`results/environment.json` records the original Kaggle runtime. The complete package list, `pip_freeze.txt`, is kept unchanged inside `evaluation_outputs.zip`. That list includes packages preinstalled by the platform that this project does not use. The repository's requirements files list the packages needed for training or checking results.

## Option 2: train from the checked public tables

Open `notebooks/LEG_SEIR_train_from_audited_data.ipynb` in a fresh Kaggle notebook or another GPU environment.

1. Enable GPU and Internet. Check `torch.cuda.is_available()` before starting.
2. Keep Kaggle's CUDA-enabled PyTorch. **Do not install `requirements-cpu.txt` on Kaggle GPU:** it contains CPU-only PyTorch.
3. If needed, install the other recorded packages from `requirements-training.txt` in a setup cell. Restart the kernel after changing the environment.
4. Save the notebook before running. If the code cannot find it, set `STUDY_NOTEBOOK` to its saved path.
5. Run all cells. New outputs go to a folder with a timestamp. Keep them separate from the original saved experiment.

The notebook downloads checked case and population tables. It does not read saved forecasts or weights as training inputs. For offline use, point `LEG_SEIR_DATA_DIR` to a previously downloaded input folder.

The model and training cells are unchanged from the original experiment. The input loader is new because it reads prepared tables instead of parsing report PDFs. The new loader passed a short CPU execution check; a full new GPU retraining was not completed for this release.

The full study has 180 development fits and 75 main final fits, plus extra checks of removal models at other learning rates. The original run had 10 such extra fits. A new run can choose different settings or numbers of extra fits because software and hardware affect training.

For a short execution check, use `python scripts/train.py --smoke`. It runs two epochs with one seed. Its output is not a reproduction of the paper's scores.

## Option 3: start from the original report PDFs

The unchanged notebook is `experiments/frozen/Dengue_final_study.ipynb`. It needs the original source files. From a fresh checkout, run:

```sh
python scripts/fetch_artifacts.py --data-only
python scripts/retrieve_original_sources.py
python scripts/train_original.py
```

The source script downloads original URLs and checks their recorded hashes. It reports files that are missing or have changed, rather than silently accepting new versions. The original team copy is kept separately and is not covered by MIT. Downloading 714 original files can be slow. This full download-and-retrain path was not rerun from start to finish for the lecturer release; the public-table notebook is the recommended training path.

The original runner refuses to overwrite an existing `evaluation_outputs/` folder. Keep each experiment in a separate checkout or output location.

## Reading model source and viewing figures

`src/leg_seir/_frozen/` contains unchanged parts of the original notebook. They share variables and must run in order. Use `leg_seir.build_candidate(config, fold, population)` instead of importing these files one at a time. [CODE_GUIDE.md](CODE_GUIDE.md) explains the functions.

`scripts/build_figures.py` rebuilds all nine figures from saved results. `notebooks/LEG_SEIR_results_gallery.ipynb` displays these results without training. The PDFs in `paper/` are maintained separately from the team's Overleaf source. Reference records remain under `references/`. The supplied public PDF still uses older dataset-availability wording; the current links are in the README.

## How to interpret a successful check

- Agreement after loading saved weights does not mean a new training run must get identical weights.
- The average of five seed scores differs from the error of an averaged forecast.
- Seed standard deviation is not a confidence interval.
- The 2026 study is retrospective, not a new unseen future outbreak test.
- The alternative printed-repeat table was not used for a completed LEG-SEIR experiment.
- Citing a paper or downloading its PDF does not mean we reproduced its experiments.
