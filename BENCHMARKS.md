# Reproduction and benchmark scope

The released causal transformer now has a training-only vocabulary, exact-overlap removal, disjoint train/validation/test data, best-validation checkpoint export and raw-text inference. A new 10,000-train/2,000-test IMDB protocol reaches **74.9% accuracy ** with 128-token inputs.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline source and generated-data training, checkpoint selection and inference reload checks; not a real-data benchmark | None |
| `train` | Configurable training on supplied fixed splits; default budget differs from named published reference protocols | data, config |
| `reference-training` | Exact published fresh-training budgets on identically prepared splits; CPU reference; checkpoint selection uses validation only | data, config |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

## Evidence boundary

This uses the current causal/final-token architecture, not the different architecture in the retained original notebook. The benchmark is a reduced-data reference run, not a full IMDB leaderboard claim.

Keep input acquisition and source credits in [DATA.md](DATA.md). Historical results, new reference runs and small contract fixtures have separate evidential scope.

## Direct commands and inputs

Obtain the [Stanford IMDB archive](https://ai.stanford.edu/~amaas/data/sentiment/) or the SHA-verified train/test Parquet mirror identified by `prepare_data.py`. Exact text overlap is removed from training/validation before fitting the vocabulary; official test labels never select the model.

```sh
python prepare_data.py --archive aclImdb_v1.tar.gz --context 128 --limit-train 10000 --limit-test 2000 --output data/imdb.npz
python train.py --data data/imdb.npz --config data/imdb.config.json --epochs 10 --batch-size 64 --seed 7 --output runs/imdb
python predict.py --checkpoint checkpoints/model.pt --vocabulary checkpoints/vocabulary.json "The film was thoughtful and moving."
```

To use the mirror, replace `--archive` with `--train-parquet train.parquet --test-parquet test.parquet`. `74.9%` describes this reduced-data protocol, not a full-data score. Vocabulary hash binding is checked at inference.


## Device selection

`train.py --device cpu` is the default reference path. `--device mps` explicitly selects an Apple Silicon GPU; `--device cuda` requires a CUDA GPU. Unsupported or unavailable devices fail explicitly. No automatic CPU fallback is enabled. Metrics record the device, runtime and elapsed time. CPU and GPU runs are separate numerical protocols; equal seeds do not imply equal weights or results across devices. Resume requires the checkpoint's original device type and saves its accelerator RNG state.

For a generated-data device integration check, set `TRAIN_TEST_DEVICE=mps` when running `python test_training_pipeline.py`. This checks training and checkpoint reload, not real-data model quality.

Strict MPS training currently fails for the transformer's embedding backward operation in the tested runtime. To explicitly accept nondeterministic operations, add `--allow-nondeterministic`; this mode is recorded and does not claim exact resumed or repeated weights. The strict CPU path remains the reference. To check this GPU route, run `TRAIN_TEST_DEVICE=mps TRAIN_TEST_ALLOW_NONDETERMINISTIC=1 python test_training_pipeline.py`.

## Recorded command definitions

### checks

Offline source and generated-data training, checkpoint selection and inference reload checks; not a real-data benchmark

```text
{python} smoke_test.py
{python} test_training_pipeline.py
```

### train

Configurable training on supplied fixed splits; default budget differs from named published reference protocols

```text
{python} train.py --data {input:data} --config {input:config} --output {output}/training
```

### reference-training

Exact published fresh-training budgets on identically prepared splits; CPU reference; checkpoint selection uses validation only

```text
{python} train.py --data {input:data} --config {input:config} --epochs 10 --batch-size 64 --seed 7 --device cpu --output {output}/reference-7
```

The observed local runtime is recorded in [reference-environment-macos.json](reference-environment-macos.json). The reproduction driver hashes source files and inputs for each run. Published result records retain their own original runtime and revision; changing the device or budget creates a different experiment.

## Pinned local reference installation

[requirements-reference-macos.txt](requirements-reference-macos.txt) pins the installed runtime/test dependency closure for Python 3.14.3 on macOS ARM64. It accompanies the observed environment record. A new empty Python3.14.3 macOS environment installed these pinned dependencies, passed `pip check`, and passed all eight offline repository check profiles. [The validation record](environment-validation.json) binds the source revisions, Python source hashes, versions and outcomes. This is a platform-specific reference, not a Linux/CUDA lock or a repetition of every full dataset benchmark. The GitHub checks use their separately declared Linux environment.

```sh
python3.14 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r requirements-reference-macos.txt
```

Use the recorded exact Python patch version when comparing numerical results. Notebook tooling and optional FID dependencies are outside this runtime/test snapshot.
