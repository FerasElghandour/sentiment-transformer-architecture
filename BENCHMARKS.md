# Reproduction and benchmark scope

The released causal transformer now has a training-only vocabulary, exact-overlap removal, disjoint train/validation/test data, best-validation checkpoint export and raw-text inference. A new 10,000-train/2,000-test IMDB protocol reaches **74.9% accuracy ** with 128-token inputs.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline behavioral regressions; not a fresh dataset benchmark | None |
| `train` | New reference training protocol on supplied fixed splits; validation-selected checkpoint and held-out metrics | data, config |

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

## Recorded command definitions

### checks

```text
{python} smoke_test.py
{python} test_training_pipeline.py
```

### train

```text
{python} train.py --data {input:data} --config {input:config} --output {output}/training
```

`{python}` is the current interpreter. `{output}` is the new run directory. `{input:name}` refers to a named absolute input path supplied to the driver. Install declared dependencies in a clean environment before executing. Never execute untrusted serialized models or notebooks.
