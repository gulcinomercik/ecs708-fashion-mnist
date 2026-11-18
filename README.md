# ECS708 Machine Learning: Fashion-MNIST investigation

Coursework for ECS708 Machine Learning (Queen Mary University of London, 2026/27).

**Research question.** What structure exists in Fashion-MNIST, how can different
machine-learning models exploit that structure, and what do their successes and
failures teach us about the data and its representations?

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Fashion-MNIST (Zalando Research): 28x28 greyscale, 10 classes, 60,000 official
train and 10,000 official test images.

The data is not stored in this repository. `src/data.py` downloads it
automatically with `torchvision` on first use into `data/fashion/`.
Alternatively, copy the four `.gz` files from
<https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion>
into `data/fashion/`.

## Part 0: fixed partitions (frozen)

```bash
python scripts/make_splits.py
```

Creates `splits/splits_seed708.npz` (index arrays) and `splits/splits_seed708.json`
(seed, method, sizes, class counts). The files are committed, so the partitions
are fixed; re-running the script reproduces them exactly and refuses to
overwrite different ones.

| Partition | Size | Source | Per class |
|---|---|---|---|
| `train_idx` | 50,000 | official train | 5,000 |
| `val_idx` | 10,000 | official train | 1,000 |
| official test | 10,000 | official test | 1,000 |
| `train_subset10k_idx` (CPU option, 0.1.4) | 10,000 | subset of `train_idx` | 1,000 |
| `cluster_train_idx` (0.1.5) | 3,000 | subset of `train_idx` | 300 |
| `cluster_val_idx` (0.1.5) | 1,000 | subset of `val_idx` | 100 |
| `cluster_test_idx` (0.1.5) | 1,000 | official test | 100 |

- Seed: **708**, used for every partition.
- Method: `sklearn.model_selection.train_test_split(stratify=labels, random_state=708)`.
- Labels are used only to stratify. They are never used to compute
  distances, fit centroids or assign clusters.
- The test set is not used until the final, pre-registered evaluation (R1, R6, R7).
- Preprocessing: pixels divided by 255 (a fixed transform with no fitted
  statistics). Any fitted scaler is fitted on the training partition only (R2).

## Repository layout

```
README.md            install, data, commands, hardware, runtimes
requirements.txt     pinned versions
configs/             one config file per experiment ID
splits/              saved split and subset indices (+ seed)
src/                 data, models, training, evaluation, clustering code
results/             logs, metrics tables, experiment_register.csv (R3)
figures/             generated Notebook figures
scripts/             make_splits.py and helper scripts
```

## Hardware and runtimes

| Item | Value |
|---|---|
| Machine | Cloud Linux container, Intel Xeon @ 2.30 GHz, 2 CPU cores, 7 GB RAM, no GPU |
| Python | 3.13 |
| PyTorch | 2.15 (CPU execution) |

Approximate training, feature-extraction and inference times are added per
experiment in `results/experiment_register.csv` and summarised here as the
parts are completed.
