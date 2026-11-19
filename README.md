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

The data is not stored in this repository. The notebook downloads it
automatically with `torchvision` on first use into `data/fashion/`.
Alternatively, copy the four `.gz` files from
<https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion>
into `data/fashion/`.

## How to run

Open `ecs708_fashion_mnist.ipynb` from the repository root and use *Restart and run all*
(about 8 minutes on a 2-core CPU, no GPU). The notebook is **self-contained**: no helper
module is imported. Part 0 inside the notebook re-creates every partition and asserts
that it equals the saved indices in `splits/`.

## Part 0: fixed partitions (frozen)

`splits/splits_seed708.npz` (index arrays) and `splits/splits_seed708.json` (seed, method,
sizes) are committed and checked by the notebook on every run.

| Partition | Size | Indexes | Per class |
|---|---|---|---|
| `train_idx` | 50,000 | official train array | 5,000 |
| `val_idx` | 10,000 | official train array | 1,000 |
| official test | 10,000 | (not indexed) | 1,000 |
| `train_subset10k_idx` (CPU option 0.1.4, unused) | 10,000 | official train array, subset of `train_idx` | 1,000 |
| `cluster_train_idx` (0.1.5) | 3,000 | official train array, subset of `train_idx` | 300 |
| `cluster_val_idx` (0.1.5) | 1,000 | official train array, subset of `val_idx` | 100 |
| `cluster_test_idx` (0.1.5) | 1,000 | **official test array** | 100 |

- Seed **708** for every partition; method `train_test_split(stratify=labels, random_state=708)`.
- Labels are used only to stratify, never to compute distances, fit centroids or assign clusters.
- The test images are not read until the guarded final cell (`RUN_FINAL_TEST`); the test plan is frozen in `results/test_plan_part1.json` before any test use (R1, R6, R7).
- Preprocessing: pixels / 255, a fixed transform with no fitted statistics; any fitted scaler is fitted on the training partition only (R2).

## Repository layout

```
ecs708_fashion_mnist.ipynb   all code, results and interpretation (task IDs in headings)
README.md                    install, data, run, hardware, assistance
requirements.txt             pinned versions
splits/                      saved split and subset indices (+ seed)
results/                     metric tables, experiment_register.csv (R3), frozen test plan
figures/                     figures produced by the notebook (+ figure index)
```

## Hardware and runtimes

Recorded by the notebook (Part 0 cell and the runtime table at the end of Part 1,
`results/part1/runtimes_part1.csv`).

| Item | Value |
|---|---|
| Machine | Cloud Linux container, Intel Xeon @ 2.10 GHz, 2 logical CPUs, 7 GB RAM, no GPU |
| Python / libraries | Python 3.13, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.5 (PyTorch 2.15 from Part 2) |
| Part 1 fits | logistic regression 7 s to 90 s depending on C (all converge); decision tree 1 s to 19 s; naive Bayes 0.2 s; prediction on 10,000 images under 0.2 s |

## Assistance and external code (acknowledgement)

I used Claude (Anthropic) as a study and coding assistant for this coursework: to explain
concepts, to draft code and notebook text, and to run experiments. I reviewed the code and
results and can explain them. Libraries: NumPy, SciPy, pandas, scikit-learn, PyTorch,
torchvision, matplotlib, seaborn. Dataset: Fashion-MNIST (Zalando Research).
