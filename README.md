# ECS708 Machine Learning: Fashion-MNIST investigation

Coursework for ECS708 (Queen Mary University of London, 2026/27). Main file: `ecs708_fashion_mnist.ipynb`.

## Install and run

```bash
pip install -r requirements.txt
jupyter notebook ecs708_fashion_mnist.ipynb   # run from this folder: Restart and run all (about 25 minutes on CPU; the final test evaluation runs only when `RUN_FINAL_TEST = True`, set in the last code section)
```

Data: Fashion-MNIST is downloaded automatically with `torchvision` into `data/fashion/` if missing (or copy the four `.gz` files from the [Zalando repository](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)).

## Reproducibility

- Seed 2208; saved split and subset indices in `splits/` (including the 500-image hierarchical subset `hier_subset_seed2208.npz`) (checked by the notebook on every run). All indices point into the official 60,000-image train array except `cluster_test_idx` (official test array).
- Experiment register: `results/experiment_register.csv` (IDs: `P1-*` classical models, `P2-CAP/OPT-*` capacity and learning rate, `P3-WD/STRAT/SEED-*` regularisation, `P4-K/STAB/INTERP/LINK/HIER-*` clustering); tables per part in `results/part1` to `results/part5`; `P5-FEAT/ENS/CLTEST-*` learned features, ensemble and test clustering; frozen test plan: `results/test_plan.json` (Parts 1, 4, 5.1, 5.2). The test set is not used until the final guarded cell.
- Hardware used: Intel Xeon @ 2.10 GHz, 2 CPU cores, 7 GB RAM, no GPU. Runtimes are recorded in the notebook and in `results/part1/runtimes_part1.csv`.

## AI and external code acknowledgement

I used Claude (Anthropic) as a study and coding assistant: to explain concepts, draft code and notebook text, and run experiments. I reviewed the code and results and can explain them. **How the assistance was checked: [GULCIN: write what you actually did, e.g. re-derived the formulas by hand, re-ran cells and compared outputs, checked numbers against the consistency cell, read every figure, explained each part aloud; delete what is not true].** All numbers quoted in the notebook text are re-computed by an assertion cell at the end of the notebook; text and results were checked against each other, and I wrote or rewrote the explanations in my own words where marked. Libraries: NumPy, SciPy, pandas, scikit-learn, PyTorch, torchvision, matplotlib, seaborn. Dataset: Fashion-MNIST (Zalando Research).
