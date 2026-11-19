# ECS708 Machine Learning: Fashion-MNIST investigation

Coursework for ECS708 (Queen Mary University of London, 2026/27). Main file: `ecs708_fashion_mnist.ipynb`.

## Install and run

```bash
pip install -r requirements.txt
jupyter notebook ecs708_fashion_mnist.ipynb   # run from this folder: Restart and run all (about 15 minutes on CPU)
```

Data: Fashion-MNIST is downloaded automatically with `torchvision` into `data/fashion/` if missing (or copy the four `.gz` files from the [Zalando repository](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)).

## Reproducibility

- Seed 2208; saved split and subset indices in `splits/` (checked by the notebook on every run). All indices point into the official 60,000-image train array except `cluster_test_idx` (official test array).
- Experiment register: `results/experiment_register.csv`; frozen test plan: `results/test_plan_part1.json`. The test set is not used until the final guarded cell.
- Hardware used: Intel Xeon @ 2.10 GHz, 2 CPU cores, 7 GB RAM, no GPU. Runtimes are recorded in the notebook and in `results/part1/runtimes_part1.csv`.

## AI and external code acknowledgement

I used Claude (Anthropic) as a study and coding assistant: to explain concepts, draft code and notebook text, and run experiments. I reviewed the code and results and can explain them. Libraries: NumPy, SciPy, pandas, scikit-learn, PyTorch, torchvision, matplotlib, seaborn. Dataset: Fashion-MNIST (Zalando Research).
